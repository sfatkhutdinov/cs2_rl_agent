import os
import time
import threading
import queue
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from datetime import datetime
import numpy as np
import cv2
from functools import lru_cache
from dataclasses import dataclass

from src.utils.logger import Logger

@dataclass
class VisionTask:
    """
    Represents a vision processing task with metadata for prioritization and tracking.
    """
    image: np.ndarray  # The image to process
    task_type: str  # Type of vision task (e.g., 'scene_analysis', 'ui_detection', 'text_recognition')
    priority: int  # Priority level (higher values = higher priority)
    callback: Callable[[Dict[str, Any]], None]  # Callback function to handle the result
    timestamp: float = 0.0  # When the task was created
    task_id: str = ""  # Unique identifier for this task
    region_of_interest: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height) or None for full image
    metadata: Dict[str, Any] = None  # Additional task metadata
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if not self.task_id:
            self.task_id = f"{self.task_type}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        if self.metadata is None:
            self.metadata = {}


class VisionCache:
    """
    Cache for vision processing results to avoid redundant computation.
    """
    def __init__(self, max_size: int = 100, ttl: float = 60.0):
        """
        Initialize the vision cache.
        
        Args:
            max_size: Maximum number of entries in the cache
            ttl: Time-to-live in seconds for cache entries
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Dict[str, Any], float]] = {}  # {key: (result, timestamp)}
        self.cache_lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _generate_key(self, image: np.ndarray, task_type: str, metadata: Dict[str, Any]) -> str:
        """
        Generate a cache key based on image content and task parameters.
        
        Args:
            image: The image data
            task_type: Type of vision task
            metadata: Additional task metadata
            
        Returns:
            A string key for cache lookup
        """
        # Create a downsampled version of the image for fingerprinting
        downsampled = cv2.resize(image, (32, 32))
        img_hash = hash(downsampled.tobytes())
        
        # Create a string representation of metadata
        meta_str = "_".join(f"{k}:{v}" for k, v in sorted(metadata.items()))
        
        return f"{task_type}_{img_hash}_{meta_str}"
    
    def get(self, image: np.ndarray, task_type: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a result from the cache if available and not expired.
        
        Args:
            image: The image data
            task_type: Type of vision task
            metadata: Additional task metadata
            
        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(image, task_type, metadata)
        
        with self.cache_lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                # Check if entry is still valid
                if time.time() - timestamp <= self.ttl:
                    self.cache_hits += 1
                    return result
                else:
                    # Remove expired entry
                    del self.cache[key]
            
            self.cache_misses += 1
            return None
    
    def put(self, image: np.ndarray, task_type: str, metadata: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Store a result in the cache.
        
        Args:
            image: The image data
            task_type: Type of vision task
            metadata: Additional task metadata
            result: The result to cache
        """
        key = self._generate_key(image, task_type, metadata)
        
        with self.cache_lock:
            # If cache is full, remove oldest entry
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (result, time.time())
    
    def clear(self) -> None:
        """Clear the cache."""
        with self.cache_lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self.cache_lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "ttl": self.ttl
            }


class ParallelVisionProcessor:
    """
    Manages parallel vision processing tasks using a thread pool and priority queue.
    """
    def __init__(self, 
                 config: Dict[str, Any], 
                 vision_processor: Any, 
                 num_workers: int = 4,
                 logger: Optional[Logger] = None):
        """
        Initialize the parallel vision processor.
        
        Args:
            config: Configuration dictionary
            vision_processor: The vision model or processor to use
            num_workers: Number of worker threads
            logger: Logger instance for metrics and debugging
        """
        self.config = config
        self.vision_processor = vision_processor
        self.num_workers = num_workers
        self.logger = logger
        
        # Set up queuing system
        self.task_queue = queue.PriorityQueue()  # (priority, task_id, VisionTask)
        self.results: Dict[str, Dict[str, Any]] = {}
        self.results_lock = threading.Lock()
        
        # Set up workers
        self.workers: List[threading.Thread] = []
        self.running = True
        self.tasks_processed = 0
        self.tasks_queued = 0
        self.processing_times: List[float] = []
        
        # Set up cache
        cache_size = config.get("vision_cache_size", 100)
        cache_ttl = config.get("vision_cache_ttl", 60.0)
        self.cache = VisionCache(max_size=cache_size, ttl=cache_ttl)
        
        # Start worker threads
        self._start_workers()
        
        if self.logger:
            self.logger.log_info(f"Started ParallelVisionProcessor with {num_workers} workers")
    
    def _start_workers(self) -> None:
        """Start the worker threads."""
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"vision_worker_{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self) -> None:
        """Main worker thread loop that processes tasks from the queue."""
        while self.running:
            try:
                # Get the next task (block for up to 1 second)
                try:
                    _, _, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the task
                start_time = time.time()
                
                # Check cache first
                cached_result = self.cache.get(task.image, task.task_type, task.metadata)
                if cached_result:
                    result = cached_result
                else:
                    # Process the image
                    result = self._process_vision_task(task)
                    # Cache the result
                    self.cache.put(task.image, task.task_type, task.metadata, result)
                
                processing_time = time.time() - start_time
                
                # Store the result
                with self.results_lock:
                    self.results[task.task_id] = result
                    self.tasks_processed += 1
                    self.processing_times.append(processing_time)
                    
                    # Keep only the last 1000 processing times for statistics
                    if len(self.processing_times) > 1000:
                        self.processing_times = self.processing_times[-1000:]
                
                # Call the callback with the result
                if task.callback:
                    task.callback(result)
                
                # Mark the task as done
                self.task_queue.task_done()
                
                # Log metrics if available
                if self.logger and self.tasks_processed % 100 == 0:
                    self._log_metrics()
                    
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Error in vision worker: {str(e)}")
                else:
                    logging.error(f"Error in vision worker: {str(e)}")
    
    def _process_vision_task(self, task: VisionTask) -> Dict[str, Any]:
        """
        Process a single vision task.
        
        Args:
            task: The vision task to process
            
        Returns:
            Dictionary containing the processing results
        """
        # Extract region of interest if specified
        image = task.image
        if task.region_of_interest:
            x, y, w, h = task.region_of_interest
            image = image[y:y+h, x:x+w]
        
        # Process based on task type
        if task.task_type == "scene_analysis":
            return self.vision_processor.analyze_scene(image, **task.metadata)
        elif task.task_type == "ui_detection":
            return self.vision_processor.detect_ui_elements(image, **task.metadata)
        elif task.task_type == "text_recognition":
            return self.vision_processor.recognize_text(image, **task.metadata)
        else:
            # Generic processing
            return self.vision_processor.process(image, task.task_type, **task.metadata)
    
    def submit_task(self, 
                   image: np.ndarray, 
                   task_type: str, 
                   callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                   priority: int = 0,
                   region_of_interest: Optional[Tuple[int, int, int, int]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a vision processing task.
        
        Args:
            image: The image to process
            task_type: Type of vision task
            callback: Function to call with the result
            priority: Priority level (higher = higher priority)
            region_of_interest: Optional region to process (x, y, width, height)
            metadata: Additional task parameters
            
        Returns:
            Task ID that can be used to retrieve the result
        """
        task = VisionTask(
            image=image,
            task_type=task_type,
            priority=priority,
            callback=callback,
            region_of_interest=region_of_interest,
            metadata=metadata or {}
        )
        
        # Check cache before queuing
        cached_result = self.cache.get(image, task_type, task.metadata)
        if cached_result:
            # Store the result immediately
            with self.results_lock:
                self.results[task.task_id] = cached_result
            
            # Call the callback if provided
            if callback:
                callback(cached_result)
            
            return task.task_id
        
        # Otherwise, queue the task
        self.task_queue.put((-priority, time.time(), task))  # Negative priority for correct ordering
        
        with self.results_lock:
            self.tasks_queued += 1
        
        return task.task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get the result for a specific task.
        
        Args:
            task_id: The task ID
            timeout: Maximum time to wait for the result (None = wait forever)
            
        Returns:
            The task result or None if not available
        """
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            with self.results_lock:
                if task_id in self.results:
                    return self.results[task_id]
            
            # Wait a bit before checking again
            time.sleep(0.01)
        
        return None
    
    def _log_metrics(self) -> None:
        """Log performance metrics."""
        if not self.logger:
            return
        
        with self.results_lock:
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
            metrics = {
                "vision_tasks_processed": self.tasks_processed,
                "vision_tasks_queued": self.tasks_queued,
                "vision_queue_depth": self.task_queue.qsize(),
                "vision_avg_processing_time": avg_processing_time,
                "vision_cache_hit_rate": self.cache.get_stats()["hit_rate"]
            }
            
            self.logger.log_info(f"Vision metrics: {metrics}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        with self.results_lock:
            avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
            return {
                "tasks_processed": self.tasks_processed,
                "tasks_queued": self.tasks_queued,
                "queue_depth": self.task_queue.qsize(),
                "avg_processing_time": avg_processing_time,
                "cache_stats": self.cache.get_stats(),
                "num_workers": self.num_workers
            }
    
    def shutdown(self) -> None:
        """Shutdown the processor and stop all workers."""
        self.running = False
        
        # Wait for all workers to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
        
        if self.logger:
            self.logger.log_info("ParallelVisionProcessor shutdown complete") 