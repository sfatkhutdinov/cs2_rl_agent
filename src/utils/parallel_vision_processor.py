import os
import time
import threading
import queue
import logging
import psutil
import hashlib
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from datetime import datetime
import numpy as np
import cv2
from functools import lru_cache
from dataclasses import dataclass, field

from src.utils.logger import Logger

@dataclass
class VisionTask:
    """
    Represents a vision processing task with metadata for prioritization and tracking.
    """
    image: np.ndarray  # The image to process
    task_type: str  # Type of vision task (e.g., 'scene_analysis', 'ui_detection', 'text_recognition')
    priority: int = 1  # Priority level (higher values = higher priority)
    callback: Optional[Callable[[Dict[str, Any]], None]] = None  # Callback function to handle the result
    timestamp: float = field(default_factory=time.time)  # When the task was created
    task_id: str = ""  # Unique identifier for this task
    region_of_interest: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height) or None for full image
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional task metadata
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"{self.task_type}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


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
        self.cache = {}  # Maps cache key to (result, timestamp)
        self.cache_lock = threading.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }
    
    def _generate_cache_key(self, image: np.ndarray, task_type: str, region_of_interest: Optional[Tuple[int, int, int, int]], metadata: Dict[str, Any]) -> str:
        """
        Generate a cache key for the given task parameters.
        
        Args:
            image: The image to process
            task_type: Type of vision task
            region_of_interest: Region of interest in the image
            metadata: Additional task metadata
            
        Returns:
            Cache key string
        """
        # Extract the region of interest if specified
        if region_of_interest:
            x, y, w, h = region_of_interest
            if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
                roi_image = image[y:y+h, x:x+w]
            else:
                roi_image = image
        else:
            roi_image = image
        
        # Downsample image for faster hashing
        scale_factor = 0.25
        small_image = cv2.resize(roi_image, (0, 0), fx=scale_factor, fy=scale_factor)
        
        # Compute image hash
        img_hash = hashlib.md5(small_image.tobytes()).hexdigest()
        
        # Combine with task type and metadata
        metadata_str = str(sorted(metadata.items())) if metadata else ""
        roi_str = str(region_of_interest) if region_of_interest else ""
        
        return f"{img_hash}_{task_type}_{roi_str}_{metadata_str}"
    
    def get(self, image: np.ndarray, task_type: str, region_of_interest: Optional[Tuple[int, int, int, int]] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Get a cached result if available.
        
        Args:
            image: The image to process
            task_type: Type of vision task
            region_of_interest: Region of interest in the image
            metadata: Additional task metadata
            
        Returns:
            Cached result or None if not in cache
        """
        if metadata is None:
            metadata = {}
        
        key = self._generate_cache_key(image, task_type, region_of_interest, metadata)
        
        with self.cache_lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                
                # Check if the entry is still valid
                if time.time() - timestamp <= self.ttl:
                    self.stats["hits"] += 1
                    return result
                else:
                    # Remove expired entry
                    del self.cache[key]
                    self.stats["evictions"] += 1
                    self.stats["size"] = len(self.cache)
            
            self.stats["misses"] += 1
            return None
    
    def put(self, image: np.ndarray, task_type: str, result: Dict[str, Any], region_of_interest: Optional[Tuple[int, int, int, int]] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a result in the cache.
        
        Args:
            image: The image that was processed
            task_type: Type of vision task
            result: The processing result
            region_of_interest: Region of interest in the image
            metadata: Additional task metadata
        """
        if metadata is None:
            metadata = {}
        
        key = self._generate_cache_key(image, task_type, region_of_interest, metadata)
        
        with self.cache_lock:
            # Evict entries if cache is full
            if len(self.cache) >= self.max_size:
                # Find the oldest entry
                oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
                del self.cache[oldest_key]
                self.stats["evictions"] += 1
            
            # Store the new entry
            self.cache[key] = (result, time.time())
            self.stats["size"] = len(self.cache)
    
    def clear(self) -> None:
        """Clear the cache."""
        with self.cache_lock:
            self.cache.clear()
            self.stats["size"] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        with self.cache_lock:
            stats = self.stats.copy()
            total_requests = stats["hits"] + stats["misses"]
            stats["hit_rate"] = stats["hits"] / total_requests if total_requests > 0 else 0.0
            return stats


class ParallelVisionProcessor:
    """
    Parallel processor for vision tasks.
    
    This class manages a pool of worker threads to process vision tasks in parallel,
    with priority queuing and caching of results.
    """
    
    def __init__(self, vision_processor: Any, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the parallel vision processor.
        
        Args:
            vision_processor: Vision processor object with processing methods
            config: Configuration dictionary
            logger: Logger instance for metrics and debugging
        """
        self.vision_processor = vision_processor
        self.config = config
        self.logger = logger or logging.getLogger("ParallelVisionProcessor")
        
        # Configuration
        self.min_workers = config.get("min_workers", 2)
        self.max_workers = config.get("max_workers", 8)
        self.queue_size = config.get("queue_size", 100)
        self.cache_size = config.get("cache_size", 200)
        self.cache_ttl = config.get("cache_ttl", 30.0)  # seconds
        self.adaptive_workers = config.get("adaptive_workers", True)
        self.worker_cpu_threshold = config.get("worker_cpu_threshold", 70.0)  # percent
        self.worker_adjustment_interval = config.get("worker_adjustment_interval", 5.0)  # seconds
        
        # Initialize cache
        self.cache = VisionCache(max_size=self.cache_size, ttl=self.cache_ttl)
        
        # Initialize task queue with priority
        self.task_queue = queue.PriorityQueue(maxsize=self.queue_size)
        
        # Initialize worker threads
        self.workers = []
        self.active_workers = self.min_workers
        self.worker_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Initialize metrics
        self.metrics = {
            "tasks_queued": 0,
            "tasks_processed": 0,
            "queue_depth": 0,
            "processing_times": [],
            "worker_count": self.active_workers,
            "cpu_usage": 0.0
        }
        self.metrics_lock = threading.Lock()
        
        # Start worker threads
        self._start_workers(self.min_workers)
        
        # Start adaptive worker manager if enabled
        if self.adaptive_workers:
            self.adaptive_thread = threading.Thread(target=self._adaptive_worker_manager, daemon=True)
            self.adaptive_thread.start()
    
    def _start_workers(self, count: int) -> None:
        """
        Start worker threads.
        
        Args:
            count: Number of worker threads to start
        """
        with self.worker_lock:
            for _ in range(count):
                worker = threading.Thread(target=self._worker_loop, daemon=True)
                worker.start()
                self.workers.append(worker)
            
            self.active_workers = len(self.workers)
            
            with self.metrics_lock:
                self.metrics["worker_count"] = self.active_workers
    
    def _stop_workers(self, count: int) -> None:
        """
        Stop worker threads.
        
        Args:
            count: Number of worker threads to stop
        """
        with self.worker_lock:
            # Can't reduce below min_workers
            count = min(count, len(self.workers) - self.min_workers)
            if count <= 0:
                return
            
            # Add sentinel values to stop workers
            for _ in range(count):
                self.task_queue.put((0, None))  # Sentinel value
            
            # Update active worker count
            self.active_workers = len(self.workers) - count
            
            with self.metrics_lock:
                self.metrics["worker_count"] = self.active_workers
    
    def _adaptive_worker_manager(self) -> None:
        """
        Adaptively adjust the number of worker threads based on system load.
        """
        last_adjustment_time = time.time()
        
        while not self.shutdown_event.is_set():
            current_time = time.time()
            
            # Check if it's time to adjust workers
            if current_time - last_adjustment_time >= self.worker_adjustment_interval:
                # Get current CPU usage
                cpu_usage = psutil.cpu_percent(interval=0.1)
                
                with self.metrics_lock:
                    self.metrics["cpu_usage"] = cpu_usage
                
                # Get current queue depth
                queue_depth = self.task_queue.qsize()
                
                with self.metrics_lock:
                    self.metrics["queue_depth"] = queue_depth
                
                # Adjust workers based on CPU usage and queue depth
                with self.worker_lock:
                    current_workers = self.active_workers
                
                if cpu_usage > self.worker_cpu_threshold and queue_depth > 0:
                    # System is busy and we have tasks, reduce workers
                    if current_workers > self.min_workers:
                        self._stop_workers(1)
                        self.logger.debug(f"Reduced workers to {self.active_workers} due to high CPU usage ({cpu_usage:.1f}%)")
                elif queue_depth > current_workers * 2:
                    # We have more tasks than workers can handle, add workers
                    if current_workers < self.max_workers:
                        self._start_workers(1)
                        self.logger.debug(f"Increased workers to {self.active_workers} due to queue depth ({queue_depth})")
                elif queue_depth == 0 and current_workers > self.min_workers:
                    # No tasks, reduce to minimum
                    self._stop_workers(current_workers - self.min_workers)
                    self.logger.debug(f"Reduced workers to minimum ({self.min_workers}) due to empty queue")
                
                last_adjustment_time = current_time
            
            # Sleep to avoid busy waiting
            time.sleep(0.1)
    
    def _worker_loop(self) -> None:
        """
        Main loop for worker threads.
        """
        while not self.shutdown_event.is_set():
            try:
                # Get a task from the queue
                priority, task = self.task_queue.get(timeout=0.1)
                
                # Check for sentinel value
                if task is None:
                    self.task_queue.task_done()
                    break
                
                # Process the task
                self._process_task(task)
                
                # Mark the task as done
                self.task_queue.task_done()
                
            except queue.Empty:
                # No tasks available, continue
                continue
            except Exception as e:
                self.logger.error(f"Error in worker thread: {str(e)}")
    
    def _process_task(self, task: VisionTask) -> None:
        """
        Process a vision task.
        
        Args:
            task: The vision task to process
        """
        # Check cache first
        cached_result = self.cache.get(
            task.image, 
            task.task_type, 
            task.region_of_interest, 
            task.metadata
        )
        
        if cached_result:
            # Use cached result
            if task.callback:
                task.callback(cached_result)
            return
        
        # Process the task
        start_time = time.time()
        
        try:
            # Extract region of interest if specified
            if task.region_of_interest:
                x, y, w, h = task.region_of_interest
                if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= task.image.shape[1] and y + h <= task.image.shape[0]:
                    image = task.image[y:y+h, x:x+w]
                else:
                    image = task.image
            else:
                image = task.image
            
            # Call the appropriate method based on task type
            if task.task_type == "scene_analysis":
                result = self.vision_processor.analyze_scene(image, **task.metadata)
            elif task.task_type == "ui_detection":
                result = self.vision_processor.detect_ui_elements(image, **task.metadata)
            elif task.task_type == "text_recognition":
                result = self.vision_processor.recognize_text(image, **task.metadata)
            else:
                # Generic processing
                result = self.vision_processor.process(image, task.task_type, **task.metadata)
            
            # Add task metadata to result
            result["task_id"] = task.task_id
            result["task_type"] = task.task_type
            result["processing_time"] = time.time() - start_time
            
            # Cache the result
            self.cache.put(
                task.image, 
                task.task_type, 
                result, 
                task.region_of_interest, 
                task.metadata
            )
            
            # Update metrics
            with self.metrics_lock:
                self.metrics["tasks_processed"] += 1
                self.metrics["processing_times"].append(result["processing_time"])
                # Keep only the last 100 processing times
                if len(self.metrics["processing_times"]) > 100:
                    self.metrics["processing_times"] = self.metrics["processing_times"][-100:]
            
            # Call the callback with the result
            if task.callback:
                task.callback(result)
                
        except Exception as e:
            self.logger.error(f"Error processing vision task: {str(e)}")
            
            # Call the callback with an error result
            if task.callback:
                error_result = {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "error": str(e),
                    "success": False
                }
                task.callback(error_result)
    
    def submit_task(self, image: np.ndarray, task_type: str, callback: Optional[Callable[[Dict[str, Any]], None]] = None, 
                   priority: int = 1, region_of_interest: Optional[Tuple[int, int, int, int]] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a vision task for processing.
        
        Args:
            image: The image to process
            task_type: Type of vision task
            callback: Callback function to handle the result
            priority: Priority level (higher values = higher priority)
            region_of_interest: Region of interest in the image (x, y, width, height)
            metadata: Additional task metadata
            
        Returns:
            Task ID
        """
        if metadata is None:
            metadata = {}
        
        # Create a task
        task = VisionTask(
            image=image,
            task_type=task_type,
            callback=callback,
            priority=priority,
            region_of_interest=region_of_interest,
            metadata=metadata
        )
        
        # Check cache first for immediate response
        cached_result = self.cache.get(
            task.image, 
            task.task_type, 
            task.region_of_interest, 
            task.metadata
        )
        
        if cached_result:
            # Use cached result
            if task.callback:
                task.callback(cached_result)
            return task.task_id
        
        # Update metrics
        with self.metrics_lock:
            self.metrics["tasks_queued"] += 1
        
        # Add to queue with priority (negative because PriorityQueue is min-heap)
        self.task_queue.put((-priority, task))
        
        # Log metrics if available
        self._log_metrics()
        
        return task.task_id
    
    def _log_metrics(self) -> None:
        """Log performance metrics."""
        # Only log occasionally to avoid spam
        if self.metrics["tasks_processed"] % 10 == 0:
            with self.metrics_lock:
                metrics = {
                    "tasks_queued": self.metrics["tasks_queued"],
                    "tasks_processed": self.metrics["tasks_processed"],
                    "queue_depth": self.task_queue.qsize(),
                    "worker_count": self.active_workers,
                    "avg_processing_time": sum(self.metrics["processing_times"]) / len(self.metrics["processing_times"]) if self.metrics["processing_times"] else 0,
                    "cache_stats": self.cache.get_stats(),
                    "cpu_usage": self.metrics["cpu_usage"]
                }
            
            self.logger.log_info(f"Vision metrics: {metrics}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        with self.metrics_lock:
            metrics = {
                "tasks_queued": self.metrics["tasks_queued"],
                "tasks_processed": self.metrics["tasks_processed"],
                "queue_depth": self.task_queue.qsize(),
                "worker_count": self.active_workers,
                "avg_processing_time": sum(self.metrics["processing_times"]) / len(self.metrics["processing_times"]) if self.metrics["processing_times"] else 0,
                "cache_stats": self.cache.get_stats(),
                "cpu_usage": self.metrics["cpu_usage"]
            }
        
        return metrics
    
    def shutdown(self) -> None:
        """Shutdown the processor and all worker threads."""
        self.logger.info("Shutting down parallel vision processor...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Add sentinel values to stop all workers
        for _ in range(len(self.workers)):
            self.task_queue.put((0, None))  # Sentinel value
        
        # Wait for all workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
        
        self.logger.info("Parallel vision processor shutdown complete.") 