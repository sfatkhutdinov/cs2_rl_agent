import unittest
import os
import sys
import time
import threading
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from queue import Queue

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.parallel_vision_processor import ParallelVisionProcessor, VisionTask, VisionCache


class TestVisionCache(unittest.TestCase):
    """Test the vision cache functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = VisionCache(max_size=10, ttl=0.5)
        
        # Create test data
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[30:70, 30:70] = 255  # White square in the middle
        
        self.test_task_type = "object_detection"
        self.test_metadata = {"threshold": 0.5, "min_size": 10}
        self.test_roi = None
        self.test_result = {"objects": [{"class": "square", "confidence": 0.95, "bbox": [30, 30, 70, 70]}]}
    
    def test_cache_key_generation(self):
        """Test that cache keys are generated consistently."""
        key1 = self.cache._generate_cache_key(self.test_image, self.test_task_type, self.test_roi, self.test_metadata)
        key2 = self.cache._generate_cache_key(self.test_image, self.test_task_type, self.test_roi, self.test_metadata)
        
        self.assertEqual(key1, key2, "Cache keys should be consistent for the same inputs")
        
        # Test with different parameters
        different_metadata = {"threshold": 0.6, "min_size": 10}
        key3 = self.cache._generate_cache_key(self.test_image, self.test_task_type, self.test_roi, different_metadata)
        
        self.assertNotEqual(key1, key3, "Cache keys should differ for different parameters")
    
    def test_cache_store_and_retrieve(self):
        """Test storing and retrieving results from the cache."""
        # Store a result using put method
        self.cache.put(self.test_image, self.test_task_type, self.test_result, self.test_roi, self.test_metadata)
        
        # Retrieve the result
        cached_result = self.cache.get(self.test_image, self.test_task_type, self.test_roi, self.test_metadata)
        
        self.assertEqual(cached_result, self.test_result, "Retrieved result should match stored result")
        self.assertEqual(self.cache.stats["hits"], 1, "Cache hit count should be incremented")
        self.assertEqual(self.cache.stats["size"], 1, "Cache size should be 1")
    
    def test_cache_miss(self):
        """Test cache miss behavior."""
        # Try to retrieve a non-existent result
        cached_result = self.cache.get(self.test_image, self.test_task_type, self.test_roi, self.test_metadata)
        
        self.assertIsNone(cached_result, "Non-existent result should return None")
        self.assertEqual(self.cache.stats["misses"], 1, "Cache miss count should be incremented")
        self.assertEqual(self.cache.stats["size"], 0, "Cache size should be 0")
    
    def test_cache_eviction(self):
        """Test cache eviction when max size is reached."""
        # Fill the cache beyond its capacity
        for i in range(15):  # Max size is 10
            image = np.ones((10, 10, 3), dtype=np.uint8) * i
            self.cache.put(image, self.test_task_type, {"value": i}, self.test_roi, self.test_metadata)
        
        self.assertLessEqual(self.cache.stats["size"], 10, "Cache size should not exceed max_size")
        self.assertGreaterEqual(self.cache.stats["evictions"], 5, "At least 5 items should have been evicted")
    
    def test_cache_expiration(self):
        """Test that entries expire after TTL."""
        # Store a result
        self.cache.put(self.test_image, self.test_task_type, self.test_result, self.test_roi, self.test_metadata)
        
        # Verify it's there
        self.assertIsNotNone(self.cache.get(self.test_image, self.test_task_type, self.test_roi, self.test_metadata))
        
        # Wait for expiration
        time.sleep(0.6)  # TTL is 0.5
        
        # Try again after expiration
        cached_result = self.cache.get(self.test_image, self.test_task_type, self.test_roi, self.test_metadata)
        self.assertIsNone(cached_result, "Expired result should be removed")
    
    def test_cache_clear(self):
        """Test clearing the cache."""
        # Store some results
        for i in range(5):
            image = np.ones((10, 10, 3), dtype=np.uint8) * i
            self.cache.put(image, self.test_task_type, {"value": i}, self.test_roi, self.test_metadata)
        
        self.assertEqual(self.cache.stats["size"], 5, "Cache should contain 5 items")
        
        # Clear the cache
        self.cache.clear()
        
        self.assertEqual(self.cache.stats["size"], 0, "Cache should be empty after clearing")
        self.assertEqual(len(self.cache.cache), 0, "Cache dictionary should be empty")


class TestVisionTask(unittest.TestCase):
    """Test the vision task functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_task_type = "object_detection"
        self.test_metadata = {"threshold": 0.5}
        self.test_callback = Mock()
        self.test_roi = (10, 10, 90, 90)
    
    def test_task_creation(self):
        """Test creating a vision task."""
        task = VisionTask(
            image=self.test_image,
            task_type=self.test_task_type,
            callback=self.test_callback,
            priority=2,
            region_of_interest=self.test_roi,
            metadata=self.test_metadata
        )
        
        self.assertEqual(task.image.shape, self.test_image.shape, "Task image should match input image")
        self.assertEqual(task.task_type, self.test_task_type, "Task type should match input")
        self.assertEqual(task.metadata, self.test_metadata, "Task metadata should match input")
        self.assertEqual(task.callback, self.test_callback, "Task callback should match input")
        self.assertEqual(task.priority, 2, "Task priority should match input")
        self.assertEqual(task.region_of_interest, self.test_roi, "Task ROI should match input")
        self.assertIsNotNone(task.task_id, "Task ID should be generated")
        self.assertIsNotNone(task.timestamp, "Task timestamp should be set")
    
    def test_task_id_uniqueness(self):
        """Test that task IDs are unique."""
        # Create tasks with different timestamps to ensure unique IDs
        task1 = VisionTask(self.test_image, self.test_task_type)
        time.sleep(0.001)  # Ensure different timestamp
        task2 = VisionTask(self.test_image, self.test_task_type)
        
        self.assertNotEqual(task1.task_id, task2.task_id, "Task IDs should be unique")


class MockVisionProcessor:
    """Mock vision processor for testing."""
    
    def analyze_scene(self, image, **kwargs):
        return {"scene_type": "test", "objects": ["test_object"]}
    
    def detect_ui_elements(self, image, **kwargs):
        return {"ui_elements": ["button", "panel"]}
    
    def recognize_text(self, image, **kwargs):
        return {"text": "test text", "confidence": 0.95}
    
    def process(self, image, task_type, **kwargs):
        return {"task_type": task_type, "processed": True}


class TestParallelVisionProcessor(unittest.TestCase):
    """Test the parallel vision processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock vision processor
        self.mock_vision_processor = MockVisionProcessor()
        
        self.config = {
            "min_workers": 2,
            "max_workers": 4,
            "queue_size": 100,
            "cache_size": 50,
            "adaptive_workers": True,
            "worker_cpu_threshold": 80.0
        }
        
        # Create test data
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[30:70, 30:70] = 255  # White square in the middle
        
        # Create a mock for the CPU usage function
        self.cpu_usage_patcher = patch('psutil.cpu_percent', return_value=50.0)
        self.mock_cpu_usage = self.cpu_usage_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.cpu_usage_patcher.stop()
    
    def test_init(self):
        """Test initialization of the processor."""
        processor = ParallelVisionProcessor(self.mock_vision_processor, self.config)
        
        self.assertEqual(processor.min_workers, 2, "Min workers should match config")
        self.assertEqual(processor.max_workers, 4, "Max workers should match config")
        self.assertEqual(processor.queue_size, 100, "Queue size should match config")
        self.assertTrue(processor.adaptive_workers, "Adaptive workers should be enabled")
        self.assertEqual(processor.worker_cpu_threshold, 80.0, "CPU threshold should match config")
        self.assertIsInstance(processor.task_queue, Queue, "Task queue should be initialized")
        self.assertIsInstance(processor.cache, VisionCache, "Cache should be initialized")
    
    def test_start_stop(self):
        """Test starting and stopping the processor."""
        processor = ParallelVisionProcessor(self.mock_vision_processor, self.config)
        
        # Initial state - workers should be started in the constructor
        self.assertTrue(len(processor.workers) > 0, 
                         "Processor should have workers started initially")
        self.assertEqual(processor.active_workers, processor.min_workers,
                         "Active workers should match min_workers initially")
        
        # Shutdown the processor
        processor.shutdown()
        
        # Check shutdown state
        self.assertTrue(processor.shutdown_event.is_set(),
                        "Shutdown event should be set after shutdown")
        
        # Wait a moment for workers to process the shutdown
        time.sleep(0.1)
        
        # All workers should have received sentinel values
        self.assertEqual(processor.task_queue.qsize(), 0,
                         "Task queue should be empty after shutdown")


if __name__ == "__main__":
    unittest.main() 