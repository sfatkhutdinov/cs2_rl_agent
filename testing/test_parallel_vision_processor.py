import unittest
import time
import numpy as np
import threading
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import the module to test
from src.utils.parallel_vision_processor import ParallelVisionProcessor, VisionCache, VisionTask


class MockVisionProcessor:
    """Mock vision processor for testing"""
    
    def __init__(self, processing_time=0.01):
        self.processing_time = processing_time
        self.process_count = 0
        self.last_processed = None
    
    def analyze_scene(self, image, **kwargs):
        self.process_count += 1
        self.last_processed = (image, 'scene_analysis', kwargs)
        time.sleep(self.processing_time)  # Simulate processing time
        return {"scene_type": "test", "objects": ["test_object"], **kwargs}
    
    def detect_ui_elements(self, image, **kwargs):
        self.process_count += 1
        self.last_processed = (image, 'ui_detection', kwargs)
        time.sleep(self.processing_time)  # Simulate processing time
        return {"ui_elements": ["button", "panel"], **kwargs}
    
    def recognize_text(self, image, **kwargs):
        self.process_count += 1
        self.last_processed = (image, 'text_recognition', kwargs)
        time.sleep(self.processing_time)  # Simulate processing time
        return {"text": "test text", "confidence": 0.95, **kwargs}
    
    def process(self, image, task_type, **kwargs):
        self.process_count += 1
        self.last_processed = (image, task_type, kwargs)
        time.sleep(self.processing_time)  # Simulate processing time
        return {"task_type": task_type, "processed": True, **kwargs}


class TestVisionCache(unittest.TestCase):
    """Test the VisionCache class"""
    
    def setUp(self):
        self.cache = VisionCache(max_size=3, ttl=0.5)
        # Create a test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = 255  # Add a white square
    
    def test_cache_put_get(self):
        """Test basic put/get operations"""
        # Initial state
        self.assertEqual(len(self.cache.cache), 0)
        
        # Put an item
        task_type = "test_task"
        metadata = {"param1": "value1"}
        result = {"result": "test_result"}
        
        self.cache.put(self.test_image, task_type, metadata, result)
        
        # Check cache state
        self.assertEqual(len(self.cache.cache), 1)
        
        # Get the item
        cached_result = self.cache.get(self.test_image, task_type, metadata)
        self.assertEqual(cached_result, result)
        
        # Check hit/miss counts
        self.assertEqual(self.cache.cache_hits, 1)
        self.assertEqual(self.cache.cache_misses, 0)
    
    def test_cache_misses(self):
        """Test cache misses"""
        # Try to get non-existent item
        result = self.cache.get(self.test_image, "nonexistent", {})
        self.assertIsNone(result)
        self.assertEqual(self.cache.cache_misses, 1)
        self.assertEqual(self.cache.cache_hits, 0)
        
        # Try with different metadata
        self.cache.put(self.test_image, "test_task", {"param1": "value1"}, {"result": "test"})
        result = self.cache.get(self.test_image, "test_task", {"param1": "different"})
        self.assertIsNone(result)
        self.assertEqual(self.cache.cache_misses, 2)
    
    def test_cache_expiration(self):
        """Test that cache entries expire after TTL"""
        # Put an item
        self.cache.put(self.test_image, "test_task", {}, {"result": "test"})
        
        # Get it immediately (should hit)
        result = self.cache.get(self.test_image, "test_task", {})
        self.assertIsNotNone(result)
        self.assertEqual(self.cache.cache_hits, 1)
        
        # Wait for expiration
        time.sleep(0.6)  # TTL is 0.5
        
        # Try again (should miss)
        result = self.cache.get(self.test_image, "test_task", {})
        self.assertIsNone(result)
        self.assertEqual(self.cache.cache_misses, 1)
    
    def test_cache_max_size(self):
        """Test that cache enforces max size"""
        # Fill the cache (max_size is 3)
        for i in range(3):
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * i
            self.cache.put(test_image, f"task_{i}", {}, {"result": f"test_{i}"})
        
        # Cache should be full
        self.assertEqual(len(self.cache.cache), 3)
        
        # Add one more item
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 99
        self.cache.put(test_image, "task_new", {}, {"result": "test_new"})
        
        # Size should still be 3 (one item was evicted)
        self.assertEqual(len(self.cache.cache), 3)
        
        # Check stats
        stats = self.cache.get_stats()
        self.assertEqual(stats["size"], 3)
        self.assertEqual(stats["max_size"], 3)
    
    def test_cache_clear(self):
        """Test cache clear functionality"""
        # Add some items
        for i in range(3):
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * i
            self.cache.put(test_image, f"task_{i}", {}, {"result": f"test_{i}"})
        
        # Clear the cache
        self.cache.clear()
        
        # Cache should be empty
        self.assertEqual(len(self.cache.cache), 0)


class TestParallelVisionProcessor(unittest.TestCase):
    """Test the ParallelVisionProcessor class"""
    
    def setUp(self):
        self.vision_processor = MockVisionProcessor(processing_time=0.05)
        self.config = {
            "vision_cache_size": 10,
            "vision_cache_ttl": 0.5
        }
        self.logger = Mock()
        self.processor = ParallelVisionProcessor(
            config=self.config,
            vision_processor=self.vision_processor,
            num_workers=2,
            logger=self.logger
        )
        # Create a test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = 255  # Add a white square
    
    def tearDown(self):
        self.processor.shutdown()
    
    def test_task_submission_and_retrieval(self):
        """Test submitting a task and retrieving the result"""
        # Submit a task
        task_id = self.processor.submit_task(
            image=self.test_image,
            task_type="scene_analysis",
            priority=1
        )
        
        # Get the result (with timeout to avoid blocking)
        result = self.processor.get_result(task_id, timeout=1.0)
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertEqual(result["scene_type"], "test")
        self.assertIn("test_object", result["objects"])
        
        # Check that task was processed
        self.assertEqual(self.vision_processor.process_count, 1)
    
    def test_task_callback(self):
        """Test that callbacks are executed"""
        # Set up a callback
        callback_result = {}
        
        def test_callback(result):
            callback_result.update(result)
        
        # Submit a task with the callback
        self.processor.submit_task(
            image=self.test_image,
            task_type="text_recognition",
            callback=test_callback,
            metadata={"language": "english"}
        )
        
        # Wait for callback to be called
        time.sleep(0.2)
        
        # Check callback result
        self.assertEqual(callback_result.get("text"), "test text")
        self.assertEqual(callback_result.get("confidence"), 0.95)
        self.assertEqual(callback_result.get("language"), "english")
    
    def test_task_prioritization(self):
        """Test that higher priority tasks are processed first"""
        # Define a tracking list
        processed_order = []
        
        # Define callbacks that record order
        def make_callback(task_name):
            def callback(_):
                processed_order.append(task_name)
            return callback
        
        # Submit tasks in reverse priority order
        for i in range(3):
            priority = 3 - i  # 3, 2, 1
            self.processor.submit_task(
                image=self.test_image,
                task_type="scene_analysis",
                callback=make_callback(f"task_{priority}"),
                priority=priority
            )
        
        # Wait for all tasks to be processed
        time.sleep(0.5)
        
        # Check processing order (highest priority first)
        self.assertEqual(processed_order[0], "task_3")
        self.assertEqual(processed_order[1], "task_2")
        self.assertEqual(processed_order[2], "task_1")
    
    def test_region_of_interest(self):
        """Test region of interest processing"""
        # Submit task with ROI
        roi = (10, 20, 30, 40)  # x, y, width, height
        self.processor.submit_task(
            image=self.test_image,
            task_type="text_recognition",
            region_of_interest=roi
        )
        
        # Wait for processing
        time.sleep(0.2)
        
        # Check that ROI was applied
        _, task_type, _ = self.vision_processor.last_processed
        self.assertEqual(task_type, "text_recognition")
    
    def test_caching(self):
        """Test that results are cached and reused"""
        # Submit the same task twice
        callback_count = [0]
        
        def callback(_):
            callback_count[0] += 1
        
        # First submission - should be processed
        self.processor.submit_task(
            image=self.test_image,
            task_type="ui_detection",
            callback=callback
        )
        
        # Wait for processing
        time.sleep(0.2)
        
        # Reset process count
        self.vision_processor.process_count = 0
        
        # Second submission with same parameters - should use cache
        self.processor.submit_task(
            image=self.test_image,
            task_type="ui_detection",
            callback=callback
        )
        
        # Wait a bit
        time.sleep(0.2)
        
        # Check that callback was called twice
        self.assertEqual(callback_count[0], 2)
        
        # But vision processor was only called once
        self.assertEqual(self.vision_processor.process_count, 0)
        
        # Check cache metrics
        metrics = self.processor.get_metrics()
        self.assertGreater(metrics["cache_stats"]["hit_rate"], 0)
    
    def test_metrics(self):
        """Test that metrics are tracked correctly"""
        # Submit multiple tasks
        for i in range(5):
            self.processor.submit_task(
                image=self.test_image,
                task_type="scene_analysis"
            )
        
        # Wait for processing
        time.sleep(0.5)
        
        # Get metrics
        metrics = self.processor.get_metrics()
        
        # Check metric properties
        self.assertIn("tasks_processed", metrics)
        self.assertIn("tasks_queued", metrics)
        self.assertIn("queue_depth", metrics)
        self.assertIn("avg_processing_time", metrics)
        self.assertIn("cache_stats", metrics)
        
        # Tasks processed should match
        self.assertEqual(metrics["tasks_processed"], 5)
        self.assertEqual(metrics["tasks_queued"], 5)
    
    def test_concurrency(self):
        """Test that tasks are processed concurrently"""
        # Use a longer processing time
        self.vision_processor.processing_time = 0.2
        
        # Create a new processor with more workers
        processor = ParallelVisionProcessor(
            config=self.config,
            vision_processor=self.vision_processor,
            num_workers=4
        )
        
        start_time = time.time()
        
        # Submit multiple tasks
        for i in range(8):
            processor.submit_task(
                image=self.test_image,
                task_type="scene_analysis"
            )
        
        # Wait for queue to empty
        while not processor.task_queue.empty():
            time.sleep(0.1)
        
        # Give workers time to complete processing
        time.sleep(0.3)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # If tasks were processed sequentially, it would take 8 * 0.2 = 1.6 seconds
        # With 4 workers, it should be much faster
        self.assertLess(duration, 1.0)
        
        # Clean up
        processor.shutdown()
    
    def test_shutdown(self):
        """Test that shutdown stops workers properly"""
        # Create a new processor
        processor = ParallelVisionProcessor(
            config=self.config,
            vision_processor=self.vision_processor,
            num_workers=2
        )
        
        # Submit some tasks
        for i in range(3):
            processor.submit_task(
                image=self.test_image,
                task_type="scene_analysis"
            )
        
        # Shutdown
        processor.shutdown()
        
        # All workers should be stopped
        for worker in processor.workers:
            self.assertFalse(worker.is_alive())
        
        # Check that running flag is set
        self.assertFalse(processor.running)


if __name__ == "__main__":
    unittest.main() 