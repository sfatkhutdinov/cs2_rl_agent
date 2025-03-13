# Performance Testing

Performance testing is a critical aspect of the CS2 reinforcement learning agent's testing infrastructure, focusing on measuring and optimizing the agent's speed, resource utilization, and scalability. The codebase implements several performance testing approaches to identify bottlenecks and ensure the agent can operate efficiently in real-time game scenarios.

## Performance Metrics Tracking

The codebase tracks several key performance metrics during testing:

```python
class PerformanceTracker:
    """Tracks performance metrics during agent operation."""
    
    def __init__(self):
        self.metrics = {
            "step_time": [],
            "observation_time": [],
            "decision_time": [],
            "action_time": [],
            "memory_usage": [],
            "gpu_memory_usage": [],
            "fps": []
        }
        self.start_time = time.time()
        
    def start_timing(self):
        """Start timing an operation."""
        self.current_time = time.time()
        
    def record_step_time(self):
        """Record the time for a complete step."""
        self.metrics["step_time"].append(time.time() - self.current_time)
        
    def record_observation_time(self):
        """Record the time to process an observation."""
        self.metrics["observation_time"].append(time.time() - self.current_time)
        self.start_timing()
        
    def record_decision_time(self):
        """Record the time to make a decision."""
        self.metrics["decision_time"].append(time.time() - self.current_time)
        self.start_timing()
        
    def record_action_time(self):
        """Record the time to execute an action."""
        self.metrics["action_time"].append(time.time() - self.current_time)
        
    def record_memory_usage(self):
        """Record current memory usage."""
        import psutil
        process = psutil.Process()
        self.metrics["memory_usage"].append(process.memory_info().rss / (1024 * 1024))  # MB
        
        # Record GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                self.metrics["gpu_memory_usage"].append(torch.cuda.memory_allocated() / (1024 * 1024))  # MB
        except (ImportError, AttributeError):
            pass
        
    def record_fps(self):
        """Record frames per second."""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.metrics["fps"].append(len(self.metrics["step_time"]) / elapsed)
        
    def get_summary(self):
        """Get a summary of performance metrics."""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_min"] = min(values)
                
        return summary
```

This performance tracking is integrated into the agent evaluation and testing processes:

```python
def evaluate_agent_performance(agent, env, episodes=10):
    """Evaluate agent performance with detailed metrics."""
    performance_tracker = PerformanceTracker()
    
    for episode in range(episodes):
        observation, info = env.reset()
        done = False
        
        while not done:
            # Track step time
            performance_tracker.start_timing()
            
            # Track observation processing time
            performance_tracker.record_observation_time()
            
            # Track decision time
            action = agent.select_action(observation)
            performance_tracker.record_decision_time()
            
            # Track action execution time
            next_observation, reward, terminated, truncated, info = env.step(action)
            performance_tracker.record_action_time()
            
            # Record complete step time
            performance_tracker.record_step_time()
            
            # Record memory usage periodically
            if np.random.random() < 0.1:  # Every 10 steps on average
                performance_tracker.record_memory_usage()
            
            done = terminated or truncated
            observation = next_observation
        
        # Record FPS at the end of each episode
        performance_tracker.record_fps()
    
    # Return performance summary
    return performance_tracker.get_summary()
```

## Performance Benchmarking

The codebase includes benchmark tests that measure performance across different configurations:

```python
def run_performance_benchmark(config_variations, episodes=5, steps_per_episode=100):
    """Run performance benchmark across different configurations."""
    results = {}
    
    for config_name, config in config_variations.items():
        print(f"Benchmarking configuration: {config_name}")
        
        # Create environment and agent
        env = CS2Environment(config)
        agent = create_agent(config)
        
        # Run benchmark
        metrics = evaluate_agent_performance(agent, env, episodes)
        
        # Store results
        results[config_name] = metrics
        
        # Clean up
        env.close()
        
    return results
```

This approach enables comparative analysis of different configurations:

```python
# Example benchmark configurations
benchmark_configs = {
    "baseline": { /* ... */ },
    "reduced_obs_space": { /* ... */ },
    "simplified_vision": { /* ... */ },
    "gpu_accelerated": { /* ... */ }
}

# Run benchmark
benchmark_results = run_performance_benchmark(benchmark_configs)

# Print results
print("\nPerformance Benchmark Results:")
print("==============================")

for config_name, metrics in benchmark_results.items():
    print(f"\n{config_name}:")
    print(f"  Mean Step Time: {metrics['step_time_mean']:.4f}s")
    print(f"  Mean FPS: {metrics['fps_mean']:.2f}")
    print(f"  Mean Memory Usage: {metrics.get('memory_usage_mean', 'N/A')} MB")
```

## Performance Bottleneck Identification

The testing infrastructure includes tools for identifying performance bottlenecks:

```python
def profile_component(component_function, args=(), kwargs={}, iterations=100):
    """Profile a specific component function."""
    import cProfile
    import pstats
    import io
    
    # Create profiler
    pr = cProfile.Profile()
    
    # Run profiling
    pr.enable()
    for _ in range(iterations):
        component_function(*args, **kwargs)
    pr.disable()
    
    # Get statistics
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions
    
    return s.getvalue()
```

This profiling capability is applied to critical components:

```python
# Profile observation processing
def profile_observation_processing():
    """Profile the observation processing pipeline."""
    env = CS2Environment({"use_fallback_mode": False})
    env.reset()
    
    # Get raw observation
    raw_obs = env.interface._capture_screenshot()
    
    # Profile observation processing
    profile_results = profile_component(
        env.interface._process_observation,
        args=(raw_obs,),
        iterations=50
    )
    
    print("Observation Processing Profile:")
    print(profile_results)
```

## GPU-Specific Performance Testing

The testing infrastructure includes specialized tests for GPU performance:

```python
def test_gpu_performance():
    """Test GPU performance for neural network operations."""
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            print("CUDA not available, skipping GPU performance test")
            return
        
        # Create test networks
        cuda_network = torch.nn.Sequential(
            torch.nn.Linear(84*84*3, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        ).cuda()
        
        cpu_network = torch.nn.Sequential(
            torch.nn.Linear(84*84*3, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )
        
        # Create test input
        test_input_cuda = torch.rand(100, 84*84*3).cuda()
        test_input_cpu = torch.rand(100, 84*84*3)
        
        # Warm-up
        for _ in range(10):
            cuda_network(test_input_cuda)
            cpu_network(test_input_cpu)
        
        # Test GPU performance
        gpu_start = time.time()
        for _ in range(100):
            cuda_network(test_input_cuda)
        gpu_time = time.time() - gpu_start
        
        # Test CPU performance
        cpu_start = time.time()
        for _ in range(100):
            cpu_network(test_input_cpu)
        cpu_time = time.time() - cpu_start
        
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU speedup: {cpu_time / gpu_time:.2f}x")
        
    except ImportError as e:
        print(f"Error in GPU performance test: {e}")
```

## Real-Time Performance Requirements

The testing infrastructure assesses whether the agent meets real-time performance requirements:

```python
def test_real_time_performance(agent, env, real_time_threshold=0.05):
    """Test if agent can operate in real-time (below threshold seconds per step)."""
    observation, info = env.reset()
    total_time = 0
    steps = 100
    
    for _ in range(steps):
        start_time = time.time()
        
        # Complete agent step
        action = agent.select_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        # Measure time
        step_time = time.time() - start_time
        total_time += step_time
        
        if terminated or truncated:
            observation, info = env.reset()
        else:
            observation = next_observation
    
    # Calculate average step time
    avg_step_time = total_time / steps
    
    # Check against threshold
    meets_real_time = avg_step_time <= real_time_threshold
    
    print(f"Average step time: {avg_step_time:.4f}s")
    print(f"Meets real-time requirement: {meets_real_time}")
    
    return meets_real_time, avg_step_time
```

## Memory Leak Detection

The performance testing includes memory leak detection:

```python
def test_memory_leaks(steps=1000, threshold=10.0):
    """Test for memory leaks during extended operation."""
    import psutil
    import gc
    
    process = psutil.Process()
    
    # Create environment and agent
    env = CS2Environment({"use_fallback_mode": True})
    agent = PPOAgent({})
    
    # Initial memory measurement
    gc.collect()
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Run for many steps
    observation, info = env.reset()
    memory_measurements = [initial_memory]
    
    for step in range(steps):
        action = agent.select_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()
        else:
            observation = next_observation
            
        # Measure memory periodically
        if step % 100 == 0:
            gc.collect()
            memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_measurements.append(memory)
            print(f"Step {step}, Memory: {memory:.2f} MB")
    
    # Final memory measurement
    gc.collect()
    final_memory = process.memory_info().rss / (1024 * 1024)  # MB
    memory_measurements.append(final_memory)
    
    # Calculate memory growth
    memory_growth = final_memory - initial_memory
    
    # Check against threshold
    has_leak = memory_growth > threshold
    
    print(f"Initial memory: {initial_memory:.2f} MB")
    print(f"Final memory: {final_memory:.2f} MB")
    print(f"Memory growth: {memory_growth:.2f} MB")
    print(f"Memory leak detected: {has_leak}")
    
    return has_leak, memory_measurements
```

## Related Sections
- [Introduction](01_testing_intro.md)
- [Testing Architecture Overview](02_testing_architecture.md)
- [Simulation Environment](05_simulation_environment.md)
- [Test Automation and CI/CD](07_test_automation.md) 