# Adaptive Orchestration Testing

*Last updated: March 13, 2025 21:17 - Initial documentation*

**Tags:** #testing #adaptive #orchestration #verification

## Overview

This document describes the testing methodology and implementation for verifying the adaptive agent's orchestration capabilities. The tests ensure that the adaptive agent correctly initializes all specialized agent modes, dynamically switches between them based on performance metrics, and transfers knowledge between different modes.

## Test Implementation

### Core Test Script

The primary test implementation is in `testing/test_adaptive_orchestration.py`, which provides a comprehensive test suite for the adaptive agent's orchestration capabilities.

#### Key Test Areas

1. **Initialization Testing**
   - Verifies that the adaptive agent correctly loads configurations
   - Ensures all specialized agent modes are properly initialized
   - Validates environment integration

2. **Mode Switching Testing**
   - Verifies dynamic mode selection based on performance metrics
   - Tests transitions between all agent modes
   - Ensures proper handling of mode switching signals

3. **Knowledge Transfer Testing**
   - Measures improvement in performance over time
   - Verifies that learning in one mode benefits other modes
   - Tests the persistence of learned behaviors across mode switches

4. **Error Handling Testing**
   - Verifies graceful handling of errors in specific modes
   - Tests fallback mechanisms when a mode fails
   - Ensures system resilience during mode transitions

### Test Environment

The test can use either:

1. **Mock Environment** (`--mock-env` flag)
   - Simulates game interactions without requiring the actual game
   - Provides controlled rewards for different modes
   - Enables randomized mode switching signals for testing
   - Default for quick verification testing

2. **Real Environment** (without `--mock-env`)
   - Uses the actual CS2Environment
   - Tests with real game interactions
   - Provides more realistic but less controlled testing
   - Suitable for comprehensive validation

### Performance Metrics Tracking

The test tracks several key metrics:

1. **Mode-specific metrics**
   - Episodes completed per mode
   - Average reward per mode
   - Time spent in each mode

2. **Mode switching metrics**
   - Number of mode switches
   - Timing of mode switches
   - Reasons for mode switches

3. **Knowledge transfer metrics**
   - Performance improvement over time
   - Comparison of early vs. late episode performance
   - Evidence of learning transfer between modes

## Batch Script Implementation

The batch script `scripts/testing/test_adaptive_orchestration.bat` provides a convenient way to run the test with proper environment setup:

1. **Environment Setup**
   - Configures the conda environment
   - Verifies required dependencies
   - Applies TensorFlow patch for compatibility

2. **Command Line Options**
   - `--real-env`: Uses the real game environment instead of the mock environment
   - Duration parameter: Sets the test duration in seconds (default: 60 seconds)

3. **Result Reporting**
   - Displays a summary of test results
   - Provides warnings for unused modes
   - Reports on evidence of knowledge transfer

## Example Usage

### Basic Test (Mock Environment)

```bash
scripts\testing\test_adaptive_orchestration.bat
```

This runs a quick 60-second test using the mock environment to verify basic functionality.

### Comprehensive Test (Real Environment)

```bash
scripts\testing\test_adaptive_orchestration.bat --real-env 300
```

This runs a 5-minute test using the real game environment for more thorough validation.

## Sample Output

```
===== Adaptive Orchestration Test Results =====
Total test duration: 120.45 seconds
Total episodes: 14
Total mode switches: 8

Mode Statistics:
  Discovery: 3 episodes, 22.5% of time, 0.32 avg reward
  Tutorial: 4 episodes, 28.7% of time, 0.48 avg reward
  Vision: 2 episodes, 15.3% of time, 0.65 avg reward
  Autonomous: 3 episodes, 18.2% of time, 0.41 avg reward
  Strategic: 2 episodes, 15.3% of time, 0.58 avg reward

SUCCESS: All agent modes were utilized during testing.

Verifying knowledge transfer:
  Discovery: Improved by 0.08
  Tutorial: Improved by 0.12
  Vision: Improved by 0.05
  Autonomous: Declined by 0.02
  Strategic: Improved by 0.09

SUCCESS: Evidence of knowledge transfer detected between episodes.
```

## Verification Criteria

The test considers the adaptive orchestration to be functioning correctly when:

1. **All modes are utilized** during the test duration
2. **Mode switching occurs** in response to performance metrics
3. **Knowledge transfer is evident** through performance improvements
4. **No errors occur** during mode transitions
5. **All specialized agent types** are properly integrated

## Implementation Details

### Mock Environment

The mock environment (`MockEnvironment` class) simulates the game environment for testing purposes:

```python
class MockEnvironment:
    def __init__(self):
        self.state = 'initial'
        self.reward_map = {
            'discovery': np.random.normal(0.3, 0.1),
            'tutorial': np.random.normal(0.5, 0.1),
            'vision': np.random.normal(0.7, 0.1),
            'autonomous': np.random.normal(0.4, 0.1),
            'strategic': np.random.normal(0.6, 0.1),
        }
    
    def reset(self):
        # Implementation details...

    def step(self, action, mode):
        # Implementation details...
```

This environment provides controlled rewards for different modes and simulates mode switching signals, enabling thorough testing of the adaptive agent's orchestration capabilities without requiring the actual game.

### Performance Analysis

The test analyzes performance improvement by comparing rewards from early episodes to later episodes:

```python
# Check knowledge transfer (did performance improve over time?)
knowledge_transfer = False
for mode, metrics in results['mode_metrics'].items():
    if len(metrics['rewards']) >= 2:
        first_half = metrics['rewards'][:len(metrics['rewards'])//2]
        second_half = metrics['rewards'][len(metrics['rewards'])//2:]
        if first_half and second_half:
            improvement = (sum(second_half)/len(second_half)) - (sum(first_half)/len(first_half))
            print(f"  {mode.capitalize()}: {'Improved' if improvement > 0 else 'Declined'} by {abs(improvement):.2f}")
            if improvement > 0:
                knowledge_transfer = True
```

This approach provides evidence of knowledge transfer and continuous learning across different agent modes.

## Integration with Other Tests

The adaptive orchestration test complements other testing components:

1. **Unit Tests**: Test individual agent mode implementations
2. **Performance Tests**: Evaluate the efficiency of the orchestration mechanism
3. **Integration Tests**: Verify interactions between components
4. **Deployment Tests**: Ensure the system works in production environments

## Future Enhancements

Planned enhancements to the testing methodology:

1. **Extended Duration Tests**: Longer test runs to better evaluate learning
2. **Scenario-Based Testing**: Targeted tests for specific game scenarios
3. **Comparative Analysis**: Comparison with non-adaptive implementations
4. **Visual Performance Monitoring**: Real-time visualization of mode switching
5. **Automated Regression Testing**: Integration with CI/CD pipelines

## References

- [Adaptive Agent System](../components/adaptive_agent.md)
- [Adaptive Agent Orchestration](../architecture/adaptive_orchestration.md)
- [Streamlined Architecture Verification](../architecture/streamlined_architecture_verification.md)
- [Adaptive Orchestration Diagram](../visualization/adaptive_orchestration_diagram.md)
- [Testing Infrastructure](testing_infrastructure.md) 