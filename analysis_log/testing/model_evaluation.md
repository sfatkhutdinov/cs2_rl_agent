# Model Evaluation Methods and Success Criteria Analysis

**Tags:** #testing #evaluation #model #metrics #analysis

## Context
This document examines how the CS2 reinforcement learning agent is evaluated beyond standard training metrics, focusing on the methodologies, success criteria, and visualization techniques used to assess agent performance in practice.

## Methodology
1. Analyzed the dedicated evaluation script (`evaluate.py`)
2. Examined evaluation callbacks and monitoring in training scripts
3. Studied custom metrics and visualization methods
4. Investigated success criteria for different agent types

## Evaluation Architecture

### Core Evaluation Framework
The system implements a comprehensive evaluation approach through a dedicated evaluation script:

```python
def evaluate(env, agent, num_episodes=10, render=False, output_dir=None):
    """
    Evaluate a trained agent in the environment for a specified number of episodes.
    
    Args:
        env: The environment to evaluate in
        agent: The trained agent to evaluate
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        output_dir: Directory to save evaluation results and visualizations
        
    Returns:
        dict: Evaluation results including rewards and metrics
    """
    # Initialize tracking variables
    episode_rewards = []
    episode_lengths = []
    episode_metrics = defaultdict(list)
    
    # Run evaluation episodes
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        # Episode metrics
        metrics = {}
        
        # Run a single episode
        while not (done or truncated):
            # Select action
            action, _states = agent.predict(obs, deterministic=True)
            
            # Execute action
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Render if specified
            if render:
                env.render()
                
            # Update tracking
            episode_reward += reward
            step_count += 1
            
            # Extract and track metrics from info
            if 'metrics' in info:
                for metric_name, metric_value in info['metrics'].items():
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(metric_value)
            
            # Update observation
            obs = next_obs
        
        # Record episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        # Process episode metrics (average over episode)
        for metric_name, values in metrics.items():
            if values:
                episode_metrics[metric_name].append(np.mean(values))
        
        # Log progress
        print(f"Episode {i+1}/{num_episodes}: Reward = {episode_reward:.2f}, Length = {step_count}")
    
    # Calculate overall results
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
    }
    
    # Add metrics to results
    for metric_name, values in episode_metrics.items():
        results[f'mean_{metric_name}'] = np.mean(values)
        results[f'std_{metric_name}'] = np.std(values)
    
    # Generate visualizations if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_results(episode_rewards, episode_lengths, episode_metrics, output_dir)
        save_results_to_csv(results, episode_rewards, episode_lengths, episode_metrics, output_dir)
    
    return results
```

### Performance Visualization
The evaluation results are visualized to provide better understanding of agent performance:

```python
def plot_results(rewards, lengths, metrics, output_dir):
    """
    Generate visualizations of evaluation results.
    
    Args:
        rewards: List of episode rewards
        lengths: List of episode lengths
        metrics: Dictionary of metrics by episode
        output_dir: Directory to save plots
    """
    # Plot reward distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(rewards, 'b-')
    plt.fill_between(range(len(rewards)), 
                     [np.mean(rewards) - np.std(rewards)] * len(rewards),
                     [np.mean(rewards) + np.std(rewards)] * len(rewards),
                     alpha=0.2)
    plt.axhline(y=np.mean(rewards), color='r', linestyle='-', label=f'Mean: {np.mean(rewards):.2f}')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot episode lengths
    plt.subplot(2, 1, 2)
    plt.plot(lengths, 'g-')
    plt.axhline(y=np.mean(lengths), color='r', linestyle='-', label=f'Mean: {np.mean(lengths):.2f}')
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rewards_and_lengths.png'))
    plt.close()
    
    # Plot metrics
    if metrics:
        for metric_name, values in metrics.items():
            if len(values) > 0:
                plt.figure(figsize=(8, 5))
                plt.plot(values, 'b-')
                plt.axhline(y=np.mean(values), color='r', linestyle='-', 
                            label=f'Mean: {np.mean(values):.2f}')
                plt.title(f'Metric: {metric_name}')
                plt.xlabel('Episode')
                plt.ylabel('Value')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'metric_{metric_name}.png'))
                plt.close()
                
    # Generate correlation matrix for metrics and rewards
    if metrics:
        correlation_data = {'reward': rewards, 'episode_length': lengths}
        for metric_name, values in metrics.items():
            if len(values) == len(rewards):  # Ensure same length
                correlation_data[metric_name] = values
                
        df = pd.DataFrame(correlation_data)
        correlation = df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()
```

### In-Training Evaluation
During training, the system uses evaluation callbacks to monitor performance and save the best models:

```python
# From train_autonomous.py
eval_env = DummyVecEnv([lambda: create_env(config, is_eval=True, seed=seed+1000)])
if config.get("policy", {}).get("type") == "CnnPolicy":
    eval_env = VecTransposeImage(eval_env)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"./models/{RUN_ID}/best_model",
    log_path=f"./logs/{RUN_ID}/eval",
    eval_freq=eval_freq,
    deterministic=True,
    render=False,
    n_eval_episodes=5
)
```

### Command-Line Interface
The evaluation system includes a command-line interface for flexible evaluation:

```python
def parse_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate a trained agent')
    
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                      help='Render the environment during evaluation')
    parser.add_argument('--output', type=str, default='./eval_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'dqn', 'a2c'], default='ppo',
                      help='RL algorithm used for the model')
    parser.add_argument('--video', action='store_true',
                      help='Record evaluation videos')
    parser.add_argument('--metrics', type=str, default=None,
                      help='Comma-separated list of specific metrics to track')
    
    return parser.parse_args()
```

## Agent-Specific Evaluation Metrics

Different agent types are evaluated using specialized metrics appropriate to their objectives:

### Discovery Agent Metrics
- **UI Element Discovery Rate**: Percentage of UI elements discovered
- **Exploration Coverage**: Percentage of game screens explored
- **Redundant Action Rate**: Frequency of repeated actions with no new discoveries
- **Novel State Discovery Rate**: Rate of discovering previously unseen states

### Tutorial Agent Metrics
- **Instruction Completion Rate**: Percentage of tutorial steps completed
- **Instruction Understanding Time**: Time taken to understand and act on instructions
- **Error Recovery Rate**: Success rate in recovering from mistakes
- **Task Completion Time**: Time to complete tutorial tasks

### Vision Agent Metrics
- **Visual Recognition Accuracy**: Accuracy in identifying game elements
- **UI Element Interaction Success**: Success rate when interacting with UI elements
- **Screen Understanding Time**: Time taken to parse and understand new screens
- **Visual Memory Utilization**: Effectiveness in remembering previously seen elements

### Autonomous Agent Metrics
- **Game Progression Metrics**: Game-specific progress indicators (population, happiness, etc.)
- **Resource Efficiency**: Effectiveness in managing game resources
- **Decision Consistency**: Consistency in making similar decisions in similar situations
- **Adaptation Speed**: How quickly the agent adapts to changing game conditions

### Strategic Agent Metrics
- **Long-term Planning Success**: Achievement of long-term objectives
- **Strategic Adaptation**: Ability to change strategies based on game state
- **Causal Understanding**: Correct identification of cause-effect relationships
- **Goal Prioritization**: Appropriate balancing of competing objectives

## Success Criteria Frameworks

The system defines multiple frameworks for evaluating agent success:

### Absolute Performance Criteria
Defines minimum performance thresholds that an agent must meet:

```python
def evaluate_against_criteria(results, criteria):
    """
    Evaluate results against absolute performance criteria.
    
    Args:
        results: Dictionary of evaluation results
        criteria: Dictionary of minimum criteria to meet
        
    Returns:
        dict: Evaluation results with pass/fail status for each criterion
    """
    evaluation = {}
    
    for metric, threshold in criteria.items():
        if metric in results:
            evaluation[metric] = {
                'value': results[metric],
                'threshold': threshold,
                'passed': results[metric] >= threshold
            }
    
    # Calculate overall pass/fail status
    evaluation['overall_passed'] = all(item['passed'] for item in evaluation.values() 
                                      if isinstance(item, dict))
    
    return evaluation
```

### Comparative Evaluation
Compares agent performance against baselines or previous versions:

```python
def comparative_evaluation(current_results, baseline_results):
    """
    Compare current agent performance against baseline results.
    
    Args:
        current_results: Results from current agent evaluation
        baseline_results: Results from baseline or previous agent
        
    Returns:
        dict: Comparative analysis with improvement metrics
    """
    comparison = {}
    
    for metric in current_results:
        if metric in baseline_results:
            current_value = current_results[metric]
            baseline_value = baseline_results[metric]
            
            # Calculate absolute and percentage difference
            abs_diff = current_value - baseline_value
            if baseline_value != 0:
                pct_diff = (abs_diff / abs(baseline_value)) * 100
            else:
                pct_diff = float('inf') if abs_diff > 0 else 0 if abs_diff == 0 else float('-inf')
            
            comparison[metric] = {
                'current': current_value,
                'baseline': baseline_value,
                'abs_diff': abs_diff,
                'pct_diff': pct_diff,
                'improved': abs_diff > 0  # Assuming higher is better
            }
    
    # Calculate overall improvement statistics
    improved_metrics = sum(1 for item in comparison.values() 
                          if isinstance(item, dict) and item['improved'])
    total_metrics = sum(1 for item in comparison.values() if isinstance(item, dict))
    
    comparison['overall_improvement'] = {
        'improved_metrics': improved_metrics,
        'total_metrics': total_metrics,
        'improvement_rate': improved_metrics / total_metrics if total_metrics > 0 else 0
    }
    
    return comparison
```

### Human-Baseline Comparisons
Compares agent performance against human benchmarks:

```python
def compare_to_human_baseline(agent_results, human_results):
    """
    Compare agent performance to human baseline performance.
    
    Args:
        agent_results: Results from agent evaluation
        human_results: Benchmark results from human players
        
    Returns:
        dict: Comparative analysis with human-relative metrics
    """
    comparison = {}
    
    for metric in agent_results:
        if metric in human_results:
            agent_value = agent_results[metric]
            human_value = human_results[metric]
            
            # Calculate relative performance (agent as % of human)
            if human_value != 0:
                relative_perf = (agent_value / human_value) * 100
            else:
                relative_perf = float('inf') if agent_value > 0 else 0 if agent_value == 0 else float('-inf')
            
            comparison[metric] = {
                'agent': agent_value,
                'human': human_value,
                'relative_perf': relative_perf,
                'human_parity': agent_value >= human_value  # Whether agent meets/exceeds human
            }
    
    # Calculate overall human parity statistics
    parity_metrics = sum(1 for item in comparison.values() 
                        if isinstance(item, dict) and item['human_parity'])
    total_metrics = sum(1 for item in comparison.values() if isinstance(item, dict))
    
    comparison['overall_human_parity'] = {
        'parity_metrics': parity_metrics,
        'total_metrics': total_metrics,
        'parity_rate': parity_metrics / total_metrics if total_metrics > 0 else 0
    }
    
    return comparison
```

## Advanced Evaluation Techniques

### Training Efficiency Analysis
Evaluates the sample efficiency of training:

```python
def analyze_training_efficiency(training_logs, model_evaluations):
    """
    Analyze training efficiency across multiple training runs.
    
    Args:
        training_logs: Dictionary of training logs by run ID
        model_evaluations: Dictionary of evaluation results by run ID
        
    Returns:
        dict: Training efficiency metrics
    """
    efficiency_metrics = {}
    
    for run_id in model_evaluations:
        if run_id in training_logs:
            # Extract data
            timesteps = training_logs[run_id]['timesteps']
            rewards = training_logs[run_id]['episode_rewards']
            final_eval = model_evaluations[run_id]
            
            # Calculate metrics
            time_to_threshold = None
            threshold = 0.8 * final_eval['mean_reward']
            
            for i, reward in enumerate(rewards):
                if reward >= threshold:
                    time_to_threshold = timesteps[i]
                    break
            
            # Calculate rate of improvement
            if len(timesteps) > 1:
                reward_slope = np.polyfit(timesteps, rewards, 1)[0]
            else:
                reward_slope = 0
                
            efficiency_metrics[run_id] = {
                'final_performance': final_eval['mean_reward'],
                'time_to_threshold': time_to_threshold,
                'improvement_rate': reward_slope,
                'sample_efficiency': final_eval['mean_reward'] / timesteps[-1] if timesteps else 0
            }
    
    return efficiency_metrics
```

### Feature Attribution Analysis
Analyzes which observation features have the most impact on agent decisions:

```python
def feature_attribution_analysis(agent, env, num_samples=1000):
    """
    Analyze which features have the most impact on agent decisions.
    
    Args:
        agent: Trained agent model
        env: Environment
        num_samples: Number of samples to collect
        
    Returns:
        dict: Feature importance metrics
    """
    # Collect samples
    observations = []
    actions = []
    
    obs, info = env.reset()
    for _ in range(num_samples):
        action, _states = agent.predict(obs, deterministic=True)
        observations.append(obs)
        actions.append(action)
        
        next_obs, _, done, truncated, _ = env.step(action)
        obs = next_obs
        
        if done or truncated:
            obs, info = env.reset()
    
    # Convert to numpy arrays
    observations = np.array(observations)
    actions = np.array(actions)
    
    # Calculate feature importance using permutation importance
    feature_importance = {}
    
    # For CNN policies, we analyze importance differently
    if isinstance(agent.policy, CnnPolicy):
        # Use saliency maps or similar techniques
        # Implementation depends on the specific CNN architecture
        pass
    else:
        # For vector observations, we can use permutation importance
        baseline_accuracy = predict_own_actions(agent, observations, actions)
        
        # Test each feature
        for feature_idx in range(observations.shape[1]):
            # Create permuted dataset
            permuted_obs = observations.copy()
            permuted_obs[:, feature_idx] = np.random.permutation(permuted_obs[:, feature_idx])
            
            # Measure new accuracy
            permuted_accuracy = predict_own_actions(agent, permuted_obs, actions)
            
            # Importance is the drop in accuracy
            importance = baseline_accuracy - permuted_accuracy
            feature_importance[f'feature_{feature_idx}'] = importance
    
    # Normalize importances
    total_importance = sum(abs(imp) for imp in feature_importance.values())
    if total_importance > 0:
        normalized_importance = {k: abs(v)/total_importance for k, v in feature_importance.items()}
    else:
        normalized_importance = {k: 0 for k in feature_importance}
    
    return {
        'raw_importance': feature_importance,
        'normalized_importance': normalized_importance
    }
```

### Behavior Characterization
Analyzes agent behavior patterns to understand decision-making strategies:

```python
def characterize_agent_behavior(episode_data, metrics_of_interest):
    """
    Analyze agent behavior patterns across evaluation episodes.
    
    Args:
        episode_data: List of episode data dictionaries
        metrics_of_interest: List of metrics to analyze
        
    Returns:
        dict: Behavior characterization metrics
    """
    characterization = {}
    
    # Aggregate action distributions
    action_counts = defaultdict(int)
    total_actions = 0
    
    for episode in episode_data:
        for action in episode['actions']:
            action_counts[action] += 1
            total_actions += 1
    
    # Calculate action distribution
    action_distribution = {
        action: count / total_actions
        for action, count in action_counts.items()
    }
    characterization['action_distribution'] = action_distribution
    
    # Calculate action entropy (measure of randomness/decisiveness)
    action_entropy = -sum(p * np.log(p) for p in action_distribution.values() if p > 0)
    characterization['action_entropy'] = action_entropy
    
    # Analyze state-dependent behavior
    state_action_pairs = defaultdict(lambda: defaultdict(int))
    
    for episode in episode_data:
        for state, action in zip(episode['states'], episode['actions']):
            state_key = state_to_key(state)  # Convert state to hashable key
            state_action_pairs[state_key][action] += 1
    
    # Calculate state-action consistency
    state_consistency = {}
    for state_key, actions in state_action_pairs.items():
        total = sum(actions.values())
        max_action_count = max(actions.values())
        consistency = max_action_count / total if total > 0 else 0
        state_consistency[state_key] = consistency
    
    characterization['mean_state_consistency'] = np.mean(list(state_consistency.values()))
    
    # Analyze metric correlations
    if metrics_of_interest and all(m in episode_data[0] for m in metrics_of_interest):
        metric_correlations = {}
        
        for i, metric1 in enumerate(metrics_of_interest):
            for metric2 in metrics_of_interest[i+1:]:
                values1 = [episode[metric1] for episode in episode_data]
                values2 = [episode[metric2] for episode in episode_data]
                
                correlation = np.corrcoef(values1, values2)[0, 1]
                metric_correlations[f'{metric1}_vs_{metric2}'] = correlation
        
        characterization['metric_correlations'] = metric_correlations
    
    return characterization
```

## Integration with Training Pipeline

The evaluation system is integrated into the training pipeline at multiple points:

1. **During Training**: Regular evaluation via callbacks to track progress and save best models
2. **Post-Training**: Comprehensive evaluation of trained models against standard benchmarks
3. **Comparative Analysis**: Evaluation against baseline agents and previous versions
4. **Performance Profiling**: Analysis of efficiency and resource utilization

## Key Findings

1. **Comprehensive Evaluation**: The system implements a sophisticated evaluation framework that goes beyond simple reward metrics, tracking game-specific performance indicators and agent behavior patterns.

2. **Visual Analytics**: Extensive visualization capabilities help understand agent performance, with plots for rewards, episode lengths, and game-specific metrics.

3. **Success Criteria Frameworks**: Multiple evaluation frameworks allow for absolute, comparative, and human-baseline evaluations.

4. **Agent-Specific Metrics**: Each agent type is evaluated with metrics tailored to its specific objectives and capabilities.

5. **Advanced Analysis**: The system includes capabilities for training efficiency analysis, feature attribution, and behavior characterization.

## Improvement Opportunities

### Enhanced Visualization
Implement more sophisticated visualization techniques:
- Interactive dashboards for exploring evaluation results
- Video recordings of agent behavior with decision annotations
- Comparative visualizations showing behavior differences between agent versions

### Automated Evaluation Pipeline
Create a fully automated evaluation pipeline:
- Scheduled evaluation of latest models against benchmarks
- Regression testing to detect performance degradation
- Performance reports with insights about improvements/regressions

### Extended Success Criteria
Develop more nuanced success criteria:
- Dynamic thresholds based on environment difficulty
- Multi-objective evaluation frameworks
- Learning curve analysis for different agent types

### Reinforcement Learning Specific Metrics
Add RL-specific evaluation metrics:
- Exploration vs. exploitation balance analysis
- Policy entropy monitoring over time
- Value function calibration assessment
- Advantage estimation accuracy

## Next Steps
1. Implement the enhanced visualization techniques
2. Develop the automated evaluation pipeline
3. Create a centralized evaluation database for tracking agent performance over time
4. Extend the success criteria framework with dynamic thresholds

## Related Analyses
- [Performance Profiling](../performance/performance_profiling.md)
- [Strategic Agent Analysis](../components/strategic_agent.md)
- [Adaptive Agent System](../components/adaptive_agent.md)
- [Comprehensive Synthesis](../architecture/comprehensive_synthesis.md) 