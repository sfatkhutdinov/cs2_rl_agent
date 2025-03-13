"""
Strategic Environment - An environment wrapper that extends the autonomous environment
with capabilities for metrics discovery, causal modeling, and goal inference.
"""

import logging
import numpy as np
import gym
from gym import spaces
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import defaultdict, deque

from src.environment.autonomous_env import AutonomousCS2Environment
from src.environment.cs2_env import CS2Environment

class StrategicEnvironment(gym.Wrapper):
    """
    A wrapper around the AutonomousCS2Environment that adds strategic capabilities:
    - Discovers and tracks game metrics
    - Builds causal models between actions and outcomes
    - Infers game goals and their relative importance
    - Provides intrinsic rewards for strategic exploration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the strategic environment wrapper.
        
        Args:
            config: Configuration dictionary
        """
        # Create the base autonomous environment first
        base_env = AutonomousCS2Environment(
            base_env=CS2Environment(config),
            exploration_frequency=config.get("exploration_frequency", 0.2),
            random_action_frequency=config.get("random_action_frequency", 0.1),
            menu_exploration_buffer_size=config.get("menu_exploration_buffer_size", 100),
            logger=logging.getLogger("StrategicEnv")
        )
        super().__init__(base_env)
        
        self.logger = logging.getLogger("StrategicEnv")
        self.config = config
        
        # Load any existing knowledge base
        self.knowledge_base = config.get("knowledge_base", {
            "game_metrics": {},
            "action_effects": {},
            "causal_links": {},
            "goal_hierarchy": {}
        })
        
        # Initialize metric tracking systems
        self.discovered_metrics = self.knowledge_base.get("game_metrics", {})
        self.metric_history = defaultdict(lambda: deque(maxlen=100))  # Track last 100 values for each metric
        self.metric_trends = {}  # Track if metrics are increasing/decreasing
        
        # Action-effect model
        self.action_effects = self.knowledge_base.get("action_effects", {})
        self.recent_actions = deque(maxlen=20)  # Track recent actions for delayed effects
        self.action_effect_window = config.get("action_effect_window", 5)  # How many steps to track effects
        
        # Causal model
        self.causal_links = self.knowledge_base.get("causal_links", {})
        self.causal_confidence = {}  # Confidence in each causal link
        
        # Goal inference
        self.inferred_goals = self.knowledge_base.get("goal_hierarchy", {})
        self.goal_importance = {}  # Relative importance of each goal
        
        # Intrinsic motivation system
        self.novelty_baseline = {}  # Baseline for novelty detection
        self.empowerment_scores = {}  # Scores for actions that lead to control
        self.progress_baselines = {}  # Baselines for measuring progress
        
        # Exploration vs exploitation balance
        self.exploration_phase = config.get("start_with_exploration", True)
        self.exploration_rate = config.get("initial_exploration_rate", 0.8)
        self.min_exploration_rate = config.get("min_exploration_rate", 0.1)
        self.exploration_decay = config.get("exploration_decay", 0.9999)
        
        # Reward weights for different components
        self.reward_weights = {
            "extrinsic": config.get("extrinsic_reward_weight", 0.5),
            "discovery": config.get("discovery_reward_weight", 0.2),
            "causal": config.get("causal_reward_weight", 0.15),
            "progress": config.get("progress_reward_weight", 0.15)
        }
        
        self.logger.info("Strategic Environment initialized")
    
    def step(self, action):
        """
        Take a step in the environment with strategic reasoning capabilities.
        
        Args:
            action: The action to take
            
        Returns:
            observation, reward, terminated, truncated, info - Standard gymnasium step return
        """
        # Store current metrics before the action
        pre_metrics = self._extract_metrics_from_observation(self.env.last_observation) if self.env.last_observation else {}
        
        # Take the action in the base environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Store the action for delayed effect tracking
        self.recent_actions.append((action, pre_metrics))
        
        # Extract metrics from the new observation
        post_metrics = self._extract_metrics_from_observation(observation)
        
        # Discover and track metrics
        self._discover_metrics(observation, post_metrics)
        
        # Update metric history and trends
        self._update_metric_history(post_metrics)
        
        # Correlate actions with metric changes
        self._correlate_actions_with_metrics(action, pre_metrics, post_metrics)
        
        # Handle delayed effects from previous actions
        self._process_delayed_effects(post_metrics)
        
        # Infer goal importance based on game feedback
        feedback_text = info.get("game_message", "") if isinstance(info, dict) else ""
        if feedback_text:
            self._extract_game_logic(feedback_text)
        
        # Calculate intrinsic rewards
        intrinsic_reward = self._calculate_intrinsic_reward(observation, post_metrics)
        
        # Combine extrinsic and intrinsic rewards
        combined_reward = (
            self.reward_weights["extrinsic"] * reward + 
            intrinsic_reward
        )
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )
        
        # Switch between exploration and exploitation phases
        if self.exploration_phase and len(self.discovered_metrics) >= self.config.get("min_metrics_to_discover", 5):
            # Check if we've learned enough about the causal model to switch to optimization
            if len(self.causal_links) >= self.config.get("min_causal_links", 10):
                self.exploration_phase = False
                self.logger.info("Switching from exploration to optimization phase")
        
        # Update info with strategic data
        info["strategic"] = {
            "discovered_metrics": len(self.discovered_metrics),
            "causal_links": len(self.causal_links),
            "intrinsic_reward": intrinsic_reward,
            "exploration_rate": self.exploration_rate,
            "exploration_phase": self.exploration_phase
        }
        
        # Update the knowledge base
        self._update_knowledge_base()
        
        return observation, combined_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset the environment and return the initial observation."""
        observation, info = self.env.reset(**kwargs)
        
        # Reset any per-episode tracking
        self.recent_actions.clear()
        
        # Extract initial metrics
        initial_metrics = self._extract_metrics_from_observation(observation)
        self._discover_metrics(observation, initial_metrics)
        self._update_metric_history(initial_metrics)
        
        return observation, info
    
    def _extract_metrics_from_observation(self, observation):
        """Extract numerical metrics from the observation."""
        if observation is None:
            return {}
        
        metrics = {}
        
        # First try to get metrics from the observation's metrics field
        if isinstance(observation, dict) and 'metrics' in observation:
            obs_metrics = observation['metrics']
            if isinstance(obs_metrics, dict):
                metrics.update(obs_metrics)
        
        # Use vision system to find numbers in the UI (if available)
        if hasattr(self.env, 'interface') and hasattr(self.env.interface, 'vision_system'):
            try:
                ui_numbers = self.env.interface.vision_system.extract_numbers_from_screen()
                for label, value in ui_numbers.items():
                    if label not in metrics:
                        metrics[label] = value
            except Exception as e:
                self.logger.debug(f"Error extracting numbers from screen: {str(e)}")
        
        return metrics
    
    def _discover_metrics(self, observation, current_metrics):
        """Identify numerical values that change in the game UI."""
        for metric_name, value in current_metrics.items():
            if metric_name not in self.discovered_metrics:
                self.discovered_metrics[metric_name] = {
                    "min_observed": value,
                    "max_observed": value,
                    "initial_value": value,
                    "discovery_time": self.env.total_steps,
                    "importance_score": 0.0,
                    "volatility": 0.0  # How much this metric tends to change
                }
                self.logger.info(f"Discovered new metric: {metric_name} = {value}")
            else:
                # Update min/max observed values
                self.discovered_metrics[metric_name]["min_observed"] = min(
                    self.discovered_metrics[metric_name]["min_observed"], value
                )
                self.discovered_metrics[metric_name]["max_observed"] = max(
                    self.discovered_metrics[metric_name]["max_observed"], value
                )
    
    def _update_metric_history(self, current_metrics):
        """Update the history and trends for each metric."""
        for metric_name, value in current_metrics.items():
            # Add to history
            self.metric_history[metric_name].append(value)
            
            # Calculate trend if we have enough history
            if len(self.metric_history[metric_name]) >= 5:
                recent_values = list(self.metric_history[metric_name])[-5:]
                if len(recent_values) >= 2:
                    # Simple linear regression slope calculation
                    x = np.array(range(len(recent_values)))
                    y = np.array(recent_values)
                    slope = np.polyfit(x, y, 1)[0] if len(set(y)) > 1 else 0
                    
                    # Calculate volatility (standard deviation of changes)
                    changes = np.diff(recent_values)
                    volatility = np.std(changes) if len(changes) > 0 else 0
                    
                    # Update metric info
                    if metric_name in self.discovered_metrics:
                        self.discovered_metrics[metric_name]["volatility"] = volatility
                    
                    # Store trend (positive, negative, or stable)
                    if slope > 0.1:
                        self.metric_trends[metric_name] = "increasing"
                    elif slope < -0.1:
                        self.metric_trends[metric_name] = "decreasing"
                    else:
                        self.metric_trends[metric_name] = "stable"
    
    def _correlate_actions_with_metrics(self, action, pre_metrics, post_metrics):
        """Learn correlations between actions and metric changes."""
        # Initialize action effects record if needed
        if action not in self.action_effects:
            self.action_effects[action] = defaultdict(lambda: {
                "positive_changes": 0,
                "negative_changes": 0,
                "no_changes": 0,
                "average_change": 0.0,
                "correlation_strength": 0.0,
                "samples": 0
            })
        
        # Compare pre and post metrics
        for metric_name, post_value in post_metrics.items():
            if metric_name in pre_metrics:
                pre_value = pre_metrics[metric_name]
                change = post_value - pre_value
                
                # Update action-effect model
                effect_data = self.action_effects[action][metric_name]
                effect_data["samples"] += 1
                
                if change > 0:
                    effect_data["positive_changes"] += 1
                elif change < 0:
                    effect_data["negative_changes"] += 1
                else:
                    effect_data["no_changes"] += 1
                
                # Update average change with exponential moving average
                if effect_data["samples"] == 1:
                    effect_data["average_change"] = change
                else:
                    alpha = 0.1  # Smoothing factor
                    effect_data["average_change"] = (
                        (1 - alpha) * effect_data["average_change"] + alpha * change
                    )
                
                # Calculate correlation strength
                total_samples = effect_data["positive_changes"] + effect_data["negative_changes"] + effect_data["no_changes"]
                if total_samples >= 3:  # Need minimum samples for meaningful correlation
                    # If most changes are in the same direction, correlation is stronger
                    max_changes = max(
                        effect_data["positive_changes"],
                        effect_data["negative_changes"],
                        effect_data["no_changes"]
                    )
                    effect_data["correlation_strength"] = max_changes / total_samples
                    
                    # Create a causal link if correlation is strong enough
                    if effect_data["correlation_strength"] >= 0.6:
                        causal_key = f"{action}→{metric_name}"
                        self.causal_links[causal_key] = {
                            "action": action,
                            "metric": metric_name,
                            "direction": "positive" if effect_data["average_change"] > 0 else "negative",
                            "strength": effect_data["correlation_strength"],
                            "average_change": effect_data["average_change"]
                        }
                        
                        # Update the causal confidence
                        self.causal_confidence[causal_key] = effect_data["correlation_strength"]
    
    def _process_delayed_effects(self, current_metrics):
        """Process potential delayed effects from recent actions."""
        # We look back through recent actions to find delayed correlations
        for i, (past_action, past_metrics) in enumerate(self.recent_actions):
            # Skip very recent actions (already covered by immediate effects)
            if i < 2:
                continue
                
            # Check for delayed effects (correlations in metrics that appear after several steps)
            delay = i + 1  # Number of steps delayed
            
            # Compare current metrics with metrics from when the action was taken
            for metric_name, current_value in current_metrics.items():
                if metric_name in past_metrics:
                    past_value = past_metrics[metric_name]
                    change = current_value - past_value
                    
                    # Only consider significant changes
                    if abs(change) > 0.1:
                        # Create a delayed effect record
                        delayed_action_key = f"{past_action}→{metric_name}→delay{delay}"
                        
                        if delayed_action_key not in self.action_effects:
                            self.action_effects[delayed_action_key] = {
                                "action": past_action,
                                "metric": metric_name,
                                "delay": delay,
                                "direction": "positive" if change > 0 else "negative",
                                "average_change": change,
                                "samples": 1,
                                "correlation_strength": 0.1  # Start with low confidence
                            }
                        else:
                            # Update existing delayed effect
                            effect = self.action_effects[delayed_action_key]
                            effect["samples"] += 1
                            
                            # Update the average change
                            alpha = 0.2  # Higher smoothing factor for delayed effects
                            effect["average_change"] = (
                                (1 - alpha) * effect["average_change"] + alpha * change
                            )
                            
                            # Update direction based on average change
                            effect["direction"] = "positive" if effect["average_change"] > 0 else "negative"
                            
                            # Increase correlation strength with more consistent observations
                            if effect["direction"] == "positive" and change > 0:
                                effect["correlation_strength"] = min(
                                    0.9, effect["correlation_strength"] + 0.05
                                )
                            elif effect["direction"] == "negative" and change < 0:
                                effect["correlation_strength"] = min(
                                    0.9, effect["correlation_strength"] + 0.05
                                )
                            else:
                                # Contradictory observation, reduce confidence
                                effect["correlation_strength"] = max(
                                    0.0, effect["correlation_strength"] - 0.1
                                )
                            
                            # Create causal link if strong enough correlation
                            if effect["correlation_strength"] >= 0.5:
                                causal_key = f"{past_action}→{metric_name}→delay{delay}"
                                self.causal_links[causal_key] = {
                                    "action": past_action,
                                    "metric": metric_name,
                                    "delay": delay,
                                    "direction": effect["direction"],
                                    "strength": effect["correlation_strength"],
                                    "average_change": effect["average_change"]
                                }
                                
                                # Update the causal confidence
                                self.causal_confidence[causal_key] = effect["correlation_strength"]
    
    def _extract_game_logic(self, game_message):
        """Extract logic and goals from in-game messages."""
        # Check if message contains relevant keywords for goals
        goal_keywords = {
            "population": ["population", "citizens", "people", "residents"],
            "money": ["money", "budget", "cash", "funds", "financial"],
            "happiness": ["happiness", "satisfaction", "happy", "content"],
            "traffic": ["traffic", "congestion", "commute"],
            "pollution": ["pollution", "environment", "environmental"],
            "education": ["education", "schools", "university"],
            "health": ["health", "healthcare", "hospital"],
            "employment": ["employment", "jobs", "work", "unemployment"]
        }
        
        # Check each goal area for mentions in the message
        for goal_area, keywords in goal_keywords.items():
            message_lower = game_message.lower()
            for keyword in keywords:
                if keyword in message_lower:
                    # Initialize goal if not present
                    if goal_area not in self.inferred_goals:
                        self.inferred_goals[goal_area] = {
                            "importance": 0.5,  # Start with medium importance
                            "mentions": 0,
                            "positive_mentions": 0,
                            "negative_mentions": 0
                        }
                    
                    # Update mentions
                    self.inferred_goals[goal_area]["mentions"] += 1
                    
                    # Try to determine if this is a positive or negative mention
                    positive_context = any(pos in message_lower for pos in [
                        "good", "great", "excellent", "increase", "more", "higher",
                        "better", "improved", "success", "achievement"
                    ])
                    negative_context = any(neg in message_lower for neg in [
                        "bad", "poor", "issue", "problem", "crisis", "decrease",
                        "less", "lower", "worse", "failed", "inadequate"
                    ])
                    
                    if positive_context:
                        self.inferred_goals[goal_area]["positive_mentions"] += 1
                    if negative_context:
                        self.inferred_goals[goal_area]["negative_mentions"] += 1
                    
                    # Update importance based on frequency and context
                    total_mentions = self.inferred_goals[goal_area]["mentions"]
                    importance_boost = 0.1 * (
                        self.inferred_goals[goal_area]["positive_mentions"] + 
                        self.inferred_goals[goal_area]["negative_mentions"]
                    ) / max(1, total_mentions)
                    
                    self.inferred_goals[goal_area]["importance"] = min(
                        1.0, self.inferred_goals[goal_area]["importance"] + importance_boost
                    )
    
    def _calculate_intrinsic_reward(self, observation, metrics):
        """Calculate intrinsic motivation rewards."""
        intrinsic_reward = 0.0
        
        # 1. Novelty reward - for discovering new metrics or UI elements
        novelty_reward = 0.0
        
        # Reward for new metrics discovered this step
        new_metrics_count = sum(
            1 for m in self.discovered_metrics.values() 
            if m["discovery_time"] == self.env.total_steps
        )
        novelty_reward += new_metrics_count * 0.5  # 0.5 reward per new metric
        
        # Novelty reward for unusual metric values
        for metric_name, value in metrics.items():
            if metric_name in self.discovered_metrics:
                metric_info = self.discovered_metrics[metric_name]
                value_range = max(0.001, metric_info["max_observed"] - metric_info["min_observed"])
                
                # Check if this is a new minimum or maximum
                if value <= metric_info["min_observed"] or value >= metric_info["max_observed"]:
                    novelty_reward += 0.2  # Reward for finding new extremes
        
        # 2. Causal discovery reward - for finding new cause-effect relationships
        causal_reward = 0.0
        new_causal_links = len(self.causal_links) - len(self.knowledge_base.get("causal_links", {}))
        if new_causal_links > 0:
            causal_reward = min(1.0, new_causal_links * 0.2)  # Cap at 1.0
        
        # 3. Progress reward - for sustained improvements in important metrics
        progress_reward = 0.0
        
        # Check progress on metrics that are deemed important
        for metric_name, metric_info in self.discovered_metrics.items():
            if metric_name in metrics and metric_name in self.metric_trends:
                # Only consider metrics with enough history
                if len(self.metric_history[metric_name]) >= 5:
                    # Get current importance score (default to 0.5 if not set)
                    importance = 0.5
                    
                    # If this metric is linked to an inferred goal, use that goal's importance
                    for goal_name, goal_info in self.inferred_goals.items():
                        if metric_name.lower() in goal_name.lower() or goal_name.lower() in metric_name.lower():
                            importance = goal_info["importance"]
                            break
                    
                    trend = self.metric_trends[metric_name]
                    
                    # For negative metrics (like pollution), a decreasing trend is good
                    is_negative_metric = any(neg in metric_name.lower() for neg in [
                        "pollution", "traffic", "congestion", "crime", "fire", "unemployment"
                    ])
                    
                    # Reward progress based on trend direction and metric type
                    if (trend == "increasing" and not is_negative_metric) or (trend == "decreasing" and is_negative_metric):
                        progress_reward += importance * 0.2  # Scale by importance
        
        # Combine all intrinsic rewards with appropriate weights
        intrinsic_reward = (
            self.reward_weights["discovery"] * novelty_reward +
            self.reward_weights["causal"] * causal_reward +
            self.reward_weights["progress"] * progress_reward
        )
        
        return intrinsic_reward
    
    def _update_knowledge_base(self):
        """Update the knowledge base with latest discoveries."""
        self.knowledge_base["game_metrics"] = self.discovered_metrics
        self.knowledge_base["action_effects"] = self.action_effects
        self.knowledge_base["causal_links"] = self.causal_links
        self.knowledge_base["goal_hierarchy"] = self.inferred_goals
    
    def run_exploration_episode(self, timesteps):
        """Run an exploration-focused episode to discover metrics and causal links."""
        # Save original exploration settings
        original_exploration_freq = self.env.exploration_frequency
        
        # Increase exploration during this phase
        self.env.exploration_frequency = min(0.9, original_exploration_freq * 1.5)
        
        # Run episode with focus on diverse actions
        observation, info = self.reset()
        episode_reward = 0
        
        for t in range(timesteps):
            # Choose action with emphasis on less-tried actions
            action = self._select_exploratory_action()
            
            # Take the action
            observation, reward, terminated, truncated, info = self.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        # Restore original exploration settings
        self.env.exploration_frequency = original_exploration_freq
        
        # Return episode stats
        return {
            "episode_reward": episode_reward,
            "episode_length": t + 1,
            "metrics_discovered": len(self.discovered_metrics),
            "causal_links_discovered": len(self.causal_links)
        }
    
    def run_optimization_episode(self, timesteps):
        """Run an optimization-focused episode to maximize goals."""
        # Save original exploration settings
        original_exploration_freq = self.env.exploration_frequency
        
        # Reduce exploration during optimization
        self.env.exploration_frequency = max(0.1, original_exploration_freq * 0.5)
        
        # Run episode with focus on maximizing goals
        observation, info = self.reset()
        episode_reward = 0
        
        for t in range(timesteps):
            # Choose action based on causal model to optimize goals
            action = self._select_optimizing_action(observation)
            
            # Take the action
            observation, reward, terminated, truncated, info = self.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        # Restore original exploration settings
        self.env.exploration_frequency = original_exploration_freq
        
        # Return episode stats
        return {
            "episode_reward": episode_reward,
            "episode_length": t + 1,
            "metrics_improved": self._count_improved_metrics()
        }
    
    def _select_exploratory_action(self):
        """Choose an action that maximizes exploration value."""
        # Get action space size
        action_space_size = self.action_space.n
        
        # Default to uniform random if we don't have enough data
        if not self.action_effects:
            return np.random.randint(0, action_space_size)
        
        # Calculate exploration value for each action
        exploration_values = np.zeros(action_space_size)
        
        for action in range(action_space_size):
            # Count how many times this action has been tried
            if action in self.action_effects:
                # Less tried actions get higher values
                tries = sum(effect["samples"] for effect in self.action_effects[action].values())
                if tries > 0:
                    exploration_values[action] = 1.0 / np.sqrt(tries)
                else:
                    exploration_values[action] = 1.0  # Untried actions get maximum value
            else:
                exploration_values[action] = 1.0  # Untried actions get maximum value
        
        # Add randomness for exploration
        exploration_values += np.random.random(action_space_size) * 0.1
        
        # Select action with highest exploration value
        return np.argmax(exploration_values)
    
    def _select_optimizing_action(self, observation):
        """Choose an action that maximizes progress toward inferred goals."""
        # Get action space size
        action_space_size = self.action_space.n
        
        # Default to base environment's action selection if we don't have enough data
        if not self.causal_links or not self.inferred_goals:
            return self.env._get_guided_exploratory_action()
        
        # Extract current metrics
        current_metrics = self._extract_metrics_from_observation(observation)
        
        # Calculate expected value for each action
        action_values = np.zeros(action_space_size)
        
        for action in range(action_space_size):
            action_value = 0.0
            
            # Evaluate each causal link involving this action
            action_causal_links = [link for link_id, link in self.causal_links.items() 
                                 if link["action"] == action]
            
            for link in action_causal_links:
                metric_name = link["metric"]
                
                # Skip if we don't have the current value of this metric
                if metric_name not in current_metrics:
                    continue
                
                # Find if this metric is related to an inferred goal
                goal_importance = 0.5  # Default medium importance
                for goal_name, goal_info in self.inferred_goals.items():
                    if metric_name.lower() in goal_name.lower() or goal_name.lower() in metric_name.lower():
                        goal_importance = goal_info["importance"]
                        break
                
                # Check if this is a negative metric (where lower is better)
                is_negative_metric = any(neg in metric_name.lower() for neg in [
                    "pollution", "traffic", "congestion", "crime", "fire", "unemployment"
                ])
                
                # Calculate the expected effect
                direction_multiplier = 1 if link["direction"] == "positive" else -1
                if is_negative_metric:
                    direction_multiplier *= -1  # Flip for negative metrics
                
                expected_change = link["average_change"] * direction_multiplier
                
                # Weighted by importance, correlation strength, and expected change
                action_value += goal_importance * link["strength"] * expected_change
            
            # Save the calculated value
            action_values[action] = action_value
        
        # Add some exploration
        exploration_noise = np.random.random(action_space_size) * self.exploration_rate
        action_values += exploration_noise
        
        # Select action with highest value
        return np.argmax(action_values)
    
    def _count_improved_metrics(self):
        """Count metrics that have improved over the episode."""
        improved_count = 0
        
        for metric_name, trend in self.metric_trends.items():
            # For negative metrics, "decreasing" is improvement
            is_negative_metric = any(neg in metric_name.lower() for neg in [
                "pollution", "traffic", "congestion", "crime", "fire", "unemployment"
            ])
            
            if (trend == "increasing" and not is_negative_metric) or (trend == "decreasing" and is_negative_metric):
                improved_count += 1
        
        return improved_count 