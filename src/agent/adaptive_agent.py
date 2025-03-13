"""
Adaptive Agent - Meta-controller that switches between different training modes
based on performance metrics and game state feedback.
"""

import time
import logging
import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

# Import the wrapper
from src.utils.observation_wrapper import FlattenObservationWrapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AdaptiveAgent")

class TrainingMode(Enum):
    """Available training modes for the agent"""
    DISCOVERY = "discovery"    # Learn UI elements 
    TUTORIAL = "tutorial"      # Learn basic mechanisms
    VISION = "vision"          # Learn to interpret visual info
    AUTONOMOUS = "autonomous"  # Basic gameplay
    STRATEGIC = "strategic"    # Advanced strategic gameplay with goal discovery

class AdaptiveAgent:
    """
    Meta-controller agent that switches between different training modes
    based on performance metrics and game feedback.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        discovery_config_path: str = "config/discovery_config.yaml",
        vision_config_path: str = "config/vision_guided_config.yaml",
        autonomous_config_path: str = "config/autonomous_config.yaml",
        tutorial_config_path: str = "config/tutorial_guided_config.yaml",
        strategic_config_path: str = "config/strategic_config.yaml"
    ):
        """
        Initialize the adaptive agent.
        
        Args:
            config: Configuration dictionary
            discovery_config_path: Path to discovery mode config
            vision_config_path: Path to vision mode config
            autonomous_config_path: Path to autonomous mode config
            tutorial_config_path: Path to tutorial mode config
            strategic_config_path: Path to strategic mode config
        """
        self.config = config
        self.config_paths = {
            TrainingMode.DISCOVERY: discovery_config_path,
            TrainingMode.VISION: vision_config_path,
            TrainingMode.AUTONOMOUS: autonomous_config_path,
            TrainingMode.TUTORIAL: tutorial_config_path,
            TrainingMode.STRATEGIC: strategic_config_path
        }
        
        # Current active mode
        self.current_mode = TrainingMode.DISCOVERY
        
        # Performance metrics for each mode
        self.mode_metrics = {
            TrainingMode.DISCOVERY: {
                "success_rate": 0.0,
                "reward_avg": 0.0,
                "ui_elements_discovered": 0,
                "stuck_episodes": 0,
                "confidence": 0.0
            },
            TrainingMode.VISION: {
                "success_rate": 0.0,
                "reward_avg": 0.0,
                "objective_completion": 0.0,
                "stuck_episodes": 0,
                "confidence": 0.0
            },
            TrainingMode.AUTONOMOUS: {
                "success_rate": 0.0,
                "reward_avg": 0.0,
                "objective_completion": 0.0,
                "stuck_episodes": 0,
                "game_cycles_completed": 0,
                "confidence": 0.0
            },
            TrainingMode.TUTORIAL: {
                "success_rate": 0.0,
                "reward_avg": 0.0,
                "tutorial_steps_completed": 0,
                "stuck_episodes": 0,
                "confidence": 0.0
            },
            TrainingMode.STRATEGIC: {
                "success_rate": 0.0,
                "reward_avg": 0.0,
                "metrics_discovered": 0,
                "causal_links_discovered": 0,
                "strategy_score": 0.0,
                "stuck_episodes": 0,
                "confidence": 0.0
            }
        }
        
        # Load sub-agents for each mode (will be initialized on demand)
        self.agents = {}
        self.environments = {}
        
        # Mode switching thresholds
        self.switching_thresholds = {
            "min_discovery_confidence": config.get("min_discovery_confidence", 0.7),
            "min_ui_elements": config.get("min_ui_elements", 20),
            "min_tutorial_steps": config.get("min_tutorial_steps", 5),
            "max_stuck_episodes": config.get("max_stuck_episodes", 5),
            "min_vision_confidence": config.get("min_vision_confidence", 0.6),
            "min_autonomous_confidence": config.get("min_autonomous_confidence", 0.8),
            "min_game_cycles": config.get("min_game_cycles", 10)
        }
        
        # Mode history for analysis
        self.mode_history = []
        self.mode_switch_reasons = []
        
        # Initialize metrics history
        self.metrics_history = {mode: [] for mode in TrainingMode}
        
        # Shared knowledge base across modes
        self.knowledge_base = {
            "ui_elements": {},           # UI elements discovered
            "game_metrics": {},          # Game metrics discovered
            "action_effects": {},        # Effects of actions on game state
            "causal_links": {},          # Causal relationships between actions and outcomes
            "strategy_patterns": {},     # Successful strategy patterns
            "game_rules": {},            # Inferred game rules
            "goal_hierarchy": {}         # Hierarchy of game goals
        }
        
        logger.info("Adaptive Agent initialized with discovery mode as default")
    
    def initialize_current_mode(self):
        """Initialize the current training mode's agent and environment"""
        from src.agent.discovery_agent import DiscoveryAgent
        from src.agent.vision_agent import VisionAgent
        from src.agent.autonomous_agent import AutonomousAgent
        from src.agent.tutorial_agent import TutorialAgent
        from src.agent.strategic_agent import StrategicAgent
        
        from src.environment.discovery_env import DiscoveryEnvironment
        from src.environment.vision_env import VisionEnvironment
        from src.environment.autonomous_env import AutonomousEnvironment
        from src.environment.tutorial_guided_env import TutorialGuidedCS2Environment as TutorialEnvironment
        from src.environment.strategic_env import StrategicEnvironment
        
        import yaml
        
        # Load configuration
        config_path = self.config_paths[self.current_mode]
        with open(config_path, 'r') as f:
            mode_config = yaml.safe_load(f)
        
        # Inject knowledge base into config
        mode_config["knowledge_base"] = self.knowledge_base
        
        # Initialize environment and agent based on current mode
        if self.current_mode == TrainingMode.DISCOVERY and TrainingMode.DISCOVERY not in self.environments:
            env = DiscoveryEnvironment(mode_config)
            # Wrap the environment to handle dictionary observations
            wrapped_env = FlattenObservationWrapper(env)
            self.environments[TrainingMode.DISCOVERY] = wrapped_env
            self.agents[TrainingMode.DISCOVERY] = DiscoveryAgent(
                self.environments[TrainingMode.DISCOVERY], mode_config
            )
            
        elif self.current_mode == TrainingMode.VISION and TrainingMode.VISION not in self.environments:
            env = VisionEnvironment(mode_config)
            # Wrap the environment to handle dictionary observations
            wrapped_env = FlattenObservationWrapper(env)
            self.environments[TrainingMode.VISION] = wrapped_env
            self.agents[TrainingMode.VISION] = VisionAgent(
                self.environments[TrainingMode.VISION], mode_config
            )
            
        elif self.current_mode == TrainingMode.AUTONOMOUS and TrainingMode.AUTONOMOUS not in self.environments:
            env = AutonomousEnvironment(mode_config)
            # Wrap the environment to handle dictionary observations
            wrapped_env = FlattenObservationWrapper(env)
            self.environments[TrainingMode.AUTONOMOUS] = wrapped_env
            self.agents[TrainingMode.AUTONOMOUS] = AutonomousAgent(
                self.environments[TrainingMode.AUTONOMOUS], mode_config
            )
            
        elif self.current_mode == TrainingMode.TUTORIAL and TrainingMode.TUTORIAL not in self.environments:
            env = TutorialEnvironment(mode_config)
            # Wrap the environment to handle dictionary observations
            wrapped_env = FlattenObservationWrapper(env)
            self.environments[TrainingMode.TUTORIAL] = wrapped_env
            self.agents[TrainingMode.TUTORIAL] = TutorialAgent(
                self.environments[TrainingMode.TUTORIAL], mode_config
            )
            
        elif self.current_mode == TrainingMode.STRATEGIC and TrainingMode.STRATEGIC not in self.environments:
            env = StrategicEnvironment(mode_config)
            # Wrap the environment to handle dictionary observations
            wrapped_env = FlattenObservationWrapper(env)
            self.environments[TrainingMode.STRATEGIC] = wrapped_env
            self.agents[TrainingMode.STRATEGIC] = StrategicAgent(
                self.environments[TrainingMode.STRATEGIC], mode_config
            )
        
        logger.info(f"Initialized {self.current_mode.value} mode")
    
    def update_metrics(self, episode_info: Dict[str, Any]):
        """
        Update performance metrics for the current mode based on episode feedback.
        
        Args:
            episode_info: Dictionary containing episode information and metrics
        """
        mode = self.current_mode
        metrics = self.mode_metrics[mode]
        
        # Update general metrics for all modes
        metrics["reward_avg"] = (metrics["reward_avg"] * 0.9) + (episode_info.get("reward", 0) * 0.1)
        metrics["success_rate"] = (metrics["success_rate"] * 0.9) + (episode_info.get("success", 0) * 0.1)
        
        # Update mode-specific metrics
        if mode == TrainingMode.DISCOVERY:
            ui_elements = episode_info.get("ui_elements_discovered", 0)
            if ui_elements > metrics["ui_elements_discovered"]:
                metrics["ui_elements_discovered"] = ui_elements
                # Reset stuck counter when discovering new elements
                metrics["stuck_episodes"] = 0
            else:
                metrics["stuck_episodes"] += 1
            
        elif mode == TrainingMode.VISION:
            completion = episode_info.get("objective_completion", 0)
            if completion > metrics["objective_completion"]:
                metrics["objective_completion"] = completion
                metrics["stuck_episodes"] = 0
            else:
                metrics["stuck_episodes"] += 1
                
        elif mode == TrainingMode.AUTONOMOUS:
            completion = episode_info.get("objective_completion", 0)
            game_cycles = episode_info.get("game_cycles_completed", 0)
            
            # Update completion metric
            if completion > metrics["objective_completion"]:
                metrics["objective_completion"] = completion
                metrics["stuck_episodes"] = 0
            else:
                metrics["stuck_episodes"] += 1
                
            # Update game cycles completed
            if game_cycles > metrics["game_cycles_completed"]:
                metrics["game_cycles_completed"] = game_cycles
                
        elif mode == TrainingMode.TUTORIAL:
            steps = episode_info.get("tutorial_steps_completed", 0)
            if steps > metrics["tutorial_steps_completed"]:
                metrics["tutorial_steps_completed"] = steps
                metrics["stuck_episodes"] = 0
            else:
                metrics["stuck_episodes"] += 1
                
        elif mode == TrainingMode.STRATEGIC:
            # Update strategic metrics
            metrics_discovered = episode_info.get("metrics_discovered", 0)
            causal_links = episode_info.get("causal_links_discovered", 0)
            strategy_score = episode_info.get("strategy_score", 0.0)
            
            # Check for progress
            progress_made = False
            
            if metrics_discovered > metrics["metrics_discovered"]:
                metrics["metrics_discovered"] = metrics_discovered
                progress_made = True
                
            if causal_links > metrics["causal_links_discovered"]:
                metrics["causal_links_discovered"] = causal_links
                progress_made = True
                
            if strategy_score > metrics["strategy_score"]:
                metrics["strategy_score"] = strategy_score
                progress_made = True
                
            if progress_made:
                metrics["stuck_episodes"] = 0
            else:
                metrics["stuck_episodes"] += 1
        
        # Update confidence metric (customized per mode)
        if mode == TrainingMode.DISCOVERY:
            # Discovery confidence based on UI elements and success rate
            metrics["confidence"] = (
                min(1.0, metrics["ui_elements_discovered"] / self.switching_thresholds["min_ui_elements"]) * 0.7 +
                metrics["success_rate"] * 0.3
            )
            
        elif mode == TrainingMode.TUTORIAL:
            # Tutorial confidence based on steps completed
            metrics["confidence"] = (
                min(1.0, metrics["tutorial_steps_completed"] / self.switching_thresholds["min_tutorial_steps"]) * 0.7 +
                metrics["success_rate"] * 0.3
            )
            
        elif mode == TrainingMode.VISION:
            # Vision confidence based on objective completion
            metrics["confidence"] = (
                metrics["objective_completion"] * 0.7 +
                metrics["success_rate"] * 0.3
            )
            
        elif mode == TrainingMode.AUTONOMOUS:
            # Autonomous confidence based on objective completion, reward, and game cycles
            metrics["confidence"] = (
                metrics["objective_completion"] * 0.4 +
                min(1.0, metrics["reward_avg"] / 100) * 0.3 +
                min(1.0, metrics["game_cycles_completed"] / self.switching_thresholds["min_game_cycles"]) * 0.1 +
                metrics["success_rate"] * 0.2
            )
            
        elif mode == TrainingMode.STRATEGIC:
            # Strategic confidence based on strategy score, metrics discovered, and causal links
            total_metrics = max(1, self.knowledge_base["game_metrics"].get("total_discovered", 1))
            total_causal_links = max(10, len(self.knowledge_base.get("causal_links", {})))
            
            metrics["confidence"] = (
                metrics["strategy_score"] * 0.5 +
                min(1.0, metrics["metrics_discovered"] / total_metrics) * 0.2 +
                min(1.0, metrics["causal_links_discovered"] / total_causal_links) * 0.2 +
                metrics["success_rate"] * 0.1
            )
        
        # Store metrics history
        self.metrics_history[mode].append(metrics.copy())
        
        # Update knowledge base with new information from episode
        self._update_knowledge_base(episode_info)
        
        logger.info(f"Updated metrics for {mode.value} mode: confidence={metrics['confidence']:.2f}, "
                    f"reward_avg={metrics['reward_avg']:.2f}, stuck={metrics['stuck_episodes']}")
    
    def _update_knowledge_base(self, episode_info: Dict[str, Any]):
        """Update the shared knowledge base with new information from the episode"""
        # Update UI elements
        if "discovered_ui_elements" in episode_info:
            for ui_element in episode_info["discovered_ui_elements"]:
                self.knowledge_base["ui_elements"][ui_element["id"]] = ui_element
        
        # Update game metrics
        if "discovered_metrics" in episode_info:
            for metric_name, metric_info in episode_info["discovered_metrics"].items():
                self.knowledge_base["game_metrics"][metric_name] = metric_info
            
            # Update total discovered count
            self.knowledge_base["game_metrics"]["total_discovered"] = len(self.knowledge_base["game_metrics"])
        
        # Update action effects
        if "action_effects" in episode_info:
            for action, effects in episode_info["action_effects"].items():
                if action not in self.knowledge_base["action_effects"]:
                    self.knowledge_base["action_effects"][action] = []
                self.knowledge_base["action_effects"][action].append(effects)
        
        # Update causal links
        if "causal_links" in episode_info:
            for cause, effect in episode_info["causal_links"].items():
                self.knowledge_base["causal_links"][cause] = effect
        
        # Update strategy patterns
        if "strategy_patterns" in episode_info:
            for pattern_name, pattern_info in episode_info["strategy_patterns"].items():
                self.knowledge_base["strategy_patterns"][pattern_name] = pattern_info
        
        # Update game rules
        if "game_rules" in episode_info:
            for rule_name, rule_info in episode_info["game_rules"].items():
                self.knowledge_base["game_rules"][rule_name] = rule_info
        
        # Update goal hierarchy
        if "goal_hierarchy" in episode_info:
            for goal, priority in episode_info["goal_hierarchy"].items():
                self.knowledge_base["goal_hierarchy"][goal] = priority
    
    def should_switch_mode(self) -> Tuple[bool, Optional[TrainingMode], str]:
        """
        Determine if the agent should switch modes based on current metrics.
        
        Returns:
            Tuple containing:
                - Boolean indicating if mode should be switched
                - The new mode to switch to (or None if no switch)
                - Reason for switching
        """
        current_mode = self.current_mode
        metrics = self.mode_metrics[current_mode]
        
        # Check if agent is stuck in current mode
        if metrics["stuck_episodes"] >= self.switching_thresholds["max_stuck_episodes"]:
            # If stuck in discovery mode, try tutorial mode
            if current_mode == TrainingMode.DISCOVERY:
                return True, TrainingMode.TUTORIAL, "Stuck in discovery mode, switching to tutorial for guidance"
            
            # If stuck in tutorial mode, try vision mode
            elif current_mode == TrainingMode.TUTORIAL:
                return True, TrainingMode.VISION, "Stuck in tutorial mode, switching to vision-guided approach"
            
            # If stuck in vision mode, try autonomous mode if we have enough confidence
            elif current_mode == TrainingMode.VISION and metrics["confidence"] >= self.switching_thresholds["min_vision_confidence"]:
                return True, TrainingMode.AUTONOMOUS, "Vision mode confident enough, upgrading to autonomous mode"
            
            # If stuck in autonomous mode, check if we have enough confidence for strategic mode
            elif current_mode == TrainingMode.AUTONOMOUS and metrics["confidence"] >= self.switching_thresholds["min_autonomous_confidence"]:
                if metrics["game_cycles_completed"] >= self.switching_thresholds["min_game_cycles"]:
                    return True, TrainingMode.STRATEGIC, "Autonomous mode mastered, transitioning to strategic gameplay"
                else:
                    # Not enough game cycles yet, continue with autonomous mode
                    return False, None, "Need more game cycles in autonomous mode before strategic transition"
            
            # If stuck in autonomous mode without enough confidence, revert to discovery
            elif current_mode == TrainingMode.AUTONOMOUS:
                return True, TrainingMode.DISCOVERY, "Stuck in autonomous mode, reverting to discovery to gather more information"
            
            # If stuck in strategic mode, revert to autonomous for more basic practice
            elif current_mode == TrainingMode.STRATEGIC:
                return True, TrainingMode.AUTONOMOUS, "Stuck in strategic mode, reverting to autonomous mode for more practice"
        
        # Progression-based switching (confidence-based upgrades)
        if current_mode == TrainingMode.DISCOVERY and metrics["confidence"] >= self.switching_thresholds["min_discovery_confidence"]:
            return True, TrainingMode.VISION, "Discovery confidence threshold reached, upgrading to vision mode"
            
        elif current_mode == TrainingMode.TUTORIAL and metrics["tutorial_steps_completed"] >= self.switching_thresholds["min_tutorial_steps"]:
            return True, TrainingMode.VISION, "Completed sufficient tutorial steps, upgrading to vision mode"
            
        elif current_mode == TrainingMode.VISION and metrics["confidence"] >= 0.8:
            return True, TrainingMode.AUTONOMOUS, "Vision mode mastered, upgrading to autonomous mode"
            
        elif current_mode == TrainingMode.AUTONOMOUS and metrics["confidence"] >= self.switching_thresholds["min_autonomous_confidence"]:
            # Check if we've completed enough game cycles to understand game dynamics
            if metrics["game_cycles_completed"] >= self.switching_thresholds["min_game_cycles"]:
                return True, TrainingMode.STRATEGIC, "Autonomous mode mastered, transitioning to strategic gameplay"
        
        # No mode switch needed
        return False, None, "Continuing with current mode"
    
    def switch_mode(self, new_mode: TrainingMode, reason: str):
        """
        Switch to a new training mode.
        
        Args:
            new_mode: The mode to switch to
            reason: Reason for the switch
        """
        old_mode = self.current_mode
        self.current_mode = new_mode
        
        # Record the mode switch
        timestamp = time.time()
        self.mode_history.append((timestamp, old_mode, new_mode))
        self.mode_switch_reasons.append((timestamp, reason))
        
        # Initialize the new mode if needed
        if new_mode not in self.agents:
            self.initialize_current_mode()
        
        logger.info(f"Switched from {old_mode.value} to {new_mode.value} mode: {reason}")
    
    def train(self, total_timesteps: int, progress_callback=None):
        """
        Train the agent, automatically switching between modes as needed.
        
        Args:
            total_timesteps: Total timesteps to train for
            progress_callback: Optional callback function for progress updates
        """
        timesteps_used = 0
        episode_count = 0
        
        # Ensure current mode is initialized
        if self.current_mode not in self.agents:
            self.initialize_current_mode()
        
        while timesteps_used < total_timesteps:
            # Get current agent and environment
            agent = self.agents[self.current_mode]
            env = self.environments[self.current_mode]
            
            # Determine timesteps for this mode (at least 100, or remaining)
            mode_timesteps = min(500, total_timesteps - timesteps_used)
            if mode_timesteps <= 0:
                break
                
            # Train current mode
            episode_info = agent.train_episode(mode_timesteps)
            episode_steps = episode_info.get("steps", 0)
            timesteps_used += episode_steps
            episode_count += 1
            
            # Update metrics based on episode feedback
            self.update_metrics(episode_info)
            
            # Check if we should switch modes
            should_switch, new_mode, reason = self.should_switch_mode()
            if should_switch and new_mode is not None:
                self.switch_mode(new_mode, reason)
            
            # Report progress
            if progress_callback:
                progress = {
                    "timesteps_used": timesteps_used,
                    "total_timesteps": total_timesteps,
                    "episode_count": episode_count,
                    "current_mode": self.current_mode.value,
                    "mode_metrics": self.mode_metrics.copy(),
                    "mode_history": self.mode_history.copy(),
                    "knowledge_base_stats": {
                        "ui_elements": len(self.knowledge_base["ui_elements"]),
                        "game_metrics": len(self.knowledge_base["game_metrics"]),
                        "action_effects": len(self.knowledge_base["action_effects"]),
                        "causal_links": len(self.knowledge_base["causal_links"]),
                        "strategy_patterns": len(self.knowledge_base["strategy_patterns"]),
                        "game_rules": len(self.knowledge_base["game_rules"]),
                        "goal_hierarchy": len(self.knowledge_base["goal_hierarchy"])
                    }
                }
                progress_callback(progress)
        
        logger.info(f"Training completed: {timesteps_used}/{total_timesteps} timesteps used over {episode_count} episodes")
        return {
            "timesteps_used": timesteps_used,
            "episode_count": episode_count,
            "final_mode": self.current_mode.value,
            "mode_history": self.mode_history,
            "mode_metrics": self.mode_metrics,
            "mode_switch_reasons": self.mode_switch_reasons,
            "knowledge_base_stats": {
                "ui_elements": len(self.knowledge_base["ui_elements"]),
                "game_metrics": len(self.knowledge_base["game_metrics"]),
                "action_effects": len(self.knowledge_base["action_effects"]),
                "causal_links": len(self.knowledge_base["causal_links"]),
                "strategy_patterns": len(self.knowledge_base["strategy_patterns"]),
                "game_rules": len(self.knowledge_base["game_rules"]),
                "goal_hierarchy": len(self.knowledge_base["goal_hierarchy"])
            }
        }
    
    def evaluate(self, eval_episodes: int = 10):
        """
        Evaluate the current agent's performance.
        
        Args:
            eval_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation metrics
        """
        # Get current agent and environment
        agent = self.agents[self.current_mode]
        
        # Run evaluation
        return agent.evaluate(eval_episodes)
    
    def save(self, save_path: str):
        """
        Save all trained agents and the meta-controller state.
        
        Args:
            save_path: Base path for saving
        """
        import os
        import json
        import pickle
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save each agent
        for mode, agent in self.agents.items():
            agent_path = os.path.join(save_path, f"{mode.value}_agent")
            agent.save(agent_path)
        
        # Save meta-controller state
        meta_state = {
            "current_mode": self.current_mode.value,
            "mode_metrics": {mode.value: metrics for mode, metrics in self.mode_metrics.items()},
            "mode_history": [(ts, old.value, new.value) for ts, old, new in self.mode_history],
            "mode_switch_reasons": self.mode_switch_reasons,
            "switching_thresholds": self.switching_thresholds
        }
        
        with open(os.path.join(save_path, "meta_controller.json"), "w") as f:
            json.dump(meta_state, f, indent=2)
        
        # Save knowledge base
        with open(os.path.join(save_path, "knowledge_base.json"), "w") as f:
            json.dump(self.knowledge_base, f, indent=2)
        
        logger.info(f"Adaptive agent saved to {save_path}")
    
    def load(self, load_path: str):
        """
        Load all trained agents and the meta-controller state.
        
        Args:
            load_path: Base path for loading
        """
        import os
        import json
        
        # Load meta-controller state
        meta_path = os.path.join(load_path, "meta_controller.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta_state = json.load(f)
            
            # Restore meta-controller state
            self.current_mode = TrainingMode(meta_state["current_mode"])
            self.mode_metrics = {TrainingMode(mode): metrics for mode, metrics in meta_state["mode_metrics"].items()}
            self.mode_history = [(ts, TrainingMode(old), TrainingMode(new)) for ts, old, new in meta_state["mode_history"]]
            self.mode_switch_reasons = meta_state["mode_switch_reasons"]
            self.switching_thresholds = meta_state["switching_thresholds"]
            
            # Load knowledge base
            kb_path = os.path.join(load_path, "knowledge_base.json")
            if os.path.exists(kb_path):
                with open(kb_path, "r") as f:
                    self.knowledge_base = json.load(f)
            
            # Initialize and load each agent
            for mode in TrainingMode:
                agent_path = os.path.join(load_path, f"{mode.value}_agent")
                if os.path.exists(agent_path):
                    # Initialize the mode
                    if mode not in self.agents:
                        self.current_mode = mode  # Temporarily set current mode
                        self.initialize_current_mode()
                    
                    # Load the agent
                    self.agents[mode].load(agent_path)
            
            # Restore current mode
            self.current_mode = TrainingMode(meta_state["current_mode"])
            
            logger.info(f"Adaptive agent loaded from {load_path}")
            return True
        else:
            logger.error(f"Meta-controller state not found at {load_path}")
            return False 