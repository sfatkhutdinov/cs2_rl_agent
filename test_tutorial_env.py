import yaml
import logging
import sys
from src.environment.tutorial_guided_env import TutorialGuidedCS2Environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("test_tutorial_env")

def main():
    """Test the TutorialGuidedCS2Environment class."""
    try:
        logger.info("Loading configuration...")
        with open("config/tutorial_guided_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        
        logger.info("Creating environment...")
        env = TutorialGuidedCS2Environment(
            base_env_config=config["environment"],
            observation_config=config["observation"],
            vision_config=config.get("vision", {}),
            tutorial_frequency=config.get("tutorial_frequency", 0.7),
            tutorial_timeout=config.get("tutorial_timeout", 300),
            tutorial_reward_multiplier=config.get("tutorial_reward_multiplier", 2.0),
            use_fallback_mode=True,
            logger=logger
        )
        logger.info("Environment created successfully!")
        
        logger.info("Resetting environment...")
        obs, info = env.reset()
        logger.info(f"Reset successful. Observation shape: {type(obs)}")
        
        logger.info("Taking a step...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(f"Step successful. Reward: {reward}")
        
        logger.info("Closing environment...")
        env.close()
        logger.info("Environment closed successfully")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main() 