import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn

# Create a custom environment with Dict observation space
class TestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define observation space similar to DiscoveryEnvironment
        self.observation_space = spaces.Dict({
            "metrics": spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32),
            "minimap": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            "screenshot": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
        })
        # Define action space
        self.action_space = spaces.Discrete(10)
        
    def reset(self, **kwargs):
        obs = {
            "metrics": np.zeros((10,), dtype=np.float32),
            "minimap": np.zeros((84, 84, 3), dtype=np.uint8),
            "screenshot": np.zeros((224, 224, 3), dtype=np.uint8)
        }
        return obs, {}
        
    def step(self, action):
        obs = {
            "metrics": np.zeros((10,), dtype=np.float32),
            "minimap": np.zeros((84, 84, 3), dtype=np.uint8),
            "screenshot": np.zeros((224, 224, 3), dtype=np.uint8)
        }
        reward = 0.0
        done = False
        truncated = False
        info = {}
        return obs, reward, done, truncated, info

print("Creating environment...")
env = make_vec_env(TestEnv, n_envs=1)

print("CombinedExtractor parameters:")
try:
    # Looking at the source code and documentation to understand valid parameters
    import inspect
    print(inspect.signature(CombinedExtractor.__init__))
except Exception as e:
    print(f"Error inspecting CombinedExtractor: {e}")

print("\nTrying to initialize PPO with combined extractor...")
try:
    # Test 1: With cnn_output_dim
    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "features_extractor_kwargs": {"cnn_output_dim": 512},
        "net_arch": [256, 256]
    }
    
    model1 = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    print("SUCCESS: PPO initialization with cnn_output_dim")
except Exception as e:
    print(f"ERROR: PPO initialization with cnn_output_dim failed: {e}")

print("\nTrying alternative parameter structure...")
try:
    # Test 2: Different parameter structure
    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
        "net_arch": [256, 256]
    }
    
    model2 = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    print("SUCCESS: PPO initialization without features_extractor_kwargs")
except Exception as e:
    print(f"ERROR: Alternative PPO initialization failed: {e}")

print("\nTests completed.") 