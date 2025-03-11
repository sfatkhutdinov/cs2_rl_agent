import gym
import torch as th
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Type, Union

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    
    This features extractor handles dictionary observations with both image and vector components.
    It processes each component separately and then combines them.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_output_dim: int = 256,
        mlp_extractor_hidden_sizes: List[int] = [256, 256]
    ):
        """
        Initialize the feature extractor.
        
        Args:
            observation_space: The observation space
            cnn_output_dim: Dimension of the CNN output features
            mlp_extractor_hidden_sizes: List of hidden sizes for MLP extractor
        """
        # Save for reference
        self.observation_space = observation_space
        self.cnn_output_dim = cnn_output_dim
        
        # Initialize
        super().__init__(observation_space, features_dim=1)  # Will be updated later
        
        extractors: Dict[str, nn.Module] = {}
        total_concat_size = 0
        
        # Identify which keys are images and which are vectors
        self.image_keys = []
        self.vector_keys = []
        
        for key, subspace in observation_space.spaces.items():
            if isinstance(subspace, gym.spaces.Box):
                # Check if it's an image (3+ dimensions) or a vector
                if len(subspace.shape) >= 3:
                    self.image_keys.append(key)
                    # For images: CNN feature extractor
                    n_input_channels = subspace.shape[2]  # RGB channels
                    cnn = nn.Sequential(
                        nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                        nn.ReLU(),
                        nn.Flatten(),
                    )
                    
                    # Get the size of the flattened features
                    # We need to make a forward pass to determine this
                    with th.no_grad():
                        sample = th.as_tensor(subspace.sample()[None]).float()
                        # Handle grayscale images
                        if len(sample.shape) == 3:
                            sample = sample.unsqueeze(1)
                        else:
                            # Move channel to the correct dimension
                            sample = sample.permute(0, 3, 1, 2)
                        cnn_output = cnn(sample)
                    
                    # Add a linear layer to match the desired output dimension
                    extractors[key] = nn.Sequential(
                        cnn,
                        nn.Linear(cnn_output.shape[1], cnn_output_dim),
                        nn.ReLU()
                    )
                    total_concat_size += cnn_output_dim
                else:
                    self.vector_keys.append(key)
                    # For vectors: Flatten + MLP feature extractor
                    dim = get_flattened_obs_dim(subspace)
                    
                    # Build MLP with provided hidden sizes
                    layers = []
                    input_size = dim
                    
                    for hidden_size in mlp_extractor_hidden_sizes:
                        layers.append(nn.Linear(input_size, hidden_size))
                        layers.append(nn.ReLU())
                        input_size = hidden_size
                    
                    extractors[key] = nn.Sequential(*layers)
                    total_concat_size += mlp_extractor_hidden_sizes[-1]
            else:
                self.vector_keys.append(key)
                # For other spaces: just flatten
                dim = 1  # Default dimension for other spaces
                
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(dim, 16),
                    nn.ReLU()
                )
                total_concat_size += 16
        
        self.extractors = nn.ModuleDict(extractors)
        
        # Update the features dim
        self._features_dim = total_concat_size
    
    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Process the observations through the feature extractors.
        
        Args:
            observations: Dictionary of observations
            
        Returns:
            Features tensor
        """
        encoded_tensor_list = []
        
        # Process each observation
        for key, extractor in self.extractors.items():
            observation = observations[key]
            
            # Handle images specially for CNN
            if key in self.image_keys:
                # If it's a batch of images, we need to ensure proper dimensions
                if len(observation.shape) == 4:
                    # Already batched with channels last, convert to channels first
                    observation = observation.permute(0, 3, 1, 2)
                elif len(observation.shape) == 3:
                    # Single image with channels last, add batch dim and convert
                    observation = observation.unsqueeze(0).permute(0, 3, 1, 2)
            
            # Extract features
            encoded_tensor_list.append(extractor(observation))
            
        # Concatenate features
        return th.cat(encoded_tensor_list, dim=1) 