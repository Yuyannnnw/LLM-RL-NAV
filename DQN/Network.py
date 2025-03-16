import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Flexible Deep Q-Network (DQN) model.
    - Accepts an architecture by specifying layer sizes.
    - Takes state as input and outputs Q-values for each action.
    """
    def __init__(self, state_dim, action_dim, hidden_layers=[256, 128]):
        """
        Initialize the Q-Network.
        :param state_dim: Number of input features (size of state space)
        :param action_dim: Number of possible actions
        :param hidden_layers: List of hidden layer sizes (e.g., [64, 128, 256])
        """
        super(QNetwork, self).__init__()
        
        # Define the first layer (input to first hidden layer)
        layers = [nn.Linear(state_dim, hidden_layers[0]), nn.ReLU()]

        # Add intermediate hidden layers dynamically
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())

        # Define the output layer (last hidden layer to action output)
        layers.append(nn.Linear(hidden_layers[-1], action_dim)) # No activation on the output layer (raw Q-values)

        # Register all layers as a Sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass through the network.
        :param state: Tensor representing the environment state
        :return: Q-values for each action
        """
        return self.model(state)  