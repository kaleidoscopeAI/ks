import numpy as np
import random
import logging
from typing import Dict

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QLearningOptimizer:
    def __init__(self, state_space_config, action_space_config, hyperparameters):
        """
        Initializes the Q-Learning Optimizer with configurations and hyperparameters.
        """
        self.state_space_config = state_space_config
        self.action_space_config = action_space_config
        self.hyperparameters = hyperparameters

        # Dynamically determine state and action space sizes from config
        state_space_size_tuple = tuple(len(config['bins']) - 1 for config in state_space_config.values())
        action_space_size_tuple = tuple(len(config) for config in action_space_config.values())

        self.state_space_size = state_space_size_tuple
        self.action_space_size = action_space_size_tuple

        self.lr = hyperparameters.get('learning_rate', 0.1)
        self.gamma = hyperparameters.get('discount_factor', 0.9)
        self.epsilon = hyperparameters.get('exploration_rate', 1.0)
        self.epsilon_decay = hyperparameters.get('exploration_decay_rate', 0.001)
        self.min_epsilon = hyperparameters.get('min_exploration_rate', 0.01)

        # Calculate total number of states and actions for Q-table dimensions
        num_states = np.prod(self.state_space_size)
        num_actions = np.prod(self.action_space_size)

        self.q_table = np.zeros((num_states, num_actions))

        # Create mappings
        self.state_index_map = self._create_index_mapper(self.state_space_size)
        self.action_index_map = self._create_index_mapper(self.action_space_size)
        self.reverse_action_index_map = self._create_reverse_index_mapper(self.action_space_size)

        logging.info(f"Q-Learning Optimizer initialized with: Hyperparameters={hyperparameters}, State Space Size={self.state_space_size}, Action Space Size={self.action_space_size}")

    def _create_index_mapper(self, dimensions):
        """Creates index mapper to convert multi-dimensional indices to flat index"""
        cumulative_products = np.cumprod([1] + list(dimensions[:-1]))
        def index_mapper(indices):
            return np.sum(np.array(indices) * cumulative_products)
        return index_mapper

    def _create_reverse_index_mapper(self, dimensions):
        """Creates reverse index mapper to convert flat index back to multi-dimensional indices"""
        cumulative_products = np.cumprod([1] + list(dimensions[:-1]))
        def reverse_index_mapper(index):
            indices = []
            for i in range(len(dimensions) - 1, -1, -1):
                dim_size = dimensions[i]
                product = cumulative_products[i]
                indices.insert(0, index // product)
                index %= product
            return tuple(indices)
        return reverse_index_mapper

    def discretize_state(self, system_state):
        """
        Discretizes the system state variables into discrete state indices using configuration.
        """
        state_indices = []

        for state_variable, config in self.state_space_config.items():
            value = system_state[state_variable]
            bins = config['bins']
            index = np.digitize([value], bins)[0] - 1
            index = max(0, min(index, len(bins) - 2))  # Ensure index is within bounds
            state_indices.append(index)

        return tuple(state_indices)

    def choose_action_and_speculate(self, state_index):
        """
        Chooses an action based on epsilon-greedy policy and generates speculations.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: Random action
            action_index = tuple(random.randint(0, dim - 1) for dim in self.action_space_size)
            action_speculation = {
                'action_index': action_index,
                'action_type': 'exploration',
                'reason': 'Exploring random action.'
            }
            logging.debug(f"State {state_index}: Exploring - Random action {action_index}")
        else:
            # Exploitation: Best action based on Q-values
            flat_state_index = self.state_index_map(state_index)
            q_values = self.q_table[flat_state_index, :]
            best_action_flat_index = np.argmax(q_values)
            action_index = self.reverse_action_index_map(best_action_flat_index)

            action_speculation = {
                'action_index': action_index,
                'action_type': 'exploitation',
                'q_value_chosen_action': float(q_values[best_action_flat_index]),
                'top_actions_q_values': {},  # Placeholder for future enhancements
                'reason': f'Exploiting policy. Action chosen with highest Q-value: {float(q_values[best_action_flat_index]):.2f}',
            }
            logging.debug(f"State {state_index}: Exploiting - Best action {action_index}, Q-values: {q_values}")

        # Add action names to speculation
        action_names = {}
        priority_actions_names = self.action_space_config.get('priority_actions')
        learning_rate_actions_names = self.action_space_config.get('learning_rate_actions')
        if priority_actions_names:
            action_names['priority_action'] = priority_actions_names[action_index[0]]
        if learning_rate_actions_names:
            action_names['learning_rate_action'] = learning_rate_actions_names[action_index[1]]
        action_speculation['action_names'] = action_names

        return action_speculation

    def update_q_table(self, state_index, action_index, reward, next_state_index):
        """Update the Q-table using the Q-Learning update rule."""
        flat_state_index = self.state_index_map(state_index)
        flat_action_index = self.action_index_map(action_index)
        flat_next_state_index = self.state_index_map(next_state_index)

        best_next_action_q_value = np.max(self.q_table[flat_next_state_index, :])

        q_value_update = self.lr * (reward + self.gamma * best_next_action_q_value - self.q_table[flat_state_index, flat_action_index])
        self.q_table[flat_state_index, flat_action_index] += q_value_update
        logging.debug(f"Q-Table updated for state {state_index}, action {action_index}, reward {reward}, next state {next_state_index}. Update value: {q_value_update:.4f}")

    def decay_exploration_rate(self):
        """Decays the exploration rate (epsilon) over time."""
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.min_epsilon)
