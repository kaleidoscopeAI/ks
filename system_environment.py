import logging

class SystemEnvironment:
    def __init__(self, action_config, reward_weights, initial_state_config):
        """
        Initializes the System Environment with action configurations and reward weights.
        """
        self.action_config = action_config
        self.reward_weights = reward_weights
        self.initial_state_config = initial_state_config
        self.reset()  # Initialize state based on config

        logging.info(f"System Environment initialized with: Action Config={action_config}, Reward Weights={reward_weights}, Initial State={initial_state_config}")

    def reset(self):
        """Resets the environment to the initial state."""
        self.energy_level = self.initial_state_config.get('energy_level', 100.0)
        self.task_load = self.initial_state_config.get('task_load', 20.0)
        self.learning_accuracy = self.initial_state_config.get('learning_accuracy', 0.3)
        self.time_step = 0
        return (self.energy_level, self.task_load, self.learning_accuracy)

    def step(self, action_index):
        """
        Simulates one time step of the system environment.
        """
        priority_action_index, learning_rate_action_index = action_index

        priority_actions = self.action_config.get('priority_actions')
        learning_rate_actions = self.action_config.get('learning_rate_actions')

        chosen_priority_action = priority_actions[priority_action_index]
        chosen_lr_action = learning_rate_actions[learning_rate_action_index]

        # --- System Dynamics ---
        energy_consumption_base = 0.5
        energy_consumption_priority = {"Prioritize_High": 2, "Prioritize_Medium": 1.5, "Prioritize_Low": 1}
        self.energy_level -= energy_consumption_base * energy_consumption_priority[chosen_priority_action]
        self.energy_level = max(0, self.energy_level)

        task_completion_rate = 0.05
        task_load_increase_rate = 0.01
        task_completion_multiplier_priority = {"Prioritize_High": 1.2, "Prioritize_Medium": 1.1, "Prioritize_Low": 0.9}
        self.task_load -= task_completion_rate * task_completion_multiplier_priority[chosen_priority_action] * 100
        self.task_load += task_load_increase_rate * 100
        self.task_load = max(0, min(100, self.task_load))

        learning_rate_base = 0.01
        learning_rate_multiplier = {"Increase_LR": 1.2, "Decrease_LR": 0.8, "Maintain_LR": 1.0}
        effective_lr = learning_rate_base * learning_rate_multiplier[chosen_lr_action]
        learning_improvement = effective_lr * (1 - self.learning_accuracy)
        self.learning_accuracy += learning_improvement
        self.learning_accuracy = min(1.0, self.learning_accuracy)

        # --- Reward Calculation ---
        reward = 0
        reward += self.reward_weights.get('task_load', 1.0) * (self.task_load / 100.0)
        reward += self.reward_weights.get('learning_accuracy', 1.0) * self.learning_accuracy
        if self.energy_level < 20:
            reward -= self.reward_weights.get('low_energy_penalty', 0.5)
        if self.energy_level < 5:
            reward -= self.reward_weights.get('critical_energy_penalty', 1.0)

        next_state = (self.energy_level, self.task_load, self.learning_accuracy)
        done = self.energy_level <= 0
        self.time_step += 1

        return next_state, reward, done

    def get_current_system_state(self):
        """Returns the current system state as a dictionary."""
        return {
            'energy_level': self.energy_level,
            'task_load': self.task_load,
            'learning_progress': self.learning_accuracy
        }
