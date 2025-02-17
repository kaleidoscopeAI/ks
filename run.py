import logging
from core.q_learning_optimizer import QLearningOptimizer
from system_environment import SystemEnvironment
import time

# Initialize configurations (example)
state_space_config = {
    'energy_level': {'bins': [0, 20, 50, 80, 100]},
    'task_load': {'bins': [0, 30, 70, 100]},
    'learning_progress': {'bins': [0, 0.5, 0.8, 1.0]}
}

action_space_config = {
    'priority_actions': ["Prioritize_High", "Prioritize_Medium", "Prioritize_Low"],
    'learning_rate_actions': ["Increase_LR", "Decrease_LR", "Maintain_LR"]
}

hyperparameters = {
    'learning_rate': 0.1,
    'discount_factor': 0.9,
    'exploration_rate': 1.0,
    'exploration_decay_rate': 0.001,
    'min_exploration_rate': 0.01
}

reward_weights = {
    'task_load': 1.0,
    'learning_accuracy': 1.0,
    'low_energy_penalty': 0.5,
    'critical_energy_penalty': 1.0
}

initial_state_config = {
    'energy_level': 100.0,
    'task_load': 20.0,
    'learning_accuracy': 0.3
}

# Initialize components
optimizer = QLearningOptimizer(state_space_config, action_space_config, hyperparameters)
env = SystemEnvironment(action_config=action_space_config, reward_weights=reward_weights, initial_state_config=initial_state_config)

# Training loop
def run_training_loop(optimizer, env, episodes=1000, log_interval=100):
    for episode in range(episodes):
        current_system_state = env.reset()
        state_index = optimizer.discretize_state(current_system_state)
        done = False
        episode_reward = 0

        while not done:
            action_speculation = optimizer.choose_action_and_speculate(state_index)
            action_index = action_speculation['action_index']
            next_state_dict, reward, done = env.step(action_index)
            next_state_index = optimizer.discretize_state(next_state_dict)
            optimizer.update_q_table(state_index, action_index, reward, next_state_index)
            state_index = next_state_index
            episode_reward += reward

        optimizer.decay_exploration_rate()

        if episode % log_interval == 0:
            logging.info(f"Episode: {episode}, Reward: {episode_reward:.2f}, Exploration Rate: {optimizer.epsilon:.2f}")

    logging.info("--- Training Finished ---")

# Run the training loop
run_training_loop(optimizer, env)
