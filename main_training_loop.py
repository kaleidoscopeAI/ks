def run_training_loop(optimizer, env, episodes, log_interval):
    """Encapsulates the training loop into a function."""
    for episode in range(episodes):
        current_system_state = env.reset()  # Get initial state as dictionary from env
        state_index = optimizer.discretize_state(current_system_state)  # Pass dictionary to discretize_state
        done = False
        episode_reward = 0

        while not done:
            action_speculation = optimizer.choose_action_and_speculate(state_index)  # Get action and speculation
            action_index = action_speculation['action_index']  # Extract action index to pass to env.step
            next_state_dict, reward, done = env.step(action_index)  # step now expects action_index
            next_state_index = optimizer.discretize_state(next_state_dict)  # Discretize next state from dictionary

            optimizer.update_q_table(state_index, action_index, reward, next_state_index)
            state_index = next_state_index
            episode_reward += reward

        optimizer.decay_exploration_rate()

        if episode % log_interval == 0:
            state_desc = env.get_state_description(state_index)  # Get human-readable state
            logging.info(f"Episode: {episode}, Reward: {episode_reward:.2f}, Exploration Rate: {optimizer.epsilon:.2f}, "
                         f"State: {state_desc}, Energy: {env.energy_level:.2f}, Task Load: {env.task_load:.2f}, Accuracy: {env.learning_accuracy:.2f}")

    logging.info("--- Training Finished ---")
    logging.info(f"Final Exploration Rate: {optimizer.epsilon:.2f}")
    logging.debug("Example Q-Table (first 10 rows, truncated for display):\n" + np.array2string(optimizer.q_table[:10, :]))
