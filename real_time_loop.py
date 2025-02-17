def run_real_time_loop(optimizer, env, control_interval_seconds=5):
    """Encapsulates the real-time control loop."""
    logging.info("Starting Real-time Control Loop...")
    import time

    while True:
        try:
            # 1. Get System State
            current_system_state_dict = env.get_current_system_state()
            state_index = optimizer.discretize_state(current_system_state_dict)

            # 2. Optimizer Action Selection and Speculation
            action_speculation = optimizer.choose_action_and_speculate(state_index)
            action_index = action_speculation['action_index']

            # 3. Apply Action (using Environment step for simulation)
            next_state_dict, reward, done = env.step(action_index)

            # 4. Log Speculation and System State
            state_description = env.get_state_description(state_index)
            logging.info(f"--- Control Step --- State: {state_description}, Action Speculation: {action_speculation}")
            time.sleep(control_interval_seconds)

        except Exception as e:
            logging.error(f"Error in real-time control loop: {e}", exc_info=True)
            time.sleep(10)
