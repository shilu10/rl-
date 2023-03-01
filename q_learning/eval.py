import numpy as np 


def check_2d_array(observation): 
    if np.array(observation, dtype="object").shape == (2, ): 
        return True 
    return False 


def eval_model(env, eval_noe, q_table, seed, max_steps=100): 
    episode_rewards = []
    for episode in range(eval_noe):
        step = 0 
        if seed:
            state = env.reset(seed=seed[episode])
        else:
            state = env.reset()
            done = False
            total_rewards_ep = 0

            if check_2d_array(state): 
                state = state[0]
            
            while not done: 
                action = np.argmax(q_table[state][:])
                new_state, reward, done, info, _ = env.step(action)
                total_rewards_ep += reward
                state = new_state
                step += 1 
                if done or step >= max_steps:
                    episode_rewards.append(total_rewards_ep)
                    break

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward