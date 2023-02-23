import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import gym 
from collections import *

blackjack_env = gym.make('Blackjack-v1')

def check_2d_obser(observation): 
    if np.array(observation, dtype="object").shape == (2, ): 
        return True 
    return False

def sample_policy(observation):
        score, dealer_score, usable_ace = observation
        return 0 if score >= 20 else 1

def generate_episode(policy, env):
    states, actions, rewards = [], [], []
    observation = env.reset()
    while True:
        if check_2d_obser(observation=observation): 
            observation = observation[0]
        states.append(observation)
        action = policy(observation)
        actions.append(action)
        observation, reward, done, info, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    return states, actions, rewards

def incremental_ev_mc(env, policy, noe, gamma, alpha): 

    value_table = defaultdict(int)
    for episode in range(noe): 
        G = 0 
        trajectory = generate_episode(policy, env)
        states, actions, rewards = trajectory     
           
        for t in range(len(states)-2, -1, -1): 
            s_t = states[t]
            r_t_1 = rewards[t+1]
            G = gamma * G + r_t_1 
                 
            curr_val = value_table.get(s_t, 0)
            mc_target = G 
            mc_error = alpha * (mc_target - curr_val)
            curr_val = curr_val + mc_error
            value_table[s_t] = curr_val
    return value_table

def convert_to_arr(V_dict, has_ace):
    V_dict = defaultdict(float, V_dict)  # assume zero if no key
    V_arr = np.zeros([10, 10])  # Need zero-indexed array for plotting 
    
    # convert player sum from 12-21 to 0-9
    # convert dealer card from 1-10 to 0-9
    for ps in range(12, 22):
        for dc in range(1, 11):
            V_arr[ps-12, dc-1] = V_dict[(ps, dc, has_ace)]
    return V_arr

def plot_3d_wireframe(axis, V_dict, has_ace):
    Z = convert_to_arr(V_dict, has_ace)
    dealer_card = list(range(1, 11))
    player_points = list(range(12, 22))
    X, Y = np.meshgrid(dealer_card, player_points)
    axis.plot_wireframe(X, Y, Z)
    
def plot_blackjack(V_dict):
    fig = plt.figure(figsize=[16,3])
    ax_no_ace = fig.add_subplot(121, projection='3d', title='No Ace')
    ax_has_ace = fig.add_subplot(122, projection='3d', title='With Ace')
    ax_no_ace.set_xlabel('Dealer Showing'); ax_no_ace.set_ylabel('Player Sum')
    ax_has_ace.set_xlabel('Dealer Showing'); ax_has_ace.set_ylabel('Player Sum')
    plot_3d_wireframe(ax_no_ace, V_dict, has_ace=False)
    plot_3d_wireframe(ax_has_ace, V_dict, has_ace=True)
    plt.show()


#value_table = incremental_fv_mc(blackjack_env, sample_policy, 1000, gamma=1, alpha=0.4) 
#plot_blackjack(value_table)

V_10k = incremental_ev_mc(blackjack_env, sample_policy, 10000, gamma=1.0, alpha=0.4)
V_100k = incremental_ev_mc(blackjack_env, sample_policy, 100000, gamma=1.0, alpha=0.4)

fig = plt.figure(figsize=[16,6])
ax_10k_no_ace = fig.add_subplot(223, projection='3d')
ax_10k_has_ace = fig.add_subplot(221, projection='3d', title='After 10,000 episodes')
ax_100k_no_ace = fig.add_subplot(224, projection='3d')
ax_100k_has_ace = fig.add_subplot(222, projection='3d', title='After 100,000 episodes')

fig.text(0., 0.75, 'Usable\n  Ace', fontsize=12)
fig.text(0., 0.25, '   No\nUsable\n  Ace', fontsize=12)

ax_100k_no_ace.set_xlabel('Dealer Showing'); ax_100k_no_ace.set_ylabel('Player Sum')

plot_3d_wireframe(ax_10k_no_ace, V_10k, has_ace=False)
plot_3d_wireframe(ax_10k_has_ace, V_10k, has_ace=True)
plot_3d_wireframe(ax_100k_no_ace, V_100k, has_ace=False)
plot_3d_wireframe(ax_100k_has_ace, V_100k, has_ace=True)

plt.tight_layout()

plt.savefig('../assets/inc_ev_mc_prediction/fig_001.png')

