import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import gym 
from collections import *


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

