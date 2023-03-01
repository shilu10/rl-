import gym
import numpy as np 
from train import *
import pickle
from eval import *
from record import *
from save_to_hf import *

TAXI_ENV = gym.make("Taxi-v3", render_mode="rgb_array")
NOA_SPACE = TAXI_ENV.action_space.n
NOS_SPACE = TAXI_ENV.observation_space.n 
NOE = 100000
MAX_EPSILON = 1
ALPHA = 0.84
GAMMA = 0.99
MIN_EPSILON = 0.5
EVAL_NOE = 100
DECAY_RATE = 0.005
EPSILON = 1 
env_id = "Taxi-v3"

eval_seed = [
    16,
    54,
    165,
    177,
    191,
    191,
    120,
    80,
    149,
    178,
    48,
    38,
    6,
    125,
    174,
    73,
    50,
    172,
    100,
    148,
    146,
    6,
    25,
    40,
    68,
    148,
    49,
    167,
    9,
    97,
    164,
    176,
    61,
    7,
    54,
    55,
    161,
    131,
    184,
    51,
    170,
    12,
    120,
    113,
    95,
    126,
    51,
    98,
    36,
    135,
    54,
    82,
    45,
    95,
    89,
    59,
    95,
    124,
    9,
    113,
    58,
    85,
    51,
    134,
    121,
    169,
    105,
    21,
    30,
    11,
    50,
    65,
    12,
    43,
    82,
    145,
    152,
    97,
    106,
    55,
    31,
    85,
    38,
    112,
    102,
    168,
    123,
    97,
    21,
    83,
    158,
    26,
    80,
    63,
    5,
    81,
    32,
    11,
    28,
    148,
]  #

env_id = "Taxi-v3"  # Name of the environment

if __name__ == "__main__": 
    """
    q_table, training_rewards = train_model(TAXI_ENV,
                    NOS_SPACE,
                    NOA_SPACE,
                    noe=NOE,
                    max_epsilon=MAX_EPSILON,
                    epsilon=EPSILON,
                    alpha=ALPHA,
                    gamma=GAMMA,
                    min_epsilon=MIN_EPSILON
                )
    with open("taxi_q_table.obj", "wb") as f: 
        pickle.dump(q_table, f) 
        print("Completed the traning of the model!!")
    """

    # For local
    q_table = pickle.load(open("taxi_q_table.obj", "rb"))
    mean_rwd, std_rwd = eval_model(
        TAXI_ENV,
        EVAL_NOE,
        q_table,
        None
    )

    print(f"Mean of reward: {mean_rwd}, std of Reward: {std_rwd}")
    #record_video(TAXI_ENV, q_table, "taxi_videos.mp4", 1)


    

    model = {
        "env_id": env_id,
        "noe": NOE,
        "n_eval_episodes": EVAL_NOE,
        "eval_seed": None,

        "alpha": ALPHA,
        "gamma": GAMMA,
        "epsilon": EPSILON,
        "max_epsilon": MAX_EPSILON,
        "min_epsilon": MIN_EPSILON,
        "eps_decay_rate": DECAY_RATE,

        "qtable": q_table,
    }

    username = "Shilash" 
    repo_name = "q-learning-taxi-v3"
    push_to_hub(
        repo_id=f"{username}/{repo_name}",
        model=model,
        env=TAXI_ENV,env_id=env_id)
     

    

