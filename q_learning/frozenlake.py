import gym
import numpy as np 
from train import *
import pickle
from eval import *
from record import *
from save_to_hf import *

FROZENLAKE_ENV = gym.make("FrozenLake-v1", render_mode="rgb_array")
NOA_SPACE = FROZENLAKE_ENV.action_space.n
NOS_SPACE = FROZENLAKE_ENV.observation_space.n 
NOE = 10000
MAX_EPSILON = 1
ALPHA = 0.83
GAMMA = 0.95
MIN_EPSILON = 0.01
EVAL_NOE = 200
DECAY_RATE = 0.005
MAX_STEP = 100
EPSILON = 1

if __name__ == "__main__": 
    """
    q_table, training_rewards = train_model(FROZENLAKE_ENV,
                        NOS_SPACE,
                        NOA_SPACE,
                        noe=NOE,
			epsilon=EPSILON,
                        max_epsilon=MAX_EPSILON,
                        alpha=ALPHA,
                        gamma=GAMMA,
                        min_epsilon=MIN_EPSILON,
                        max_steps = MAX_STEP
                    )
    with open("frozenlake_q_table.obj", "wb") as f: 
        pickle.dump(q_table, f) 
        print("Completed the traning of the model!!")
   # """

   
    # With this QTable got
    #Mean of reward: 0.695, std of Reward: 0.460407428263272

    q_table = pickle.load(open("frozenlake_q_table.obj", "rb"))
    mean_rwd, std_rwd = eval_model(
        FROZENLAKE_ENV,
        EVAL_NOE,
        q_table,
        None,
        100
    )

    print(f"Mean of reward: {mean_rwd}, std of Reward: {std_rwd}")
    record_video(FROZENLAKE_ENV, q_table, "videos.mp4", 1)


   

    model = {
            "env_id": "frozenlake-v1",
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
    repo_name = "q-learning-frozenLake-v1-4x4-noSlippery"
    push_to_hub(
        repo_id=f"{username}/{repo_name}",
        model=model,
        env=FROZENLAKE_ENV,env_id="frozenlake-v1")
     

    

    




    
