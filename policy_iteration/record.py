import random 
import numpy as np 
import imageio
import os


def check_2d_array(observation): 
    if np.array(observation, dtype="object").shape == (2, ): 
        return True 
    return False 


def record_video(env, policy, out_directory, fps=1):
    images = []  
    done = False
    state = env.reset(seed=random.randint(0,500))
    if check_2d_array(state): 
        state = state[0]
    img = env.render()
    
    images.append(img)
    while not done:
        # Take the action (index) that have the maximum expected future reward given that state
        action = policy[state]
        state, reward, done, info, _ = env.step(action) # We directly put next_state = state for recording logic
        
        img = env.render()
        
        images.append(img)
      #  print(images)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)