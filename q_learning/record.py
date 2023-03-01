import random 
import numpy as np 
import imageio
import os


def check_2d_array(observation): 
    if np.array(observation, dtype="object").shape == (2, ): 
        return True 
    return False 


def record_video(env, Qtable, out_directory, fps=1):
    images = []  
    done = False
    state = env.reset(seed=random.randint(0,500))
    if check_2d_array(state): 
        state = state[0]
    img = env.render()
    images.append(img)
    while not done:
        action = np.argmax(Qtable[state][:])
        state, reward, done, info, _ = env.step(action) 
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)