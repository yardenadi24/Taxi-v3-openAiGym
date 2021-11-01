import numpy as np;
import gym;

def modify_env(env):

    def new_reset(state=None):
        env.orig_reset()  
        if state is not None:
            env.env.s = state
        return np.array(env.env.s)
    
    env.orig_reset =  env.reset
    env.reset = new_reset
    return env