
import gym
import random as rnd
import numpy as np
import time



#env initiaalize
env = gym.make("Taxi-v3")

#gamma:
gamma = 0.95

#env modify:
def modify_env(env):
    def new_reset(state=None):
        env.orig_reset()
        if state is not None:
            env.env.s = state
        return np.array(env.env.s)

    env.orig_reset = env.reset
    env.reset = new_reset
    return env

def decode_sate(i):
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    return reversed(out)

#transition and reward functions extractor:
def get_P_R(env):
    P={}
    R={}

    for s in range (0,500):
        P[s] = {0: [0,False], 1: [0,False], 2: [0,False], 3: [0,False], 4: [0,False], 5: [0,False]}
        R[s] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for a in range(0,6):
            env.reset(s)
            m=env.step(a)
            P[s][a],R[s][a] = (m[0],m[2]),m[1]
    return P,R


def print_values(policy,env,check_state,P,R):
    V = policy_eval(policy,P,R)
    for s in check_state:
        print("state: ",s,", has Value: ",V[s])

def print_sim(to_print,step,reward):
    print("total steps:",step,"\ntotal reward:",reward)

         



def policy_iter(P,R):
    print("Training agent please wait...")
    #random policy
    policy = {s: rnd.randint(0,5) for s in range(0,500)}
    iter = 1 

    while True:
        
        old_policy= policy.copy()
        #evaluate the value function for the current policy
        V = policy_eval(policy,P,R)

        #imporve the policy bye acting greedy
        policy = policy_improve(V,P,R)

        # breaking condition
        if all(old_policy[s] == policy[s] for s in range (0,500)):
            break

        iter = iter+1    
    return policy
        
def policy_eval(policy,P,R):
    
    #initioal value function
    V = {s:0 for s in range (0,500)}

    while True:
        oldV = V.copy()

        for s in range(0,500):
            a = policy[s]
            V[s] = R[s][a] + gamma*oldV[P[s][a][0]]
            if P[s][policy[s]][1] == True :
                V[P[s][policy[s]][0]] = 0
        
        if all(oldV[s]==V[s] for s in range (0,500)):
            break
    return V

def policy_improve(V,P,R):
    policy = {s:0 for s in range (0,500)}

    for s in range (0,500):
        Q = {}
        for a in range(0,6):
            Q[a] = R[s][a] + gamma*V[P[s][a][0]]

        policy[s] = max(Q,key=Q.get)
    return policy

def action_to_string(a,env):
    switcher = {0:"move south",1:"move north",2:"move east",3:"move west",4:"pickup passenger",5:"drop off passenger"}
    return switcher[a]

def print_iter(iter,env,step):
    print("------step-----",step)
    print(iter['locations'])
    print(iter['action'])
    print(iter['reward'])
    env.render()
    print("\n\n")


def rand_sim(policy,env,P,R):
    print("Starting random simulation:")
    time.sleep(2)
    s = rnd.randint(0,499)
    to_print={}
    env.reset(s)
    print("---------- initial state render: ----------")
    env.render()
    total_Reward = 0
    #step counter
    step = 0
    #first iteration:
    a = policy[s]
    data = env.step(a)
    step = step+1
    to_print[step]=[s,action_to_string(a,env),R[s][a]]
    terminate = P[s][a][1]
    reward = R[s][a]
    total_Reward = total_Reward+reward
    s = P[s][a][0]
    time.sleep(1)
    env.render()
    while not(terminate):
        time.sleep(1)
        a  = policy[s]
        data = env.step(a)
        env.render()
        step = step+1
        to_print[s]=[s,action_to_string(a,env),R[s][a]]
        terminate = P[s][a][1]
        reward = R[s][a]
        total_Reward = total_Reward+reward
        s = P[s][a][0]

    print_sim(to_print,step,total_Reward)    

#---------------------- main: ----------------#

#modiff env:
env = modify_env(env)
#get transition function and reward function
P,R = get_P_R(env)

#calculate optimal policy
opt_policy = policy_iter(P,R)

#run random simulation
rand_sim(opt_policy,env,P,R)

#check state array
#check_state = {s:0 for s in range (0,500)}
#print_values(opt_policy,env,check_state,P,R)
