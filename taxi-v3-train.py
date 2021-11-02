#imports:
import gym
import numpy as np
import random

#env initialize:
env = gym.make("Taxi-v3")

#how many action for each state
actionSize = env.action_space.n
#how many states there is
stateSize = env.observation_space
#discount factor:
discount = 0.95
numOfEpisodes = 200

#S: set of steps
#A: set of actions
#P: transition function
#R: instant reward function
def policy_iteration(S,A,P,R):
    #intial policy(0)
    policy = {s:A[0] for s in S}
    V = {s:0 for s in S}

    while True:
        #hold old policy
        old_pilicy = policy.copy()

        #evaluate the value function according to the current policy
        V = policy_evaluation(policy,S,R,P)

        #impove and update the policy
        policy = policy_improvement(V,S,A,R,P)

        #Breaking condition, old policy equals new policy, so we are at optimal policy
        if all(old_pilicy[s] == policy[s] for s in S): 
            break

    return policy
        
def policy_evaluation(policy, S,R,P):
    #initial value_function
    V = {s:0 for s in S}

    while True:
        oldV = V.copy()

        for s in S:
            # saves in a the actions from state s 
            # according to the currnt policy
            a = policy[s]
            V[s]=R(s,a) + sum(P(s_next,s,a)*oldV[s_next] for s_next in S)

            if all(oldV[s] == V[s] for s in S):
                break
        return V    

def policy_improvement(V,S,A,R,P):
    policy = {s:A[0] for s in S}

    for s in S:
        Q = {}
        for a in A:
            Q[a] = R(s,a)+sum(P(s_next,s,a)*V[s_next] for s_next in S)

        policy[s] = max(Q,key=Q.get)
        
    return policy 

def modify_env(env):
    
    def new_reset(state=None):
        env.orig_reset()  
        if state is not None:
            env.env.s = state
        return np.array(env.env.s)
    
    env.orig_reset =  env.reset
    env.reset = new_reset
    P={}
    R={}
    for s in range (0,500):
        P[s]={0:0,1:0,2:0,3:0,4:0,5:0}
        R[s]={0:0,1:0,2:0,3:0,4:0,5:0}
        env.reset(s)
        for a in range (0,6):
            env.env.state = s
            P[s][a] = env.step(a)[0]
            env.env.state = s
            R[s][a] = env.step(a)[1]
        

    print(env.P[1])
    print(P[1])
    print(R[1])
    return env ,P ,R

modify_env(env)    