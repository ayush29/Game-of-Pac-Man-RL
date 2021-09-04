import numpy as np
import random


def getPolicy(Q):
    
    policy={}
    for s in range(0,Q.shape[0]):
        policy[s] = np.argmax(Q[s,:])
    
    def greedyAction(s):
        action = policy[s]
        return action
        
    return greedyAction

def getEpsilonGreedyPolicy(Q,epsilon):
    policy={}
    for s in range(0,Q.shape[0]):
        policy[s] = np.argmax(Q[s,:])
    num_actions = Q.shape[1]
        
    def epsilonGreedyAction(s):
#         prob = np.zeros(num_actions)+(epsilon/num_actions)
#         prob[policy[s]] += 1-epsilon
    
#         action = np.random.choice(a=num_actions,p=prob)

        if np.random.rand() < epsilon:
            action = policy[s]
        else:
            action = np.random.randint(0, num_actions)
            
        return action
    
    return epsilonGreedyAction

def greedyAction(Q,s):
    return np.argmax(Q[s,:])

def epsilonGreedyAction(Q,s,epsilon):
#     greedy_action = np.argmax(Q[s,:])
#     prob = np.zeros(Q.shape[1],dtype = float)+(epsilon/Q.shape[1])
#     prob[greedy_action] += 1-epsilon
    
#     action = np.random.choice(a=Q.shape[1],p=prob)
    
    num_actions = Q.shape[1]
    if np.random.rand() < epsilon:
        action = np.argmax(Q[s, :])
    else:
        action = np.random.randint(0, num_actions)
    
    return action

def sarsa(env,gamma, alpha, epsilon, episodes, max_steps):
    Q = np.random.rand(env.observation_space.n,env.action_space.n)
    total_rewards = []
    for e in range(episodes):
        #start state, assuming enviroment return integer as observation
        s =env.reset()
#         s = random.sample(range(0,env.observation_space.n),1)[0]
        a = epsilonGreedyAction(Q,s,epsilon)
        done = False
        total_reward =0
        i=0
        while not done and i< max_steps:
            s2, r, done,info = env.step(a)
            total_reward += r
            if done:
                target = r #for terminal state target = reward only, as no look ahead state exist
                update = alpha*(target - Q[s,a])
                Q[s,a] += update
            else:
                a2 = epsilonGreedyAction(Q,s2,epsilon)
                target = r + gamma*Q[s2,a2]
                update = alpha*(target - Q[s,a])
                Q[s,a] += update
                s = s2
                a = a2
            i +=1
        total_rewards.append(total_reward)
    policy =  getPolicy(Q)     
    return policy, total_rewards
#     return Q, total_rewards

def qLearning(env,gamma, alpha, epsilon, episodes, max_steps):
    Q = np.random.rand(env.observation_space.n,env.action_space.n)
    total_rewards = []
    for e in range(episodes):
        #start state, assuming enviroment return integer as observation
        s =env.reset()
#         s = random.sample(range(0,env.observation_space.n),1)[0]
        a = epsilonGreedyAction(Q,s,epsilon)
        done = False
        total_reward = 0
        i=0
        while not done and i< max_steps:
            s2, r, done,info = env.step(a) 
            total_reward += r
            if done:
                target = r #for terminal state target = reward only, as no look ahead state exist
                update = alpha*(target - Q[s,a])
                Q[s,a] += update
            else:
                a2 = greedyAction(Q,s2)
                target = r + gamma*Q[s2,a2]
                update = alpha*(target - Q[s,a])
                Q[s,a] += update
                s = s2
                a = epsilonGreedyAction(Q,s2,epsilon)
            i += 1
        total_rewards.append(total_reward)
    policy =  getPolicy(Q)      
    return policy, total_rewards
#     return Q, total_rewards        
        