import gym
from learning import sarsa, qLearning   
import argparse
import random
import numpy as np
np.random.seed(0)
random.seed(0)

from matplotlib import pyplot as plt

def play(episodes, env, policy, max_steps):
    #playing 
    test_rewards = []
    for e in range(episodes):
        print('\nNew episode starting..\n')
        state = env.reset()
        done = False
        #start state
#         state = random.sample(range(0,env.observation_space.n),1)[0]
        env.render()
        total_reward = 0
        i=0
        while not done and i<max_steps:
            action = policy(state)
#             action = np.argmax(Q[state,:])
            print('\nTaking action:{}(0:left, 1:right, 2:up, 3:down)'.format(action))
            state, r, done, info = env.step(action)
            total_reward += r
            if done:
                print('\nEpisode finished') 
                print('\nFinal score: {}'.format(info['score']))
            else:
                env.render()
                print('\nCurrent score: {}'.format(info['score']))
            i += 1
        test_rewards.append(total_reward)
    return test_rewards

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_episodes', '-eps1', nargs = 1, type = int, default = [100], help = 'Number of episodes for training Default:100')
    parser.add_argument('-test_episodes', '-eps2', nargs = 1, type = int, default = [10], help = 'Number of episodes for testing Default:10')
    parser.add_argument('-gridsize','-gs', nargs = 1, type = int, required = True, help = 'Grid size of Pacman game')
    parser.add_argument('-num_food','-f', nargs = 1, type = int, required = True, help = 'Number of food objects in Pacman game')
    parser.add_argument('-learning_rate', '-lr', nargs = 1, type = float, default = [0.4], help = 'Learning rate Default:0.4')
    parser.add_argument('-gamma', '-gm', nargs = 1, type = float, default = [0.99], help = 'Discount factor Default:0.99')
    parser.add_argument('-epsilon', '-epn', nargs = 1, type = float, default = [0.9], help = '**Here epsilon is prob with which greedy action is selected** thats why its default value is high Default:0.9')
    parser.add_argument('-max_steps', '-ms', nargs = 1, type = int, default = [500], help = 'max steps in an episode Default:500')
   
    args = parser.parse_args()
    
    env = gym.make('pacman:pacman-v0',grid_size = args.gridsize[0], num_food =args.num_food[0])
    
    gamma = args.gamma[0]
    alpha = args.learning_rate[0]
    epsilon = args.epsilon[0]
    episodes = args.train_episodes[0]
    max_steps = args.max_steps[0]
    
    #Training & Testing
    print('**SARSA Agent**')
    sarsa_policy, sarsa_train_rewards = sarsa(env,gamma, alpha, epsilon, episodes,max_steps)
    sarsa_test_rewards = play(args.test_episodes[0], env, sarsa_policy, max_steps)
    
    print('**QLearning Agent**')
    qlearning_policy, qlearning_train_rewards = qLearning(env,gamma, alpha, epsilon, episodes,max_steps)
    qlearning_test_rewards = play(args.test_episodes[0], env, qlearning_policy, max_steps)

        
    #evaluation
    print('\nAverage Test Rewards(SARSA Agent):',end='')
    print(np.mean(sarsa_test_rewards))
    print('\nAverage Train Rewards(SARSA Agent):',end='')
    print(np.mean(sarsa_train_rewards))
    
    print('\nAverage Test Rewards(QLearning Agent):',end='')
    print(np.mean(qlearning_test_rewards))
    print('\nAverage Train Rewards(QLearning Agent):',end='')
    print(np.mean(qlearning_train_rewards))
    
    #save Plots of convergence of rewards during trainig
    plt.plot(range(1,episodes+1,100),sarsa_train_rewards[::100],label = 'SARSA')
    plt.plot(range(1,episodes+1,100),qlearning_train_rewards[::100],label = 'QLearninig')
    plt.xlabel('Episodes')
    plt.xticks(range(0,episodes+1,1000))
#     plt.yticks(range(-10,50,2))
    plt.ylabel('Train Rewards')
    plt.title('SARSA vs QLearning')
    plt.legend()
    plt.savefig('pacman-sarsa-vs-qlearninig.png')
    
if __name__=="__main__": 
    main()
    