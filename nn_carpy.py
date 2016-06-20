# nearest neighbor majority vote implementation


import numpy as np
import gym
from collections import deque


class NNAgent(object):

    def __init__(self, action_space):
    
        self.action_space = action_space
        assert isinstance(action_space, gym.spaces.discrete.Discrete), 'Yo, not our space!'
        
        # hyperparameters
        self.epsilon = 0.1  # exploration percentage
        self.epsilon_decay = 0.98  # exploration decay
        self.nn_num = 25 # number of nearest neighbors to vote 
        self.max_mem = 2000 # number of state_action pairs to keep in memory
        self.mem_b4_exploit = 200 # amount of experience before exploiting our knows
        self.highest_episode_rewards = 0  # keep record of highest episodes, to decide what memories to keep
        
        # our memory
        self.memory = deque(maxlen=2000)  # self-pruning memory with max length of 2000
        self.iteration = 0 # how many actions have we taken

    
    def find_voters(self, state):
        
        mem_4_state = []
        
        for mem in self.memory:
            
            sum_o_sqrs = 0  # distance
            action = mem[1]
            
            for iter in range(len(state)): # get distance 
                
                param_dist = (state[iter] - mem[0][iter])**2
                sum_o_sqrs += param_dist
          
            mem_4_state.append(sum_o_sqrs, action) # (distance,action) tuple
          
        sort_mem_4_state = sorted(mem_4_state, key = lambda x: x[0])
        
        voters_4_state = sort_mem_4_state[:self.nn_num]
        
        return voters_4_state
        
        
    def get_action(self, state):    
    
        voters = self.find_voters(state)
        votes = []
        
        for voter in voters:
            
            votes.append(voter[1])
            
        avg_action = np.average(votes)
        
        if avg_action > 0.5:
        
            return 1
          
        if avg_action <= 0.5:
        
            return 0  
            
            
    def should_we_exploit(self):    
    
        if self.iteration > self.mem_b4_exploit:
        
            random_fate = np.random.random()
        
            if random_fate < self.epsilon:
            
                return False
            
            return True  # we have enough memory, and we aren't exploring this time
    
        return False  # not enough memory to exploit 
        
    
    def should_we_add_to_memory(self, episode_rewards):   
    
        if episode_rewards >= self.highest_episode_rewards:
        
            return True
          
        return False  
    
        
    def decay_epsilon(self):
    
        self.epsilon *= self.epsilon_decay
        
    
    def add_episode_to_memory(self, episode):
    
        for instance in episode:
        
            self.memory.append(instance)
        

env = gym.make('CartPole-v0')
wondering_gnome = NNAgent(env.action_space)        
            
for i_episode in xrange(400):
    observation = env.reset()
    
    episode_rewards = 0
    episode_state_action_list = []
    
    for t in xrange(200):
        env.render()
        
        current_state = observation  
        
        # choose action 
        if wondering_gnome.should_we_exploit():
        
            action = wondering_gnome.get_action(current_state)
            
        if not wondering_gnome.should_we_exploit():
        
            action = env.action_space.sample()   
        
        
        observation, reward, done, info = env.step(action)
        
 
        episode_rewards += reward
        
        episode_state_action_list.append((current_state,action))
        
        wondering_gnome.iteration += 1
        
        print "Iteration_number: "
        print wondering_gnome.iteration 
 
        if done:
            
            print "Episode finished after {} timesteps".format(t+1)
            break
    print "Episode Rewards: "
    print episode_rewards
    
    
    if wondering_gnome.should_we_add_to_memory(episode_rewards):
    
        wondering_gnome.add_episode_to_memory(episode_state_action_list)  
      
    