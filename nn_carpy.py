# nearest neighbor majority vote implementation


import numpy as np
import gym
from collections import deque


class NNAgent(object):

    def __init__(self, action_space):
    
        self.action_space = action_space
        assert isinstance(action_space, gym.spaces.discrete.Discrete), 'Yo, not our space!'
        
        # hyperparameters
        self.epsilon = 0.3  # exploration percentage
        self.epsilon_decay = 0.95 # exploration decay
        self.nn_num = 17 # number of nearest neighbors to vote 
        self.max_mem = 4000 # number of state_action pairs to keep in memory
        self.mem_b4_exploit = 600 # amount of experience before exploiting our knows
        self.highest_episode_rewards = 0  # keep record of highest episode, to decide what memories to keep
        self.did_we_do_well_threshold = 0.5 # percentage of highest episode rewards needed for pat on back
        self.novelty_threshold = 5 # should be greater than or equal to 1( ='s 1 => no effect)
        
        # our memory
        self.memory = deque(maxlen=self.max_mem)  # self-pruning memory with max length of 2000
        self.iteration = 0 # how many actions have we taken

    
    def is_episode_novel(self, episode_states, novelty_threshold):
        
        
        avg_state_in_memory = [0]*len(self.memory[0][0][0]) 
        
        for mem in self.memory: # sum every state in memory       
            
            for iter in range(len(avg_state_in_memory)):
            
                avg_state_in_memory[iter] += mem[0][0][iter]    
         
        for iter in range(len(avg_state_in_memory)): # divide by memory length for avg
            
            avg_state_in_memory[iter] /= len(self.memory) 


        avg_state_this_episode = [0]*len(self.memory[0][0][0]) 
        
        for state in episode_states: # sum every state in memory       
            
            for iter in range(len(avg_state_this_episode)): # sum states in episode list
            
                avg_state_this_episode[iter] += state[iter]    
         
        for iter in range(len(avg_state_this_episode)): # divide by episode length for avg
            
            avg_state_this_episode[iter] /= len(episode_states) 

        
        distance_tween_mem_avg_episode_avg = [0]*len(self.memory[0][0][0]) 
        
        for iter in range(len(distance_tween_mem_avg_episode_avg)):
        
            distance_tween_mem_avg_episode_avg[iter] = abs(avg_state_this_episode[iter] - avg_state_in_memory)
      
        # new novelty calc    
        
        if np.sum(distance_tween_mem_avg_episode_avg) > self.novelty_threshold:
            
            print "distance"
            print np.sum(distance_tween_mem_avg_episode_avg)
            return True
        
        return False
                 
    
    def find_voters(self, state):
        
        mem_4_state = []
        
        for mem in self.memory:
            
            sum_o_sqrs = 0  # distance
            action = mem[0][1]
            
            for iter in range(len(state)): # get distance 
                
                param_dist = (state[iter] - mem[0][0][iter])**2
                sum_o_sqrs += param_dist
          
            mem_4_state.append((sum_o_sqrs, action)) # (distance,action) tuple
          
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
    
        if len(self.memory) == 0:
        
            return True
        
        if len(self.memory) > 0:
    
            if episode_rewards >= self.memory[0][1]:
        
                return True
          
        return False  
    
    
        
    def decay_epsilon(self):
    
        if self.iteration > self.mem_b4_exploit : 
        
            self.epsilon *= self.epsilon_decay
        
    
    def add_episode_to_memory(self, episode):
    
        for instance in episode:
        
            self.memory.append(instance)
        
        # put memories with lowest reward near front, will be pruned first    
        self.memory = deque(sorted(self.memory, key = lambda x: x[1]),maxlen=self.max_mem)
        
        
    def did_we_do_well(current_episode_rewards):
    
        if current_episode_rewards < self.highest_episode_rewards*self.did_we_do_well_threshold:
      
            return False
        
        return True
        
      
    def correct_our_actions_list(episode_state_action_list):
    
        corrected_state_action_list = []
    
        for iter in range(len(episode_state_action_list)):
         
            corrected_action = 0.5 # not an integer
         
            if episode_state_action_list[iter][1] == 1:
            
                corrected_action = 0
                
            if episode_state_action_list[iter][1] == 0:
            
                corrected_action = 1
                
            assert (corrected_action==1 or corrected_action==0)        
        
            corrected_state_action_list.append((episode_state_action_list[iter][0], corrected_action))  
            
        assert (len(corrected_state_action_list) == len(episode_state_action_list))
        
        return corrected_state_action_list                  
    
            
        
        

env = gym.make('CartPole-v0')
wondering_gnome = NNAgent(env.action_space)        
            
episode_rewards_list = []            
            
for i_episode in xrange(40):
    observation = env.reset()
    
    episode_rewards = 0
    episode_state_list = []
    episode_state_action_list = []
    
    for t in xrange(200):
        env.render()
        
        current_state = observation  
        
        episode_state_list.append(current_state)
        
        # choose action 
        if wondering_gnome.should_we_exploit():
        
            action = wondering_gnome.get_action(current_state)
            
        if not wondering_gnome.should_we_exploit():
        
            action = env.action_space.sample()   
        
        print "Action: "
        print action
        
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
    episode_rewards_list.append(episode_rewards)
    
    
    episode_state_action_rewards_list = []
    
    
    for iter in range(len(episode_state_action_list)):
    
        episode_state_action_rewards_list.append((episode_state_action_list[iter],episode_rewards)) 
    
    append_bool = False
    
    if wondering_gnome.should_we_add_to_memory(episode_rewards):
    
        wondering_gnome.add_episode_to_memory(episode_state_action_rewards_list)  
    
        append_bool = True 
      
    if not append_bool: # if we haven't already added to memory   
        
        if wondering_gnome.is_episode_novel(episode_state_list, wondering_gnome.novelty_threshold):   
            
            print "1"
            #assert False
    
            if True: # wondering_gnome.should_we_add_to_novelty_memory(episode_rewards):
        
                print "2"
                #assert False
        
                wondering_gnome.add_episode_to_novelty_memory(episode_state_action_rewards_list)
        
    wondering_gnome.decay_epsilon()    
           
      
print "Rewards List: "
print episode_rewards_list
print "Average Overall: "
print np.average(episode_rewards_list)   
print "Average of last 30 episodes: "
print np.average(episode_rewards_list[-30:]) 
print "Len of Nov Mem: "
print len(wondering_gnome.novelty_memory)
print len(wondering_gnome.memory)
