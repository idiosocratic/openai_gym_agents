# nearest neighbor majority vote implementation


import numpy as np
import gym
from collections import deque


class NnValAgent(object):

    def __init__(self, action_space):
    
        self.action_space = action_space
        assert isinstance(action_space, gym.spaces.discrete.Discrete), 'Yo, not our space!'
        
        # hyperparameters
        self.epsilon = 0.3  # exploration percentage
        self.epsilon_decay = 0.95 # exploration decay
        self.learning_rate = 0.37  # our learning rate
        self.nearest_neighbors_number = 6 # number of neighbors to look at when deciding actions
        self.similarity_threshold = 2 # how close states can be before we consider them roughly equivalent
        self.state_value_memory = {} # dictionary containing our state archetypes and their corresponding estimated values
        self.sas_tuple_list = [] # dictionary linking state-action keys to next state
        self.iteration = 0 # how many actions have we taken
        
        # our memory
        #self.memory = deque(maxlen=self.max_mem)  # self-pruning memory with max length of 2000

    
    
    def should_we_add_to_sas_memory(self, state, action): # only need first state and action, deterministic environment
    
        for sas in self.sas_tuple_list:
        
            if (self.are_these_states_the_same(state, sas[0])) and (action == sas[1]):
            
                return False
        
        return True
            
    
    
    def have_we_been_here(self, state):
        
        for sas in self.sas_tuple_list:
            
            if self.are_these_states_the_same(state, sas[0]):
            
                return True
        
        return False
                 
    
    
    def are_these_states_the_same(self, state1, state2):
        
        sum_o_sqrs_dist = self.get_L2_distance(state1, state2)
            
        if sum_o_sqrs_dist < self.similarity_threshold:
        
            return True
            
        return False        
        
        
        
    def add_2_state_val_memory(self, state):    
    
        self.state_value_memory[state] = 0
        
        
        
    def get_episode_archetypes(self, episode_states): # get representative episode states
        
        archetype_list = []
    
        redundant_indices = []
    
        for index1, state1 in enumerate(episode_states):   
            
            if index1 not in redundant_indices:
                
                archetype_list.append(state1)
                
                for index2, state2 in enumerate(episode_states):
                    
                    if index2 not in redundant_indices:
                        
                        if self.are_these_states_the_same(state1, state2):
                     
                            redundant_indices.append(index2) 
        
        return archetype_list    
        
        
        
    def update_state_values(self, episode_archetypes, episode_rewards):  
    
        for episode_state in episode_archetypes:
        
            for memory_state in self.state_value_memory:
            
                if self.self.are_these_states_the_same(episode_state, memory_state):
                
                    current_value = self.state_value_memory[memory_state] 
                    
                    update_weight = episode_rewards*self.learning_rate
                    
                    self.state_value_memory[memory_state] = current_value*(1 - self.learning_rate) + update_weight
                    
                    
                    
    def get_best_action(self, state):
    
        closest_sas_tuples = self.get_closest_states(states)
        
        sas_value_list = []  # stores values of next state and the actions to get there
        
        for sas in closest_sas_tuples:
        
            for state_key in self.state_value_memory:
            
                if self.are_these_states_the_same(sas[2], state_key):
                
                    sas_value_list.append((state_key, self.state_value_memory[state_key], sas[1]))
                    
        sas_value_list = sorted(sas_value_list, key = lambda x: x[1], reverse=True) # sort our list            
                     
        highest_valued_action = sas_value_list[0][1] # get best action
        
        return highest_valued_action
            
    
    
    def get_L2_distance(self, state1, state2):
    
        sum_o_sqrs_dist = 0
        
        for param1, param2 in zip(state1, state2):   
        
            summed_diff = (param1 - param2)**2
            
            sum_o_sqrs_dist += summed_diff
            
        return sum_o_sqrs_dist    
        
        
    
    def get_closest_states(self, state):  
        
        nearest_sas_tuples = []
        
        for sas in self.sas_tuple_list:
        
            if self.are_these_states_the_same(sas[0], state):
            
                nearest_sas_tuples.append(sas)
        
        return nearest_sas_tuples       
              
        
            
    def should_we_exploit(self, state):    
    
        if self.have_we_been_here(state): 
            
            random_fate = np.random.random()
        
            if random_fate < self.epsilon:
            
                return False
            
            return True  # we have enough knowledge, and we aren't exploring this time
    
        return False  # not enough knowledge to exploit 
    
    
        
    def decay_epsilon(self):
    
        self.epsilon *= self.epsilon_decay
        
    
  
        

env = gym.make('CartPole-v0')
wondering_gnome = NNAgent(env.action_space)        
            
episode_rewards_list = []            
            
for i_episode in xrange(100):
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
    
    wondering_gnome.update_highest_rewards(episode_rewards)
    
    
    episode_state_action_rewards_list = []
    
    
    for iter in range(len(episode_state_action_list)):
    
        episode_state_action_rewards_list.append((episode_state_action_list[iter],episode_rewards)) 
    
    append_bool = False
    
    if wondering_gnome.should_we_add_to_memory(episode_rewards):
    
        wondering_gnome.add_episode_to_memory(episode_state_action_rewards_list)  
    
        append_bool = True 
      
    if wondering_gnome.iteration < 501: #not append_bool: # if we haven't already added to memory   
        
        #if wondering_gnome.is_episode_novel(episode_state_list, wondering_gnome.novelty_threshold):   
        
        #if not wondering_gnome.did_we_do_well(episode_rewards): # wondering_gnome.should_we_add_to_novelty_memory(episode_rewards):
                
        episode_state_action_rewards_list = wondering_gnome.correct_our_actions_list(episode_state_action_rewards_list)
                
        #wondering_gnome.add_episode_to_corrected_memory(episode_state_action_rewards_list) 
                
        
    wondering_gnome.decay_epsilon()    
    
           
      
print "Rewards List: "
print episode_rewards_list
print "Average Overall: "
print np.average(episode_rewards_list)   
print "Average of last 30 episodes: "
print np.average(episode_rewards_list[-30:]) 
print "Avg Rwd of Episodes in Mem: "
print wondering_gnome.avg_reward_of_episodes_in_mem(wondering_gnome.memory)
print len(wondering_gnome.corrected_memory)
print wondering_gnome.avg_reward_of_episodes_in_mem(wondering_gnome.corrected_memory)