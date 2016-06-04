# first deep q-net 

import numpy as np


# account for terminal states
     
# hyperparameters
discount = 0.9
iteration_number = 0

# function for calculating decaying learning rate
def get_learning_rate(iteration):
 
  power = 1 - iteration_number*0.01
  return np.exp(power) 
  
# function for calculating exploration rate
def get_exploration_rate(iteration):

  initial_rate = 0.5
  decay = 0 
  
  if iteration > 15:
    decay = 0.1
  if iteration > 30:
    decay = 0.2 
  if iteration > 50:
    decay = 0.3 
  if iteration > 75:
    decay = 0.4 
  if iteration > 150:
    decay = 0.45      
  current_rate = initial_rate - decay
  
  return current_rate   

# Layers
input_size = 0
layer1_size = 16
layer2_size = 16
layer3_size = 16
output_size = 0
  
Wxl1 = np.random.randn(layer1_size, input_size)*0.01 # input to layer1
Wl1l2 = np.random.randn(layer2_size, layer1_size)*0.01 # layer1 to layer2
Wl2l3 = np.random.randn(layer3_size, layer2_size)*0.01 # layer2 to layer3
Wl3y = np.random.randn(output_size, layer3_size)*0.01 # layer3 to output
bl1 = np.zeros((hidden_size, 1)) # layer1 bias
bl2 = np.zeros((hidden_size, 1)) # layer2 bias
bl3 = np.zeros((hidden_size, 1)) # layer3 bias
by = np.zeros((output_size, 1)) # output bias


weights = [Wxl1,Wl1l2,Wl2l3,Wl3y]
biases = [bl1,bl2,bl3,by]

def forward_pass(_input, _weights, _biases):

  for w, b in zip(_weights, _biases):
    _input = np.tanh(np.dot(w, _input) + b)
    
  return _input
  
# back-propagation function  
def backprop(input, target):

  nabla_w = [np.zeros(w.shape) for w in weights] # initialize list of weight updates by layer
  nabla_b = [np.zeros(b.shape) for b in biases]  # initialize list of bias updates by layer
  
  activation = input
  activations = [input] # keep list of activations by layer
      
  zees = [] # list of z-value vectors by layer -> (np.dot(w, input) + b)
  
  for w, b in zip(weights, biases):
    zee = np.dot(w, activation) + b
    activation = np.tanh(zee)
  
    zees.append(zee)
    activations.append(activation)  
     
  delta = cost_derivative(activations[-1], target) * tanh_prime(zees[-1])
  
  nabla_b[-1] = delta
  nabla_w[-1] = np.dot(delta, activations[-2].transpose()) 
  
  for layer in xrange(2, len(weights)+1):
    zee = zees[-layer]
    tp = tanh_prime(zee)
    delta = np.dot(weights[-layer+1].transpose(), delta) * tp
    nabla_b[-layer] = delta
    nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
  return (nabla_w, nabla_b)
   
   
     
def tanh_prime(zee):  # derivative function for tanh

  return 1-(np.tanh(zee)*np.tanh(zee))    
    



#np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients


current_state = 

best_q_val_current_state = 0
best_current_action = random.choice(actions)
predicted_next_state = 

for action1 in actions:

  next_state, reward = predict_next_state_reward(state, action1)
  
  best_q_ns = 0
  
  
  for action2 in actions:
  
    current_q_ns = predict_q_val(next_state, action2)
    
    if current_q_ns > best_q_ns:
    
      best_q_ns = current_q_ns
      
  q_val_current_state = reward + discount*best_q_ns      
  
  if q_val_current_state > best_q_val_current_state:
  
    best_q_val_current_state = q_val_current_state   
  
    best_current_action = action1
    
    predicted_next_state = next_state
    
    
old_state = current_state    
current_state, new_reward = agent.act(best_current_action)     

# run rnn loss function using predicted reward/next_state vs actual 
# backprop rnn

old_state_q_val = predict_q_val(old_state, best_current_action)

# run nn loss function using predicted best_q and received reward
# backprop nn 



# Replay Memory (list of SARS' tuples)
replay_memory = []

# Add to replay
def add_to_replay(old_state, action, reward, new_state):

  replay_memory.append((old_state, action, reward, new_state))

# Function for reward clipping
current_max_reward = 0.0
current_min_reward = 0.0

def scale_reward(unscaled_reward):
  
  if unscaled_reward > current_max_reward:
    current_max_reward = unscaled_reward
    
  if unscaled_reward < current_min_reward:
    current_min_reward = unscaled_reward  
  
  if unscaled_reward == current_max_reward:
    return 0.99
    
  if unscaled_reward == current_min_reward:
    return -0.99   
    
    
  scale = current_max_reward - current_min_reward
  dif_from_min = unscaled_reward - current_min_reward 

  scaled_reward = ((dif_from_min/scale)*1.98)-0.99
  if (-1 < scaled_reward < 1):
    return scaled_reward

# Function for pruning replay memory
def prune_memory(replay_memory_list, max_memory):
  
  if len(replay_memory_list) > max_memory:
    num_2_prune = len(replay_memory_list) - max_memory  # number of tuples to prune(pop)
    
    for cut in range(num_2_prune):
      indx = np.random.randint(0, len(replay_memory_list))
      branch = replay_memory_list.pop(indx)
    
    
# Function for getting random minibatch

def get_minibatch(replay_mem, batch_size):

  minibatch = []
  for each in range(batch_size):
    index = np.random.randint(0,len(replay_mem))
    rand_sample = replay_mem[index]
    target = rand_sample[3] + discount*(calculate_optimal_q_value(rand_sample[3])[0])
    minibatch.append(((rand_sample[0],rand_sample[1]), target))

  return minibatch  
  
  #add targets to get_minibatch function
  # currently returning sars' instead of ((state, action), (reward + disco*Q(s',a*)))

# function for calculating optimal q_value
def calculate_optimal_q_value(state):

  optimal_q_val = 0
  optimal_action = 0
  
  actions = [0,1]
  for action in actions:
    q_val = calculate_q_value(state, action)
    if q_val > optimal_q_val:
      optimal_q_val = q_val  
      optimal_action = action
      
  best_q_val_tuple = (optimal_q_val, optimal_action)    
   
  return best_q_val_tuple     
  
# function for calculating q_value  
def calculate_q_value(state, action):

  input = format_input(state, action)
  
  q_val = forward_pass(input, weights, biases)
  
  return q_val

  
  
# function for formatting input from (state, action)
def format_input(state, action):

  if action == 0:
    state_list = list(state)
    state_list.append(0)
    state_list.append(0)
    input = np.array(state_list)
    return input  
    
  if action == 1:
    state_list = list(state)
    state_list.append(1)
    state_list.append(1)
    input = np.array(state_list)
    return input    


# function for running minibatch
def run_minibatch(minibatch, learning_rate):

  batch_len = len(minibatch)
  nabla_w = [np.zeros(w.shape) for w in weights] # initialize list of weight updates by layer
  nabla_b = [np.zeros(b.shape) for b in biases]  # initialize list of bias updates by layer
  
  for input, target in minibatch:
  
    delta_n_w, delta_n_b = backprop(input, target)  
    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_n_w)]
    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_n_b)]
    
  #update weights & biases
  weights = [w-(learning_rate/batch_len)*nw for w, nw in zip(weights, nabla_w)]
  biases = [b-(learning_rate/batch_len)*nb for b, nb in zip(biases, nabla_b)]

  
  
  
# Create standalone q-network to calculate q-value targets from frozen weights
# update weights periodically
  
  
  # generic script for loading gym environment and running an agent
import gym


env = gym.make('CartPole-v0')
for i_episode in xrange(20):
    observation = env.reset()
    for t in xrange(100):
        env.render()
        print observation
        
        old_state = observation  # retain old state for updates
        
        if iteration_number < 16:
        
          #pick random action
          action = env.action_space.sample()
        
        if iteration_number > 15:
        
          #pick best action for state
        
        
        observation, reward, done, info = env.step(action)
        
        new_state = observation
        
        add_to_replay(old_state, action, reward, new_state)
        
        print "Old state, action, new state, reward: "
        print old_state, action, new_state, reward
        print "Shape: "
        print observation.shape
        
        s_a_count_update(old_state,action)
        add_s_to_sas(old_state,action,new_state)
        update_sas_trans_count(old_state,action,new_state)
        
        if (old_state,action) not in q_val_dict:
          q_val_dict[(old_state,action)] = 0  # initializing
        
        trans_reward_update(old_state,action,reward,new_state)
        update_Q_sa(old_state, action)  
        
        if done:
            print "Q-value Dict:"
            print q_val_dict
            print "Episode finished after {} timesteps".format(t+1)
            break
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  