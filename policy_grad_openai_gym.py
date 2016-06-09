# first policy gradient implementation

import numpy as np


# account for terminal states
     
# hyperparameters
discount = 0.9
iteration_number = 0

# function for calculating decaying learning rate
def get_learning_rate(iteration):
 
  power = 1 - iteration_number*0.01
  return np.exp(power) 
  
 

# Layers
input_size = 4
layer1_size = 10
layer2_size = 10
output_size = 1
  
Wxl1 = np.random.randn(layer1_size, input_size)*0.01 # input to layer1
Wl1l2 = np.random.randn(layer2_size, layer1_size)*0.01 # layer1 to layer2
Wl2y = np.random.randn(output_size, layer2_size)*0.01 # layer2 to output
bl1 = np.zeros((layer1_size, 1)) # layer1 bias
bl2 = np.zeros((layer2_size, 1)) # layer2 bias
by = np.zeros((output_size, 1)) # output bias

weights = [Wxl1,Wl1l2,Wl2l3,Wl3y]
biases = [bl1,bl2,bl3,by]

input_shape_zeros = np.zeros((4, 1))
 
def forward_pass(_input, _weights, _biases):

  for w, b in zip(_weights, _biases):
    _input = sigmoid(np.dot(w, _input) + b)
    
  return _input
  
# sigmoid function
def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x))  
  
# derivative of sigmoid function
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))  
  
# back-propagation function  
def backprop(input, target):

  nabla_w = [np.zeros(w.shape) for w in weights] # initialize list of weight updates by layer
  nabla_b = [np.zeros(b.shape) for b in biases]  # initialize list of bias updates by layer
  
  activation = input
  print "Input: "
  print input
  print input.shape
  activations = [input] # keep list of activations by layer
      
  zees = [] # list of z-value vectors by layer -> (np.dot(w, input) + b)
  
  for w, b in zip(weights, biases):
    zee = np.dot(w, activation) + b
    activation = sigmoid(zee)
  
    zees.append(zee)
    activations.append(activation)  
     
  delta = cost_derivative(activations[-1], target) * sigmoid_prime(zees[-1])
  
  nabla_b[-1] = delta
  nabla_w[-1] = np.dot(delta, activations[-2].transpose()) 
  
  for layer in xrange(2, len(weights)+1):
    zee = zees[-layer]
    sp = sigmoid_prime(zee)
    delta = np.dot(weights[-layer+1].transpose(), delta) * sp
    nabla_b[-layer] = delta
    nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
  return (nabla_w, nabla_b)
   
   
def cost_derivative(output_activations, targets):
        
  return (output_activations - targets)   
     

# keep list of episode rewards
episode_rewards_list = []
     
# calculate average from a list     
def avg_rewards_per_epi(rwd_list):

  return np.average(rwd_list)     
     

# function for formatting input 
def format_input(state):

  state_list = list(state)
  input = input_shape_zeros
  for i in range(len(state_list)):
    input[i] = state_list[i]
  return input   


# function for running minibatch
def run_batch(batch, learning_rate, _weights, _biases):

  batch_len = len(batch)
  nabla_w = [np.zeros(w.shape) for w in _weights] # initialize list of weight updates by layer
  nabla_b = [np.zeros(b.shape) for b in _biases]  # initialize list of bias updates by layer
  
  for state, target in batch:
  
    delta_n_w, delta_n_b = backprop(state, target)  
    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_n_w)]
    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_n_b)]
    
  #update weights & biases
  weights = [w-(learning_rate*nw) for w, nw in zip(_weights, nabla_w)]
  biases = [b-(learning_rate*nb) for b, nb in zip(_biases, nabla_b)]


# script for loading gym environment and running agent
import gym


env = gym.make('CartPole-v0')
for i_episode in xrange(300):
    observation = env.reset()
    episode_rewards = 0
    episode_state_action_list = []
    episode_state_target_list = []
    for t in xrange(20):
        env.render()
        print observation
        
        current_state = format_input(observation)  
        
        # pick action
        if iteration_number < 13:
        
          #pick random action on first few, to initialize replay
          action = env.action_space.sample()
        
        if iteration_number > 12:
          
          # implement epsilon-greedy
          random_fate = np.random.random()
          
          # probability of choosing one
          prob_for_one = forward_pass(current_state, weights, biases)
          if random_fate <= prob_for_one:
            action = 1
          
          if random_fate > prob_for_one:
            action = 0
        
        observation, reward, done, info = env.step(action)
        
        episode_state_action_list.append((current_state, action))
        
        episode_rewards += reward
        
        add_to_replay(old_state, action, reward, new_state)
        
        print "Old state, action, new state, reward: "
        print old_state, action, new_state, reward
        print "Shape: "
        print observation.shape
        
        iteration_number += 1
        print "iteration_number: "
        print iteration_number
        
        # run mini-batch
        if iteration_number % 11 == 0:
          mini_b = get_minibatch(replay_memory, 10) 
          eta = get_learning_rate(iteration_number)
          run_minibatch(mini_b, eta, weights, biases)
         
          max_replay_size = 50 
          prune_memory(replay_memory, max_replay_size)
        
        if done:
            
            print "Episode finished after {} timesteps".format(t+1)
            break
    print "E Rs: "
    print episode_rewards
    
    if iteration_number > 12:
    
      if episode_rewards > avg_rewards_per_epi(episode_rewards_list):
    
        # everything marked correct
        episode_state_target_list = episode_state_action_list
      
        eta = get_learning_rate(iteration_number)
        run_batch(episode_state_target_list, eta, weights, biases)
      
      if episode_rewards <= avg_rewards_per_epi(episode_rewards_list):
    
        # everything marked incorrect
        corrections = []
        for entry in episode_state_action_list:
          if entry[1] == 0:
            corrections.append((entry[0], 1))
          if entry[1] == 1:
            corrections.append((entry[0], 0))    
      
        episode_state_target_list = corrections  
      
        eta = get_learning_rate(iteration_number)
        run_batch(episode_state_target_list, eta, weights, biases)
  
    episode_rewards_list.append(episode_rewards)
      
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  