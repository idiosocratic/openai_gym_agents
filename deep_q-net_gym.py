# first deep q-net 

import numpy as np

# hyperparameters
hidden_size = 50 # size of hidden layer
seq_length = 15 # number of steps to unroll the RNN for # PLAG
learning_rate = 1e-1


# account for terminal states
# RNN for predicting next states/reward
# NN for predicting q-values 

class gym_env_variables

class my_rnn(input_size, hidden_size, output_size):

  Wxh = np.random.randn(hidden_size, input_size)*0.01 # input to hidden
  Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
  Why = np.random.randn(output_size, hidden_size)*0.01 # hidden to output
  bh = np.zeros((hidden_size, 1)) # hidden bias
  by = np.zeros((output_size, 1)) # output bias
  
  # forward pass
  def forward_pass(input):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

  # backward pass: compute gradients going backwards
  def backward_pass():
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

class my_nn(object):
    """class for classic neural net"""


    def __init__(self, state=None, period=None):
        self.state = state if state is not None else np.random.choice(self.valid_states)
        self.period = period if period is not None else np.random.choice([3, 4, 5])
        self.last_updated = 0


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
    minibatch.append(sample)

  return minibatch  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  