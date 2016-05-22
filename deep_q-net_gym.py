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


class my_nn(object):
    """class for classic neural net"""


    def __init__(self, state=None, period=None):
        self.state = state if state is not None else random.choice(self.valid_states)
        self.period = period if period is not None else random.choice([3, 4, 5])
        self.last_updated = 0


best_q_val_current_state = 0
best_current_action = random.choice(actions)

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
    
    
new_state, new_reward = agent.act(best_current_action)      
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  