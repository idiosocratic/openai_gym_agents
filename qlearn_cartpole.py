import gym
import numpy as np

env = gym.make('CartPole-v0')
for i_episode in xrange(20):
    observation = env.reset()
    for t in xrange(100):
        env.render()
        print observation
        # need to retain old observation for transition dict
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break
            
            
# create transition dictionary, model building

trans_dict = {}

if observation not in trans_dict:

  trans_dict.append(observation:{})              
  
trans_dict[old_observation][observation]+=1

# create scaling mechanism to keep track of transition probabilities
observation_count = {}

if observation not in observation_count:

  observation_count[observation] = 0
else:
  observation_count[observation] += 1
  
for trans in trans_dict[old_observation]:

  len(trans_dict[old_observation])
  adjust = 1/observation_count[old_observation]  
  
  #want to add weight to trans_dict entry of new observation 
  # & subtract (1/len(entries)) from other entries  
  
# need transition reward dictionary to keep track of rewards for (s,a,s') tuples
trans_reward_dict = {}  

# need code for keeping track of stochastic rewards

def trans_reward_update(old_observation,action,observation):
  tdc = trans_dict_count[(old_observation, action, observation)]
  trd = trans_reward_dict[(old_observation, action, observation)]
  if tdc == 0:
    trans_reward_dict[(old_observation, action, observation)] == reward
  else:
    trans_reward_dict[(old_observation, action, observation)] == ((tdc-1)*trd + reward)/tdc 
  
# implement e-greedy
if np.random.random() < 0.1:
  action = env.action_space.sample()
else:
  action =     
  
  
# q_dic = {}
# state1 = "here"
# state2 = "there"
# state3 = "er'where"
# a1 = "act1"
# a2 = "act2"
# q_dic[(state1,a1)] = 17
# q_dic[(state1,a2)] = 1
# q_dic[(state2,a2)] = 13
# q_dic[(state2,a1)] = 14
# q_dic[(state3,a1)] = 3
# q_dic[(state3,a2)] = 33

# function for returning best action based on q_function
def maxQ_for_s(q_dict, state): # will arg-max our action
  high_q = 0
  high_act = ''
  for kee in q_dict:
    if kee[0] == state:
      if q_dict[kee] > high_q:
        high_q = q_dic[kee]
        high_act = kee[1]
  print(high_q)
  print(high_act)

# function for updating Q  
def update_Q(old_state, reward, new_state):
  
  learning_rate = 0.1
  discount = 0.9
  q_val_dict[old_state] = q_val_dict[old_state] + learning_rate*(reward + discount*max_q[new_state] - q_val_dict[old_state]) 
  
  

  
  
  
  