import gym
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
            
            
# create transition dictionary

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

if trans_reward_dict[(old_observation, action, observation)] == reward:
  pass
else:
  # re-balance
  
    