# generic script for loading gym environment and running an agent
import gym


env = gym.make('CartPole-v0')
for i_episode in xrange(20):
    observation = env.reset()
    for t in xrange(100):
        env.render()
        print observation
        
        old_state = state_to_int_func(observation)  # retain old state for updates
        action = env.action_space.sample()  #get_action(old_state)
        
        observation, reward, done, info = env.step(action)
        
        new_state = state_to_int_func(observation)  
        
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
     