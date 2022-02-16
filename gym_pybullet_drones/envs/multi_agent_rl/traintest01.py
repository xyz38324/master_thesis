import numpy as np
import sys 
sys.path.append('/content/drive/MyDrive/xyz_master/master_thesis')
from maddpg import MADDPG
from Buffer import MultiAgentReplayBuffer
from  xyzMultiAviary import xyzMultiAviary
import gym
from  torch.utils.tensorboard import SummaryWriter
import argparse
def obs_list_to_state_vector(observation):
    state = np.array([])
    L = len(observation)
    for i in range(L):
        state = np.concatenate([state, observation[i]])
    return state
'''
def run_episode(env,maddpg_agent,memory,max_step):
    step=0
    score_history=[]
    best_score = -np.inf
    obs = env.reset()

    while True:

        actions = maddpg_agent.choose_action(obs)
        obs_, reward, done, info = env.step(actions)
        reward = list(reward.values())
        state = obs_list_to_state_vector(obs)
        state_ = obs_list_to_state_vector(obs_)
      #  done_list = list(done.values())
        done_terminal =all(done)
        terminal = (step>max_step)
        if terminal or done_terminal:
            break
        memory.store_transition(obs, state, actions, reward, obs_, state_, done)
        obs = obs_
        if step % 50 == 0:
            maddpg_agent.learn(memory)

        score = sum(reward)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        # if step %500 ==0:
        #     print(f"--------------------------------------------------average_score:{avg_score}")
        if avg_score > best_score:
            maddpg_agent.save_checkpoint()
            best_score = avg_score
        step += 1



    return best_score,step
'''
def main(args):
    evaluate = args.evaluate
    MAX_EPISODE = args.max_episode
    MAX_STEP = args.max_step

    env = gym.make('xyzmultidrone-aviary-v0')
    n_agents = env.NUM_DRONES
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims)
    n_actions = env.action_space[0].shape[0]
    #observation_space = env.observation_space[0].shape[0]
    #scenario = 'xyzMultiAviary'
    maddpg_agent = MADDPG(actor_dims=actor_dims, critic_dims=critic_dims, n_agents=n_agents, n_actions=n_actions,
                          fc1=64, gamma=args.gamma,
                          fc2=64, alpha=args.alpha, tau=args.tau,
                          beta=args.beta,chkpt_dir='models',evaluate=evaluate)
    memory = MultiAgentReplayBuffer(args.buffer_size, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=args.batch_size)

    if not evaluate:
        writer = SummaryWriter("xyz_logs")
        rewards = []
        for episode in range(MAX_EPISODE):
            obs = env.reset()
            episode_reward = 0
            actor_L = []
            critic_L=[]
            for step in range(1,MAX_STEP+1):
                actions = maddpg_agent.choose_action(obs)
                obs_, reward, done, info = env.step(actions)
                reward = list(reward.values())
                state = obs_list_to_state_vector(obs)
                state_ = obs_list_to_state_vector(obs_)
                done_terminal = all(done)
                memory.store_transition(obs, state, actions, reward, obs_, state_, done)
                if step % args.learn_interval == 0:
                    maddpg_agent.learn(memory)
                    actor_loss = maddpg_agent.actor_loss
                    critic_loss = maddpg_agent.critic_loss
                    actor_L.append(sum(actor_loss.values()))
                    critic_L.append(sum(critic_loss.values()))
                obs = obs_
                reward_all = sum(reward)
                episode_reward += reward_all
                #print(env.target_reward[0])
                if done_terminal:
                    break


            rewards.append(episode_reward)
            writer.add_scalar("episode_reward/episode", rewards[-1], episode)
            writer.add_scalar("episode_reward_mean/episode", np.mean(rewards), episode)
            writer.add_scalar("actor_loss",np.mean(actor_L),episode)
            writer.add_scalar("critic_loss",np.mean(critic_L),episode)
            # if episode % args.save_interval == 0:
            #     maddpg_agent.save_checkpoint()
            if len(rewards)>1:
                if rewards[-1]>rewards[-2]:
                    maddpg_agent.save_checkpoint()
        env.close()

    else:
        maddpg_agent.load_checkpoint()
        obs = env.reset()
        step = 0
        rewards = 0
        while True:
            actions = maddpg_agent.choose_action(obs)
            obs_, reward, done, info = env.step(actions)
            reward = list(reward.values())
            done_terminal = all(done)
            obs = obs_
            score = sum(reward)
            rewards += score

            print(reward)

            step += 1
            if done_terminal or step>3000:
                break
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-max_episode',type=int,default=100)
#     parser.add_argument('-evaluate', type=bool, default=False)
#     parser.add_argument('-max_step',type=int,default=3000)
#     parser.add_argument('-alpha',type=float,default=1e-4)
#     parser.add_argument('-beta',type=float,default=1e-3)
#     parser.add_argument('-batch_size',type=int,default=1024)
#     parser.add_argument('-gamma',type=float,default=0.98)
#     parser.add_argument('-tau', type=float, default=1e-2)
#     parser.add_argument('-buffer_size',type=int,default=1000000)
#     parser.add_argument('-save_interval', type=int, default=2)
#     parser.add_argument('-learn_interval',type=int,default=20)
#     args = parser.parse_args()
#     main(args)
#tensorboard --logdir xyz_logs --host=127.0.0.1 --port=6969

        #env.close()

    #total_steps = 0

'''
    for episode in range(MAX_EPISODE):
        obs = env.reset()
        score_history = []
        memory.mem_cntr=0
        





        for step in range(MAX_STEP):
        
            
            actions = maddpg_agent.choose_action(obs)
            obs_,reward,done,info = env.step(actions)

            reward =list(reward.values())
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            memory.store_transition(obs,state,actions,reward,obs_,state_,done)
            obs = obs_

            if step % 200 ==0:
                maddpg_agent.learn(memory)

            score =sum(reward)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            print(avg_score)
            if avg_score > best_score:
                maddpg_agent.save_checkpoint()
                best_score = avg_score
            if step% PRINT_INTERVAL==0 and step>0:
                print('episode', episode, 'average score {:.1f}'.format(avg_score))
'''



'''
            best_score, total_step = run_episode(env,maddpg_agent,memory,MAX_STEP)
            print(f"best score= {best_score},total step = {total_step}")

            episode_reward.append(best_score)
            average_episode_reward = np.mean(episode_reward[-100:])
            total_episode +=1
            print(f"********************************* total_episode ={total_episode}")

            writer.add_scalar("bestscore_episode", best_score, total_episode)
            #writer.add_scalar("step_episode",total_step,total_episode)
    '''
