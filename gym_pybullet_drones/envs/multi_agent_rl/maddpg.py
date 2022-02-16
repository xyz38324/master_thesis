import numpy as np
import torch
import torch as T
import torch.nn.functional as F
import torch.nn as nn
from Agent import Agent
class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,alpha,beta,gamma,tau,chkpt_dir,evaluate,fc1=64,fc2=64):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.evaluate = evaluate
        self.critic_criterion = nn.MSELoss()
        self.actor_loss = {}
        self.critic_loss = {}
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                            n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                            chkpt_dir=chkpt_dir,gamma=gamma,tau=tau,fc1=fc1,fc2=fc2))


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
            print(agent)

    def choose_action(self, raw_obs):
        # actions = []
        # for agent_idx, agent in enumerate(self.agents):
        #     action = agent.choose_action(raw_obs[agent_idx])
        #     actions.append(action)
        # return actions

        actions = {}
        for agent_idx, agent in enumerate(self.agents):
            actions[agent_idx] = agent.choose_action(raw_obs[agent_idx])

            if self.evaluate==False:
                noise = np.random.randn(self.n_actions)
                actions[agent_idx]= noise+actions[agent_idx]
            #actions[agent_idx]=np.clip(actions[agent_idx],-1,1)
        return actions

    # def choose_action_evaluation(self, raw_obs):
    #     # actions = []
    #     # for agent_idx, agent in enumerate(self.agents):
    #     #     action = agent.choose_action(raw_obs[agent_idx])
    #     #     actions.append(action)
    #     # return actions
    #     actions = {}
    #     for agent_idx, agent in enumerate(self.agents):
    #         pp=raw_obs[agent_idx]
    #         actions[agent_idx] = agent.choose_action_evaluation(raw_obs[agent_idx])
    #     return actions


    def learn(self, memory):
        if not memory.ready():
            return
        learn_epoch=0
        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        states = T.FloatTensor(states).to(device)#(1024,54)
        actions = T.FloatTensor(actions).to(device)#(3,1024,3)
        rewards  =T.FloatTensor(rewards).to(device)#(1024,3)
        states_ = T.FloatTensor(states_).to(device)#(1024,54)
        dones = T.FloatTensor(dones).to(device)#(1042,3)

        # states = T.tensor(states, dtype=T.float).to(device)
        # actions = T.tensor(actions, dtype=T.float).to(device)
        # rewards = T.tensor(rewards,dtype=T.float).to(device)
        # states_ = T.tensor(states_, dtype=T.float).to(device)
        # dones = T.tensor(dones, dtype=T.float).to(device)

        all_agents_new_actions = []#3*(1024,3)
        all_agents_new_mu_actions = []#3*(1024,3)
        old_agents_actions = []#3*(1024,3)

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.FloatTensor(actor_new_states[agent_idx]).to(device)
            # new_states = T.tensor(actor_new_states[agent_idx],
            #                       dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)

            mu_states = T.FloatTensor(actor_states[agent_idx]).to(device)
            # mu_states = T.tensor(actor_states[agent_idx],
            #                      dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)#(1024,9)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)#(1024,9)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)#(1024,9)


        for agent_idx, agent in enumerate(self.agents):
            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = actor_loss.mean()
            self.actor_loss[agent_idx] = actor_loss.detach().cpu().numpy()
            actor_loss = -actor_loss


            critic_value_ = agent.target_critic.forward(states_, new_actions.detach()).flatten()#(1024,)
            ppp = 1 - dones#(1024,3)
            target = rewards[:, agent_idx] + ppp[:, agent_idx] * agent.gamma * critic_value_#(1024,)
            #critic_value_.require_grad = False
            #critic_value_[dones[:, 0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()#(1024,)
            critic_loss = self.critic_criterion(critic_value,target.detach())
            self.critic_loss[agent_idx]=critic_loss.detach().cpu().numpy()

            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()

            agent.update_network_parameters()
'''
    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()
        # print(f"states.shape={np.array(states).shape}")
        # print(f"actions.shape={np.array(actions).shape}")
        # print(f"rewards.shape={np.array(rewards).shape}")
        # print(f"actor_new_states.shape={np.array(actor_new_states).shape}")
        # print(f"states_.shape={np.array(states_).shape}")
        # print(f"dones.shape={np.array(dones).shape}")
        # print(f"dones={dones}")

        device = self.agents[0].actor.device

        #states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards,dtype=T.float).to(device)
        #states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones,dtype=T.float).to(device)

        
        actions = actions.permute(1,0,2).reshape(memory.batch_size,memory.n_actions*memory.n_agents)
        for agent_idx, agent in enumerate(self.agents):
            next_obs = T.tensor(actor_new_states[agent_idx],dtype=T.float).to(device)
            obs = T.tensor(actor_states[agent_idx],dtype=T.float).to(device)

            next_action = agent.target_actor.forward(next_obs)   #[1024,3]
            next_Q = agent.target_critic.forward(next_obs,next_action).flatten()

            ppp=1-dones
            qq=ppp[:,agent_idx]
            target_Q = rewards[:,agent_idx] +qq*agent.gamma*next_Q
            # 阻止梯度传播函数
            #target_Q.requires_grad=False
            target_Q.detach()
            Q = agent.critic.forward(obs, actions).flatten()

            critic_loss = F.mse_loss(Q,target_Q)

            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.zero_grad()
            agent.critic.optimizer.step()



            #actor_loss = agent.critic.forward(states, mu).flatten()#mu是action么？
            actor_loss = -T.mean(-1.0*Q)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()
'''