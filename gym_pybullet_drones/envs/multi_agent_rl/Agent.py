import torch as T
import sys 
sys.path.append('/content/drive/MyDrive/xyz_master/master_thesis')
from xyzModel import CriticNetwork,ActorNetwork
class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha, beta, fc1,
                    fc2, gamma, tau):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor.pth').to(device)
        self.critic = CriticNetwork(beta, critic_dims,
                            fc1, fc2, n_agents, n_actions,
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic.pth').to(device)
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, name=self.agent_name+'_target_actor.pth').to(device)
        self.target_critic = CriticNetwork(beta, critic_dims,
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir, name=self.agent_name+'_target_critic.pth').to(device)
        #device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        # self.actor.to(device)
        # self.critic.to(device)
        # self.target_actor.to(device)
        # self.target_critic.to(device)
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)#(1,18)
        actions = self.actor.forward(state)#(1,3)
        # noise = T.randn(self.n_actions).to(self.actor.device)
        # action = actions + noise

        return actions.detach().cpu().numpy()[0]

    # def choose_action_evaluation(self, observation):
    #     state = T.tensor([observation], dtype=T.float).to(self.actor.device)
    #     actions = self.actor.forward(state)
    #
    #
    #     return actions.detach().cpu().numpy()[0]

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        self.target_actor.load_checkpoint()
