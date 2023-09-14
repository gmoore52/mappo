import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.num_enemies = config["num_enemies"]
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.policy1 = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)
        
        self.policy2 = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)
    

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer1 = TrainAlgo(self.all_args, self.policy1, device = self.device)
        self.trainer2 = TrainAlgo(self.all_args, self.policy2, device = self.device)
        
        # buffer
        self.buffer1 = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])
        
        # buffer for enemy side
        self.buffer2 = SharedReplayBuffer(self.all_args,
                                        self.num_enemies,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer1.prep_rollout()
        next_values1 = self.trainer1.policy.get_values(np.concatenate(self.buffer1.share_obs[-1]),
                                                np.concatenate(self.buffer1.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer1.masks[-1]))
        next_values1 = np.array(np.split(_t2n(next_values1), self.n_rollout_threads))
        self.buffer1.compute_returns(next_values1, self.trainer1.value_normalizer)
        
        self.trainer2.prep_rollout()
        next_values2 = self.trainer2.policy.get_values(np.concatenate(self.buffer2.share_obs[-1]),
                                                np.concatenate(self.buffer2.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer2.masks[-1]))
        next_values2 = np.array(np.split(_t2n(next_values2), self.n_rollout_threads))
        self.buffer2.compute_returns(next_values2, self.trainer2.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer1.prep_training()
        train_infos1 = self.trainer1.train(self.buffer1)      
        self.buffer1.after_update()
        
        self.trainer2.prep_training()
        train_infos2 = self.trainer2.train(self.buffer2)      
        self.buffer2.after_update()
        return train_infos1, train_infos2

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer1.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor1.pt")
        policy_critic = self.trainer1.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic1.pt")
        if self.trainer1._use_valuenorm:
            policy_vnorm = self.trainer1.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm1.pt")
            
        policy_actor = self.trainer2.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor2.pt")
        policy_critic = self.trainer2.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic2.pt")
        if self.trainer2._use_valuenorm:
            policy_vnorm = self.trainer2.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm2.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        print("restoring model.......")
        print(str(self.run_dir))
        cur_dir = os.getcwd()
        print('=======================================',self.model_dir)
        policy_actor_state_dict = torch.load(self.model_dir  + 'actor1.pt')
        self.policy1.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(self.model_dir  + 'critic1.pt')
            self.policy1.critic.load_state_dict(policy_critic_state_dict)
            
        policy_actor_state_dict = torch.load(self.model_dir  + 'actor2.pt')
        self.policy2.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(self.model_dir  + 'critic2.pt')
            self.policy2.critic.load_state_dict(policy_critic_state_dict)
            # if self.trainer._use_valuenorm:
            #     policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm.pt')
            #     self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
