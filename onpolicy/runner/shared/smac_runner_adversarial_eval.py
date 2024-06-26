from cmath import inf
import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.multi_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)
        self.last_battle_end = 0
        self.episode_threshold = 50
        self.episodes_since_guide_window_reduction = 0
        self.jsrl_guide_windows = {}
        self.explore_policy_active = False
        self.multi_player = True
        self.seed = config['all_args'].seed
        self.opp1_policy = None
        self.opp2_policy = None
        # self.num_enemies = config["num_enemies"]
        self.all_args.reward_speed = 1
        if self.seed > 1:
            share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
            enemy_share_observation_space = self.envs.enemy_share_observation_space[0] if self.use_centralized_V else self.envs.enemy_observation_space[0]
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
            self.opp1_policy = Policy(self.all_args,
                                self.envs.enemy_observation_space[0],
                                enemy_share_observation_space,
                                self.envs.enemy_action_space[0],
                                device = self.device)
            self.opp2_policy = Policy(self.all_args,
                                self.envs.observation_space[0],
                                share_observation_space,
                                self.envs.action_space[0],
                                device = self.device)
            
            self.opponent_dir = config["all_args"].opp_model_dir
            
            # load Opponents
            print("Loading Opponents........")
            print(str(self.opponent_dir))
            policy_state_dict = torch.load(self.opponent_dir + 'actor1.pt')
            self.policy1.actor.load_state_dict(policy_state_dict)
            policy_state_dict = torch.load(self.opponent_dir + 'actor2.pt')
            self.policy2.actor.load_state_dict(policy_state_dict)

        # load opponent model
        # print("restoring model.......")
        # print(str(self.opponent_dir))
        # print('=======================================',self.opponent_dir)
        # policy_actor_state_dict = torch.load(self.opponent_dir  + 'actor.pt')
        # self.opp_policy.actor.load_state_dict(policy_actor_state_dict)
        # policy_critic_state_dict = torch.load(self.opponent_dir  + 'critic.pt')
        # self.opp_policy.critic.load_state_dict(policy_critic_state_dict)

        for i in range(self.n_rollout_threads):
            # [guide_window, last_battle]
            self.jsrl_guide_windows[i] = [inf, 0]

    def run(self):
        self.warmup()
        
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)
        
        enemy_last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        enemy_last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)
        
        self.guide_index = 0

        for episode in range(episodes):
            # # if self.all_args.jump_start_model_pool_dir:
            # #     self.jump_start_policy = self.jump_start_policy_pool[self.guide_index]
            # #     # cycle through guide windows.
            # #     if self.guide_index < 4:
            # #         self.guide_index += 1
            # #     else:
            # #         self.guide_index = 0

            # if self.use_linear_lr_decay:
            #     self.trainer1.policy.lr_decay(episode, episodes)
            #     self.trainer2.policy.lr_decay(episode, episodes)
            
            # if self.multi_player:
            #     self.available_enemy_actions = np.ones((self.n_rollout_threads, self.num_enemies, 6+self.num_agents), dtype=np.float32)
            #     self.enemy_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            #     self.enemy_obs = np.zeros((self.n_rollout_threads, self.num_enemies, 204), dtype=np.float32)

            # for step in range(self.episode_length):
            #     # Sample actions
            #     # if self.all_args.jump_start_model_dir or self.all_args.jump_start_model_pool_dir:
            #     #     values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect_guide(step)
            #     #     explore_values, explore_actions, explore_action_log_probs, explore_rnn_states, explore_rnn_states_critic = self.collect(step)
            #     # else:
            #     values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
            #     enemy_values, enemy_actions, enemy_action_log_probs, enemy_rnn_states, enemy_rnn_states_critic = self.collect_enemy(step)
                

            #     # for i in range(self.n_rollout_threads):
            #     #     if (self.all_args.jump_start_model_dir or self.all_args.jump_start_model_pool_dir) and (step - self.jsrl_guide_windows[i][1] >= self.jsrl_guide_windows[i][0]):
            #     #         values[i] = explore_values[i]
            #     #         actions[i] = explore_actions[i]
            #     #         action_log_probs[i] = explore_action_log_probs[i]
            #     #         rnn_states[i] = explore_rnn_states[i]
            #     #         rnn_states_critic[i] = explore_rnn_states_critic[i]

            #     # Obser reward and next obs
            #     # enemy_actions = None
            #     # if self.multi_player:
            #     #     enemy_values, enemy_actions, enemy_action_log_probs, enemy_rnn_states, enemy_rnn_states_critic = self.collect_enemy(step)
            #         # enemy_actions, enemy_rnn_states = self.collect_enemy(self.enemy_obs,self.enemy_rnn_states,self.buffer2.masks[step],self.available_enemy_actions)
            #     # print("Environments:    ", self.envs)
            #     obs, enemy_obs, agent_state, enemy_state, rewards, enemy_rewards, \
            #     dones, enemy_dones, infos, enemy_infos, available_actions, available_enemy_actions \
            #         = self.envs.step(actions,enemy_actions) if self.multi_player \
            #             else self.envs.step(actions)
                    
            #     if self.multi_player:
            #         self.available_enemy_actions = available_enemy_actions.copy()
            #         self.enemy_rnn_states = enemy_rnn_states.copy()
            #         self.enemy_obs = enemy_obs.copy()

            #     # if the policy is swapped to explore, and the sum rewards are greater, then great, move on to the next guide_window, I think? This is one option.
            #     # if self.guide_window > 0 and self.explore_policy_active and rewards[0][0][0] > 10:
            #     #    self.guide_window = self.guide_window - 1
            #     #    self.episodes_since_guide_window_reduction = 0 # set the threshold to 0.
            #     #    self.guide_policy_last_rewards[0][0][0] = inf

            #     for index, done in enumerate(dones):
            #         if done.all():
            #             if self.all_args.reward_speed and infos[index][0]['won']:
            #                 max_reward = 1840 # Hardcoded from env
            #                 scale_rate = 20 # Hardcoded from env.
            #                 map_length = step - self.jsrl_guide_windows[index][1]
            #                 upper_map_length_limit = 150
            #                 norm_max = 30
            #                 z1 = (map_length / upper_map_length_limit) * norm_max
            #                 z1 = (z1 - norm_max) * -1 # invert reward to reward lower map_length
            #                 speed_reward = z1 / (max_reward / scale_rate)
            #                 rewards[index] += speed_reward

            #             self.jsrl_guide_windows[index][1] = step

            #             # If we haven't set the guide window yet, set it to the last frame of last attempt, that's our starting point.
            #             if self.jsrl_guide_windows[index][0] is inf:
            #                 self.jsrl_guide_windows[index][0] = step

            #     for index, done in enumerate(enemy_dones):
            #         if done.all():
            #             if self.all_args.reward_speed and infos[index][0]['won']:
            #                 max_reward = 1840 # Hardcoded from env
            #                 scale_rate = 20 # Hardcoded from env.
            #                 map_length = step - self.jsrl_guide_windows[index][1]
            #                 upper_map_length_limit = 150
            #                 norm_max = 30
            #                 z1 = (map_length / upper_map_length_limit) * norm_max
            #                 z1 = (z1 - norm_max) * -1 # invert reward to reward lower map_length
            #                 speed_reward = z1 / (max_reward / scale_rate)
            #                 rewards[index] += speed_reward

            #             self.jsrl_guide_windows[index][1] = step

            #             # If we haven't set the guide window yet, set it to the last frame of last attempt, that's our starting point.
            #             if self.jsrl_guide_windows[index][0] is inf:
            #                 self.jsrl_guide_windows[index][0] = step


            #     data = obs, enemy_obs, agent_state, enemy_state, rewards, enemy_rewards, \
            #            dones, enemy_dones, infos, enemy_infos, available_actions, available_enemy_actions, \
            #            values, enemy_values, actions, enemy_actions, action_log_probs, enemy_action_log_probs, \
            #            rnn_states, enemy_rnn_states, rnn_states_critic, enemy_rnn_states_critic 


            #     # insert data into buffer
            #     self.insert(data)

            # # compute return and update network
            # self.compute()
            # agent_train_infos, enemy_train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name in ("StarCraft2", "StarCraft2v2", "StarCraft2v2_Random"):
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []        
                    
                    enemy_battles_won = []
                    enemy_battles_game = []
                    enemy_incre_battles_won = []
                    enemy_incre_battles_game = []                

                #     for i, info in enumerate(enemy_infos):
                #         if 'battles_won' in info[0].keys():
                #             enemy_battles_won.append(info[0]['battles_won'])
                #             enemy_incre_battles_won.append(info[0]['battles_won']-enemy_last_battles_won[i])
                #         if 'battles_game' in info[0].keys():
                #             enemy_battles_game.append(info[0]['battles_game'])
                #             enemy_incre_battles_game.append(info[0]['battles_game']-enemy_last_battles_game[i])

                #     enemy_incre_win_rate = np.sum(enemy_incre_battles_won)/np.sum(enemy_incre_battles_game) if np.sum(enemy_incre_battles_game)>0 else 0.0
                #     print("enemy incre win rate is {}.".format(enemy_incre_win_rate))
                #     if self.use_wandb:
                #         wandb.log({"enemy_incre_win_rate": enemy_incre_win_rate}, step=total_num_steps)
                #         # wandb.log({"enemy_guide_window": self.jsrl_guide_windows[0][0]}, step=total_num_steps)
                #     else:
                #         self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)


                #     for i, info in enumerate(infos):
                #         if 'battles_won' in info[0].keys():
                #             battles_won.append(info[0]['battles_won'])
                #             incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                #         if 'battles_game' in info[0].keys():
                #             battles_game.append(info[0]['battles_game'])
                #             incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                #     incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                #     print("agent incre win rate is {}.".format(incre_win_rate))
                #     if self.use_wandb:
                #         wandb.log({"agent_incre_win_rate": incre_win_rate}, step=total_num_steps)
                #         # wandb.log({"agent_guide_window": self.jsrl_guide_windows[0][0]}, step=total_num_steps)
                #     else:
                #         self.writter.add_scalars("agent_incre_win_rate", {"agent_incre_win_rate": incre_win_rate}, total_num_steps)
                    
                #     last_battles_game = battles_game
                #     last_battles_won = battles_won
                    
                #     enemy_last_battles_game = enemy_battles_game
                #     enemy_last_battles_won = enemy_battles_won

                # agent_train_infos['dead_ratio'] = 1 - self.buffer1.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer1.active_masks.shape)) 
                # enemy_train_infos['dead_ratio'] = 1 - self.buffer2.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer2.active_masks.shape)) 
                
                # self.log_train(agent_train_infos, enemy_train_infos, total_num_steps)

            # If we make it through some number of episodes, just adjust guide window anyway
            # if (self.all_args.jump_start_model_dir or self.all_args.jump_start_model_pool_dir) and self.episodes_since_guide_window_reduction >= self.episode_threshold:
            #     for key in self.jsrl_guide_windows.keys():
            #         if self.jsrl_guide_windows[key][0] > 0:
            #             self.jsrl_guide_windows[key][0] = self.jsrl_guide_windows[key][0] - 1

                self.episodes_since_guide_window_reduction = -1

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                # rewrite evaluate function to work with evaluating against benchmark model
                if self.env_name == "StarCraft2v2_Random":
                    # self.eval_mp_random(total_num_steps)
                    self.eval(total_num_steps)
                    self.eval_envs.envs[0].save_replay()
                else:
                    self.eval_static(total_num_steps)
                    self.eval_envs[0][0].save_replay()

            self.episodes_since_guide_window_reduction += 1

    def warmup(self):
        # reset env
        obs, opp_obs, agent_state, enemy_state, available_actions, avail_enemy_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            agent_state = obs
            enemy_state = opp_obs

        self.buffer1.share_obs[0] = agent_state.copy()
        self.buffer1.obs[0] = obs.copy()
        self.buffer1.available_actions[0] = available_actions.copy()
        
        self.buffer2.share_obs[0] = enemy_state.copy()
        self.buffer2.obs[0] = opp_obs.copy()
        self.buffer2.available_actions[0] = avail_enemy_actions.copy()
        # self.enemy_obs = opp_obs


    @torch.no_grad()
    def collect(self, step):
        self.trainer1.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer1.policy.get_actions(np.concatenate(self.buffer1.share_obs[step]),
                                            np.concatenate(self.buffer1.obs[step]),
                                            np.concatenate(self.buffer1.rnn_states[step]),
                                            np.concatenate(self.buffer1.rnn_states_critic[step]),
                                            np.concatenate(self.buffer1.masks[step]),
                                            np.concatenate(self.buffer1.available_actions[step]))
            
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    # @torch.no_grad()
    # def collect_guide(self, step):
    #     self.trainer.prep_rollout()
    #     value, action, action_log_prob, rnn_state, rnn_state_critic \
    #         = self.trainer.jump_start_policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
    #                                         np.concatenate(self.buffer.obs[step]),
    #                                         np.concatenate(self.buffer.rnn_states[step]),
    #                                         np.concatenate(self.buffer.rnn_states_critic[step]),
    #                                         np.concatenate(self.buffer.masks[step]),
    #                                         np.concatenate(self.buffer.available_actions[step]))
    #     # [self.envs, agents, dim]
    #     values = np.array(np.split(_t2n(value), self.n_rollout_threads))
    #     actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
    #     action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
    #     rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
    #     rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

    #     return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    @torch.no_grad()
    def collect_enemy(self, step):
        self.trainer2.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer2.policy.get_actions(np.concatenate(self.buffer2.share_obs[step]),
                                            np.concatenate(self.buffer2.obs[step]),
                                            np.concatenate(self.buffer2.rnn_states[step]),
                                            np.concatenate(self.buffer2.rnn_states_critic[step]),
                                            np.concatenate(self.buffer2.masks[step]),
                                            np.concatenate(self.buffer2.available_actions[step]))
            
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, opp_obs, agent_state, enemy_state, rewards, enemy_rewards, dones, enemy_dones, infos, enemy_infos, available_actions, enemy_avail_actions, \
        values, enemy_values, actions, enemy_actions, action_log_probs, enemy_action_log_probs, rnn_states, enemy_rnn_states, \
        rnn_states_critic, enemy_rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer1.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            agent_state = obs

        self.buffer1.insert(agent_state, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)
        
        # Insert data to buffer for opposing model
        enemy_dones_env = np.all(enemy_dones, axis=1)
        
        enemy_rnn_states[enemy_dones_env == True] = np.zeros(((enemy_dones_env == True).sum(), self.num_enemies, self.recurrent_N, self.hidden_size), dtype=np.float32)
        enemy_rnn_states_critic[enemy_dones_env == True] = np.zeros(((enemy_dones_env == True).sum(), self.num_enemies, *self.buffer2.rnn_states_critic.shape[3:]), dtype=np.float32)

        enemy_masks = np.ones((self.n_rollout_threads, self.num_enemies, 1), dtype=np.float32)
        enemy_masks[enemy_dones_env == True] = np.zeros(((enemy_dones_env == True).sum(), self.num_enemies, 1), dtype=np.float32)

        enemy_active_masks = np.ones((self.n_rollout_threads, self.num_enemies, 1), dtype=np.float32)
        enemy_active_masks[enemy_dones == True] = np.zeros(((enemy_dones == True).sum(), 1), dtype=np.float32)
        enemy_active_masks[enemy_dones_env == True] = np.ones(((enemy_dones_env == True).sum(), self.num_enemies, 1), dtype=np.float32)

        enemy_bad_masks = np.array([[[0.0] if info[enemy_id]['bad_transition'] else [1.0] for enemy_id in range(self.num_enemies)] for info in enemy_infos])
        
        if not self.use_centralized_V:
            enemy_state = opp_obs

        # print("buffer size: ", np.shape(enemy_state))
        
        self.buffer2.insert(enemy_state, opp_obs, enemy_rnn_states, enemy_rnn_states_critic,
                           enemy_actions, enemy_action_log_probs, enemy_values, enemy_rewards, 
                           enemy_masks, enemy_bad_masks, enemy_active_masks, enemy_avail_actions)
        

    def log_train(self, agent_train_infos, enemy_train_infos, total_num_steps):
        agent_train_infos["average_step_rewards"] = np.mean(self.buffer1.rewards)
        enemy_train_infos["average_step_rewards"] = np.mean(self.buffer2.rewards)
        # info_we_want_to_keep = ['average_step_rewards', 'dead_ratio']
        
        for k, v in agent_train_infos.items():
            # if k not in info_we_want_to_keep:
            #     continue # Skip info we don't care about.

            if self.use_wandb:
                wandb.log({f"agent_{k}": v}, step=total_num_steps)
            else:
                self.writter.add_scalars(f"agent_{k}", {f"agent_{k}": v}, total_num_steps)
                
        for k, v in enemy_train_infos.items():
            # if k not in info_we_want_to_keep:
            #     continue # Skip info we don't care about.

            if self.use_wandb:
                wandb.log({f"enemy_{k}": v}, step=total_num_steps)
            else:
                self.writter.add_scalars(f"enemy_{k}", {f"enemy_{k}": v}, total_num_steps)
    

    @torch.no_grad()
    def eval(self, total_num_steps):        
        trainers = [self.trainer1, self.trainer2]
        teams = ["agent", "enemy"]

    
        # num_agents  = self.num_agents  if team == "agent" else self.num_enemies
        # num_enemies = self.num_enemies if team == "agent" else self.num_agents
        # modes = ("random", "bot") if self.seed > 1 else ("random")
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        eval_episode_healths = []
        
        agent_obs, enemy_obs, agent_state, enemy_state, agent_available_actions, enemy_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_opp_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_enemies, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_opp_masks = np.ones((self.n_eval_rollout_threads, self.num_enemies, 1), dtype=np.float32)


        # eval_env_infos = {f'total_health_remaining_{team}': 0, f'eval_average_episode_rewards_{team}': 0}
            
                
        while True:
        #     if team == "agent":
        #         eval_obs = agent_obs
        #         eval_state = agent_state
        #         eval_available_actions = agent_available_actions
        #         enemy_avail_actions = enemy_available_actions
        #         policy = self.opp1_policy
        #         eval_opp_obs = enemy_obs
        #         eval_opp_state = enemy_state
        #     elif team == "enemy":
        #         eval_obs = enemy_obs
        #         eval_state = enemy_state
        #         eval_available_actions = enemy_available_actions
        #         enemy_avail_actions = agent_available_actions
        #         policy = self.opp2_policy
        #         eval_opp_obs = agent_obs
        #         eval_opp_state = agent_state
                
            
            self.trainer1.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer1.policy.act(np.concatenate(agent_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(agent_available_actions),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            # print("EVAL ENVIRONMENT", self.eval_envs)
            
            # opp_rnn_states = self.collect_enemy(self.enemy_obs,opp_rnn_states,opp_masks)
                    
            
            # Obser reward and next obs
            previous_state = agent_state
            previous_opp_state = enemy_state
            # eval_obs, eval_agent_state, eval_rewards, eval_dones, \
            # eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            
            # Make random selection for enemy actions to perform against random agents
            # if self.seed == 1 or mode == "random":
            #     enemy_actions = []
            #     for actions in enemy_avail_actions[0]:
            #         avail_actions_ind = np.nonzero(actions)[0]
            #         action = np.random.choice(avail_actions_ind)
            #         enemy_actions.append(action)
            #     #     print("Indices: ", avail_actions_ind)
            #     #     print("Action: ", action)
            #     enemy_actions = np.array(enemy_actions)

            #     # print("Enemy_actions: ", enemy_actions)
            #     # print("Eval actions: ", eval_actions)
            #     # if type(enemy_actions) == tensor
            #     enemy_actions = np.array(np.split(enemy_actions, self.n_eval_rollout_threads))
            # else:
            self.trainer2.prep_rollout()
            enemy_actions, eval_opp_rnn_states = \
                self.trainer2.policy.act(np.concatenate(enemy_obs),
                                    np.concatenate(eval_opp_rnn_states),
                                    np.concatenate(eval_opp_masks),
                                    np.concatenate(enemy_available_actions),
                                    deterministic=True)
            enemy_actions = np.array(np.split(_t2n(enemy_actions), self.n_eval_rollout_threads))
            eval_opp_rnn_states = np.array(np.split(_t2n(eval_opp_rnn_states), self.n_eval_rollout_threads))
                

            self.eval_envs.envs[0].render()

            # Just to check and print any actions that may be invalid
            if torch.is_tensor(enemy_actions):
                print(enemy_actions)
            
            agent_obs, enemy_obs, agent_state, enemy_state, agent_rewards, enemy_rewards, \
            agent_dones, enemy_dones, agent_infos, enemy_infos, agent_available_actions, enemy_available_actions \
                = self.eval_envs.step(eval_actions, enemy_actions) #if team == "agent"\
                    #else self.eval_envs.step(enemy_actions, eval_actions)
            
            time.sleep(0.02)
            
            
            # if team == "agent":
            #     # eval_state = agent_state
            #     eval_infos = agent_infos
            #     eval_rewards = agent_rewards
            #     eval_dones = agent_dones
            #     eval_opp_dones = enemy_dones
            # else:
            #     # eval_state = enemy_state
            #     eval_infos = enemy_infos
            #     eval_rewards = enemy_rewards
            #     eval_dones = enemy_dones
            #     eval_opp_dones = agent_dones
                
            
            one_episode_rewards.append(agent_rewards)

            eval_dones_env = np.all(agent_dones, axis=1)
            eval_opp_dones_env = np.all(enemy_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            eval_opp_masks = np.ones((self.n_eval_rollout_threads, self.num_enemies, 1), dtype=np.float32)
            eval_opp_masks[eval_opp_dones_env == True] = np.zeros(((eval_opp_dones_env == True).sum(), self.num_enemies, 1), dtype=np.float32)

            # Get relative health and shield values for units, this will only work with protoss?
            featureCount = 22

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1

                    total_relative_health = 0
                    total_relative_shield = 0
                    for agent in range(self.num_agents):
                        healthIdx = agent * featureCount
                        shieldIdx = healthIdx + 4
                        total_relative_health += previous_state[eval_i][agent][healthIdx]
                        total_relative_shield += previous_state[eval_i][agent][shieldIdx]

                    # if eval_infos[eval_i][0]['won']:
                    #     eval_env_infos[f'total_health_remaining_{team}'] = (total_relative_shield + total_relative_health) / (num_agents * 2)
                    # else:
                    #     eval_env_infos[f'total_health_remaining_{team}'] = 0 


                    # eval_episode_healths.append(eval_env_infos[f'total_health_remaining_{team}'])
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    # if eval_infos[eval_i][0]['won']:
                    #     eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_episode_healths = np.array(eval_episode_healths)
                # eval_env_infos[f'eval_average_episode_rewards_{team}'] = eval_episode_rewards
                # eval_env_infos[f'total_health_remaining_{team}'] = eval_episode_healths
        
                # self.log_env(agent_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                # print("{} {} eval win rate is {}.".format(team, mode, eval_win_rate))

                # if self.use_wandb:
                #     # wandb.log({f"{mode}_eval_win_rate_{team}": eval_win_rate}, step=total_num_steps)
                #     # wandb.log({f"total_health_remaining_{team}": eval_episode_healths.mean()}, step=total_num_steps)
                # else:
                #     self.writter.add_scalars(f"eval_win_rate_{team}", {f"eval_win_rate_{team}": eval_win_rate}, total_num_steps)
                break

    @torch.no_grad()
    def eval_mp_random(self, total_num_steps):        
        trainers = [self.trainer1, self.trainer2]
        teams = ["agent", "enemy"]

        for trainer, team in zip(trainers, teams):
            num_agents  = self.num_agents  if team == "agent" else self.num_enemies
            num_enemies = self.num_enemies if team == "agent" else self.num_agents
            modes = ("random", "bot") if self.seed > 1 else (["random"])
            print(modes)
            print(type(modes))
            modes = [modes] if type(modes) == str else modes
            for mode in modes:
                eval_battles_won = 0
                eval_episode = 0

                eval_episode_rewards = []
                one_episode_rewards = []
                eval_episode_healths = []
                
                agent_obs, enemy_obs, agent_state, enemy_state, agent_available_actions, enemy_available_actions = self.eval_envs.reset()

                eval_rnn_states = np.zeros((self.n_eval_rollout_threads, num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_opp_rnn_states = np.zeros((self.n_eval_rollout_threads, num_enemies, self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_eval_rollout_threads, num_agents, 1), dtype=np.float32)
                eval_opp_masks = np.ones((self.n_eval_rollout_threads, num_enemies, 1), dtype=np.float32)


                eval_env_infos = {f'total_health_remaining_{team}': 0, f'eval_average_episode_rewards_{team}': 0}
                    
                        
                while True:
                    if team == "agent":
                        eval_obs = agent_obs
                        eval_state = agent_state
                        eval_available_actions = agent_available_actions
                        enemy_avail_actions = enemy_available_actions
                        policy = self.opp1_policy
                        eval_opp_obs = enemy_obs
                        eval_opp_state = enemy_state
                    elif team == "enemy":
                        eval_obs = enemy_obs
                        eval_state = enemy_state
                        eval_available_actions = enemy_available_actions
                        enemy_avail_actions = agent_available_actions
                        policy = self.opp2_policy
                        eval_opp_obs = agent_obs
                        eval_opp_state = agent_state
                        
                    
                    trainer.prep_rollout()
                    eval_actions, eval_rnn_states = \
                        trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                np.concatenate(eval_available_actions),
                                                deterministic=True)
                    eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
                    eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
                    # print("EVAL ENVIRONMENT", self.eval_envs)
                    
                    # opp_rnn_states = self.collect_enemy(self.enemy_obs,opp_rnn_states,opp_masks)
                            
                    
                    # Obser reward and next obs
                    previous_state = eval_state
                    previous_opp_state = eval_opp_state
                    # eval_obs, eval_agent_state, eval_rewards, eval_dones, \
                    # eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
                    
                    # Make random selection for enemy actions to perform against random agents
                    if self.seed == 1 or mode == "random":
                        enemy_actions = []
                        for actions in enemy_avail_actions[0]:
                            avail_actions_ind = np.nonzero(actions)[0]
                            action = np.random.choice(avail_actions_ind)
                            enemy_actions.append(action)
                        #     print("Indices: ", avail_actions_ind)
                        #     print("Action: ", action)
                        enemy_actions = np.array(enemy_actions)

                        # print("Enemy_actions: ", enemy_actions)
                        # print("Eval actions: ", eval_actions)
                        # if type(enemy_actions) == tensor
                        enemy_actions = np.array(np.split(enemy_actions, self.n_eval_rollout_threads))
                    else:
                        enemy_actions, eval_opp_rnn_states = \
                            policy.act(np.concatenate(eval_opp_obs),
                                                np.concatenate(eval_opp_rnn_states),
                                                np.concatenate(eval_opp_masks),
                                                np.concatenate(enemy_avail_actions),
                                                deterministic=True)
                        enemy_actions = np.array(np.split(_t2n(enemy_actions), self.n_eval_rollout_threads))
                        eval_opp_rnn_states = np.array(np.split(_t2n(eval_opp_rnn_states), self.n_eval_rollout_threads))
                        
                    # Just to check and print any actions that may be invalid
                    if torch.is_tensor(enemy_actions):
                        print(enemy_actions)
                    
                    agent_obs, enemy_obs, agent_state, enemy_state, agent_rewards, enemy_rewards, \
                    agent_dones, enemy_dones, agent_infos, enemy_infos, agent_available_actions, enemy_available_actions \
                        = self.eval_envs.step(eval_actions, enemy_actions) if team == "agent"\
                            else self.eval_envs.step(enemy_actions, eval_actions)
                    
                    
                    if team == "agent":
                        # eval_state = agent_state
                        eval_infos = agent_infos
                        eval_rewards = agent_rewards
                        eval_dones = agent_dones
                        eval_opp_dones = enemy_dones
                    else:
                        # eval_state = enemy_state
                        eval_infos = enemy_infos
                        eval_rewards = enemy_rewards
                        eval_dones = enemy_dones
                        eval_opp_dones = agent_dones
                        
                    
                    one_episode_rewards.append(eval_rewards)

                    eval_dones_env = np.all(eval_dones, axis=1)
                    eval_opp_dones_env = np.all(eval_opp_dones, axis=1)

                    eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

                    eval_masks = np.ones((self.all_args.n_eval_rollout_threads, num_agents, 1), dtype=np.float32)
                    eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), num_agents, 1), dtype=np.float32)

                    eval_opp_masks = np.ones((self.n_eval_rollout_threads, num_enemies, 1), dtype=np.float32)
                    eval_opp_masks[eval_opp_dones_env == True] = np.zeros(((eval_opp_dones_env == True).sum(), num_enemies, 1), dtype=np.float32)

                    # Get relative health and shield values for units, this will only work with protoss?
                    featureCount = 22

                    for eval_i in range(self.n_eval_rollout_threads):
                        if eval_dones_env[eval_i]:
                            eval_episode += 1

                            total_relative_health = 0
                            total_relative_shield = 0
                            for agent in range(num_agents):
                                healthIdx = agent * featureCount
                                shieldIdx = healthIdx + 4
                                total_relative_health += previous_state[eval_i][agent][healthIdx]
                                total_relative_shield += previous_state[eval_i][agent][shieldIdx]

                            if eval_infos[eval_i][0]['won']:
                                eval_env_infos[f'total_health_remaining_{team}'] = (total_relative_shield + total_relative_health) / (num_agents * 2)
                            else:
                                eval_env_infos[f'total_health_remaining_{team}'] = 0 


                            eval_episode_healths.append(eval_env_infos[f'total_health_remaining_{team}'])
                            eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                            one_episode_rewards = []
                            if eval_infos[eval_i][0]['won']:
                                eval_battles_won += 1

                    if eval_episode >= self.all_args.eval_episodes:
                        eval_episode_rewards = np.array(eval_episode_rewards)
                        eval_episode_healths = np.array(eval_episode_healths)
                        eval_env_infos[f'eval_average_episode_rewards_{team}'] = eval_episode_rewards
                        eval_env_infos[f'total_health_remaining_{team}'] = eval_episode_healths
                
                        self.log_env(eval_env_infos, total_num_steps)
                        eval_win_rate = eval_battles_won/eval_episode
                        print("{} {} eval win rate is {}.".format(team, mode, eval_win_rate))

                        if self.use_wandb:
                            wandb.log({f"{mode}_eval_win_rate_{team}": eval_win_rate}, step=total_num_steps)
                            wandb.log({f"total_health_remaining_{team}": eval_episode_healths.mean()}, step=total_num_steps)
                        else:
                            self.writter.add_scalars(f"eval_win_rate_{team}", {f"eval_win_rate_{team}": eval_win_rate}, total_num_steps)
                        break
                
                # break

    @torch.no_grad()
    def eval_static(self, total_num_steps):
        # TODO: Edit this method to accomodate for two different configurations per side, currently only allows identical configurations
        trainers = [self.trainer1, self.trainer2]
        teams = ["agent", "enemy"]

        for trainer, team, eval_env in zip(trainers, teams, self.eval_envs):
            num_agents = self.num_agents if team == "agent" else self.num_enemies
            eval_battles_won = 0
            eval_episode = 0

            eval_episode_rewards = []
            one_episode_rewards = []
            eval_episode_healths = []

            eval_obs, eval_agent_state, eval_available_actions = eval_env.reset()

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, num_agents, 1), dtype=np.float32)

            print(np.shape(eval_obs), np.shape(eval_rnn_states), np.shape(eval_masks))
            eval_env_infos = {f'total_health_remaining_{team}': 0, f'eval_average_episode_rewards_{team}': 0}
            while True:
                    
                trainer.prep_rollout()
                eval_actions, eval_rnn_states = \
                    trainer.policy.act(np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
                # print("EVAL ENVIRONMENT", eval_env)
                
                # opp_rnn_states = self.collect_enemy(self.enemy_obs,opp_rnn_states,opp_masks)
                        
                
                # Obser reward and next obs
                previous_state = eval_agent_state
                eval_obs, eval_agent_state, eval_rewards, eval_dones, \
                eval_infos, eval_available_actions = eval_env.step(eval_actions)
                one_episode_rewards.append(eval_rewards)

                eval_dones_env = np.all(eval_dones, axis=1)

                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), num_agents, 1), dtype=np.float32)

                # Get relative health and shield values for units, this will only work with protoss?
                featureCount = 22

                for eval_i in range(self.n_eval_rollout_threads):
                    if eval_dones_env[eval_i]:
                        eval_episode += 1

                        total_relative_health = 0
                        total_relative_shield = 0
                        for agent in range(num_agents):
                            healthIdx = agent * featureCount
                            shieldIdx = healthIdx + 4
                            total_relative_health += previous_state[eval_i][agent][healthIdx]
                            total_relative_shield += previous_state[eval_i][agent][shieldIdx]

                        if eval_infos[eval_i][0]['won']:
                            eval_env_infos[f'total_health_remaining_{team}'] = (total_relative_shield + total_relative_health) / (num_agents * 2)
                        else:
                            eval_env_infos[f'total_health_remaining_{team}'] = 0 


                        eval_episode_healths.append(eval_env_infos[f'total_health_remaining_{team}'])
                        eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                        one_episode_rewards = []
                        if eval_infos[eval_i][0]['won']:
                            eval_battles_won += 1

                if eval_episode >= self.all_args.eval_episodes:
                    eval_episode_rewards = np.array(eval_episode_rewards)
                    eval_episode_healths = np.array(eval_episode_healths)
                    eval_env_infos[f'eval_average_episode_rewards_{team}'] = eval_episode_rewards
                    eval_env_infos[f'total_health_remaining_{team}'] = eval_episode_healths
            
                    self.log_env(eval_env_infos, total_num_steps)
                    eval_win_rate = eval_battles_won/eval_episode
                    print("eval win rate is {}.".format(eval_win_rate))

                    if self.use_wandb:
                        wandb.log({f"eval_win_rate_{team}": eval_win_rate}, step=total_num_steps)
                        wandb.log({f"total_health_remaining_{team}": eval_episode_healths.mean()}, step=total_num_steps)
                    else:
                        self.writter.add_scalars(f"eval_win_rate_{team}", {f"eval_win_rate_{team}": eval_win_rate}, total_num_steps)
                    break
                
                # break
