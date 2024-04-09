#!/usr/bin/env python
import sys
import os
import wandb
import shutil
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from onpolicy.envs.starcraft2.StarCraft2_Env_multiplayer import StarCraft2Env as StarCraft2EnvMPlayer
from onpolicy.envs.starcraft2.SMACv2_modified import SMACv2 as SMACv2_Mod
from onpolicy.envs.starcraft2.SMACv2 import SMACv2 as SMACv2_sp
from onpolicy.envs.starcraft2.SMACv2_multiplayer import SMACv2 as SMACv2_mp
from onpolicy.envs.starcraft2.smac_maps import get_map_params
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, ShareMultiDummyVecEnv, ShareMultiSubprocVecEnv
from absl import logging

# logging.set_verbosity(logging.DEBUG)


"""Train script for SMAC."""

def parse_smacv2_distribution(args):
    units = args.units.split('v')
    map_size = 32
    # map_size = 8+(args.seed*8)
    distribution_config = {
        "n_units": int(units[0]),
        "n_enemies": int(units[1]),
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "map_x": map_size,
            "map_y": map_size,
        }
    }
    if 'protoss' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            "unit_types": ["stalker", "zealot"],#, "colossus"],
            "weights": [0.4, 0.6],# 0.1],
            "observe": True,
        }
    elif 'zerg' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            "unit_types": ["zergling", "baneling", "hydralisk"],
            "weights": [0.45, 0.1, 0.45],
            "observe": True,
        } 
    elif 'terran' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            # "unit_types": ["marine", "marauder", "medivac"],
            "unit_types": ["marine"],
            "weights": [1],
            # "weights": [0.45, 0.45, 0.1],
            "observe": True,
        } 
    return distribution_config

def parse_smacv2_distribution_reverse(args):
    units = args.units.split('v')
    # map_size = 8+(args.seed*8)
    map_size = 32
    distribution_config = {
        "n_units": int(units[1]),
        "n_enemies": int(units[0]),
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "map_x": map_size,
            "map_y": map_size,
        }
    }
    if 'protoss' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            "unit_types": ["stalker", "zealot"],#, "colossus"],
            "weights": [0.2, 0.8],#, 0.1],
            "observe": True,
        }
    elif 'zerg' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            "unit_types": ["zergling", "baneling", "hydralisk"],
            "weights": [0.45, 0.1, 0.45],
            "observe": True,
        } 
    elif 'terran' in args.map_name:
        distribution_config['team_gen'] = {
            "dist_type": "weighted_teams",
            # "unit_types": ["marine", "marauder", "medivac"],
            "unit_types": ["marine"],
            "weights": [1],
            # "weights": [0.45, 0.45, 0.1],
            "observe": True,
        } 
    return distribution_config

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                env = StarCraft2EnvMPlayer(all_args)
            elif all_args.env_name in ("StarCraft2v2", "StarCraft2v2_Random"):
                env = SMACv2_mp(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareMultiDummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name in ("StarCraft2v2", "StarCraft2v2_Random"):
            return ShareMultiSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
        
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env(): # KEEP THIS ENVIRONMENT AS A BASE ENVIRONMENT AND THE TRAIN ONE AS A MULTIPLAYER ONE
            if all_args.env_name == "StarCraft2":
                env = StarCraft2Env(all_args)
            elif all_args.env_name == "StarCraft2v2":
                env = SMACv2_Mod(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            elif all_args.env_name == "StarCraft2v2_Random":
                env = SMACv2_mp(capability_config=parse_smacv2_distribution(all_args), map_name=all_args.map_name)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env
    
    # Only defined for base SMACv2
    def get_env_fn_reverse(rank):
        def init_env(): # KEEP THIS ENVIRONMENT AS A BASE ENVIRONMENT AND THE TRAIN ONE AS A MULTIPLAYER ONE
            if all_args.env_name == "StarCraft2v2":
                env = SMACv2_Mod(capability_config=parse_smacv2_distribution_reverse(all_args), map_name=all_args.map_name)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        if all_args.env_name == "StarCraft2v2_Random":
            return ShareMultiDummyVecEnv([get_env_fn(0)])
        elif all_args.env_name == "StarCraft2v2":
            return ShareDummyVecEnv([get_env_fn(0)]), ShareDummyVecEnv([get_env_fn_reverse(0)])
        else:
            return ShareDummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name == "StarCraft2v2_Random":
            return ShareMultiSubprocVecEnv([get_env_fn(0)])
        elif all_args.env_name == "StarCraft2v2": #or \
        # (all_args.env_name == "StarCraft2v2_Random" and all_args.seed == 3):
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]), \
                ShareSubprocVecEnv([get_env_fn_reverse(i) for i in range(all_args.n_eval_rollout_threads)])
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m',
                        help="Which smac map to run on")
    parser.add_argument('--units', type=str, default='10v10') # for smac v2
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)
    parser.add_argument("--jumpstart_model_dir")
    parser.add_argument("--opp_model_dir")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    if all_args.env_name == "SMAC":
        num_agents = get_map_params(all_args.map_name)["n_agents"]
        num_enemies = get_map_params(all_args.map_name)["n_enemies"]
        print('smac map details  == ', get_map_params(all_args.map_name))
    elif all_args.env_name == 'StarCraft2':
        num_agents = get_map_params(all_args.map_name)["n_agents"]
        num_enemies = get_map_params(all_args.map_name)["n_enemies"]
        print('smac map details  == ', get_map_params(all_args.map_name))
    elif all_args.env_name in ('SMACv2', 'StarCraft2v2', 'StarCraft2v2_Random'):
        num_agents = parse_smacv2_distribution(all_args)['n_units']
        num_enemies = parse_smacv2_distribution(all_args)["n_enemies"]
        print('smac map details  == ', parse_smacv2_distribution(all_args))
        
    # num_agents = get_map_params(all_args.map_name)["n_agents"]
    # num_enemies = get_map_params(all_args.map_name)["n_enemies"]

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "num_enemies": num_enemies,
        "device": device,
        "run_dir": run_dir
    }
    print("RUN DIR: ", wandb.run.dir)

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.smac_runner_adversarial_eval import SMACRunner as Runner
    else:
        from onpolicy.runner.separated.smac_runner import SMACRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

    # Put current run in a file for the next seed to use
    target_dir = os.path.join(run_dir, f"{str(all_args.seed+1)}_bot/")
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        
    for file in os.listdir(runner.save_dir):
        origin = os.path.join(runner.save_dir, file)
        outpath = os.path.join(target_dir, file)
        shutil.copy(origin, outpath)
    
    outpath = os.path.join()    
    shutil.copy(runner.save_dir, all_args.seed)


if __name__ == "__main__":
    main(sys.argv[1:])
