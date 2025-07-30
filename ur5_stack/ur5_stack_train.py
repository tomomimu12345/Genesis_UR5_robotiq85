import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from ur5_stack_env import UR5StackEnv


def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [256, 128],
            "critic_hidden_dims": [256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "experiment_name": exp_name,
            "run_name": "",
            "checkpoint": -1,
            "resume": False,
            "resume_path": None,
            "log_interval": 1,
            "record_interval": -1,
            "max_iterations": max_iterations,
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 16,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 42,
    }


def get_env_cfg():
    return {
        "num_envs": 32,
        "num_blocks": 10,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="ur5_stack")
    parser.add_argument("--max_iterations", type=int, default=1000)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg = get_env_cfg()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump((env_cfg, train_cfg), f)

    env = UR5StackEnv(num_envs=env_cfg["num_envs"], num_blocks=env_cfg["num_blocks"], show_viewer=False)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
