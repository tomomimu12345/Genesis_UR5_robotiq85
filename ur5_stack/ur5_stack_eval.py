import argparse
import os
import pickle
from importlib import metadata

import torch

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="ur5_stack")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, train_cfg = pickle.load(f)

    env = UR5StackEnv(
        num_envs=1,
        num_blocks=env_cfg["num_blocks"],
        show_viewer=True
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rew, done, infos = env.step(actions)

if __name__ == "__main__":
    main()

"""
# evaluation
python eval_ur5_stack.py --exp_name ur5_stack --ckpt 100
"""