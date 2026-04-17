import os
import argparse
import numpy as np
print(np.__version__)
import gymnasium as gym
from gymnasium import spaces
import wandb
import time
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Pyflyt 
from cyberFly.gym_envs.quadx_envs.quadx_hover_env import QuadXHoverEnv
from cyberFly.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv

# Global Defaults
ENV_REGISTRY = {
    "hover": QuadXHoverEnv,
    "waypoints": QuadXWaypointsEnv,
}

DEFAULT_ENV = 'hover'
DEFAULT_RETRAIN = False
DEFAULT_FLIGHT_MODE = 0
DEFAULT_OUTPUT_FOLDER = os.path.join('results', 'ma')


def find_model_file(base_path, env):
    if not os.path.isdir(base_path):
        return None

    zip_candidates = []
    for root, _, files in os.walk(base_path):
        for file_name in files:
            if file_name.endswith('.zip'):
                zip_candidates.append(os.path.join(root, file_name))

    if not zip_candidates:
        return None

    preferred_names = [
        'final_model.zip',
        'final_agent_0_model.zip',
        'final_agent_1_model.zip',
        'final_agent_2_model.zip',
        'final_agent_3_model.zip',
    ]

    for name in preferred_names:
        for candidate in zip_candidates:
            if candidate.lower().endswith(name):
                return candidate

    if env == 'hover':
        for candidate in zip_candidates:
            if 'hover' in candidate.lower():
                return candidate
    elif env == 'waypoints':
        for candidate in zip_candidates:
            if 'waypoints' in candidate.lower():
                return candidate

    return zip_candidates[0]


def main(env=DEFAULT_ENV, flight_mode=DEFAULT_FLIGHT_MODE, output_folder=DEFAULT_OUTPUT_FOLDER):
    print('[INFO] Starting Simulation...')

    # Load Model
    if env == 'waypoints':
        filename = 'save-07.09.2025_01.05.40'
        filename = os.path.join(output_folder, filename)
    elif env == 'hover':
        filename = 'save-hover-0-08.04.2025_14.19'
        filename = os.path.join(output_folder, filename)
    else:
        print("[ERROR]: no file specified for the environment", env)
        exit()

    print("[INFO] Loading model from", filename)

    preferred_paths = [
        os.path.join(filename, 'final_model.zip'),
        os.path.join(filename, 'final_agent_0_model.zip'),
        os.path.join(filename, 'final_agent_1_model.zip'),
        os.path.join(filename, 'final_agent_2_model.zip'),
        os.path.join(filename, 'final_agent_3_model.zip'),
    ]

    path = next((p for p in preferred_paths if os.path.isfile(p)), None)
    if path is None:
        print("[WARNING]: no model under the specified path", filename)
        path = find_model_file(output_folder, env)
        if path is None:
            print("[ERROR]: no model under", output_folder)
            exit()
        print("[INFO] Fallback model found:", path)

    model = SAC.load(path)

    # Initiate test environment 
    env_class = ENV_REGISTRY[env]
    test_env = env_class(render_mode="human", flight_mode=flight_mode, max_duration_seconds=15.0)
    test_env_no_gui = env_class(render_mode=None, flight_mode=flight_mode)

    if model.observation_space.shape != test_env_no_gui.observation_space.shape:
        print("[ERROR]: model observation shape", model.observation_space.shape,
              "does not match env observation shape", test_env_no_gui.observation_space.shape)
        exit()

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_no_gui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    # Simulation
    obs, info = test_env.reset(seed=7)
    print("[INFO] Obs:", obs)
    print("[INFO] Start Pos:", test_env.start_pos)

    while True:
    # for i in range(200):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)

        # print("Obs:", obs, "\tAction:", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)

         # Render frame
        test_env.render()
        time.sleep(1.0 / test_env.agent_hz)

        # Exit if either agent is done
        if terminated or truncated:
            break

    test_env.close()

    # Plot


def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single agent reinforcement learning in cyberFly Gymnasium Environments")
    parser.add_argument('--env',           default=DEFAULT_ENV,           type=str,      help='Single agent gymnasium environment to train (default: hover).')
    parser.add_argument('--flight_mode',   default=DEFAULT_FLIGHT_MODE,   type=int,      help='Flight mode (0=default).')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,      help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()
    main(**vars(ARGS))

   