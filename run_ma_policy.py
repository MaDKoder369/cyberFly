import os
import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import Counter
import wandb
import time
import re
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.utils import check_for_correct_spaces, get_device
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.vec_env.patch_gym import _convert_space
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
import torch as th

# Envs
from cyberFly.pz_envs import MAFixedwingDogfightEnvV2
from cyberFly.pz_envs.quadx_envs.ma_combat_env import CombatWaypointPursuitEnv
from cyberFly.pz_envs.quadx_envs.ma_quadx_hover_env import MAQuadXHoverEnv
from cyberFly.pz_envs.quadx_envs.ma_quadx_dogfight_env import MAQuadXDogfightEnv

DARKNESS = '#060606' #040404 too dark,# 131313 too bright

# Global Defaults
ENV_REGISTRY = {
    "dogfight": MAFixedwingDogfightEnvV2,
    # "dogfight_QX": MAQuadXDogfightEnv,       # Implementation not finished
    "combat": CombatWaypointPursuitEnv,      # Results Questionable
    "hover": MAQuadXHoverEnv,
}

class DummyGymEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        observation = self.observation_space.sample()
        return observation, 0.0, True, False, {}

    def render(self, mode="human"):
        return None

    def close(self):
        pass


def load_sac_compat(path, env, device="auto", custom_objects=None, print_system_info=False, force_reset=True, **kwargs):
    data, params, pytorch_variables = load_from_zip_file(
        path,
        device=device,
        custom_objects=custom_objects,
        print_system_info=print_system_info,
    )

    assert data is not None, "No data found in the saved file"
    assert params is not None, "No params found in the saved file"

    if "policy_kwargs" in data:
        if "device" in data["policy_kwargs"]:
            del data["policy_kwargs"]["device"]
        saved_net_arch = data["policy_kwargs"].get("net_arch")
        if saved_net_arch and isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
            data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

    if "observation_space" not in data or "action_space" not in data:
        if env is None:
            raise KeyError(
                "The observation_space and action_space were not given, can't verify new environments"
            )
        data["observation_space"] = env.observation_space
        data["action_space"] = env.action_space

    for key in {"observation_space", "action_space"}:
        data[key] = _convert_space(data[key])

    if env is not None:
        check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        if force_reset:
            data["_last_obs"] = None
        try:
            data["n_envs"] = env.num_envs
        except AttributeError:
            data["n_envs"] = 1
    else:
        if "env" in data:
            env = data["env"]

    if "policy_class" not in data and "policy" in data:
        data["policy_class"] = data["policy"]

    model = SAC(
        policy=data["policy_class"],
        env=env,
        device=device,
        _init_setup_model=False,
    )

    model.__dict__.update(data)
    model.__dict__.update(kwargs)
    model._setup_model()

    try:
        model.set_parameters(params, exact_match=True, device=device)
    except RuntimeError as e:
        if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
            model.set_parameters(params, exact_match=False, device=device)
            warnings.warn(
                "You are probably loading a A2C/PPO model saved with SB3 < 1.7.0, "
                "we deactivated exact_match so you can save the model "
                "again to avoid issues in the future (see SB3 issue #1233). "
                f"Original error: {e} \n"
                "Note: the model should still work fine, this is only a warning."
            )
        else:
            raise
    except ValueError as e:
        saved_optim_params = params["policy.optimizer"]["param_groups"][0]["params"]
        n_params_saved = len(saved_optim_params)
        n_params = len(model.policy.optimizer.param_groups[0]["params"])
        if n_params_saved == 2 * n_params:
            params["policy.optimizer"]["param_groups"][0]["params"] = saved_optim_params[:n_params]
            model.set_parameters(params, exact_match=True, device=device)
            warnings.warn(
                "You are probably loading a DQN model saved with SB3 < 2.4.0, "
                "we truncated the optimizer state so you can save the model "
                "again to avoid issues in the future (see SB3 issue #1963). "
                f"Original error: {e} \n"
                "Note: the model should still work fine, this is only a warning."
            )
        else:
            raise

    if pytorch_variables is not None:
        for name in pytorch_variables:
            if pytorch_variables[name] is None:
                continue
            recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

    if model.use_sde:
        model.policy.reset_noise()

    return model


def extract_env_type(path: str) -> str:
    """Extract env_type from a results path and validate against ENV_REGISTRY."""
    match = re.search(r'save-([^-]+)-', path)
    if not match:
        raise ValueError(f"Could not extract environment type from path: {path}")
    
    env_type = match.group(1)
    
    if env_type not in ENV_REGISTRY:
        raise KeyError(
            f"Environment type '{env_type}' not found in ENV_REGISTRY. "
            f"Available: {list(ENV_REGISTRY.keys())}"
        )
    
    return env_type

def evaluate_competitive_game(env, models, num_episodes=100, seed=None):
    """
    Evaluate a multi-agent competitive game environment (dogfight, combat, etc.).
    
    Args:
        env: Environment instance.
        num_episodes: Number of episodes to evaluate.
        seed: Optional seed.

    Returns:
        dict: {'team_0_wins': int, 'team_1_wins': int, 'ties': int}
    """
    results = {
        "team_0_wins": 0,
        "team_1_wins": 0,
        "ties": 0
    }

    for ep in range(num_episodes):
        print(f"[INFO] Evaluating Episode {ep}...")
        if seed is not None:
            obs, _ = env.reset(seed=seed + ep)
        else:
            obs, _ = env.reset()

        dones = {agent: False for agent in env.agents}
        actions = {}

        while not all(dones.values()):
            for agent in env.agents:
                agent_obs = obs[agent]
                actions[agent], _ = models[agent].predict(agent_obs, deterministic=True)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            dones = {agent: terminations[agent] or truncations[agent] for agent in env.agents}

        # Check info dicts for team wins
        team_win_flags = [infos[ag].get("team_win", False) for ag in infos]
        if any(team_win_flags):
            # Determine which team won
            print(f"[INFO] flag:", env.unwrapped.team_flag)
            # if env.unwrapped.team_flag[0]:  # uav_0's team is True
            # Map win flags to team indexes
            team_idx = [int(env.unwrapped.team_flag[env.unwrapped.agent_name_mapping[ag]]) for ag in infos]
            winning_teams = {team_idx[i] for i, win in enumerate(team_win_flags) if win}
            if len(winning_teams) == 1:
                if 0 in winning_teams:
                    results["team_0_wins"] += 1
                else:
                    results["team_1_wins"] += 1
            else:
                    results["ties"] += 1
            
            # raise ValueError("Unexpected team_flag format")
        else:
            results["ties"] += 1

        # env.close()
    env.close()
    return dict(results)

def plot_win_rates(strategies, eval_results):

    if not eval_results:
        raise ValueError("eval_results must contain at least one result")

    if len(strategies) < len(eval_results):
        strategies = list(strategies) + [f"Strategy {i+1}" for i in range(len(strategies), len(eval_results))]
    elif len(strategies) > len(eval_results):
        warnings.warn(
            "The number of strategies does not match the number of evaluation results. "
            "Only the first {len(eval_results)} strategies will be plotted.",
            UserWarning,
        )
        strategies = strategies[: len(eval_results)]

    # Extract Results
    team_0_wins = [r['team_0_wins'] for r in eval_results]
    ties        = [r['ties'] for r in eval_results]
    team_1_wins = [r['team_1_wins'] for r in eval_results]

    iters = sum(eval_results[0].values())

    x = np.arange(len(eval_results))
    bar_width = 0.6

    # Apply rcParams BEFORE creating figure
    plt.style.use('dark_background')
    plt.rcParams.update({
        "axes.grid": True,
        "grid.color": '#444444',
        "text.color": '#e0e0e0',
        "axes.labelcolor": '#d0d0d0',
        "xtick.color": '#d0d0d0',
        "ytick.color": '#d0d0d0',
        "legend.edgecolor": '#444444'
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARKNESS) 
    ax.set_facecolor(DARKNESS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(False)

    p1 = ax.bar(x, team_0_wins, bar_width, label="Team 0 Wins", color="cyan")
    p2 = ax.bar(x, ties, bar_width, bottom=team_0_wins, label="Ties", color="#F5F5F5")
    p3 = ax.bar(x, team_1_wins, bar_width,
                bottom=np.array(team_0_wins) + np.array(ties),
                label="Team 1 Wins", color="magenta")
    
    # Add value labels on each segment
    for i in range(len(strategies)):
        # Team 0 Wins labels (middle of their bars)
        if team_0_wins[i] > 0:
            ax.text(x[i], team_0_wins[i] / 2, str(team_0_wins[i]), ha='center', va='center', color='black', fontsize=9)

        # Ties labels (middle of ties segment, offset by team_0_wins)
        if ties[i] > 0:
            ax.text(x[i], team_0_wins[i] + ties[i] / 2, str(ties[i]), ha='center', va='center', color='black', fontsize=9)

        # Team 1 Wins labels (middle of team_1_wins segment, offset by team_0_wins + ties)
        if team_1_wins[i] > 0:
            ax.text(x[i], team_0_wins[i] + ties[i] + team_1_wins[i] / 2, str(team_1_wins[i]), ha='center', va='center', color='black', fontsize=9)


    # Labels and formatting
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Number of Games')
    ax.set_title(f'Game Outcomes per Strategy ({iters} games)({DARKNESS})')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, iters)
    # ax.legend()

    plt.tight_layout()
    plt.show()

def plot_training_rewards(data_sets):
    # Example data
    x = np.arange(1, 31)  # 30 evaluation episodes # TODO Read from data_sets

    # Apply rcParams BEFORE creating figure
    plt.style.use('dark_background')
    plt.rcParams.update({
        "axes.grid": True,
        "grid.color": '#444444',
        "text.color": '#e0e0e0',
        "axes.labelcolor": '#d0d0d0',
        "xtick.color": '#d0d0d0',
        "ytick.color": '#d0d0d0',
        "legend.edgecolor": '#444444'
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARKNESS) #040404 too dark,# 131313 too bright
    ax.set_facecolor(DARKNESS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot 
    for data in data_sets:
        ax.plot(data["x"], data["mean"], label=data["label"], color=data["color"])
        ax.fill_between(
            data["x"],
            np.array(data["mean"]) - np.array(data["std"]),
            np.array(data["mean"]) + np.array(data["std"]),
            color=data["color"],
            alpha=0.2
        )


    ax.set_xlabel("Evaluation Episode")
    ax.set_ylabel("Win Rate (%)")
    # ax.set_title("Strategy Performance with Std. Dev. Shading")
    plt.suptitle(f"Rollout Reward {DARKNESS}")
    ax.legend()

    

    plt.show()

if __name__ == "__main__":

    print("[INFO] Beginning Evaluation...")
    strategies = ['Base Case', 'Vanilla', 'Fictitious', 'D-Uniform']

    # === Loading ===
    # TODO Fix File path loading format

    save_dirs =[]

    save_dir = './results/ma/save-dogfight-a-07.23.2025_22.28'
    # save_dir = './results/ma/save-hover-0-08.02.2025_16.46'
    # save_dir = './results/ma/save-combat-0-07.31.2025_09.38'
    model_filename_template = 'final_agent_{agent_num}_model.zip'
    # === End of Loading ===

    # === Initialize environment with rendering enabled ===
    # Initiate test environment 
    env_type = extract_env_type(save_dir)
    env_class = ENV_REGISTRY[env_type]
    test_env = env_class(render_mode="human", max_duration_seconds=15.0)
    test_env_no_gui = env_class(render_mode=None, max_duration_seconds=60.0)
    
    test_env_no_gui.reset()

    # Monkey patch for numpy compatibility
    import sys
    import numpy
    sys.modules['numpy._core'] = numpy.core
    sys.modules['numpy._core.numeric'] = numpy.core.numeric

    # Fix for numpy random bit generator
    import numpy.random._pickle
    original_ctor = numpy.random._pickle.__bit_generator_ctor
    def new_ctor(bit_generator_name):
        if isinstance(bit_generator_name, str):
            return original_ctor(bit_generator_name)
        else:
            return bit_generator_name()
    numpy.random._pickle.__bit_generator_ctor = new_ctor

    ### Load Models for all agents
    models = {}
    for i, agent in enumerate(test_env_no_gui.agents):
        model_path = os.path.join(save_dir, model_filename_template.format(agent_num=i))
        assert os.path.exists(model_path), f"Missing model for agent {i}: {model_path}"

        if callable(getattr(test_env_no_gui, 'observation_space', None)):
            obs_space = test_env_no_gui.observation_space(agent)
        else:
            obs_space = test_env_no_gui.observation_space

        if callable(getattr(test_env_no_gui, 'action_space', None)):
            act_space = test_env_no_gui.action_space(agent)
        else:
            act_space = test_env_no_gui.action_space

        dummy_env = DummyGymEnv(obs_space, act_space)
        models[agent] = load_sac_compat(model_path, env=dummy_env)

    ### Statistical Evaluation
    eval_results = []
    eval_results.append(evaluate_competitive_game(test_env_no_gui, models, num_episodes=1))
    print(f"[INFO] Consider your results, evaluated \n", eval_results)
    plot_win_rates(strategies, eval_results)
    plot_training_rewards()
    exit()
    

    ### Visual Evaluation
    obs, _ = test_env.reset(seed=7)
    while True:

        actions = {}
        for agent in test_env.agents:
            agent_obs = obs[agent]
            actions[agent], _ = models[agent].predict(agent_obs, deterministic=True)

        obs, rewards, dones, truncs, infos = test_env.step(actions)

        # Render frame
        time.sleep(1.0 / 40)

        # Exit if either agent is done
        if any(dones.values()) or any(truncs.values()):
            break

    # Optional: show trajectories
    test_env.render_trajectory()

   