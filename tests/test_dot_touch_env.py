"""Tests for the QuadX Dot Touch game environment.

Covers environment creation, observation/action spaces, seeding, gymnasium
check_env compatibility, and all game-specific mechanics:
  - first-touch reward and dot-touched mask
  - completion bonus and truncation
  - truncate_on_completion=False behaviour
  - double-touch termination
  - lingering-inside-dot (no double-touch penalty)
  - proximity shaping reward
"""

from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.error import Error
from gymnasium.utils.env_checker import check_env, data_equivalence

import cyberFly.gym_envs  # noqa

_ENV_ID = "cyberFly/QuadX-DotTouch-v1"

# Warnings that are acceptable and should not fail the check_env test
_CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is -infinity. This is probably too high.",
        "A Box observation space minimum value is infinity. This is probably too low.",
        "A Box observation space maximum value is infinity. This is probably too high.",
        "Human rendering should return `None`, got <class 'numpy.ndarray'>",
        "RGB-array rendering should return a numpy array in which the last axis has three dimensions, got 4",
    ]
]

# Dot placed directly at the drone spawn so it is always inside touch radius
_SPAWN_POS = np.array([0.0, 0.0, 1.0])
# Dot placed far from spawn so it is never accidentally touched during tests
_FAR_POS = np.array([4.5, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Gymnasium API compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle_representation", ["euler", "quaternion"])
def test_check_env(angle_representation):
    """DotTouch passes the gymnasium check_env for both angle representations."""
    env = gym.make(_ENV_ID, angle_representation=angle_representation)

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env.unwrapped)

    for w in caught_warnings:
        assert isinstance(w.message, Warning)
        if w.message.args[0] not in _CHECK_ENV_IGNORE_WARNINGS:
            raise Error(f"Unexpected warning: {w.message}")

    env.close()


def test_seeding_is_deterministic():
    """Same seed produces identical observations and rewards across two instances."""
    env1 = gym.make(_ENV_ID)
    env2 = gym.make(_ENV_ID)

    obs1, info1 = env1.reset(seed=42)
    obs2, info2 = env2.reset(seed=42)
    assert data_equivalence(obs1, obs2)
    assert data_equivalence(info1, info2)

    for _ in range(10):
        action = env1.action_space.sample()
        o1, r1, t1, tr1, i1 = env1.step(action)
        o2, r2, t2, tr2, i2 = env2.step(action)
        assert data_equivalence(o1, o2)
        assert r1 == r2
        assert t1 == t2 and tr1 == tr2
        assert data_equivalence(i1, i2)
        if t1 or tr1:
            break

    env1.close()
    env2.close()


# ---------------------------------------------------------------------------
# Observation / action space shape
# ---------------------------------------------------------------------------


def test_observation_keys_present():
    """Observation dict contains exactly the three required keys."""
    env = gym.make(_ENV_ID)
    obs, _ = env.reset(seed=0)
    assert set(obs.keys()) == {"attitude", "dot_deltas", "dot_touched"}
    env.close()


@pytest.mark.parametrize("num_dots", [1, 3, 6, 10])
def test_observation_shapes_match_num_dots(num_dots):
    """dot_deltas and dot_touched observation shapes reflect num_dots."""
    env = gym.make(_ENV_ID, num_dots=num_dots)
    obs, _ = env.reset(seed=0)
    assert obs["dot_deltas"].shape == (num_dots, 3)
    assert obs["dot_touched"].shape == (num_dots,)
    env.close()


def test_observation_in_observation_space():
    """Every observation returned by reset/step is contained in the declared space."""
    env = gym.make(_ENV_ID, num_dots=3)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)

    for _ in range(5):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            break

    env.close()


def test_action_in_action_space():
    """Sampled actions are always inside the declared action space."""
    env = gym.make(_ENV_ID)
    env.reset(seed=0)
    for _ in range(20):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
    env.close()


# ---------------------------------------------------------------------------
# Info dict
# ---------------------------------------------------------------------------


def test_info_dict_initial_values():
    """After reset the game-specific info fields are initialised to zero / False."""
    env = gym.make(_ENV_ID)
    _, info = env.reset(seed=0)
    assert info["dots_touched"] == 0
    assert info["double_touch"] is False
    assert info["all_dots_touched"] is False
    env.close()


# ---------------------------------------------------------------------------
# First-touch mechanics
# ---------------------------------------------------------------------------


def test_first_touch_increments_dots_touched():
    """Touching a dot for the first time increments info['dots_touched']."""
    env = gym.make(_ENV_ID, num_dots=2)
    env.reset(seed=0)
    # Place dot 0 at spawn so it is touched on the very first step
    env.unwrapped.dots = np.array([_SPAWN_POS, _FAR_POS])

    _, _, _, _, info = env.step(env.action_space.sample() * 0)
    assert info["dots_touched"] == 1
    env.close()


def test_first_touch_adds_reward():
    """First touch adds +50 to the reward (on top of the base living-penalty)."""
    env = gym.make(_ENV_ID, num_dots=2)
    env.reset(seed=0)
    env.unwrapped.dots = np.array([_SPAWN_POS, _FAR_POS])

    _, reward, _, _, _ = env.step(env.action_space.sample() * 0)
    # base reward is -0.1; first touch adds +50 → should be well above 0
    assert reward > 0.0
    env.close()


def test_dot_touched_mask_updates_in_observation():
    """dot_touched observation reflects which dots have been touched."""
    env = gym.make(_ENV_ID, num_dots=2)
    env.reset(seed=0)
    env.unwrapped.dots = np.array([_SPAWN_POS, _FAR_POS])

    obs, _, _, _, _ = env.step(env.action_space.sample() * 0)
    assert obs["dot_touched"][0] == 1.0, "Touched dot should have mask=1"
    assert obs["dot_touched"][1] == 0.0, "Untouched dot should have mask=0"
    env.close()


def test_untouched_dot_mask_is_zero_at_reset():
    """After reset, dot_touched observation is all zeros."""
    env = gym.make(_ENV_ID, num_dots=4)
    obs, _ = env.reset(seed=0)
    assert np.all(obs["dot_touched"] == 0.0)
    env.close()


# ---------------------------------------------------------------------------
# Completion mechanics
# ---------------------------------------------------------------------------


def test_all_dots_touched_truncates_episode():
    """Touching the last dot truncates (not terminates) the episode."""
    env = gym.make(_ENV_ID, num_dots=1)
    env.reset(seed=0)
    env.unwrapped.dots = np.array([_SPAWN_POS])

    _, _, term, trunc, info = env.step(env.action_space.sample() * 0)
    assert info["all_dots_touched"]
    assert trunc, "Episode should be truncated on successful completion"
    assert not term, "Successful completion is not a termination"
    env.close()


def test_all_dots_touched_gives_completion_bonus():
    """Completing all dots adds the +200 bonus on top of per-dot rewards."""
    env = gym.make(_ENV_ID, num_dots=1)
    env.reset(seed=0)
    env.unwrapped.dots = np.array([_SPAWN_POS])

    _, reward, _, _, _ = env.step(env.action_space.sample() * 0)
    # +50 (touch) + 200 (bonus) - 0.1 (living) ≈ 249.9 — definitely above 200
    assert reward > 200.0
    env.close()


def test_truncate_on_completion_false_does_not_end_episode():
    """With truncate_on_completion=False the episode continues after all dots are touched."""
    env = gym.make(_ENV_ID, num_dots=1, truncate_on_completion=False)
    env.reset(seed=0)
    env.unwrapped.dots = np.array([_SPAWN_POS])

    _, _, term, trunc, info = env.step(env.action_space.sample() * 0)
    assert info["all_dots_touched"]
    assert not trunc, "Episode should NOT be truncated when truncate_on_completion=False"
    assert not term
    env.close()


# ---------------------------------------------------------------------------
# Double-touch mechanics
# ---------------------------------------------------------------------------


def test_double_touch_terminates_with_negative_reward():
    """Re-entering a touched dot (after leaving) terminates with reward=-100."""
    env = gym.make(_ENV_ID, num_dots=2)
    env.reset(seed=0)
    env.unwrapped.dots = np.array([_SPAWN_POS, _FAR_POS])

    # Step 1 — first touch of dot 0
    _, _, term, _, info = env.step(env.action_space.sample() * 0)
    assert info["dots_touched"] == 1
    assert not term, "Episode should still be running after first touch"

    # Simulate the drone having flown away from dot 0 and re-approaching
    env.unwrapped._was_inside_radius[0] = False

    # Step 2 — drone re-enters dot 0 → double-touch
    _, reward, term, _, info = env.step(env.action_space.sample() * 0)
    assert term, "Double-touch should terminate the episode"
    assert info["double_touch"]
    assert reward == -100.0
    env.close()


def test_lingering_inside_dot_does_not_trigger_double_touch():
    """Staying continuously inside a dot does not cause a double-touch penalty."""
    env = gym.make(_ENV_ID, num_dots=2)
    env.reset(seed=0)
    env.unwrapped.dots = np.array([_SPAWN_POS, _FAR_POS])

    action = env.action_space.sample() * 0

    # Step 1 — first touch
    _, _, term, _, info = env.step(action)
    assert info["dots_touched"] == 1
    assert not term

    # Step 2 — still inside the same dot (_was_inside_radius stays True)
    _, _, term, _, info = env.step(action)
    assert not term, "Lingering should not trigger double-touch termination"
    assert not info["double_touch"]
    assert info["dots_touched"] == 1, "Dots-touched count must not increase from lingering"
    env.close()


def test_double_touch_exempt_when_all_dots_already_touched():
    """Re-entering a touched dot after task completion does not terminate the episode."""
    env = gym.make(_ENV_ID, num_dots=1, truncate_on_completion=False)
    env.reset(seed=0)
    env.unwrapped.dots = np.array([_SPAWN_POS])

    action = env.action_space.sample() * 0

    # Step 1 — touch the only dot (task complete)
    _, _, term, trunc, info = env.step(action)
    assert info["all_dots_touched"]
    assert not term and not trunc

    # Simulate leaving the dot's radius
    env.unwrapped._was_inside_radius[0] = False

    # Step 2 — re-enter; all dots already touched → no penalty
    _, reward, term, _, info = env.step(action)
    assert not term, "Re-entry after task completion should not terminate the episode"
    assert not info["double_touch"]
    assert reward != -100.0
    env.close()


# ---------------------------------------------------------------------------
# Proximity shaping reward
# ---------------------------------------------------------------------------


def test_proximity_reward_present_when_untouched_dots_remain():
    """A proximity shaping reward is added whenever at least one dot is untouched."""
    env = gym.make(_ENV_ID, num_dots=2)
    env.reset(seed=0)
    # Place dot 0 far from spawn so first step does NOT trigger a touch
    env.unwrapped.dots = np.array([_FAR_POS, _FAR_POS + np.array([0.1, 0, 0])])

    _, reward, term, _, _ = env.step(env.action_space.sample() * 0)
    if not term:
        # reward = -0.1 (living) + 0.1/distance (proximity) > -0.1
        assert reward > -0.1, "Proximity reward should push reward above -0.1"
    env.close()


def test_proximity_reward_higher_when_closer():
    """Proximity reward is inversely proportional to distance: closer → higher reward."""
    # Two fresh envs, identical except dot distance from spawn
    env_near = gym.make(_ENV_ID, num_dots=1)
    env_far = gym.make(_ENV_ID, num_dots=1)

    env_near.reset(seed=0)
    env_far.reset(seed=0)

    # Near dot: 1 m away from spawn along x-axis
    env_near.unwrapped.dots = np.array([[1.0, 0.0, 1.0]])
    # Far dot: 4 m away from spawn along x-axis
    env_far.unwrapped.dots = np.array([[4.0, 0.0, 1.0]])

    action = env_near.action_space.sample() * 0
    _, reward_near, term_near, _, _ = env_near.step(action)
    _, reward_far, term_far, _, _ = env_far.step(action)

    if not term_near and not term_far:
        assert reward_near > reward_far, "Reward should be higher when closer to the dot"

    env_near.close()
    env_far.close()
