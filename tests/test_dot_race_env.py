"""Tests for MAQuadXDotRaceEnv — two-drone competitive dot-claiming game."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test

from cyberFly.pz_envs.quadx_envs.ma_quadx_dot_race_env import MAQuadXDotRaceEnv

# Ignore standard infinite-obs-space warnings from pettingzoo validator
_IGNORE_WARNINGS = [
    "Agent's minimum observation space value is -infinity. This is probably too low.",
    "Agent's maximum observation space value is infinity. This is probably too high",
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    e = MAQuadXDotRaceEnv(num_dots=4, max_duration_seconds=10.0, render_mode=None)
    yield e
    e.close()


# ── PettingZoo API ─────────────────────────────────────────────────────────────

def test_parallel_api():
    """Environment passes the official PettingZoo parallel API test."""
    e = MAQuadXDotRaceEnv(num_dots=4, max_duration_seconds=10.0, render_mode=None)
    with warnings.catch_warnings(record=True) as caught:
        parallel_api_test(e, num_cycles=500)
    for w in caught:
        assert isinstance(w.message, Warning)
        if w.message.args[0] not in _IGNORE_WARNINGS:
            raise AssertionError(f"Unexpected warning: {w.message}")
    e.close()


# ── Reset ─────────────────────────────────────────────────────────────────────

def test_reset_returns_two_agents(env: MAQuadXDotRaceEnv):
    obs, infos = env.reset(seed=0)
    assert set(obs.keys()) == {"uav_0", "uav_1"}
    assert set(infos.keys()) == {"uav_0", "uav_1"}


def test_reset_observation_in_space(env: MAQuadXDotRaceEnv):
    obs, _ = env.reset(seed=0)
    for ag, ob in obs.items():
        assert env.observation_space(ag).contains(ob), (
            f"{ag} reset observation not in observation_space"
        )


def test_reset_initial_scores_zero(env: MAQuadXDotRaceEnv):
    env.reset(seed=0)
    assert list(env.scores) == [0, 0]


def test_reset_all_dots_unclaimed(env: MAQuadXDotRaceEnv):
    env.reset(seed=0)
    assert np.all(env.dot_owner == -1)


def test_reset_dots_shape(env: MAQuadXDotRaceEnv):
    env.reset(seed=0)
    assert env.dots.shape == (env.num_dots, 3)


def test_reset_dots_inside_dome(env: MAQuadXDotRaceEnv):
    env.reset(seed=7)
    dists = np.linalg.norm(env.dots, axis=1)
    assert np.all(dists <= env.flight_dome_size), "A dot was placed outside the dome"


def test_reset_dots_above_ground(env: MAQuadXDotRaceEnv):
    env.reset(seed=7)
    assert np.all(env.dots[:, 2] >= 0.0), "A dot was placed below ground level"


# ── Observation / action spaces ───────────────────────────────────────────────

def test_observation_space_shape(env: MAQuadXDotRaceEnv):
    """Observation size must equal the formula: 21 + D*3 + D + 3 + 2."""
    D = env.num_dots
    expected = 21 + D * 3 + D + 3 + 2
    for ag in env.possible_agents:
        assert env.observation_space(ag).shape == (expected,)


def test_action_space_shape(env: MAQuadXDotRaceEnv):
    for ag in env.possible_agents:
        assert env.action_space(ag).shape == (4,)


def test_observations_in_space_during_episode(env: MAQuadXDotRaceEnv):
    obs, _ = env.reset(seed=1)
    for _ in range(50):
        if not env.agents:
            break
        actions = {ag: env.action_space(ag).sample() for ag in env.agents}
        obs, _, _, _, _ = env.step(actions)
        for ag, ob in obs.items():
            assert env.observation_space(ag).contains(ob)


# ── Dot ownership ─────────────────────────────────────────────────────────────

def test_dot_owner_values(env: MAQuadXDotRaceEnv):
    """dot_owner must always be -1, 0, or 1."""
    env.reset(seed=2)
    for _ in range(100):
        if not env.agents:
            break
        actions = {ag: env.action_space(ag).sample() for ag in env.agents}
        env.step(actions)
        assert np.all(np.isin(env.dot_owner, [-1, 0, 1]))


def test_claimed_dot_cannot_be_stolen(env: MAQuadXDotRaceEnv):
    """Once a dot is owned it must never change owner."""
    env.reset(seed=3)
    prev_owner = env.dot_owner.copy()
    for _ in range(200):
        if not env.agents:
            break
        actions = {ag: env.action_space(ag).sample() for ag in env.agents}
        env.step(actions)
        for i, (old, new) in enumerate(zip(prev_owner, env.dot_owner)):
            if old >= 0:
                assert new == old, f"Dot {i} changed owner from {old} to {new}"
        prev_owner = env.dot_owner.copy()


def test_scores_match_dot_owner(env: MAQuadXDotRaceEnv):
    """scores[k] must equal the number of dots owned by uav_k."""
    env.reset(seed=4)
    for _ in range(200):
        if not env.agents:
            break
        actions = {ag: env.action_space(ag).sample() for ag in env.agents}
        env.step(actions)
        assert env.scores[0] == int(np.sum(env.dot_owner == 0))
        assert env.scores[1] == int(np.sum(env.dot_owner == 1))


# ── Episode termination ───────────────────────────────────────────────────────

def test_episode_ends_within_time_limit(env: MAQuadXDotRaceEnv):
    """Episode must end at or before max_steps."""
    env.reset(seed=5)
    steps = 0
    while env.agents:
        actions = {ag: env.action_space(ag).sample() for ag in env.agents}
        env.step(actions)
        steps += 1
    assert steps <= env.max_steps + 1


def test_two_agents_present_at_start(env: MAQuadXDotRaceEnv):
    env.reset(seed=6)
    assert len(env.agents) == 2


def test_possible_agents_are_uav_0_and_uav_1(env: MAQuadXDotRaceEnv):
    assert env.possible_agents == ["uav_0", "uav_1"]


# ── Seeding ───────────────────────────────────────────────────────────────────

def test_same_seed_same_dots():
    """Two resets with the same seed should produce identical dot layouts."""
    e1 = MAQuadXDotRaceEnv(num_dots=4, render_mode=None)
    e2 = MAQuadXDotRaceEnv(num_dots=4, render_mode=None)
    e1.reset(seed=99)
    e2.reset(seed=99)
    np.testing.assert_array_equal(e1.dots, e2.dots)
    e1.close()
    e2.close()


def test_different_seeds_different_dots():
    """Two resets with different seeds should (almost certainly) differ."""
    e = MAQuadXDotRaceEnv(num_dots=4, render_mode=None)
    e.reset(seed=1)
    dots_a = e.dots.copy()
    e.reset(seed=2)
    dots_b = e.dots.copy()
    assert not np.array_equal(dots_a, dots_b)
    e.close()
