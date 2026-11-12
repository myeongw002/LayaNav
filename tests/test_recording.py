"""Deterministic saved/replayed episodes are verified against observations."""

import copy

import numpy as np
import pytest

from layanav.envs import GridConfig, GridWorldEnv
from layanav.recording import EpisodeRecorder, ReplaySession


def make_recording():
    n = 9
    prior = np.zeros((n, n), dtype=bool)
    prior[0, :] = prior[-1, :] = True
    prior[:, 0] = prior[:, -1] = True
    hidden = np.zeros_like(prior)
    hidden[4, 4] = True
    env = GridWorldEnv(
        GridConfig(size=n, sensor_mode="raycast", observation_radius=2, max_steps=30)
    )
    env.reset(options={
        "static_map": prior,
        "hidden_obstacles": hidden,
        "start": (2, 4),
        "goal": (6, 4),
    })
    recorder = EpisodeRecorder(env)
    observations = []
    for action in (3, 3, 0, 3, 3, 1, 3):
        obs, reward, terminated, truncated, info = env.step(action)
        recorder.record_step(action, obs, reward, terminated, truncated, info)
        observations.append(obs)
        if terminated or truncated:
            break
    return env, recorder, observations


def test_episode_roundtrip_verified(tmp_path):
    env, recorder, originals = make_recording()
    file = tmp_path / "run.json"
    recorder.save(file)
    replay = ReplaySession.load(file)
    assert replay.env.config.sensor_mode == "raycast"
    assert replay.env.debug_snapshot()["step"] == 0

    for original in originals:
        obs, _, _, _, _ = replay.step()
        np.testing.assert_array_equal(obs, original)
    assert replay.finished
    with pytest.raises(RuntimeError):
        replay.step()
    assert replay.env.debug_snapshot()["robot"] == env.debug_snapshot()["robot"]


def test_replay_detects_reward_corruption():
    _, recorder, _ = make_recording()
    trace = copy.deepcopy(recorder.to_dict())
    trace["steps"][0]["reward"] += 1.0
    replay = ReplaySession(trace)
    with pytest.raises(ValueError, match="diverged"):
        replay.step()


def test_replay_detects_belief_observation_corruption():
    _, recorder, _ = make_recording()
    trace = copy.deepcopy(recorder.to_dict())
    trace["steps"][0]["observation_sha256"] = "0" * 64
    replay = ReplaySession(trace)
    with pytest.raises(ValueError, match="diverged"):
        replay.step()


def test_recording_requires_fresh_reset():
    env, _, _ = make_recording()
    with pytest.raises(ValueError):
        EpisodeRecorder(env)


def test_replay_invalid_trace_version():
    _, recorder, _ = make_recording()
    trace = recorder.to_dict()
    trace["schema_version"] = 99
    with pytest.raises(ValueError, match="Unsupported"):
        ReplaySession(trace)


def test_eight_way_replay_and_legacy_mode(tmp_path):
    import json

    n = 9
    prior = np.zeros((n, n), dtype=bool)
    hidden = np.zeros_like(prior)
    options = {
        "static_map": prior, "hidden_obstacles": hidden,
        "start": (1, 1), "goal": (7, 7),
    }
    env = GridWorldEnv(
        GridConfig(size=n, movement_mode=8, observation_radius=2, max_steps=20)
    )
    env.reset(options=options)
    recorder = EpisodeRecorder(env)
    obs, reward, done, truncated, info = env.step(8)
    recorder.record_step(8, obs, reward, done, truncated, info)
    dest = tmp_path / "eight.json"
    recorder.save(dest)
    loaded = ReplaySession.load(dest)
    assert loaded.env.action_space.n == 9
    loaded.step()
    assert loaded.env.debug_snapshot()["robot"] == (2, 2)

    tampered = json.loads(dest.read_text(encoding="utf-8"))
    tampered["config"]["movement_mode"] = 4
    with pytest.raises(ValueError, match="Invalid trace action"):
        ReplaySession(tampered).step()

    old = recorder.to_dict()
    old["config"].pop("movement_mode")
    old["steps"].clear()
    old["initial_observation_sha256"] = recorder.initial_digest
    assert ReplaySession(old).env.config.movement_mode == 4
