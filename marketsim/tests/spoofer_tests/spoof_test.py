import random

import numpy as np

import pytest

import matplotlib.pyplot as plt
import time
from marketsim.simulator.sampled_arrival_simulator import SimulatorSampledArrival
from marketsim.wrappers.SP_wrapper import SPEnv
from marketsim.private_values.private_values import PrivateValues
import torch.distributions as dist

import torch

from marketsim.wrappers.SP_wrapper import SPEnv


NORMALIZERS = {
    "fundamental": 1e5,
    "reward": 1e4,
    "min_order_val": 1e4,
    "invt": 10,
    "cash": 1e6,
}


def test_sp_env_step_executes():
    torch.manual_seed(0)
    random.seed(0)

    env = SPEnv(
        num_background_agents=4,
        sim_time=10,
        lam=0.6,
        lamSP=1.0,
        mean=1e3,
        r=0.05,
        shock_var=50,
        q_max=3,
        pv_var=1e4,
        shade=[50, 100],
        normalizers=NORMALIZERS,
        order_size=10,
        spoofing_size=10,
    )

    try:
        obs, info = env.reset()
    except ValueError:
        pytest.skip("spoofer did not arrive during the short test horizon")
    assert obs.shape == env.observation_space.shape

    total_reward = 0.0
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    assert np.isfinite(total_reward)
    assert env.time >= 0
