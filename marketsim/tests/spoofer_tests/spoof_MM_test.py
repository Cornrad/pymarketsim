
import gymnasium as gym
from marketsim.wrappers import MMSP_wrapper


def test_mmsp_env_is_available():
    assert hasattr(MMSP_wrapper, "MMSPEnv")
    assert issubclass(MMSP_wrapper.MMSPEnv, gym.Env)
