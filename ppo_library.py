import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.vec_env import VecEnvWrapper
from environment import GroceryStoreEnv  # import your environment here

# Custom wrapper to flatten or resize observations to (8,)
class ObservationWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super(ObservationWrapper, self).__init__(venv)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    
    def reset(self):
        obs = self.venv.reset()
        return self.process_obs(obs)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return self.process_obs(obs), reward, done, info
    
    def process_obs(self, obs):
        # Flatten or reduce the original observation to a shape of (8,)
        return obs[:, :8]  # Taking the first 8 elements as a workaround

# Initialize the environment and wrap it
env = GroceryStoreEnv()
vec_env = DummyVecEnv([lambda: env])
wrapped_env = ObservationWrapper(vec_env)

# Train the PPO model
model = PPO('MlpPolicy', wrapped_env, verbose=1)
model.learn(total_timesteps=10000)
