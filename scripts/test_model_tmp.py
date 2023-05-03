
import gym
import highway_env
import numpy as np
from rl_agents.agents.common.factory import load_agent, load_environment

#env = gym.make("intersection-v0")
environment_config = "configs/HighwayEnv/env_obs_attention.json"
agent_config = "configs/HighwayEnv/agents/DQNAgent/ego_attention.json"


env = load_environment(environment_config)
agent = load_agent(agent_config, env)
agent.load(".\out\HighwayEnv\DQNAgent\saved_models\latest.tar")

while True:
  terminated = truncated = False
  obs= env.reset()
  while not (terminated or truncated):
    action = agent.act(obs)
    action = int(action)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()