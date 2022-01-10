import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
import Config
import Agent
from test_process import TestProcess

# CHECK IF WE NEED TO NULLIFY REWARD WHEN DONE, LAST BATCH WITHOUT DONEWILL BE BIASED!!!!!!!!!!!!!!!!!!!

env = gym.make(Config.ENV_NAME)
state = env.reset()
agent = Agent.Agent(state_size=state.shape[0], action_size=env.action_space.shape[0], batch_size=Config.BATCH_SIZE)
writer = None
if Config.WRITER_FLAG:
    writer = SummaryWriter(log_dir='content/runs/test-'+Config.WRITER_NAME)
test_process = TestProcess(state_size=state.shape[0], action_size=env.action_space.shape[0], writer=writer)
n_episodes = 0
ep_reward = 0
ep_reward_mean = [0]*100
for n_step in range(Config.NUMBER_OF_STEPS):
    if n_step % 50 == 0 and n_step > 0:
        test_process.test(agent.agent_control.policy_nn, n_step)
    # Collect samples
    for n_batch_step in range(Config.BATCH_SIZE):
        actions, actions_logprob = agent.get_action(state)
        new_state, reward, done, _ = env.step(actions)
        ep_reward += reward
        agent.add_to_memory(state, actions, actions_logprob, new_state, reward, done, n_batch_step)
        state = new_state
        if done:
            state = env.reset()
            ep_reward_mean[n_episodes % 100] = ep_reward
            n_episodes += 1
            ep_reward = 0
    agent.calculate_advantage()

    batch_indices = np.arange(Config.BATCH_SIZE)
    for _ in range(Config.UPDATE_STEPS):
        np.random.shuffle(batch_indices)
        for i in range(0, Config.BATCH_SIZE, Config.MINIBATCH_SIZE):
            agent.update(batch_indices[i: i + Config.MINIBATCH_SIZE])

    agent.policy_loss_mm[n_step % 100] = np.mean(agent.policy_loss_m)
    agent.critic_loss_mm[n_step % 100] = np.mean(agent.critic_loss_m)
    
    print("Step " + str(n_step) + "/" + str(Config.NUMBER_OF_STEPS) + " Mean 100 policy loss: " + str(
        np.mean(agent.policy_loss_mm[:min(n_step + 1, 100)])) + " Mean 100 critic loss: " + str(
        np.mean(agent.critic_loss_mm[:min(n_step + 1, 100)])) + " Mean 100 reward: " + str(np.round(np.mean(ep_reward_mean[:min(n_episodes, 100)]), 2)) + " Last reward: " + str(np.round(ep_reward_mean[(n_episodes - 1) % 100], 2)))
    if Config.WRITER_FLAG:
        writer.add_scalar('pg_loss', np.mean(agent.policy_loss_m), n_step)
        writer.add_scalar('vl_loss', np.mean(agent.critic_loss_m), n_step)
        writer.add_scalar('rew', ep_reward_mean[(n_episodes - 1) % 100], n_step)
        writer.add_scalar('100rew', np.mean(ep_reward_mean[:min(n_episodes, 100)]), n_step)
    agent.critic_loss_m = []
    agent.policy_loss_m = []

# PROMENI CONFIG NA KRAJU
# ENTORPY KOD NORMAL
# NORMALIZACIJA ADVANTAGE
print(n_episodes)

env.close()

