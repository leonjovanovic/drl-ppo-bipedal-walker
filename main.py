import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
import Config
import Agent
from test_process import TestProcess

# CHECK IF WE NEED TO NULLIFY REWARD WHEN DONE, LAST BATCH WITHOUT DONEWILL BE BIASED!!!!!!!!!!!!!!!!!!!

env = gym.make(Config.ENV_NAME)
if Config.ENV_SCALE_CROP:
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env)
    #env = gym.wrappers.TransformReward(env, lambda rew: np.clip(rew, -10, 10))
state = env.reset()
agent = Agent.Agent(state_size=state.shape[0], action_size=env.action_space.shape[0], batch_size=Config.BATCH_SIZE)
writer = SummaryWriter(log_dir='content/runs/test-'+Config.WRITER_NAME) if Config.WRITER_FLAG else None
test_process = TestProcess(state_size=state.shape[0], action_size=env.action_space.shape[0], writer=writer)
n_episodes = 0
ep_reward = 0
for n_step in range(Config.NUMBER_OF_STEPS):
    # Anneal lr
    agent.set_optimizer_lr(n_step)
    # Test the model after 50 steps
    #if n_step % 50 == 0 and n_step > 0:
    #    state = test_process.test(agent.agent_control.policy_nn, n_step, writer, env)
    # Collect batch_size number of samples
    for n_batch_step in range(Config.BATCH_SIZE):
        if n_step % 50 == 0 and n_step > 0:
            env.render()
        actions, actions_logprob = agent.get_action(state)
        new_state, reward, done, _ = env.step(actions)
        ep_reward += reward * np.sqrt(env.return_rms.var + env.epsilon)
        reward = np.clip(reward, -10, 10)# ----------------------------------------------------------------------------------------
        agent.add_to_memory(state, actions, actions_logprob, new_state, reward, done, n_batch_step)
        state = new_state
        if done:
            state = env.reset()
            agent.ep_reward_mean[n_episodes % 100] = ep_reward
            n_episodes += 1
            ep_reward = 0
    #
    agent.calculate_old_value_state()
    # Calculate advantage
    agent.calculate_advantage()

    batch_indices = np.arange(Config.BATCH_SIZE)
    for _ in range(Config.UPDATE_STEPS):
        np.random.shuffle(batch_indices)
        for i in range(0, Config.BATCH_SIZE, Config.MINIBATCH_SIZE):
            agent.update(batch_indices[i: i + Config.MINIBATCH_SIZE])

    agent.record_results(n_step, writer, n_episodes)

# PROMENI CONFIG NA KRAJU
# ENTORPY KOD NORMAL
#test_process.env.close()
env.close()

#tensorboard --logdir="D:\Users\Leon Jovanovic\Documents\Computer Science\Reinforcement Learning\drl-trpo-ppo-bipedal-walker\content\runs" --host=127.0.0.1
