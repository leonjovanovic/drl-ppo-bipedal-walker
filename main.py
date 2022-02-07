import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
import Config
import Agent
from test_process import TestProcess
import itertools

# --------------------------------------------------- Initialization ---------------------------------------------------
# Create Bipedal Walker enviroment and add wrappers to record statistics and clip action if its extreme
env = gym.make(Config.ENV_NAME)
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ClipAction(env)
# If we want to normalize and clip state and scale and clip reward
if Config.ENV_SCALE_CROP:
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda rew: np.clip(rew, -10, 10))
env = gym.wrappers.RecordVideo(env, "recordings", name_prefix="rl-video" + Config.date_time,)
state = env.reset()
# Create agent which will use PPO to train NNs
agent = Agent.Agent(state_size=state.shape[0], action_size=env.action_space.shape[0], batch_size=Config.BATCH_SIZE)
# Create writer for Tensorboard
writer = SummaryWriter(log_dir='content/runs/'+Config.WRITER_NAME) if Config.WRITER_FLAG else None
print(Config.WRITER_NAME)
# Initialize test process which will be occasionally called to test whether goal is met
test_process = TestProcess(state_size=state.shape[0], action_size=env.action_space.shape[0], writer=writer)
# ------------------------------------------------------ Training ------------------------------------------------------
for n_step in range(Config.NUMBER_OF_STEPS):
    # Implement learning rate and epsilon decay for Adam optimizer for both NNs
    agent.set_optimizer_lr_eps(n_step)
    # Test the model after 50 steps
    if (n_step + 1) % 50 == 0 or (len(env.return_queue) >= 100 and np.mean(list(itertools.islice(env.return_queue, 90, 100))) >= 300):
        end_train = test_process.test(agent.agent_control.policy_nn, n_step, writer, env)
        if end_train:
            break
    # Collect batch_size number of samples
    for n_batch_step in range(Config.BATCH_SIZE):
        if (n_step + 1) % 50 == 0:
            env.render()
        # Feed current state to the policy NN and get action and its probability
        actions, actions_logprob = agent.get_action(state)
        # Use given action and retrieve new state, reward agent recieved and whether episode is finished flag
        new_state, reward, done, _ = env.step(actions)
        # Store step information to memory for future use
        agent.add_to_memory(state, actions, actions_logprob, new_state, reward, done, n_batch_step)
        state = new_state
        if done:
            state = env.reset()

    # For value (critic) function clipping, we need NN output before update, which we will use as baseline to see
    # how much new output is different and to clip it if its too much different
    agent.calculate_old_value_state()
    # Calculate advantage for policy NN loss
    agent.calculate_advantage()

    # Instead of shuffling whole memory, we will create indices and shuffle them after each update
    batch_indices = np.arange(Config.BATCH_SIZE)
    # We will use every collected step to update NNs Config.UPDATE_STEPS times
    for _ in range(Config.UPDATE_STEPS):
        np.random.shuffle(batch_indices)
        # Split the memory to mini-batches and use them to update NNs
        for i in range(0, Config.BATCH_SIZE, Config.MINIBATCH_SIZE):
            agent.update(batch_indices[i: i + Config.MINIBATCH_SIZE])

    # Record losses and rewards and print them to console and SummaryWriter for nice Tensorboard graphs
    agent.record_results(n_step, writer, env)
if writer is not None:
    writer.close()
test_process.env.close()
env.close()

#tensorboard --logdir="PATH_TO_FOLDER\drl-trpo-ppo-bipedal-walker\content\runs" --host=127.0.0.1
