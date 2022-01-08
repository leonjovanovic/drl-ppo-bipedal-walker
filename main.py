import gym
import numpy as np

import Config
import Agent

# CHECK IF WE NEED TO NULLIFY REWARD WHEN DONE, LAST BATCH WITHOUT DONEWILL BE BIASED!!!!!!!!!!!!!!!!!!!
env = gym.make(Config.ENV_NAME)
state = env.reset()
agent = Agent.Agent(state_size=state.shape[0], action_size=env.action_space.shape[0], batch_size=Config.BATCH_SIZE)
n_episodes = 0
ep_reward = 0
ep_reward_mean = []
for n_step in range(Config.NUMBER_OF_STEPS):
    # Collect samples
    for n_batch_step in range(Config.BATCH_SIZE):
        actions, actions_logprob = agent.get_action(state)
        # env.render()
        new_state, reward, done, _ = env.step(actions)
        ep_reward += reward
        # print(reward)
        agent.add_to_memory(state, actions, actions_logprob, new_state, reward, done, n_batch_step)
        state = new_state
        if done:
            state = env.reset()
            n_episodes += 1
            ep_reward_mean.append(ep_reward)
            ep_reward = 0

    agent.calculate_advantage()
    batch_indices = np.arange(Config.BATCH_SIZE)
    for _ in range(Config.UPDATE_STEPS):
        np.random.shuffle(batch_indices)
        for i in range(0, Config.BATCH_SIZE, Config.MINIBATCH_SIZE):
            agent.update(batch_indices[i: i + Config.MINIBATCH_SIZE])
    agent.policy_loss_mm.append(np.mean(agent.policy_loss_m))
    agent.critic_loss_mm.append(np.mean(agent.critic_loss_m))
    print("Step " + str(n_step) + "/4000 Mean 100 policy loss: " + str(
        np.mean(agent.policy_loss_mm[-100:])) + " Mean 100 critic loss: " + str(
        np.mean(agent.critic_loss_mm[-100:])) + " Mean 100 reward: " + str(np.round(np.mean(ep_reward_mean[-100:]), 2)) + " Last reward: " + str(np.round(ep_reward_mean[-1], 2)))
    agent.critic_loss_m = []
    agent.policy_loss_m = []
# PROMENI CONFIG NA KRAJU
# ENTORPY KOD NORMAL
# NORMALIZACIJA ADVANTAGE
print(n_episodes)

env.close()
