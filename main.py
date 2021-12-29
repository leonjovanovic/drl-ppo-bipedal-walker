import gym
import Config
import Agent

env = gym.make(Config.ENV_NAME)
state = env.reset()
agent = Agent.Agent(state_size=state.shape[0], action_size=env.action_space.shape[0])
n_episodes = 0
for n_step in range(1, Config.NUMBER_OF_STEPS):
    action = agent.get_action(state)
    env.render()
    new_state, reward, done, _ = env.step(action)
    agent.add_to_memory(state, action, new_state, reward)
    if done or n_step % Config.TRAJECTORY_LENGTH == 0:
        agent.update()
        agent.reset()
        if done:
            state = env.reset()
            n_episodes += 1
print(n_episodes)

env.close()