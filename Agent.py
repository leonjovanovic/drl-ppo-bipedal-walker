import AgentControl
import Memory


class Agent:

    def __init__(self, state_size, action_size):
        self.agent_control = AgentControl.AgentControl(state_size=state_size, action_size=action_size)
        self.memory = Memory.Memory() #mozda u numpy ili odmah u tensor, na osnovu TRAJECTORY_LENGTH velicina

    def get_action(self, state):
        action = self.agent_control.get_action(state)
        return action.cpu().detach().numpy()

    def add_to_memory(self, state, action, new_state, reward):
        self.memory.add(state, action, new_state, reward)

    def update(self):
        while len(self.memory.ep_state) != 0:
            state, action, new_state, reward = self.memory.get()
            advantage = self.agent_control.advantage(state, new_state, reward)
            ratio = self.agent_control.ratio(state, action)

    def reset(self):
        self.memory.reset()
        self.agent_control.sync_nns()



