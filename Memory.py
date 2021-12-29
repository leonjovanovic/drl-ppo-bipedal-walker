import Config

class Memory:

    def __init__(self):
        self.ep_state = []
        self.ep_action = []
        self.ep_action_prob = []
        self.ep_new_state = []
        self.ep_reward = []

    def add(self, state, action, new_state, reward):
        self.ep_state.append(state)
        self.ep_action.append(action)
        self.ep_new_state.append(new_state)
        self.ep_reward.append(reward)

    def get(self):
        if len(self.ep_state) == 0:
            return
        size = Config.MINIBATCH_SIZE
        if len(self.ep_state) < Config.MINIBATCH_SIZE:
            size = len(self.ep_state)
        l1 = self.ep_state[:size]
        l2 = self.ep_action[:size]
        l3 = self.ep_new_state[:size]
        l4 = self.ep_reward[:size]

        del self.ep_state[:size]
        del self.ep_action[:size]
        del self.ep_new_state[:size]
        del self.ep_reward[:size]
        return l1, l2, l3, l4

    def reset(self):
        self.ep_state = []
        self.ep_action = []
        self.ep_action_prob = []
        self.ep_new_state = []
        self.ep_reward = []
