import AgentControl
import Memory
import numpy as np
import sys

class Agent:

    def __init__(self, state_size, action_size, batch_size):
        self.agent_control = AgentControl.AgentControl(state_size=state_size, action_size=action_size)
        self.memory = Memory.Memory(state_size, action_size, batch_size)
        self.policy_loss_m = []
        self.critic_loss_m = []
        self.policy_loss_mm = [0] * 100
        self.critic_loss_mm = [0] * 100

    def get_action(self, state):
        actions, actions_logprob = self.agent_control.get_action(state)
        return actions.cpu().detach().numpy(), actions_logprob

    def add_to_memory(self, state, action, actions_logprob, new_state, reward, done, n_batch_step):
        self.memory.add(state, action, actions_logprob, new_state, reward, done, n_batch_step)

    def calculate_advantage(self):
        next_value = self.agent_control.get_critic_value(self.memory.new_states[-1]).squeeze(-1).detach()
        values = self.agent_control.get_critic_value(self.memory.states).squeeze(-1).detach()
        self.memory.calculate_advantage(next_value, values)

    def update(self, indices):
        # ratio = new/old log prob
        # za isto stanje ubacimo u policy NN da bi dobili novu raspodelu a verovatnocu dobijamo
        # kada trazimo ver za istu akciju u novoj raspodelu
        new_action_logprob = self.agent_control.calculate_logprob(self.memory.states[indices], self.memory.actions[indices])
        ratios = self.agent_control.calculate_ratio(new_action_logprob, self.memory.action_logprobs[indices])
        policy_loss = self.agent_control.update_policy(self.memory.advantages[indices], ratios)
        critic_loss = self.agent_control.update_critic(self.memory.gt[indices], self.memory.states[indices])

        if np.isnan(policy_loss.item()):
            print("ERROR! advantage: " + str(self.memory.advantages[indices]) + " ratios: " + str(ratios))
            print("ERROR! new_action_logprob: " + str(new_action_logprob) + " action_logprobs: " + str(self.memory.action_logprobs[indices]))
            print("ERROR! actions: " + str(self.memory.actions[indices]))
            print("ERROR! Parameters: " + str(self.agent_control.policy_nn.parameters()))
            sys.exit()
        self.policy_loss_m.append(policy_loss.detach().item())
        self.critic_loss_m.append(critic_loss.detach().item())




