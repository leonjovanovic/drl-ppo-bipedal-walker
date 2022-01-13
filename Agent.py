import AgentControl
import Config
import Memory
import numpy as np

class Agent:

    def __init__(self, state_size, action_size, batch_size):
        self.agent_control = AgentControl.AgentControl(state_size=state_size, action_size=action_size)
        self.memory = Memory.Memory(state_size, action_size, batch_size)
        self.policy_loss_m = []
        self.critic_loss_m = []
        self.policy_loss_mm = [0] * 100
        self.critic_loss_mm = [0] * 100
        self.ep_reward_mean = [0]*100

    def set_optimizer_lr(self, n_step):
        self.agent_control.set_optimizer_lr(n_step)

    def get_action(self, state):
        actions, actions_logprob = self.agent_control.get_action(state)
        return actions.cpu().detach().numpy(), actions_logprob

    def add_to_memory(self, state, action, actions_logprob, new_state, reward, done, n_batch_step):
        self.memory.add(state, action, actions_logprob, new_state, reward, done, n_batch_step)

    def calculate_old_value_state(self):
        self.memory.set_old_value_state(self.agent_control.get_critic_value(self.memory.states).squeeze(-1).detach())

    def calculate_advantage(self):
        values = self.agent_control.get_critic_value(self.memory.states).squeeze(-1).detach()
        if Config.GAE:
            next_values = self.agent_control.get_critic_value(self.memory.new_states).squeeze(-1).detach()
            self.memory.calculate_gae_advantage(values, next_values)
        else:
            next_value = self.agent_control.get_critic_value(self.memory.new_states[-1]).squeeze(-1).detach()
            self.memory.calculate_advantage(next_value, values)

    def update(self, indices):
        # ratio = new/old log prob
        # za isto stanje ubacimo u policy NN da bi dobili novu raspodelu a verovatnocu dobijamo
        # kada trazimo ver za istu akciju u novoj raspodelu
        new_action_logprob, entropy = self.agent_control.calculate_logprob(self.memory.states[indices], self.memory.actions[indices])
        ratios = self.agent_control.calculate_ratio(new_action_logprob, self.memory.action_logprobs[indices])
        policy_loss = self.agent_control.update_policy(self.memory.advantages[indices], ratios, entropy)
        critic_loss = self.agent_control.update_critic(self.memory.gt[indices], self.memory.states[indices], self.memory.old_value_state[indices])

        self.policy_loss_m.append(policy_loss.detach().item())
        self.critic_loss_m.append(critic_loss.detach().item())

    def record_results(self, n_step, writer, n_episodes):
        self.policy_loss_mm[n_step % 100] = np.mean(self.policy_loss_m)
        self.critic_loss_mm[n_step % 100] = np.mean(self.critic_loss_m)
        print("Step " + str(n_step) + "/" + str(Config.NUMBER_OF_STEPS) + " Mean 100 policy loss: " + str(
            np.mean(self.policy_loss_mm[:min(n_step + 1, 100)])) + " Mean 100 critic loss: " + str(
            np.mean(self.critic_loss_mm[:min(n_step + 1, 100)])) + " Mean 100 reward: " + str(
            np.round(np.mean(self.ep_reward_mean[:min(n_episodes, 100)]), 2)) + " Last reward: " + str(
            np.round(self.ep_reward_mean[(n_episodes - 1) % 100], 2)))
        if Config.WRITER_FLAG:
            writer.add_scalar('pg_loss', np.mean(self.policy_loss_m), n_step)
            writer.add_scalar('vl_loss', np.mean(self.critic_loss_m), n_step)
            writer.add_scalar('rew', self.ep_reward_mean[(n_episodes - 1) % 100], n_step)
            writer.add_scalar('100rew', np.mean(self.ep_reward_mean[:min(n_episodes, 100)]), n_step)
        self.critic_loss_m = []
        self.policy_loss_m = []





