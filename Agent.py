import AgentControl
import Config
import Memory
import numpy as np
import itertools

class Agent:
    # Role of Agent class is to coordinate between AgentControll where we do all calculations
    # and Memory where we store all of the data
    def __init__(self, state_size, action_size, batch_size):
        self.agent_control = AgentControl.AgentControl(state_size=state_size, action_size=action_size)
        self.memory = Memory.Memory(state_size, action_size, batch_size)
        self.policy_loss_m = []
        self.critic_loss_m = []
        self.policy_loss_mm = [0] * 100
        self.critic_loss_mm = [0] * 100
        self.max_reward = -300
        self.ep_count = 0

    def set_optimizer_lr_eps(self, n_step):
        self.agent_control.set_optimizer_lr_eps(n_step)

    def get_action(self, state):
        actions, actions_logprob = self.agent_control.get_action(state)
        return actions.cpu().detach().numpy(), actions_logprob

    def add_to_memory(self, state, action, actions_logprob, new_state, reward, done, n_batch_step):
        self.memory.add(state, action, actions_logprob, new_state, reward, done, n_batch_step)

    def calculate_old_value_state(self):
        # Get NN output from collected states and pass it to the memory
        self.memory.set_old_value_state(self.agent_control.get_critic_value(self.memory.states).squeeze(-1).detach())

    def calculate_advantage(self):
        # For basic advantage function we have to calculate future rewards we got from each state, where reward from
        # last state is estimation (since we only know rewards in steps we took, not after), discount them and
        # subtract from baseline which in this case will be estimated value of each state.
        # GAE advantage gives us to decide we want each state advantage to be calculated with
        # reward + estimate(next state) - estimate(state) which has low variance but high bias or with
        # reward + gamma*next_reward + ... + gamma^n * estimate(last next state) - estimate(state) which has high
        # variance but low bias. We can decide to calculate advantage with somethig between those two and Config.LAMBDA
        # will be hyperparameter for that
        values = self.agent_control.get_critic_value(self.memory.states).squeeze(-1).detach()
        if Config.GAE:
            next_values = self.agent_control.get_critic_value(self.memory.new_states).squeeze(-1).detach()
            self.memory.calculate_gae_advantage(values, next_values)
        else:
            next_value = self.agent_control.get_critic_value(self.memory.new_states[-1]).squeeze(-1).detach()
            self.memory.calculate_advantage(next_value, values)

    def update(self, indices):
        # Main PPO point is updating policy NN. This is done by calculating derivative of loss function and doing
        # gradient descent. First we have to calculate ratio. Second to find minimum between ratio*advantage and
        # clipped_ratio*advantage. Third to find mean of Config.MINIBATCH_SIZE losses.

        # To calculate ratio we need new and old action probability. We already have old when we fed states to
        # policy NN when we wanted to get action from it. We can get new action probabilities if we give same states
        # but also actions we got. With states NN can create Normal distribution and with action he will sample the same
        # part of distribution, but now with different probability because Normal distribution is different.
        new_action_logprob, entropy = self.agent_control.calculate_logprob(self.memory.states[indices], self.memory.actions[indices])
        ratios = self.agent_control.calculate_ratio(new_action_logprob, self.memory.action_logprobs[indices])
        policy_loss = self.agent_control.update_policy(self.memory.advantages[indices], ratios, entropy)
        # Similar to ratio in policy loss, we also clipped values from critic. For that we need old_value_state which
        # represent old estimate of states before updates.
        critic_loss = self.agent_control.update_critic(self.memory.gt[indices], self.memory.states[indices], self.memory.old_value_state[indices])

        # Calculating mean losses for statistics
        self.policy_loss_m.append(policy_loss.detach().item())
        self.critic_loss_m.append(critic_loss.detach().item())

    def record_results(self, n_step, writer, env):
        self.max_reward = np.maximum(self.max_reward, np.max(env.return_queue))
        self.policy_loss_mm[n_step % 100] = np.mean(self.policy_loss_m)
        self.critic_loss_mm[n_step % 100] = np.mean(self.critic_loss_m)
        print("Step " + str(n_step) + "/" + str(Config.NUMBER_OF_STEPS) + " Mean 100 policy loss: " + str(
            np.round(np.mean(self.policy_loss_mm[:min(n_step + 1, 100)]), 4)) + " Mean 100 critic loss: " + str(
            np.round(np.mean(self.critic_loss_mm[:min(n_step + 1, 100)]), 4)) + " Max reward: " + str(
            np.round(self.max_reward, 2)) + " Mean 100 reward: " + str(
            np.round(np.mean(env.return_queue), 2)) + " Last rewards: " + str(
            np.round(list(itertools.islice(env.return_queue, min(env.episode_count, 100)-(env.episode_count-self.ep_count), min(env.episode_count, 100))), 2)) + " Ep" + str(env.episode_count))

        if Config.WRITER_FLAG:
            writer.add_scalar('pg_loss', np.mean(self.policy_loss_m), n_step)
            writer.add_scalar('vl_loss', np.mean(self.critic_loss_m), n_step)
            writer.add_scalar('rew', env.return_queue[-1], n_step)
            writer.add_scalar('100rew', np.mean(env.return_queue), n_step)
        self.critic_loss_m = []
        self.policy_loss_m = []
        self.ep_count = env.episode_count



