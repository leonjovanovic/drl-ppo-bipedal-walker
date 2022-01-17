import Config
import NN
import torch

class AgentControl:

    def __init__(self, state_size, action_size):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_nn = NN.PolicyNN(input_shape=state_size, output_shape=action_size).to(self.device)
        self.critic_nn = NN.CriticNN(input_shape=state_size).to(self.device)
        self.optimizer_policy = torch.optim.Adam(params=self.policy_nn.parameters(), lr=Config.LEARNING_RATE_POLICY, eps=Config.EPSILON_START)
        self.optimizer_critic = torch.optim.Adam(params=self.critic_nn.parameters(), lr=Config.LEARNING_RATE_CRITIC, eps=Config.EPSILON_START)
        self.loss_critic = torch.nn.MSELoss()

    def set_optimizer_lr_eps(self, n_step):
        # Annealing the rate if instructed to do so.
        if Config.ANNEAL_LR:
            frac = 1.0 - n_step / Config.NUMBER_OF_STEPS
            lr_policy = frac * Config.LEARNING_RATE_POLICY
            lr_critic = frac * Config.LEARNING_RATE_CRITIC
            self.optimizer_policy.param_groups[0]["lr"] = lr_policy
            self.optimizer_critic.param_groups[0]["lr"] = lr_critic

        self.optimizer_critic.param_groups[0]["eps"] = Config.EPSILON_START - n_step * (Config.EPSILON_START - Config.EPSILON_END) / Config.NUMBER_OF_STEPS

    def get_action(self, state):
        actions, actions_logprob, _ = self.policy_nn(torch.tensor(state, dtype=torch.float, device=self.device))
        return actions, actions_logprob

    def get_critic_value(self, state):
        return self.critic_nn(state)

    def calculate_logprob(self, states, actions):
        _, new_actions_logprob, entropy = self.policy_nn(states, actions)
        return new_actions_logprob, entropy

    def calculate_ratio(self, new_action_logprob, action_logprobs):
        # Since probabilites are log probabilities and we have to calculate ratio, we can simply subtract logs of new
        # and old. Before that, because we have 4 actions and not 1, we have to calculate joint probability which will
        # represent single action in R^4. Since actions are independent, the joint probability will be a product of
        # the probabilites. Product of probs is equal to sum of log probs. At the end we need to e^log prob to get prob.
        return torch.exp(torch.sum(new_action_logprob, dim=1) - torch.sum(action_logprobs, dim=1).detach())

    def update_policy(self, advantages, ratios, entropy):
        # Normalize the advantages to reduce variance
        advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss = torch.minimum(ratios * advantages_norm, torch.clamp(ratios, 1-Config.CLIPPING_EPSILON, 1+Config.CLIPPING_EPSILON) * advantages_norm)
        policy_loss = -policy_loss.mean()
        policy_loss -= entropy.mean() * Config.ENTROPY_COEF

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        # The gradients of the policy network are clipped so that the “global l2 norm” (i.e. the norm of the
        # concatenated gradients of all parameters) does not exceed 0.5
        torch.nn.utils.clip_grad_norm_(self.policy_nn.parameters(), Config.MAX_GRAD_NORM)
        self.optimizer_policy.step()
        return policy_loss

    def update_critic(self, gt, states, old_value_state):
        # We need to calculate regular loss where we just find squared difference between estimated value and return
        estimated_value = self.critic_nn(states).squeeze(-1)
        critic_loss1 = torch.square(estimated_value - gt)
        # and we need to calculate clipped loss where estimated value is replaced with old estimated value + clipped difference
        estimated_value_clipped = old_value_state + torch.clamp(self.critic_nn(states).squeeze(-1) - old_value_state, - Config.CLIPPING_EPSILON, Config.CLIPPING_EPSILON)
        critic_loss2 = torch.square(estimated_value_clipped - gt)
        # Compare two losses and take bigger and calculate mean to get final critic loss
        critic_loss = 0.5 * (torch.maximum(critic_loss1, critic_loss2)).mean()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        # The gradients of the value (critic) network are clipped so that the “global l2 norm” (i.e. the norm of the
        # concatenated gradients of all parameters) does not exceed 0.5
        torch.nn.utils.clip_grad_norm_(self.critic_nn.parameters(), Config.MAX_GRAD_NORM)
        self.optimizer_critic.step()
        return critic_loss


