import torch
import numpy as np

class Policy:
    def pi(self, s_t):
        '''
        returns the probability distribution over actions
        (torch.distributions.Distribution)

        s_t (np.ndarray): the current state
        '''
        raise NotImplementedError

    def act(self, s_t):
        '''
        s_t (np.ndarray): the current state
        Because of environment vectorization, this will produce
        E actions where E is the number of parallel environments.
        '''
        pi = self.pi(s_t)
        a_t = pi.sample()
        log_prob = pi.log_prob(a_t).detach().numpy()
        return a_t, log_prob

    def learn(self, states, actions, rewards_or_advantages, log_probs_old=None, epsilon=0.2, update_method='PPO'):
        # Convert inputs to tensors
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards_or_advantages = torch.tensor(rewards_or_advantages, dtype=torch.float32)
        if log_probs_old is not None:
            log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)

        # Calculate new log probabilities
        log_probs = self.pi(states).log_prob(actions)

        if update_method == 'PPO':
            # PPO-specific calculations
            r_theta = torch.exp(log_probs - log_probs_old)
            clipped = torch.where(rewards_or_advantages > 0,
                                torch.min(r_theta, torch.tensor(1 + epsilon)),
                                torch.max(r_theta, torch.tensor(1 - epsilon)))
            loss = -torch.mean(clipped * rewards_or_advantages)
        elif update_method in ['A2C', 'PG']:
            # A2C and VPG share the same loss calculation,
            # but A2C uses advantages and VPG uses returns.
            loss = -torch.mean(log_probs * rewards_or_advantages)

        # Common steps for optimization
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Optional: Return additional information based on update method
        if update_method == 'PPO':
            approx_kl = (log_probs_old - log_probs).mean().item()  # Calculate approximate KL divergence
            return approx_kl
        else:
            return loss.item()
