import torch
from .policy import Policy

class CategoricalPolicy(Policy):
    def __init__(self, env, lr=1e-2, hidden_sizes=[16], activation=torch.nn.ReLU):
        '''
        env (gym.Env): the environment
        lr (float): learning rate
        hidden_sizes (list of int): sizes of hidden layers
        activation (callable): activation function to use in the neural network, default is torch.nn.ReLU
        '''
        self.N = env.observation_space.shape[0]
        self.M = env.action_space.n

        # Construct the layers of the MLP
        layers = []
        input_size = self.N
        for size in hidden_sizes:
            layers.append(torch.nn.Linear(input_size, size))
            layers.append(activation())  # Dynamically use the specified activation function
            input_size = size
        # Output layer
        layers.append(torch.nn.Linear(input_size, self.M))

        self.p = torch.nn.Sequential(*layers).double()

        # Initialize weights and biases to 0 for the output layer
        with torch.no_grad():
            self.p[-1].weight.fill_(0)
            self.p[-1].bias.fill_(0)

        self.opt = torch.optim.Adam(self.p.parameters(), lr=lr)

    def pi(self, s_t):
        '''
        returns the probability distribution over actions
        s_t (np.ndarray): the current state
        '''
        s_t = torch.as_tensor(s_t).double()
        logits = self.p(s_t)
        pi = torch.distributions.Categorical(logits=logits)
        return pi
