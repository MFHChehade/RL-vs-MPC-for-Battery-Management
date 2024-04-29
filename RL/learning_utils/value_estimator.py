import torch

class ValueEstimator:
    def __init__(self, env, lr=1e-2, hidden_sizes=[64]):
        self.N = env.observation_space.shape[0]
        self.hidden_sizes = hidden_sizes
        
        # Create a list of layers starting with the input layer
        layers = []
        input_dim = self.N
        
        for hidden_size in self.hidden_sizes:
            layers.append(torch.nn.Linear(input_dim, hidden_size))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_size  # Set the input dimension for the next layer
            
        # Add the output layer (single output neuron)
        layers.append(torch.nn.Linear(input_dim, 1))
        
        # Compile the sequential model
        self.V = torch.nn.Sequential(*layers).double()
        
        # Initialize the output layer weights and biases to zero
        with torch.no_grad():
            self.V[-1].weight.fill_(0)
            self.V[-1].bias.fill_(0)

        # Optimizer for learning
        self.opt = torch.optim.Adam(self.V.parameters(), lr=lr)

    def predict(self, s_t):
        s_t = torch.tensor(s_t, dtype=torch.double)
        return self.V(s_t).squeeze()

    def learn(self, V_pred, returns):
        returns = torch.tensor(returns, dtype=torch.double)
        loss = torch.mean((V_pred - returns) ** 2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss
