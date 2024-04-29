from collections import deque
import random

class ReplayBuffer:
    def __init__(self, batch_size=32, size=1000000):
        '''
        batch_size (int): number of data points per batch
        size (int): size of replay buffer.
        '''
        self.batch_size = batch_size
        self.memory = deque(maxlen=size)

    def remember(self, s_t, a_t, r_t, s_t_next, d_t):
        '''
        s_t (np.ndarray double): state
        a_t (np.ndarray int): action
        r_t (np.ndarray double): reward
        d_t (np.ndarray float): done flag
        s_t_next (np.ndarray double): next state
        '''
        self.memory.append((s_t, a_t, r_t, s_t_next, d_t))

    def sample(self):
        '''
        random sampling of data from buffer
        '''
        # if we don't have enough samples yet
        size = min(self.batch_size, len(self.memory))
        return random.sample(self.memory, size)