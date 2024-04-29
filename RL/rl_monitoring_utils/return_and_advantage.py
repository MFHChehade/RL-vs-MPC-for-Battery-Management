import numpy as np

def calculate_returns(rewards, dones, gamma):
    result = np.empty_like(rewards)
    result[-1] = rewards[-1]
    for t in range(len(rewards)-2, -1, -1):
        result[t] = rewards[t] + gamma*(1-dones[t])*result[t+1]
    return result

def calculate_advantages(TD_errors, lam, gamma):
    result = np.empty_like(TD_errors)
    result[-1] = TD_errors[-1]
    for t in range(len(TD_errors)-2, -1, -1):
        result[t] = TD_errors[t] + gamma*lam*result[t+1]
    return result