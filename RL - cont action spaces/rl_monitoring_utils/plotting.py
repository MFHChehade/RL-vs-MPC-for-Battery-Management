import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import seaborn as sns

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()

def plot_results(log_folders, title="Learning Curves"):
    """
    Plot the results for multiple log folders on the same plot using Seaborn.

    :param log_folders: (list of str) List of save locations of the results to plot
    :param title: (str) The title of the plot
    """
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(12, 8))
    max_steps = 0
    all_interpolated_results = []

    for log_folder in log_folders:
        x, y = ts2xy(load_results(log_folder), "timesteps")
        y = moving_average(y, window=1000)
        x = x[len(x) - len(y):]

        max_steps = max(max_steps, x[-1])

        # Interpolation
        common_x = np.linspace(0, x[-1], num=1000)
        interpolated_y = np.interp(common_x, x, y)
        all_interpolated_results.append((common_x, interpolated_y))

    for i, (x, y) in enumerate(all_interpolated_results):
        sns.lineplot(x=x, y=y, label=log_folders[i].split('\\')[-1], linewidth=2)

    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.legend()
    plt.show()