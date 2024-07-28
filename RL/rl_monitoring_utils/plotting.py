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


# def plot_results(log_folder, title="Learning Curve"):
#     """
#     plot the results

#     :param log_folder: (str) the save location of the results to plot
#     :param title: (str) the title of the task to plot
#     """
#     x, y = ts2xy(load_results(log_folder), "timesteps")
#     y = moving_average(y, window=50)
#     # Truncate x
#     x = x[len(x) - len(y) :]

#     fig = plt.figure(title)
#     plt.plot(x, y)
#     plt.xlabel("Number of Timesteps")
#     plt.ylabel("Rewards")
#     plt.title(title + " Smoothed")
#     plt.show()

def plot_results(log_folders, title="Learning Curves", window=10):
    """
    Plot the mean rewards and the standard deviation across multiple runs from separate directories.
    """
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(12, 8))
    
    all_data = []
    max_steps = 0

    for log_folder in log_folders:
        x, y = ts2xy(load_results(log_folder), "timesteps")
        max_steps = max(max_steps, x[-1])
        all_data.append((x, y))
    
    common_x = np.linspace(0, max_steps, num=1000)
    interpolated_ys = []

    for x, y in all_data:
        y_smooth = moving_average(y, window)
        x_truncated = x[len(x) - len(y_smooth):]
        interpolated_y = np.interp(common_x, x_truncated, y_smooth)
        interpolated_ys.append(interpolated_y)

    mean_rewards = np.mean(interpolated_ys, axis=0)
    std_rewards = np.std(interpolated_ys, axis=0)

    plt.fill_between(common_x, mean_rewards - std_rewards, mean_rewards + std_rewards, color='gray', alpha=0.3)
    plt.plot(common_x, mean_rewards, label="Mean Rewards", color="blue")
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.legend()
    plt.show()

# def plot_results(log_folders, title="Learning Curves"):
#     """
#     Plot the results for multiple log folders on the same plot using Seaborn.

#     :param log_folders: (list of str) List of save locations of the results to plot
#     :param title: (str) The title of the plot
#     """
#     sns.set(style="whitegrid", font_scale=1.2)
#     plt.figure(figsize=(12, 8))
#     max_steps = 0
#     all_interpolated_results = []

#     for log_folder in log_folders:
#         x, y = ts2xy(load_results(log_folder), "timesteps")
#         y = moving_average(y, window=10)
#         x = x[len(x) - len(y):]

#         max_steps = max(max_steps, x[-1])

#         # Interpolation
#         common_x = np.linspace(0, x[-1], num=1000)
#         interpolated_y = np.interp(common_x, x, y)
#         all_interpolated_results.append((common_x, interpolated_y))

#     for i, (x, y) in enumerate(all_interpolated_results):
#         sns.lineplot(x=x, y=y, label=log_folders[i].split('\\')[-1], linewidth=2)

#     plt.xlabel("Number of Timesteps")
#     plt.ylabel("Rewards")
#     plt.title(title + " Smoothed")
#     plt.legend()
#     plt.show()

def plot_results(log_folders_by_alg, title="Learning Curves", window=1000):
    """
    Plot the mean rewards and the standard deviation for multiple algorithms across multiple runs.
    
    :param log_folders_by_alg: (dict) Dictionary with algorithm names as keys and lists of log folder paths as values.
    :param title: (str) The title of the plot
    :param window: (int) The window size for the moving average
    """
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(12, 8))

    for alg_name, log_folders in log_folders_by_alg.items():
        all_data = []
        max_steps = 0

        # Load and interpolate data for each algorithm
        for log_folder in log_folders:
            x, y = ts2xy(load_results(log_folder), "timesteps")
            y_smooth = moving_average(y, window)
            x = x[len(x) - len(y_smooth):]  # Adjust x to match length of smoothed y
            max_steps = max(max_steps, max(x))
            all_data.append((x, y_smooth))

        # Create a common x-axis for interpolation
        common_x = np.linspace(0, max_steps, num=1000)
        interpolated_ys = []

        # Interpolate all data series to the common x-axis
        for x, y in all_data:
            interpolated_y = np.interp(common_x, x, y)
            interpolated_ys.append(interpolated_y)

        # Calculate mean and standard deviation
        mean_rewards = np.mean(interpolated_ys, axis=0)
        std_rewards = np.std(interpolated_ys, axis=0)

        # Plotting the mean and filling the standard deviation
        plt.plot(common_x, mean_rewards, label=f'Mean Rewards - {alg_name}')
        plt.fill_between(common_x, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.legend()
    plt.show()