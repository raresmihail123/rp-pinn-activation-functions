"""
********************************************************************************
figs
********************************************************************************
"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime

def plot_sol0(X_star, phi1):
    lb = X_star.min(0); ub = X_star.max(0); nn = 200
    x, y = np.linspace(lb[0], ub[0], nn), np.linspace(lb[1], ub[1], nn); x, y = np.meshgrid(x, y)
    PHI_I = griddata(X_star, phi1.flatten(), (x, y), method = "linear")

    plt.figure(figsize = (8, 4))
    plt.xlabel("t"); plt.ylabel("x")
    plt.xticks(np.arange(-30, 30, 1)); plt.yticks(np.arange(-30, 30, 1))
    plt.pcolor(x, y, PHI_I, cmap = "coolwarm", shading = "auto")
    plt.colorbar()
    plt.show()
    
def plot_sol1(X_star, phi1, v0, v1, ticks, file_path):
    lb = X_star.min(0); ub = X_star.max(0); nn = 200
    x, y = np.linspace(lb[0], ub[0], nn), np.linspace(lb[1], ub[1], nn); x, y = np.meshgrid(x, y)
    PHI_I = griddata(X_star, phi1.flatten(), (x, y), method = "linear")

    plt.figure(figsize = (8, 4))
    plt.xlabel("t"); plt.ylabel("x")
    if "loss" not in file_path:
        plt.title("PDE solution")
    else:
        plt.title("PDE loss of solution")
    plt.xticks(np.arange(-30, 30, 1)); plt.yticks(np.arange(-30, 30, 1))
    plt.pcolor(x, y, PHI_I, cmap = "coolwarm", shading = "auto", vmin = v0, vmax = v1)
    plt.colorbar(ticks = np.arange(-30, 30, ticks))
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()

def read_data(file_path):
    epochs = []
    losses = []
    minimum = 100
    with open(file_path, 'r') as file:
        for line in file:
            # Extract epoch and loss values using simple string parsing
            if "Execution" in line:
                continue
            parts = line.strip().split(", ")
            epoch = int(parts[0].split(": ")[1])
            loss = float(parts[1].split(": ")[1])
            epochs.append(epoch)
            losses.append(loss)
            minimum = min(minimum, loss)
    return np.array(epochs), np.array(losses), minimum

def plot_all__no_adaptive_sol(file_name_tanh, file_name_sigmoid, file_name_relu):
    os.makedirs(f"figures/aggregated_results", exist_ok=True)
    # File paths for the three data files
    file1 = file_name_tanh
    file2 = file_name_sigmoid
    file3 = file_name_relu
    save_file = f"figures/aggregated_results/loss_history_aggregated_no_adaptive.jpg"

    # Read data from the files
    epochs1, losses1, min_tanh = read_data(file1)
    epochs2, losses2, min_sigmoid = read_data(file2)
    epochs3, losses3, min_relu = read_data(file3)

    print(min_tanh)
    print(min_sigmoid)
    print(min_relu)

    # Plotting the data
    plt.figure(figsize=(10, 6))

    # Plot each file with specific colors
    plt.plot(epochs1, losses1, label='tanh', color='b', linewidth=1)  # Blue for file1
    plt.plot(epochs2, losses2, label='sigmoid', color='g', linewidth=1)  # Green for file2
    plt.plot(epochs3, losses3, label='relu', color='r', linewidth=1)  # Red for file3

    # Set the scale to log-log for the loss axis
    plt.xscale('linear')
    plt.yscale('log')

    # Set the limits for the y-axis
    plt.ylim(1e-4, 1e2)
    plt.grid(alpha=.5)
    # Labeling the axes
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Title and legend
    plt.title('Loss vs Epoch (Log Scale)')
    plt.legend(loc='upper right')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    # Display the plot
    plt.grid(True)
    plt.show()


def plot_all_sol(file_name_no_af, file_name_laaf, file_name_n_laaf):
    os.makedirs(f"figures/aggregated_results", exist_ok=True)
    # File paths for the three data files
    file1 = file_name_no_af
    file2 = file_name_laaf
    file3 = file_name_n_laaf
    save_file = f"figures/aggregated_results/loss_history_aggregated.jpg"

    # Read data from the files
    epochs1, losses1, min_no_af = read_data(file1)
    epochs2, losses2, min_laaf = read_data(file2)
    epochs3, losses3, min_nlaaf = read_data(file3)

    print(min_no_af)
    print(min_laaf)
    print(min_nlaaf)

    # Plotting the data
    plt.figure(figsize=(10, 6))

    # Plot each file with specific colors
    plt.plot(epochs1, losses1, label='no af - tanh', color='b', linewidth=1)  # Blue for file1
    plt.plot(epochs2, losses2, label='laaf', color='g', linewidth=1)  # Green for file2
    plt.plot(epochs3, losses3, label='n_laaf', color='r', linewidth=1)  # Red for file3

    # Set the scale to log-log for the loss axis
    plt.xscale('linear')
    plt.yscale('log')

    # Set the limits for the y-axis
    plt.ylim(1e-6, 1e2)
    plt.grid(alpha=.5)
    # Labeling the axes
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Title and legend
    plt.title('Loss vs Epoch (Log Scale)')
    plt.legend(loc='upper right')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    # Display the plot
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_name_no_af = Path.cwd() / "figures" / "no_af" / "2025-01-11-16-41-21.797975_tanh" / "log.txt"
    file_name_laaf = Path.cwd() / "figures" / "laaf" / "2025-01-11-21-08-25.704512" / "log.txt"
    file_name_n_laaf = Path.cwd() / "figures" / "n_laaf" / "2025-01-11-18-35-05.164718" / "log.txt"

    plot_all_sol(file_name_no_af, file_name_laaf, file_name_n_laaf)
    # file_name_tanh = Path.cwd() / "figures" / "no_af" / "2025-01-11-16-41-21.797975_tanh" / "log.txt"
    # file_name_sigmoid = Path.cwd() / "figures" / "no_af" / "2025-01-18-19-52-48.319224_sigmoid" / "log.txt"
    # file_name_relu = Path.cwd() / "figures" / "no_af" / "2025-01-18-18-45-12.196251_relu" / "log.txt"
    # plot_all__no_adaptive_sol(file_name_tanh, file_name_sigmoid, file_name_relu)

