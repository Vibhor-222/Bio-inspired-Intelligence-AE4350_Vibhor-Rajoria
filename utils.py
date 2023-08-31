import matplotlib.pyplot as plt
import gym
import numpy as np

def plot_learning_curve(x, scores, epsilons):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    
    ax.plot(x, epsilons, color=color1, label='Epsilon', linewidth=2)
    ax.set_xlabel("Training Steps", color=color1)
    ax.set_ylabel("Epsilon", color=color1)
    ax.tick_params(axis='x', colors=color1)
    ax.tick_params(axis='y', colors=color1)
    ax.grid(True)  # Add grid lines
    
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2 = ax.twinx()
    ax2.plot(x, running_avg, color=color2, label='Avg. Score', linewidth=2)
    ax2.set_ylabel('Score', color=color2)
    ax2.tick_params(axis='y', colors=color2)
    
    # Improved legend placement
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    # Adjust margins for better spacing
    plt.subplots_adjust(right=0.85)
    
    # Increase font size of axis labels and tick labels
    ax.set_xlabel("Training Steps", color=color1, fontsize=14)
    ax.set_ylabel("Epsilon", color=color1, fontsize=14)
    ax2.set_ylabel('Score', color=color2, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.show()

