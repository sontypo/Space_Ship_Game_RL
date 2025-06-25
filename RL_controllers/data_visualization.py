import pandas as pd
import matplotlib.pyplot as plt
import argparse

plt.style.use('seaborn-darkgrid')

parser = argparse.ArgumentParser()
# usage: python3 data_visualization.py --view_mode combined
parser.add_argument('--view_mode', type=str, default='separate', choices=['separate', 'combined'],
                    help="Choose 'separate' for individual plots or 'combined' for all-in-one figure.")
parser.add_argument('--save_path', type=str, default='d3qn_data', help='Loading save path.')
args = parser.parse_args()
view_mode = args.view_mode

saved_path = 'saved_models/' + args.save_path + '/'

# Load data
score = pd.read_csv(saved_path + 'score_per_episode.csv', header=None, comment='/', names=['Score'])
reward = pd.read_csv(saved_path + 'reward_per_episode.csv', header=None, comment='/', names=['Reward'])
health = pd.read_csv(saved_path + 'health_per_episode.csv', header=None, comment='/', names=['Health'])

ma_window = 10

def plot_metric(csv_file, ylabel, color, ma_window=10, title=None):
    data = pd.read_csv(csv_file, header=None, comment='/', names=[ylabel])
    values = data[ylabel].values

    plt.figure(figsize=(12, 5))
    plt.plot(values, label=f'{ylabel} per Episode', color=color, alpha=0.7)
    # Moving average
    if len(values) >= ma_window:
        ma = pd.Series(values).rolling(ma_window).mean()
        plt.plot(ma, label=f'{ylabel} {ma_window}-Episode MA', color='black', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.title(title if title else f'{ylabel} per Episode')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if view_mode == "combined":
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].plot(score['Score'], label='Score per Episode', color='tab:orange', alpha=0.7)
    if len(score) >= ma_window:
        score_ma = score['Score'].rolling(ma_window).mean()
        axs[0].plot(score_ma, label=f'Score {ma_window}-Episode MA', color='black', linewidth=2)
    axs[0].set_title('Score per Episode')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Score')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)

    axs[1].plot(reward['Reward'], label='Reward per Episode', color='tab:green', alpha=0.7)
    if len(reward) >= ma_window:
        reward_ma = reward['Reward'].rolling(ma_window).mean()
        axs[1].plot(reward_ma, label=f'Reward {ma_window}-Episode MA', color='black', linewidth=2)
    axs[1].set_title('Reward per Episode')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Reward')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.5)

    axs[2].plot(health['Health'], label='Health per Episode', color='tab:blue', alpha=0.7)
    if len(health) >= ma_window:
        health_ma = health['Health'].rolling(ma_window).mean()
        axs[2].plot(health_ma, label=f'Health {ma_window}-Episode MA', color='black', linewidth=2)
    axs[2].set_title('Health per Episode')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Health')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
else:
    # Separate plots for each metric
    plt.figure(figsize=(8, 4))
    plt.plot(score['Score'], label='Score per Episode', color='tab:orange', alpha=0.7)
    if len(score) >= ma_window:
        score_ma = score['Score'].rolling(ma_window).mean()
        plt.plot(score_ma, label=f'Score {ma_window}-Episode MA', color='black', linewidth=2)
    plt.title('Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    plt.plot(reward['Reward'], label='Reward per Episode', color='tab:green', alpha=0.7)
    if len(reward) >= ma_window:
        reward_ma = reward['Reward'].rolling(ma_window).mean()
        plt.plot(reward_ma, label=f'Reward {ma_window}-Episode MA', color='black', linewidth=2)
    plt.title('Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    plt.plot(health['Health'], label='Health per Episode', color='tab:blue', alpha=0.7)
    if len(health) >= ma_window:
        health_ma = health['Health'].rolling(ma_window).mean()
        plt.plot(health_ma, label=f'Health {ma_window}-Episode MA', color='black', linewidth=2)
    plt.title('Health per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Health')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

plt.show()