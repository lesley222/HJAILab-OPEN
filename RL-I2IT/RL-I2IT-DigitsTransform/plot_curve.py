import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import seaborn.timeseries

def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm)*1.0/sm
            d = np.convolve(y, d, "same")

            smooth_data.append(d)

    return smooth_data

def _plot_std_bars(*args, central_data=None, ci=None, data=None, **kwargs):
    std = data.std(axis=0)
    ci = np.asarray((central_data - std, central_data + std))
    kwargs.update({"central_data": central_data, "ci": ci, "data": data})
    seaborn.timeseries._plot_ci_bars(*args, **kwargs)

def _plot_std_band(*args, central_data=None, ci=None, data=None, **kwargs):
    std = data.std(axis=0)
    ci = np.asarray((central_data - std, central_data + std))
    kwargs.update({"central_data": central_data, "ci": ci, "data": data})
    seaborn.timeseries._plot_ci_band(*args, **kwargs)

seaborn.timeseries._plot_std_bars = _plot_std_bars
seaborn.timeseries._plot_std_band = _plot_std_band

def _plot_range_band(*args, central_data=None, ci=None, data=None, **kwargs):
    upper = data.max(axis=0)
    lower = data.min(axis=0)
    # import pdb; pdb.set_trace()
    ci = np.asarray((lower, upper))
    kwargs.update({"central_data": central_data, "ci": ci, "data": data})
    sns.timeseries._plot_ci_band(*args, **kwargs)

sns.timeseries._plot_range_band = _plot_range_band

if __name__ == "__main__":
    sac_path = 'curve/mnist_sac.npy'
    # ppo_path = 'curve/old_ppo_reward.npy'
    # ppo1_path = 'curve/ppo_reward_1.npy'
    # ppo2_path = 'curve/ppo_reward_2.npy'
    # ppo3_path = 'curve/ppo_reward_3.npy'

    sac_rewards = np.load(sac_path)
    # ppo_rewards = np.load(ppo_path)
    # ppo_rewards1 = np.load(ppo1_path)
    # ppo_rewards2 = np.load(ppo2_path)
    # ppo_rewards3 = np.load(ppo3_path)
    # ppo_rewards = np.vstack((ppo_rewards1, ppo_rewards2))
    # ppo_rewards = np.vstack((ppo_rewards1, ppo_rewards2, ppo_rewards3))
    # ppo_rewards = np.vstack((ppo_rewards[0], ppo_rewards[2], ppo_rewards[4]))

    time = np.arange(len(sac_rewards[0][::1]))
    sac_rewards = smooth(sac_rewards[:, :], sm=10)
    # ppo_rewards = smooth(ppo_rewards[:, :], sm=10)
    # ppo_rewards1 = smooth(ppo_rewards1[1:2, ::10], sm=5)
    # ppo_rewards2 = smooth(ppo_rewards2[2:3, ::10], sm=5)

    # df = pd.DataFrame(rewards).melt()
    # print(df.head())
    sns.tsplot(time=time, data=sac_rewards, color='r', n_boot=0, err_style="range_band", linewidth=1., linestyle='-', condition='sac')
    # sns.tsplot(time=time, data=ppo_rewards, color='b', n_boot=0, err_style="range_band", linewidth=1., linestyle='-', condition='ppo')
    # sns.tsplot(time=time, data=ppo_rewards2, color='b', linestyle='-', condition='ppo2')
    # sns.lineplot(x='variable', y='value', data=df)

    plt.ylabel("Dice Reward", fontsize=15)
    plt.xlabel("Episode", fontsize=15)
    plt.title("deformable registration", fontsize=20)

    plt.legend(loc='lower right')
    plt.savefig("curve/test.png")
    plt.show()


