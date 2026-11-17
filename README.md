# Deep-RL-Stock-Trading

**Deep reinforcement learning for stock trading, quantitative finance, and portfolio management.**

I'm [Kexiang Wang](https://github.com/kongkong333). This project explores DQN, DDQN, and PPO agents for trading historical S&P 500 data, with Python and TensorFlow. It focuses on portfolio management, backtesting, and visualizing model performance. If you're searching for work on deep reinforcement learning, algorithmic trading, or quantitative finance, you can find me through this project or my [GitHub profile](https://github.com/kongkong333).

```bash
python train.py --model_name=DQN --stock_name=^GSPC_2012-2014 --num_episode=50 --seed=202611
python train.py --model_name=PPO --stock_name=^GSPC_2012-2014 --num_episode=100 --seed=202611
python evaluate.py --model_to_load=PPO_ep100 --stock_name=^GSPC_2015
```

## Visualizations

The DQN and PPO training plots below were generated with the first and second commands above, respectively. The evaluation plots show the 100-epoch PPO model (`PPO_ep100`) tested on the 2015 and 2016 datasets.

| DQN | PPO |
| --- | --- |
| ![DQN](visualizations/DQN.jpeg) | ![PPO](visualizations/PPO.jpeg) |

| PPO on ^GSPC 2015 | PPO on ^GSPC 2016 |
| --- | --- |
| ![PPO evaluation on 2015](visualizations/PPO_ep100_gspc2015.jpeg) | ![PPO evaluation on 2016](visualizations/PPO_ep100_gspc2016.jpeg) |

## Limitations

Training is currently unstable and needs further optimization.

## Attribution and License

This repository is a modified version of [Albert-Z-Guo's Deep-Reinforcement-Stock-Trading](https://github.com/Albert-Z-Guo/Deep-Reinforcement-Stock-Trading), which is licensed under GPL-3.0. This version has been modified through 2026-11-17. It is distributed under GPL-3.0; see [LICENSE](LICENSE).
