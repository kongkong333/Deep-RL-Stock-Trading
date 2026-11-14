import argparse
import csv
import importlib
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from utils import *


parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_name', action="store", dest="model_name", default='DQN', choices=['DQN', 'DDQN', 'PPO'], help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2010-2015', help="stock name")
parser.add_argument('--train_stocks', nargs='+', default=None,
                    help='one or more training CSV names; defaults to --stock_name')
parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int, help="span (days) of observation")
parser.add_argument('--num_episode', action="store", dest="num_episode", default=10, type=int, help='episode number')
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int, help='initial balance')
parser.add_argument('--episode_length', type=int, default=500, help='trading days per episode; shorter datasets use the full available range; use 0 for the full CSV')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducible training (default: 42)')
inputs = parser.parse_args()

set_global_seed(inputs.seed)

model_name = inputs.model_name
stock_name = inputs.stock_name
train_stock_names = inputs.train_stocks or [stock_name]
window_size = inputs.window_size
num_episode = inputs.num_episode
initial_balance = inputs.initial_balance
episode_length = inputs.episode_length

prices_by_stock = {name: stock_close_prices(name) for name in train_stock_names}
dates_by_stock = {
    name: pd.read_csv(Path('data') / f'{name}.csv')['Date'].astype(str).tolist()
    for name in train_stock_names
}
returns_across_episodes = []
num_experience_replay = 0
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

# select learning model
model = importlib.import_module(f'agents.{model_name}')
agent = model.Agent(state_dim=window_size + 3, balance=initial_balance)

def buy(t):
    if agent.balance >= stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        return 'Buy: ${:.2f}'.format(stock_prices[t])

def sell(t):
    if len(agent.inventory) > 0:
        agent.balance += stock_prices[t]
        bought_price = agent.inventory.pop(0)
        profit = stock_prices[t] - bought_price
        return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)

# configure logging. TensorFlow may install a root handler before this point,
# in which case basicConfig() without force=True would silently do nothing.
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f'{model_name}_training_{stock_name}.log'
logging.basicConfig(
    filename=str(log_path),
    filemode='w',
    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    force=True,
)
print(f'Training log: {log_path}', flush=True)

metrics_path = log_dir / f'{model_name}_episode_metrics_{stock_name}_seed{inputs.seed}.csv'
metric_fields = [
    'episode', 'seed', 'stock', 'window_start_date', 'window_end_date',
    'total_return', 'final_portfolio_value', 'maximum_drawdown', 'sharpe_ratio',
    'buy_count', 'sell_count', 'hold_count', 'trade_count',
    'mean_daily_return', 'mean_training_loss', 'replay_updates',
]
if hasattr(agent, 'epsilon'):
    metric_fields.append('epsilon')
with metrics_path.open('w', newline='', encoding='utf-8') as metrics_file:
    csv.DictWriter(metrics_file, fieldnames=metric_fields).writeheader()
print(f'Episode metrics CSV: {metrics_path}', flush=True)

logging.info(f'Training CSVs:            {", ".join(train_stock_names)}')
logging.info('Episode windows: randomly sampled from the selected training CSVs')
logging.info(f'Window Size:              {window_size} days')
logging.info(f'Training Episode:         {num_episode}')
logging.info(f'Model Name:               {model_name}')
logging.info(f'Random Seed:              {inputs.seed}')
logging.info('Initial Portfolio Value: ${:,}'.format(initial_balance))

start_time = time.time()
for e in range(1, num_episode + 1):
    logging.info(f'\nEpisode: {e}/{num_episode}')

    agent.reset() # reset to initial balance and hyperparameters
    episode_losses = []
    action_counts = {'buy': 0, 'sell': 0, 'hold': 0}
    stock_name = np.random.choice(train_stock_names)
    stock_prices = prices_by_stock[stock_name]
    total_trading_period = len(stock_prices) - 1
    trading_period = min(episode_length, total_trading_period) if episode_length > 0 else total_trading_period
    logging.info(f'Training interval: {stock_name}; episode length: {trading_period} days')
    start_index = np.random.randint(0, total_trading_period - trading_period + 1)
    end_index = start_index + trading_period
    state = generate_combined_state(start_index, window_size, stock_prices, agent.balance,
                                    len(agent.inventory))

    for t in range(start_index + 1, end_index + 1):
        if (t - start_index) % 100 == 0:
            logging.info(f'\n-------------------Period: {t - start_index}/{trading_period}-------------------')

        reward = 0.0
        previous_portfolio_value = len(agent.inventory) * stock_prices[t - 1] + agent.balance
        action_mask = valid_action_mask(stock_prices[t - 1], agent.balance, len(agent.inventory))

        if model_name == 'PPO':
            action = agent.act(state, action_mask)
            actions = agent.action_probs
        else:
            actions = agent.model.predict(state, verbose=0)[0]
            action = agent.act(state, action_mask)
        
        # execute position
        logging.info('Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}'.format(t, actions[0], actions[1], actions[2]))
        if action != np.argmax(actions): logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
        if action == 0:
            execution_result = None
        elif action == 1:
            execution_result = buy(t)
        else:
            execution_result = sell(t)

        executed_action = action if execution_result is not None else 0
        if executed_action == 0:
            action_counts['hold'] += 1
        elif executed_action == 1:
            action_counts['buy'] += 1
        else:
            action_counts['sell'] += 1
        
        # check execution result
        if execution_result is not None:
            logging.info(execution_result)                

        # calculate reward
        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        reward = (current_portfolio_value - previous_portfolio_value) / initial_balance

        next_state = generate_combined_state(t, window_size, stock_prices, agent.balance,
                                             len(agent.inventory))
        next_action_mask = valid_action_mask(stock_prices[t], agent.balance, len(agent.inventory))

        agent.portfolio_values.append(current_portfolio_value)
        agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)

        done = t == end_index
        if model_name == 'PPO':
            # PPO must retain the sampled action paired with its old policy probability.
            agent.remember(state, action, reward, next_state, done, action_mask)
        else:
            agent.remember(state, action, reward, next_state, done, next_action_mask)

        # update state
        state = next_state

        # experience replay
        if model_name == 'PPO' and done:
            num_experience_replay += 1
            loss = agent.experience_replay()
            episode_losses.append(float(loss))
            logging.info('Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}'.format(e, loss, action_dict[executed_action], reward, agent.balance, len(agent.inventory)))
            agent.tensorboard.on_batch_end(num_experience_replay, {'loss': loss, 'portfolio value': current_portfolio_value})
        elif model_name != 'PPO' and len(agent.memory) >= agent.learning_starts and (t - start_index) % agent.train_freq == 0:
            num_experience_replay += 1
            loss = agent.experience_replay()
            episode_losses.append(float(loss))
            logging.info('Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}'.format(e, loss, action_dict[executed_action], reward, agent.balance, len(agent.inventory)))
            agent.tensorboard.on_batch_end(num_experience_replay, {'loss': loss, 'portfolio value': current_portfolio_value})

        if done:
            portfolio_return = evaluate_portfolio_performance(agent, logging)
            returns_across_episodes.append(portfolio_return)
            trade_count = action_counts['buy'] + action_counts['sell']
            metric_row = {
                'episode': e,
                'seed': inputs.seed,
                'stock': stock_name,
                'window_start_date': dates_by_stock[stock_name][start_index],
                'window_end_date': dates_by_stock[stock_name][end_index],
                'total_return': portfolio_return,
                'final_portfolio_value': agent.portfolio_values[-1],
                'maximum_drawdown': maximum_drawdown(agent.portfolio_values),
                'sharpe_ratio': sharpe_ratio(
                    np.asarray(agent.return_rates),
                    risk_free=treasury_bond_daily_return_rate(),
                ),
                'buy_count': action_counts['buy'],
                'sell_count': action_counts['sell'],
                'hold_count': action_counts['hold'],
                'trade_count': trade_count,
                'mean_daily_return': np.mean(agent.return_rates),
                'mean_training_loss': np.mean(episode_losses) if episode_losses else '',
                'replay_updates': len(episode_losses),
            }
            if hasattr(agent, 'epsilon'):
                metric_row['epsilon'] = agent.epsilon
            with metrics_path.open('a', newline='', encoding='utf-8') as metrics_file:
                csv.DictWriter(metrics_file, fieldnames=metric_fields).writerow(metric_row)

    # save models periodically
    if e % 5 == 0:
        Path('saved_models').mkdir(exist_ok=True)
        if model_name == 'DQN':
            agent.model.save('saved_models/DQN_ep' + str(e) + '.h5')
            agent.model_target.save('saved_models/DQN_ep{}_target.h5'.format(e))
        elif model_name == 'DDQN':
            agent.model.save('saved_models/DDQN_ep' + str(e) + '.h5')
            agent.model_target.save('saved_models/DDQN_ep{}_target.h5'.format(e))
        elif model_name == 'PPO':
            Path('saved_models').mkdir(exist_ok=True)
            agent.model.save_weights('saved_models/PPO_ep{}.h5'.format(e))
        logging.info('model saved')

logging.info('total training time: {0:.2f} min'.format((time.time() - start_time)/60))
plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)
