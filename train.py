import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

from utils import *


parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_name', action="store", dest="model_name", default='DQN', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2010-2015', help="stock name")
parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int, help="span (days) of observation")
parser.add_argument('--num_episode', action="store", dest="num_episode", default=10, type=int, help='episode number')
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int, help='initial balance')
inputs = parser.parse_args()

model_name = inputs.model_name
stock_name = inputs.stock_name
window_size = inputs.window_size
num_episode = inputs.num_episode
initial_balance = inputs.initial_balance

stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1
returns_across_episodes = []
num_experience_replay = 0
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

# select learning model
model = importlib.import_module(f'agents.{model_name}')
agent = model.Agent(state_dim=window_size + 3, balance=initial_balance)

def buy(t):
    if agent.balance > stock_prices[t]:
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

logging.info(f'Trading Object:           {stock_name}')
logging.info(f'Trading Period:           {trading_period} days')
logging.info(f'Window Size:              {window_size} days')
logging.info(f'Training Episode:         {num_episode}')
logging.info(f'Model Name:               {model_name}')
logging.info('Initial Portfolio Value: ${:,}'.format(initial_balance))

start_time = time.time()
for e in range(1, num_episode + 1):
    logging.info(f'\nEpisode: {e}/{num_episode}')

    agent.reset() # reset to initial balance and hyperparameters
    state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))

    for t in range(1, trading_period + 1):
        if t % 100 == 0:
            logging.info(f'\n-------------------Period: {t}/{trading_period}-------------------')

        reward = 0
        previous_portfolio_value = len(agent.inventory) * stock_prices[t - 1] + agent.balance

        if model_name == 'DDPG':
            actions = agent.act(state, t)
            action = np.argmax(actions)
        else:
            actions = agent.model.predict(state)[0]
            action = agent.act(state)
        
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
        
        # check execution result
        if execution_result is None:
            reward -= treasury_bond_daily_return_rate() * agent.balance  # missing opportunity
        else:
            logging.info(execution_result)                

        # calculate reward
        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        unrealized_profit = current_portfolio_value - agent.initial_portfolio_value
        reward += unrealized_profit

        next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))

        agent.portfolio_values.append(current_portfolio_value)
        agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)

        done = True if t == trading_period else False
        replay_action = executed_action if model_name in ('DQN', 'DDQN') else np.eye(agent.action_dim)[executed_action]
        agent.remember(state, replay_action, reward, next_state, done)

        # update state
        state = next_state

        # experience replay
        if len(agent.memory) > agent.buffer_size:
            num_experience_replay += 1
            loss = agent.experience_replay()
            logging.info('Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}'.format(e, loss, action_dict[executed_action], reward, agent.balance, len(agent.inventory)))
            agent.tensorboard.on_batch_end(num_experience_replay, {'loss': loss, 'portfolio value': current_portfolio_value})

        if done:
            portfolio_return = evaluate_portfolio_performance(agent, logging)
            returns_across_episodes.append(portfolio_return)

    # save models periodically
    if e % 5 == 0:
        if model_name == 'DQN':
            agent.model.save('saved_models/DQN_ep' + str(e) + '.h5')
            agent.model_target.save('saved_models/DQN_ep{}_target.h5'.format(e))
        elif model_name == 'DDQN':
            agent.model.save('saved_models/DDQN_ep' + str(e) + '.h5')
            agent.model_target.save('saved_models/DDQN_ep{}_target.h5'.format(e))
        elif model_name == 'DDPG':
            agent.actor.model.save_weights('saved_models/DDPG_ep{}_actor.h5'.format(str(e)))
            agent.critic.model.save_weights('saved_models/DDPG_ep{}_critic.h5'.format(str(e)))
        logging.info('model saved')

logging.info('total training time: {0:.2f} min'.format((time.time() - start_time)/60))
plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)
