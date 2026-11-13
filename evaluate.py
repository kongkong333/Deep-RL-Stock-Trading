import argparse
import importlib
import logging
import sys

import numpy as np
# np.random.seed(3)  # for reproducible Keras operations

from utils import *


parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_to_load', action="store", dest="model_to_load", default='DQN_ep10', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='^GSPC_2018', help="stock name")
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int, help='initial balance')
parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int, help='window_size')
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
inputs = parser.parse_args()

set_global_seed(inputs.seed)

model_to_load = inputs.model_to_load
model_name = model_to_load.split('_')[0]
if model_name == 'DDPG':
    parser.error('DDPG checkpoints are incompatible with the discrete action space; evaluate a DQN, DDQN, or PPO checkpoint.')
stock_name = inputs.stock_name
initial_balance = inputs.initial_balance
display = True
window_size = inputs.window_size
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

# select evaluation model
model = importlib.import_module(f'agents.{model_name}')

def buy(t):
    if agent.balance >= stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        agent.buy_dates.append(t)
        logging.info('Buy:  ${:.2f}'.format(stock_prices[t]))

def sell(t):
    agent.balance += stock_prices[t]
    bought_price = agent.inventory.pop(0)
    profit = stock_prices[t] - bought_price
    global reward
    reward = profit
    agent.sell_dates.append(t)
    logging.info('Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit))

# configure logging
logging.basicConfig(filename=f'logs/{model_name}_evaluation_{stock_name}.log', filemode='w',
                    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

portfolio_return = 0
while True:
    agent = model.Agent(state_dim=window_size+2, balance=initial_balance, is_eval=True, model_name=model_to_load)
    stock_prices = stock_close_prices(stock_name)
    trading_period = len(stock_prices) - 1
    state = generate_combined_state(0, window_size, stock_prices, agent.balance,
                                    len(agent.inventory))

    for t in range(1, trading_period + 1):
        action_mask = valid_action_mask(stock_prices[t - 1], agent.balance, len(agent.inventory))
        if model_name == 'PPO':
            action = agent.act(state, action_mask)
            actions = agent.action_probs
        else:
            actions = agent.model.predict(state, verbose=0)[0]
            action = agent.act(state, action_mask)

        # print('actions:', actions)
        # print('chosen action:', action)

        previous_portfolio_value = len(agent.inventory) * stock_prices[t - 1] + agent.balance
        
        # execute position
        logging.info(f'Step: {t}')
        if action != np.argmax(actions): logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
        if action == 1 and agent.balance >= stock_prices[t]:
            buy(t)
        elif action == 2 and len(agent.inventory) > 0:
            sell(t)

        current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
        agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
        agent.portfolio_values.append(current_portfolio_value)
        next_state = generate_combined_state(t, window_size, stock_prices, agent.balance,
                                             len(agent.inventory))
        state = next_state

        done = True if t == trading_period else False
        if done:
            portfolio_return = evaluate_portfolio_performance(agent, logging)

    break

if display:
    # plot_portfolio_transaction_history(stock_name, agent)
    # plot_portfolio_performance_comparison(stock_name, agent)
    plot_all(stock_name, agent)
