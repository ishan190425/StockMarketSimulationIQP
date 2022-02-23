import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import math
import numpy as np
import pandas as pd
import random
from collections import deque
from os import walk

padding = "--------------------------------"
class Reinforcment():
    def __init__(self, state_size, action_size, memory_size, indicators, is_eval=False, model_name="ANN", gamma=0.95, epsilon=1.0, eplsion_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size  # normalized previous days
        self.action_size = action_size  # hold, buy, sell
        self.indicators = indicators
        self.memory = deque(maxlen=memory_size)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = eplsion_min
        self.epsilon_decay = epsilon_decay
        self.model = load_model(model_name) if is_eval else self._model(model_name)

    def _model(self, type, learning_rate=0.001):
        model = Sequential()
        if type == 'ANN':
            model.add(
                Dense(units=64, input_dim=8, activation="relu"))
            model.add(Dense(units=32, activation="relu"))
            model.add(Dense(units=8, activation="relu"))
            model.add(Dense(self.action_size, activation="linear"))
            model.compile(loss="mse", optimizer=Adam(
                learning_rate=learning_rate), metrics='accuracy')

        if type == 'RNN':
            model.add(LSTM(units=50, return_sequences=True,
                      input_shape=(self.state_size, self.indicators)))
            model.add(Dropout(rate=0.2))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(rate=0.2))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(rate=0.2))
            model.add(LSTM(units=50))
            model.add(Dropout(rate=0.2))
            model.add(Dense(units=self.action_size))
            model.compile(optimizer=Adam(learning_rate=learning_rate),
                          loss='mse', metrics='accuracy')
        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def getStockDataVec(self, path):
        data = pd.read_csv(path)
        vec = data["CLOSE"].values
        return vec

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def getState(self, data, t, n):
        d = t - n + 1
        block = data[d:t + 1] if d >= 0 else np.pad(data[0:t+1], (-d, 0), 'constant',constant_values = data[0])  # pad with t0
        res = []
        for i in range(n - 2):
            res.append(self.sigmoid(block[i+1] - block[i]))
        return np.array([res])

    def run(self, stock, window, episodes):
        stock_name = stock
        window_size = window
        episode_count = episodes
        data = self.getStockDataVec(stock_name)
        l = len(data) - 1
        batch_size = 32
        maxProfit = float("-inf")
        for e in range(episode_count + 1):
            print("Episode {}/{}".format(e,episode_count))
            state = self.getState(data, 0, window_size + 1)
            total_profit = 0
            self.inventory = []
            for t in range(l):
                action = self.act(state) #sit
                next_state = self.getState(data, t + 1, window_size + 1)
                reward = 0
                if action == 1 and len(self.inventory) == 0:  #buy
                    self.inventory.append(data[t])
                    print("Buy: {}".format(data[t]))
                elif action == 2 and len(self.inventory) > 0:  #sell
                    bought_price = self.inventory.pop(0)
                    profit = data[t] - bought_price
                    reward = max(profit, 0)
                    total_profit += profit
                    print("Sell: {} | Profit: {} | Total Profit: {}".format(data[t], profit, total_profit))
                done = True if t == l - 1 else False
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                if done:
                    print("{}\nTotal Profit: {}\n{}".format(padding,total_profit,padding))
                    maxProfit = max(maxProfit, total_profit)
                if len(self.memory) > batch_size:
                    self.expReplay(batch_size)
            if e % 10 == 0:
                self.model.save("{}-{}".format(stock_name,e))

companies = []
data_path = "/Users/ishan/Coding/Wpi/StockMarketSimulationIQP/BriannaCode/30y_stock_csvs/"
for (dirpath, dirnames, filenames) in walk(data_path):
    companies.extend(filenames)
    break 
for company in companies:
    print("{}\nTraining Company: {}\n{}".format(padding,company[:-4],padding))
    r = Reinforcment(10,3,1000,1)
    r.run("{}{}".format(data_path,company),9,10)

