import keras
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2D, Flatten
from collections import deque
import numpy as np
import random
import sys
import tensorflow as tf
import pandas as pd

MIN_REWARD = -1000
MAX_REWARD = 1000

class Agent:
	def __init__(self, width, height, layers, num_of_orientations, epsilon_episodes, discount=0.95):
		self.state_size = height*width*layers
		self.height = height
		self.width = width
		self.layers = layers
		self.number_of_actions = width*height*num_of_orientations
		self.memory = deque(maxlen=30000)
		self.discount = discount
		self.epsilon = 1.0
		self.epsilon_min = 0.001 
		self.epsilon_end_episode = epsilon_episodes
		self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_end_episode

		self.batch_size = 30
		self.epochs = 30

		self.model = self.build_model()


	def build_model(self):
		model = keras.Sequential([
				Reshape((self.width, self.height, self.layers), input_shape=(self.state_size,)),
				Conv2D(4, 3, activation='relu', kernel_initializer='glorot_uniform'),
				Conv2D(8, 3, activation='relu', kernel_initializer='glorot_uniform'),
				Flatten(),
				Dense(128, activation='relu', kernel_initializer='glorot_uniform'),
				Dense(64, activation='relu', kernel_initializer='glorot_uniform'),
				Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
				Dense(self.number_of_actions, activation='linear')
		])

		# model.compile(loss=tf.keras.losses.Huber(delta=1.0), optimizer='adam')
		model.compile(loss='mse', optimizer='adam')
		return model

	def add_to_memory(self, current_state, action, reward, next_state, done):
		self.memory.append([current_state, action, reward, next_state, done])

	def act(self, state, action_mask):
		max_value = -sys.maxsize - 1
		best = None

		if random.random() <= self.epsilon:
			return np.random.choice(np.nonzero(action_mask)[0])
		else:
			action_q_values = self.model.predict(np.reshape(state, [1, self.state_size]))
			masked_q_values = action_q_values*(action_mask)
			best = np.argmax(masked_q_values)
	
		# returns action index
		return best

	def replay(self):
		
		if len(self.memory) < 1:
			print("skipped...")
			return

		print("Weight Updated")

		batch = self.memory

		actions = [s[1] for s in batch]
		rewards = [s[2] for s in batch]
		dones = [s[4] for s in batch]
		current_q_values = self.model.predict(np.array([s[0] for s in batch])).tolist()
		next_q_values = self.model.predict(np.array([s[3] for s in batch])).tolist()


		x = [s[0] for s in batch]
		y = []

		for i in range(len(actions)):
			action = actions[i]
			reward = rewards[i]
			done = dones[i]
			current_q_value = current_q_values[i]
			next_q_value = next_q_values[i]

			if done:
				current_q_value[action] = reward
			else:
				current_q_value[action] = self.epsilon*reward + (1-self.epsilon)*self.discount*max(next_q_value)

			current_q_value[action] = max(MIN_REWARD, current_q_value[action])
			current_q_value[action] = min(MAX_REWARD, current_q_value[action])

			y.append(current_q_value)

		self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size, epochs=self.epochs, verbose=0)
		self.memory = deque(maxlen=30000)

	def save_model(self, path):
		self.model.save(path)
	
	def load_model(self, path):
		self.model = keras.models.load_model(path)

