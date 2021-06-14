import keras
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2D, Flatten
from collections import deque
import numpy as np
import random
import sys

class Agent:
	def __init__(self, state_size, epsilon_episodes, discount=0.95):
		self.state_size = state_size
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
				Reshape((20, 20, 2), input_shape=(self.state_size,)),
				Conv2D(4, 3, activation='relu', kernel_initializer='glorot_uniform'),
				Conv2D(8, 3, activation='relu', kernel_initializer='glorot_uniform'),
				Flatten(),
				#Dense(128, activation='relu', kernel_initializer='glorot_uniform'),
				Dense(64, activation='relu', kernel_initializer='glorot_uniform'),
				Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
				Dense(1, activation='linear')
		])

		model.compile(loss='mse', optimizer='adam')
		return model

	def add_to_memory(self, current_state, next_state, reward, done):
		self.memory.append([current_state, next_state, reward, done])

	def act(self, states):
		max_value = -sys.maxsize - 1
		best = None

		if random.random() <= self.epsilon:
			return random.choice(list(states))
		else:
			for state in states:
				value = self.model.predict(np.reshape(state, [1, self.state_size]))
				if value > max_value:
					max_value = value
					best = state
		
		return best

	def replay(self):
		
		if len(self.memory) < 1:
			print("skipped...")
			return

		print("Weight Updated")

		batch = self.memory

		next_states = np.array([s[1] for s in batch])
		next_qvalue = np.array([s[0] for s in self.model.predict(next_states)])

		x = []
		y = []


		for i in range(len(batch)):
			state, _, reward, done = batch[i][0], None, batch[i][2], batch[i][3]
			new_q = reward
			#if not done:
			#	new_q = reward + self.discount * next_qvalue[i]
			#else:
			#	new_q = reward

			x.append(state)
			y.append(new_q)

		self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size, epochs=self.epochs, verbose=0)
		self.memory = deque(maxlen=30000)

	def save_model(self, path):
		self.model.save(path)
	
	def load_model(self, path):
		self.model = keras.models.load_model(path)

