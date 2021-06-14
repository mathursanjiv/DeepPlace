from agent import Agent
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import numpy as np
import cv2 as cv
import random
from tabulate import tabulate
import pickle as pk


parser = argparse.ArgumentParser("""Implementation of Deep Q Network for block placement""")
parser.add_argument("--width", type=int, default=100, help="The common width for all images")
parser.add_argument("--height", type=int, default=100, help="The common height for all images")
parser.add_argument("--numofgames", type=int, default=10000, help="Number of placement")
parser.add_argument("--discount", type=float, default=0.95)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--replay_interval", type=int, default=50)
parser.add_argument("--saved_path", type=str, default="trained_models")
parser.add_argument("--load_model", type=int, default=0)
parser.add_argument("--view_placement", type=int, default=0)

args = parser.parse_args()

def file_parser(filename):

    count = 0
    macros = True
    shapes = {}
    adjency_matrix = []
    adjency_list = []
    numberOfMacros = 0
    numberOfEdges = 0
    OptimalWL = 0

    with open(filename,encoding="utf8") as f:

        for line in f:

            splitLine = line.split()

            if(splitLine[0]=='Macros'):
                macros = True
                numberOfMacros = int(splitLine[1])
                continue
            if(splitLine[0] == 'Edges'):
                macros = False
                numberOfEdges = int(splitLine[1])
                OptimalWL = int(splitLine[2])

                adjency_matrix = [[0 for x in range(len(shapes))] for y in range(len(shapes))]
                adjency_list = [[] for _ in range(len(shapes))]

                continue

            if(macros):
                index = int(splitLine[0])
                XL = int(splitLine[1])
                YL = int(splitLine[2])
                XR = int(splitLine[3])
                YR = int(splitLine[4])
                shapes[index] = ((XL, YL), (XR, YR))

            else:
                index1 = int(splitLine[0])
                index2 = int(splitLine[1])
                numberOfConnections = int(splitLine[2])
                adjency_matrix[index1][index2] = numberOfConnections
                adjency_matrix[index2][index1] = numberOfConnections

                adjency_list[index1].append(index2)
                adjency_list[index2].append(index1)


    print("Parsed File:", filename)
    print("numberOfMacros:", numberOfMacros, "numberOfEdges:", numberOfEdges, "OptimalWL:", OptimalWL, "\n")
    return shapes, adjency_matrix, adjency_list


color_list = []
for _ in range(10000):
    color_list.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
color_list.append((0, 0, 0))
black =  (0, 0, 0)
white = (255, 255, 255)

def rotated(shape):
    return [(-j, i) for i, j in shape]

def is_occupied(shape, anchor, board):
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        if y < 0 or x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
            return True
    return False

class DeepPlace:
    def __init__(self, width, height, filename):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=float)
        self.indexboard =  [[-1 for _ in range(width)] for _ in range(height)]

        self.state_size = self.width * self.height *2 

        # For running the engine
        self.score = -1
        self.anchor = None
        self.shape = None
        self.shape_index = -1

        self.filename = filename
        # Reset after initializing
        #self.reset()

    def _choose_shape(self):

        use_shape = self.shape_list[0]
        self.shape_list.remove(use_shape)
        return self.shapes[use_shape], use_shape

    def _new_piece(self):
        self.anchor = (self.width / 2, 1)
        self.shape, self.shape_index = self._choose_shape()

    def _reset_piece(self):
        self.anchor = None
        self.shape = None
        self.shape_index = -1

    def _manhanttan_distance(self, x, y):
        x1, x2 = x
        y1, y2 = y

        return abs(x1-y1) + abs(x2-y2)

    def _parse_shape(self):
        self.shapes, self.adjency_matrix, self.adjency_list = file_parser("train_data/" + str(self.filename) + ".txt")
        self.shape_list = list(self.shapes.keys())

    def _calculate_wirelength(self):

        hpwl = 0
        # Store positions of each macro ids
        indexes = {}

        for i in range(self.width):
            for j in range(self.height):
                key = self.indexboard[i][j]
                if key not in indexes:
                    indexes[key] = []
                indexes[key].append((i,j))

        for i in range(len(self.adjency_list)):
            for j in self.adjency_list[i]:
                
                if i not in indexes or j not in indexes:
                    continue

                i_ = indexes[i]
                j_ = indexes[j]

                distance = 1e10
                for k in i_:
                    for l in j_:
                        distance = min(distance, self._manhanttan_distance(k, l))

                hpwl += distance*self.adjency_matrix[i][j]


        return hpwl/2

    def step(self, action):
        
        reward = 0
        done = False

        self._set_piece(True)
        self._set_index_piece()

        if len(self.shape_list)==0:

            # print(tabulate(self.indexboard))
            reward -= self._calculate_wirelength()/1000
            self.reset()
            done = True
        else:
            self._new_piece()

        return reward, done

    def reset(self):
        self.time = 0
        self.score = 0
        self._reset_piece()
        self._parse_shape()
        self.shape_list = list(self.shapes.keys())
        self.board = np.zeros_like(self.board)
        self.indexboard =  [[-1 for _ in range(self.width)] for _ in range(self.height)]
        

        initBoard = pk.load(open("train_data/" + str(self.filename) + ".pkl", "rb"))

        for i in range(self.width):
            for j in range(self.width):
                self.board[i, j] = initBoard[i][j]

        return np.array([0 for _ in range(self.state_size)])

    def _set_piece(self, on):
        """To lock a piece in the board"""
        if(self.shape!=None):

            XL = self.shape[0][0]
            YL = self.shape[0][1]
            XR = self.shape[1][0]
            YR = self.shape[1][1]

            for x in range(XL, XR+1):
                for y in range(YL, YR+1):
                    self.board[x][y] = True

    def _set_index_piece(self):
        """To lock index piece in the index board"""
        if(self.shape!=None):

            XL = self.shape[0][0]
            YL = self.shape[0][1]
            XR = self.shape[1][0]
            YR = self.shape[1][1]

            for x in range(XL, XR+1):
                for y in range(YL, YR+1):
                    self.indexboard[x][y] = self.shape_index

    def get_available_blocks(self):

        return np.array([[len(self.shape_list) for _ in range(self.width)] for _ in range(self.height)])

    def get_connections(self):

        connections = [[0 for _ in range(self.width)] for _ in range(self.height)]

        # Store positions of each macro ids
        indexes = {}

        if(self.shape_index == -1):
            return np.array(connections)

        for i in range(self.width):
            for j in range(self.height):
                key = self.indexboard[i][j]
                if key not in indexes:
                    indexes[key] = []
                indexes[key].append((i,j))

        for i in self.adjency_list[self.shape_index]:
            if i in indexes:
                for pos in indexes[i]:
                    x, y = pos
                    connections[x][y] = self.adjency_matrix[self.shape_index][i]

        return np.array(connections)

    def get_current_state(self, board):

        available_blocks = self.get_available_blocks()
        connections = self.get_connections()/10

        return np.stack((board, connections), axis=-1).flatten()


    def get_next_states(self):
        """To get all possible state from current shape"""
        states = {}
        if(self.shape!=None):
            self._set_piece(True)
            states[self.shape] = self.get_current_state(self.board[:])
            self._set_piece(False)
        # Loop to try each posibilities
        
        return states

    def render(self, score, string=None, view=0):
        
        #if(view == 0):
        #    return

        score = self._calculate_wirelength()
        origboard = self.board

        indexboard = np.array(self.indexboard)
        # displayboard = indexboard
        indexboard = indexboard.reshape(-1,1)
        unique_vals = np.unique(indexboard)

        #board1 = [[green if len(self.indexboardshape[i][j]) >= 12 else black for j in range(self.width)] for i in range(self.height)]
        #board2 = [[blue if len(self.indexboardshape[i][j]) < 12 else board1[i][j] for j in range(self.width)] for i in range(self.height)]
        #board3 = [[yellow if len(self.indexboardshape[i][j]) <= 4 else board2[i][j] for j in range(self.width)] for i in range(self.height)]
        #board4 = [[brown if len(self.indexboardshape[i][j]) <= 1 else board3[i][j] for j in range(self.width)] for i in range(self.height)]
        #board = [[black if len(self.indexboardshape[i][j]) == '' else board4[i][j] for j in range(self.width)] for i in range(self.height)]

        board = np.array(self.indexboard)[:]
        board = [[color_list[board[i][j]] for j in range(self.width)] for i in range(self.height)]

        img = np.array(board).reshape((self.height, self.width, 3)).astype(np.uint8)
        img = cv.resize(img, (self.width * 25, self.height * 25), interpolation=cv.INTER_NEAREST)

        # To draw lines every 25 pixels
        img[[i * 25 for i in range(self.height)], :, :] = 0
        img[:, [i * 25 for i in range(self.width)], :] = 0
        # FLY LINE CODE

        # if(len(indexes) == 22):

        points_dict = {}

        onIndexes = {}
        for i in range(self.width):
            for j in range(self.height):
                if origboard[i][j] == True:
                    key = self.indexboard[i][j]
                    if key not in onIndexes:
                        onIndexes[key] = []
                    onIndexes[key].append((i,j))
                    points_dict[key] = [i,j]

        #print(points_dict)
        new_matrix = np.array(self.adjency_matrix)
        edgelist = {}

        for i in range(new_matrix.shape[0]):
            for j in range(new_matrix.shape[1]):
                # if(new_matrix[i][j] != 0 and (j,i) not in edgelist.keys()):
                if(new_matrix[i][j] != 0):
                    edgelist[(i,j)] = new_matrix[i][j]
        

        if(points_dict):
            for key,value in edgelist.items():
                if(key[0] in unique_vals and key[1] in unique_vals):
                    y1 = points_dict[key[0]][0]
                    x1 = points_dict[key[0]][1]
                    y2 = points_dict[key[1]][0]
                    x2 = points_dict[key[1]][1]
                    # print(x1,y1,x2,y2)
                    line_weight = value
                    
                    scale_factor = 25
                    x_position_1 = int(np.mean([x1,x2],axis=0).item())*scale_factor
                    y_position_1 = int(np.mean([y1,y2],axis=0).item())*scale_factor
                    pos1 = (x_position_1,y_position_1)
                    
                    color = (255,255,255)
                    img = cv.line(img,(int(x1*scale_factor),int(y1*scale_factor)),((int(x2*scale_factor),int(y2*scale_factor))),color,1,lineType=cv.LINE_AA)
                    img = cv.putText(img,str(line_weight),pos1,cv.FONT_HERSHEY_SIMPLEX,0.3,color,1)

        # To draw lines every 25 pixels
        img[[i * 25 for i in range(self.height)], :, :] = 0
        img[:, [i * 25 for i in range(self.width)], :] = 0

        # Add extra spaces on the top to display game score
        extra_spaces = np.zeros((2 * 25, self.width * 25, 3))
        cv.putText(extra_spaces, "Score: " + str(score), (15, 35), cv.FONT_HERSHEY_SIMPLEX, 1, white, 2, cv.LINE_AA)

        # Add extra spaces to the board image
        img = np.concatenate((extra_spaces, img), axis=0)

        # Draw horizontal line to separate board and extra space area
        img[50, :, :] = white

        if(view == 1):
            cv.imshow('DQN DeepPlace', img)
            cv.waitKey(5)
        else:
            cv.imwrite('./images_train/DQN' + str(string) + '.jpg', img)


# Initialize Placement enviroment
env = DeepPlace(args.width, args.height, 0)

# Initialize training variable
max_episode = args.numofgames
max_steps = args.width * args.height * 8 # 8 orientations

agent = Agent(env.state_size, args.numofgames, args.discount)

if(args.load_model):
    agent.load_model(args.saved_path + '/model.h5')

episodes = []
rewards = []

current_max = 0

start = 0

for episode in range(start, max_episode):
    for fplan_type in ["min", "max"]:

        env.filename = fplan_type + str(episode)
        current_state = env.reset()
        done = False
        steps = 0
        total_reward = 0
        env._new_piece()
        print("Running episode " + str(episode))

        while not done and steps < max_steps:   
            # Render the board for visualization
            #env.render(total_reward, env.filename + "_" + str(steps), args.view_placement )

            # Get all possible tetromino placement in current board
            next_states = env.get_next_states()

            # If the dict is empty, meaning game is over
            if not next_states:
                next_state = env.reset()
                done = True
                reward = -1
                total_reward += reward
                agent.add_to_memory(current_state, next_state, reward, done)
                break
                
            # Tell agent to choose the best possible state
            best_state = agent.act(next_states.values())

            # Grab best tetromino position and its rotation chosen by the agent
            best_action = None
            for action, state in next_states.items():
                if (best_state==state).all():
                    best_action = action
                    break

            reward, done = env.step(best_action)
            total_reward += reward

            # Add to memory for replay
            if fplan_type == "min":
                reward_adjust = -0.1
            else:
                reward_adjust = -0.9
            agent.add_to_memory(current_state, next_states[best_action], reward_adjust, done)

            # Set current new state 
            current_state = next_states[best_action]

            steps += 1

        print("Total reward: " + str(total_reward))
        episodes.append(episode)
        rewards.append(total_reward)

        if((episode+1)%args.replay_interval==0):
            agent.replay()

        if((episode+1)%args.save_interval==0):
            agent.save_model(args.saved_path + '/model.h5')


        if agent.epsilon > agent.epsilon_min:
            agent.epsilon -= agent.epsilon_decay



def plot_running_avg(totalrewards):
    # pdb.set_trace()
    df = pd.Series(totalrewards)
    df1 = df.rolling(window=10).mean()
    print("Running Average")
    print(df1)

plot_running_avg(rewards)

