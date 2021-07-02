from agent import Agent
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import cv2 as cv
import random
from tabulate import tabulate
import pickle as pk

parser = argparse.ArgumentParser("""Implementation of Deep Q Network for block placement""")
parser.add_argument("--width", type=int, default=20, help="The common width for all images")
parser.add_argument("--height", type=int, default=20, help="The common height for all images")
parser.add_argument("--numofgames", type=int, default=100, help="Number of placement")
parser.add_argument("--discount", type=float, default=0.95)
parser.add_argument("--save_interval", type=int, default=100000000000)
parser.add_argument("--replay_interval", type=int, default=10)
parser.add_argument("--saved_path", type=str, default="trained_models")
parser.add_argument("--load_model", type=int, default=1)
parser.add_argument("--view_placement", type=int, default=0)
parser.add_argument("--filename", type=int, default=0)

args = parser.parse_args()

def file_parser(filename='block.txt'):

    count = 0
    macros = 0
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
                macros = 0
                numberOfMacros = int(splitLine[1])
                continue
            if(splitLine[0] == 'Edges'):
                macros = 1
                numberOfEdges = int(splitLine[1])
                OptimalWL = int(splitLine[2])
                adjency_matrix = [[0 for x in range(len(shapes))] for y in range(len(shapes))]
                adjency_list = [[] for _ in range(len(shapes))]
                continue

            
            if(macros == 0):
                # macro
                index = int(splitLine[0])
                width = int(splitLine[1])
                height = int(splitLine[2])
                shape = []
                for w in range(width):
                    for h in range(height):
                        shape.append((-w, -h))

                shapes[index] = shape

            else:
                # edge

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
black = (0, 0, 0)
white = (255, 255, 255)


def rotated(shape):
    return [(-j, i) for i, j in shape]

def is_occupied(shape, anchor, board):
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        if y < 0 or x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
            return True
    return False

def get_action_index(x, y, width, height, rotation):
    return (width*height*rotation) + y*width + x

def get_action_position(action_index, width, height):
    rotation = action_index//(width*height)
    action_index = action_index%(width*height)
    y = action_index//width
    action_index = action_index%width
    x = action_index
    return (x, y, rotation)


# Environment
class DeepPlace:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=float)
        self.indexboard =  [[-1 for _ in range(width)] for _ in range(height)]
        self.dir_prefix = "test_data/block_"
        #self.dir_prefix = "train_test_data/block_"

        self.state_size = self.width * self.height *2 

        # For running the engine
        self.score = -1
        self.anchor = None
        self.shape = None
        self.shape_index = -1

        # Reset after initializing
        self.reset()

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
        self.shapes, self.adjency_matrix, self.adjency_list = file_parser(self.dir_prefix + str(args.filename) + ".txt")
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
        pos = (action[0], action[1])

        # Rotate shape n times
        for rot in range(action[2]):
            self.shape = rotated(self.shape)

        self.anchor = pos
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
        self.score = 0
        self._reset_piece()
        self._parse_shape()
        self.board = np.zeros_like(self.board)
        self.indexboard =  [[-1 for _ in range(self.width)] for _ in range(self.height)]

        initBoard = pk.load(open(self.dir_prefix + str(args.filename) + ".pkl", "rb"))

        for i in range(self.width):
            for j in range(self.height):
                self.board[i, j] = initBoard[i][j]


        return np.stack((self.board, np.array([[0 for _ in range(self.width)] for _ in range(self.height)])), axis=-1).flatten()


    def _set_piece(self, on):
        """To lock a piece in the board"""
        if(self.shape!=None and self.anchor!=None):
            for i, j in self.shape:
                x, y = i + self.anchor[0], j + self.anchor[1]
                if x < self.width and x >= 0 and y < self.height and y >= 0:
                    self.board[int(self.anchor[0] + i), int(self.anchor[1] + j)] = on
                else:
                    pass
                    # print(x, y, self.width, self.height)
                    # print("Debug - Some Error in Next State and Current State Functions")


    def _set_index_piece(self):
        """To lock index piece in the board"""
        if(self.shape!=None and self.anchor!=None):
            for i, j in self.shape:
                x, y = i + self.anchor[0], j + self.anchor[1]
                if x < self.width and x >= 0 and y < self.height and y >= 0:
                    self.indexboard[int(self.anchor[0] + i)][int(self.anchor[1] + j)] = self.shape_index
                else:
                    pass
                    # print(x, y, self.width, self.height)
                    # print("Debug - Some Error in Next State and Current State Functions")


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


    def get_next_actions(self):
        """To get all possible actions from current state"""
        # Loop to try each posibilities

        cur_shape = self.shape
        rotations = 2
        action_mask = [0 for _ in range(self.width*self.height*rotations)]

        for rotation in range(rotations):
            for x in range(0, self.width):
                for y in range(0, self.height):
                    if(not is_occupied(cur_shape, (x, y), self.board)):
                        action_mask[get_action_index(x, y, self.width, self.height, rotation)] = 1
            cur_shape = rotated(cur_shape)
        return action_mask

    def render(self, score, string=None, view=0):

        #if(view == 0):
            #return

        score = self._calculate_wirelength()
        origboard = self.board

        indexboard = np.array(self.indexboard)
        # displayboard = indexboard
        indexboard = indexboard.reshape(-1,1)
        unique_vals = np.unique(indexboard)


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

        # Add extra spaces on the top to display game score
        extra_spaces = np.zeros((2 * 25, self.width * 25, 3))
        cv.putText(extra_spaces, "Wire Length: " + str(score), (15, 35), cv.FONT_HERSHEY_SIMPLEX, 1, white, 2, cv.LINE_AA)

        # Add extra spaces to the board image
        img = np.concatenate((extra_spaces, img), axis=0)

        # Draw horizontal line to separate board and extra space area
        img[50, :, :] = white

        if(view == 1):
            cv.imshow('DQN DeepPlace', img)
            cv.waitKey(1)
        else:
            cv.imwrite('./images_test/DQN' + str(string) + '.jpg', img)



# Initialize Placement enviroment
env = DeepPlace(args.width, args.height)

# Initialize training variable
max_episode = args.numofgames
max_steps = args.width * args.height * 8 # 8 orientations

agent = Agent(args.width, args.height, 2, 2, args.numofgames, args.discount) # 2 layers in convolution and #orientations
agent.epsilon = 0

if(args.load_model):
    print('Loading model...')
    agent.load_model(args.saved_path + '/model.h5')
    print('Done.')

episodes = []
rewards = []
current_max = 0

for episode in range(max_episode):
    current_state = env.reset()
    done = False
    steps = 0
    total_reward = 0
    env._new_piece()
    print("Running episode " + str(episode))
    agent_memory = []

    while not done and steps < max_steps:   
        # Render the board for visualization
        env.render(total_reward, str(args.filename) + str(episode) + "_" + str(steps), args.view_placement )

        #print("Running step " + str(steps))

        # Get all possible tetromino placement actions in the current board
        action_mask = env.get_next_actions()

        # If all entries are 0, meaning no action possible
        if sum(action_mask) == 0:
            done = True
            reward = -1
            total_reward += reward
            agent_memory.append([current_state, 0, reward, current_state, done])
            break
            
        # Tell agent to choose the best possible action
        best_action = agent.act(current_state, np.array(action_mask))

        reward, done = env.step(get_action_position(best_action, env.width, env.height))
        total_reward += reward
        #print('Current Reward = {}', reward)

        next_state = env.get_current_state(env.board)


        # Add to memory for replay
        agent_memory.append((current_state, best_action, reward, next_state, done))

        # Set current new state 
        current_state = next_state

        steps += 1

    print("Total reward: " + str(total_reward))
    episodes.append(episode)
    rewards.append(total_reward)

    reward_adjustment_factor = agent_memory[len(agent_memory) -1][2]*(args.discount**(len(agent_memory) -1))


    # Only train using equal or better results. 
    if(total_reward >= rewards[0]):
        for current_state, action, reward, next_state, done in agent_memory:
            agent.add_to_memory(current_state, action, reward_adjustment_factor, next_state, done)
            reward_adjustment_factor/= args.discount

    if((episode+1)%args.replay_interval==0):
        print("Best WL: ", max(rewards)*1000, "Best WL episode: ", rewards.index(max(rewards)) + 1)
        agent.replay()

    if((episode+1)%args.save_interval==0):
        agent.save_model(args.saved_path + '/model.h5')


    #if agent.epsilon > agent.epsilon_min:
    #    agent.epsilon -= agent.epsilon_decay
    if agent.epsilon < 1:
        agent.epsilon += agent.epsilon_decay



def plot_running_avg(totalrewards):
    # pdb.set_trace()
    df = pd.Series(totalrewards)
    df1 = df.rolling(window=10).mean()
    print("Running Average")
    print(df1)

plot_running_avg(rewards)

f = open("results", "a")
f.write("Best WL: " + str(max(rewards)*1000) + " Number: " + str(args.filename) + "\n")
f.write("OneShot WL: " + str(rewards[0]*1000) + " Number: " + str(args.filename) + "\n")
f.close()
print("Best WL: ", max(rewards)*1000,"Best WL episode: ", rewards.index(max(rewards)) + 1)
print("OneShot WL: ", rewards[0]*1000,"OneShot WL episode: ", 1)

