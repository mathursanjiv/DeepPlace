import imageio
import os
from os import path
import argparse

parser = argparse.ArgumentParser("""Generate Gifs""")
parser.add_argument("--filename", type=int, default=0)
parser.add_argument("--numofgames", type=int, default=100, help="Number of placement")


args = parser.parse_args()


def create_gif(basename, filenames, duration): 
    images = [] 
    for filename in filenames: 
        images.append(imageio.imread(filename)) 
    output_file = '%s.gif' % basename 
    imageio.mimsave(output_file, images, duration=duration)


max_episode = args.numofgames
max_steps = 100

for episode in range(max_episode):

    filenames = []
    steps = 0
    done = False
    while not done and steps < max_steps:   
        # Render the board for visualization
        string = str(args.filename) + str(episode) + '_' + str(steps)
        filename = './images_test/DQN' + str(string) + '.jpg'
        if path.exists(filename):
           filenames.append(filename)
        else:
           done = True
           if len(filenames) > 1 :
              print("Creating for ", './images_test/DQN' + '_' + str(args.filename) + '_' + str(episode))
              create_gif('./images_test/DQN' + '_' + str(args.filename) + '_' + str(episode), filenames, 0.5)

        steps += 1


  
