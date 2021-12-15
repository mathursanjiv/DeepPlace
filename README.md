# DeepPlace
VLSI Block Placement using Reinforcement Learning

This is an attempt to use Reinforcement Learning to place rectangular blocks in a rectangular die area. The technique uses training to prepare a model based on training data set to minimize wirelength and come up with legal floorplan. Use "make gen" to generate test floorplans for training and then "make train" to train the model. Then "make test" can be used to try model on unseen floorplans and place blocks using RL. WL of generated floorplan is compared with that of reference.

This is an early development and can be used a foundation to build matured 2D placement engine sensitive to wire length, congestion, channel spacing, alignment etc.

Demo
https://github.com/mathursanjiv/DeepPlace/blob/main/fp18.gif
