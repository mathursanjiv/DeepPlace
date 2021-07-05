import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import namedtuple
from itertools import product
from functools import reduce
import random
import pickle as pk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--grid_size", type=int, default=20)
parser.add_argument("--min_size", type=int, default=1)
parser.add_argument("--max_size", type=int, default=4)
parser.add_argument("--train_size", type=int, default=10000)
parser.add_argument("--test_size", type=int, default=100)
args = parser.parse_args()


Block = namedtuple("Block", ['Type', 'Loc', 'Width', 'Height', 'Rotated', 'Id', 'BlockId', 'BBox'])

def manhanttan_distance(x, y):
        x1, x2 = x
        y1, y2 = y
        return abs(x1-y1) + abs(x2-y2)

def getBlockDistMatrix(inpBlocks, min_max) :
    placeableBlocks =[x for x in inpBlocks if x.BlockId != -1]
    distDict = {}
    indexes = {}

    for block1 in placeableBlocks :
        b1 = block1.BBox
        XL = b1[0][0]
        YL = b1[0][1]
        XR = b1[1][0]-1
        YR = b1[1][1]-1

        if(block1.BlockId not in indexes):
            indexes[block1.BlockId] = []

        for x in range(XL, XR+1):
            for y in range(YL, YR+1):
                indexes[block1.BlockId].append((x, y))

    for block1 in placeableBlocks :
        list1 = indexes[block1.BlockId]
        for block2 in placeableBlocks :
            list2 = indexes[block2.BlockId]
            if(block1.BlockId != block2.BlockId):
                dist = float('inf')
                if min_max == "max":
                    dist = 0
                for k in list1:
                    for l in list2:
                        if min_max == "min":
                            dist = min(dist, manhanttan_distance(k, l))
                        else:
                            dist = max(dist, manhanttan_distance(k, l))
                distDict[(block1.BlockId), (block2.BlockId)] = dist
    return distDict

def getOptConnGraph(blocks, edgeConnPercent, min_max) :
    placeableBlocks =[x for x in blocks if x.BlockId != -1]
    distMatrix = getBlockDistMatrix(blocks, min_max)
    dists = list(distMatrix.items())
    blockIds = [x.BlockId for x in placeableBlocks]
    optGraph = nx.Graph()
    optGraph.add_nodes_from(blockIds)
    if min_max == "min":
        optEdges = sorted(dists, key=lambda x : x[1])[0:int(len(dists)*edgeConnPercent)]
    if min_max == "max":
        optEdges = sorted(dists, key=lambda x : x[1], reverse=True)[0:int(len(dists)*edgeConnPercent)]
    optDist = 0
    for e in optEdges :
        optGraph.add_edge(*e[0])
        optDist += e[1]
    return optGraph, optDist


def generateBoard(boardShape, rowIds, colIds, typeDistr=None, 
                  edgeConnPercent=0.4):
    board = np.zeros(boardShape, dtype=int)
    startRow = 0
    rows = np.hstack([rowIds, boardShape[0]])
    cols = np.hstack([colIds, boardShape[-1]])
    objTypes = np.random.choice(['Blockage', 'WhiteSpace', 'Block'], len(rows)*len(cols), p=typeDistr)
    blocks = []
    blockId = 1 
    objId = 0
    for row in rows :
        startCol = 0
        for col in cols :
            blockType = objTypes[objId]
            curBlockId = -1
            if blockType == 'Block' :
                board[startRow:row, startCol:col] = blockId
                curBlockId = blockId
                blockId += 1
            elif blockType == 'Blockage' :
                board[startRow:row, startCol:col] = -1
            rotateBlock = np.random.choice([False], 1)[0]
            blockWidth = row-startRow
            blockHeight = col-startCol
            if rotateBlock :
                blockWidth, blockHeight = blockHeight, blockWidth
            curBlock = Block(Type=blockType, Loc=(startRow, startCol), 
                             Width=blockWidth, Height=blockHeight, 
                             Rotated=rotateBlock, 
                             Id=objId, BlockId=curBlockId,BBox=((startRow, startCol), (row, col)))
            blocks.append(curBlock)
            startCol = col
            objId += 1
        startRow = row
    placeableBlocks =[x for x in blocks if x.BlockId != -1]
    return (board, blocks, getOptConnGraph(blocks, edgeConnPercent, "min"), getOptConnGraph(blocks, edgeConnPercent, "max"))

def writeBlocks(baseDir, baseName, num, initialBlock_, blocks_, connGraphs_):
    pk.dump(initialBlock_, open(baseDir + baseName + str(num) + ".pkl","wb"))
    blocks_.sort(key=lambda x: int(x[0]), reverse=True)

    f = open(baseDir + baseName + str(num) + ".txt", "w")
    f.write("Macros " + str(len(blocks_)) + "\n")
    for block in blocks_:
        _, BBox, Id = block

        XL = BBox[0][0]
        YL = BBox[0][1]
        XR = BBox[1][0]-1
        YR = BBox[1][1]-1
        s = str(Id-1) + " " + str(XL) + " " + str(YL) + " " + str(XR) + " " + str(YR) + "\n"
        f.write(s)

    f.write("Edges " + str(len(connGraphs_[0].edges)) + " " + str(connGraphs_[1]) + "\n")
    for edge in connGraphs_[0].edges:
        s = str(edge[0]-1) + " " + str(edge[1]-1) + " " + str(1) + "\n"
        f.write(s)

    f.close()



trainSize = args.train_size

for i in range(trainSize):
    if i%1000 == 0:
        print("Done ", i)
    horizontalLines = []
    number = 0
    while(True):
        number += random.randint(args.min_size, args.max_size)
        if(number < args.grid_size):
            horizontalLines.append(number)
        else:
            break

    verticalLines = []
    number = 0
    while(True):
        number += random.randint(args.min_size, args.max_size)
        if(number < args.grid_size):
            verticalLines.append(number)
        else:
            break


    boardN, blocksN, connGraphsMin, connGraphsMax = generateBoard((args.grid_size, args.grid_size), horizontalLines, verticalLines, [0.2, 0.55, 0.25], 0.25)

    blockages, blocks = [], []

    for block in blocksN:

        if(block.Type == 'Block'):
            blocks.append((block.Width*block.Height, block.BBox, block.BlockId))
        if(block.Type == 'Blockage'):
            blockages.append(block.BBox) 


    initialBlock = [[False for _ in range(args.grid_size)] for _ in range(args.grid_size)]

    for block in blockages:
        XL = block[0][0]
        YL = block[0][1]
        XR = block[1][0]-1
        YR = block[1][1]-1

        for x in range(XL, XR+1):
            for y in range(YL, YR+1):
                initialBlock[x][y] = True

    writeBlocks("train_data/", "min", i, initialBlock, blocks, connGraphsMin)
    writeBlocks("train_data/", "max", i, initialBlock, blocks, connGraphsMax)




testSize = args.test_size
print("Generating test data.")
for i in range(testSize):

    horizontalLines = []
    number = 0
    while(True):
        number += random.randint(args.min_size, args.max_size)
        if(number < args.grid_size):
            horizontalLines.append(number)
        else:
            break

    verticalLines = []
    number = 0
    while(True):
        number += random.randint(args.min_size, args.max_size)
        if(number < args.grid_size):
            verticalLines.append(number)
        else:
            break

    boardN, blocksN, connGraphs, connGraphsMax = generateBoard((args.grid_size, args.grid_size), horizontalLines, verticalLines, [0.2, 0.55, 0.25], 0.4)

    blockages, blocks = [], []

    for block in blocksN:

        if(block.Type == 'Block'):
            blocks.append((block.Width*block.Height, block.BBox, block.BlockId, block.Width, block.Height))
        if(block.Type == 'Blockage'):
            blockages.append(block.BBox)

    initialBlock = [[False for _ in range(args.grid_size)] for _ in range(args.grid_size)]

    for block in blockages:
        XL = block[0][0]
        YL = block[0][1]
        XR = block[1][0]-1
        YR = block[1][1]-1

        for x in range(XL, XR+1):
            for y in range(YL, YR+1):
                initialBlock[x][y] = True

    #writeBlocks("test_data/", "block_", i, initialBlock, blocks, connGraphsMin)
    pk.dump(initialBlock, open("test_data/block_" + str(i) + ".pkl","wb"))
    blocks.sort(key=lambda x: int(x[0]), reverse=True)

    f = open("test_data/block_" + str(i) + ".txt", "w")
    f.write("Macros " + str(len(blocks)) + "\n")
    for block in blocks:
        area , BBox, Id, width, height = block

        XL = BBox[0][0]
        YL = BBox[0][1]
        XR = BBox[1][0]-1
        YR = BBox[1][1]-1
        
        s = str(Id-1) + " " + str(width) + " " + str(height) + "\n"
        f.write(s)

    f.write("Edges " + str(len(connGraphs[0].edges)) + " " + str(connGraphs[1]) + "\n")
    for edge in connGraphs[0].edges:
        s = str(edge[0]-1) + " " + str(edge[1]-1) + " " + str(1) + "\n"
        f.write(s)

    f.close()


