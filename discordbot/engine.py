import torch
import cv2
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import math
import chess

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import re
import tensorflow as tf
import chess.svg
import sys

nFEN = str(sys.argv[1])
print(nFEN)
print("brav")

fenSplits = re.split(" ", nFEN)
nturn = fenSplits[1]
nstr = "lmao"

board = chess.Board(nFEN)

def strfix(fen, tr):
    
    fstr = str(fen)
    
    if '#' in str(tr):
        if '-' in tr:
            t = -10000
        else:
            t = 10000
    elif '\ufeff+23' in str(tr):
        t = 0
    else:
        t = int(tr)
    if "b" in nturn:
        t = t*-1
    t = t/10

    return t

def pieces_to_bits(boardstr):
    for i, char in enumerate(boardstr):
        if char == 'P':
            boardstr = boardstr.replace(char, "|1")
        elif char == 'p':
            boardstr = boardstr.replace(char, "|-1")
        elif char == 'N':
            boardstr = boardstr.replace(char, "|3")
        elif char == 'n':
            boardstr = boardstr.replace(char, "|-3")
        elif char == 'B':
            boardstr = boardstr.replace(char, "|4")
        elif char == 'b':
            boardstr = boardstr.replace(char, "|-4")
        elif char == 'Q':
            boardstr = boardstr.replace(char, "|9")
        elif char == 'q':
            boardstr = boardstr.replace(char, "|-9")
        elif char == 'K':
            boardstr = boardstr.replace(char, "|100")
        elif char == 'k':
            boardstr = boardstr.replace(char, "|-100")
        elif char == 'R':
            boardstr = boardstr.replace(char, "|5")
        elif char == 'r':
            boardstr = boardstr.replace(char, "|-5")
        elif char == '.':
            boardstr = boardstr.replace(char, "|0")
    boardstr = boardstr.replace("'", " ")
    boardstr = boardstr.replace("|", ",")
    boardstr = boardstr.replace("\n", "")
    boardstr = boardstr.replace(" ", "")
    boardstr = boardstr[1:]
    boardstr = eval(boardstr)
    boardstr = list(boardstr)

    return boardstr

def board_to_bits(fen):
    components = re.split(" ", fen[0])
    turn = components[1]
    castlingRights = components[2]
    enPassant = components[3]
    board = chess.Board(fen[0])
    boardString = str(board)

    boardString = pieces_to_bits(boardString)
    # print(boardString)

    if 'K' in castlingRights:
        WCK = 1
    else:
        WCK = 0
    if 'Q' in castlingRights:
        WCQ = 1
    else:
        WCQ = 0
    if 'k' in castlingRights:
        BCK = 1
    else:
        BCK = 0
    if 'k' in castlingRights:
        BCQ = 1
    else:
        BCQ = 0
    blackAux = [BCK, BCQ]
    whiteAux = [WCK, WCQ]
    
    if "w" not in turn:
        for i in range(len(boardString)):
            boardString[i] = -1*boardString[i]
        temp = blackAux
        blackAux = whiteAux
        whiteAux = temp


    return blackAux + whiteAux + boardString

boardInfoMatrix = board_to_bits([nFEN, 0])
print(boardInfoMatrix)

bitMatrix = np.array([boardInfoMatrix[4:]])
auxMatrix = np.array([boardInfoMatrix[:4]])

loaded_model = tf.keras.models.load_model("./engine01.keras")
legalMovesTuples = list(board.legal_moves)
legalMoves = []

for i, moves in enumerate(legalMovesTuples):
    legalMoves.append(str(moves))

potentialPositions = []

for i, move in enumerate(legalMoves):
    val = chess.Move.from_uci(move)
    board.push(val)
    potentialPositions.append(board.fen())
    board.pop()

for i, potential in enumerate(potentialPositions):
    potentialSplits = re.split(" ", potential)
    if nstr == "b":
        potentialSplits[1] = potentialSplits[1].replace("b", "w")
    elif nstr == "w":
        potentialSplits[1] = potentialSplits[1].replace("w", "b")
    else:
        if potentialSplits[1] == "b":
            potentialSplits[1] = potentialSplits[1].replace("b", "w")
        else:
            potentialSplits[1] = potentialSplits[1].replace("w", "b")
    potentialPositions[i] = " ".join(potentialSplits)

predictions = []

for i, position in enumerate(potentialPositions):
    # print(position)
    boardInfoMatrix = board_to_bits([position, 0])
    bitMatrix = np.array([boardInfoMatrix[4:]])
    auxMatrix = np.array([boardInfoMatrix[:4]])
    predictions.append(loaded_model.predict([(bitMatrix, (auxMatrix))]))

# potentialPositions

if len(predictions) == 0:
    file1 = open('./cache/FEN.txt',"w")
    file1.write('CHECKMATE')
    file1.close
else:
    index = predictions.index(max(predictions))
    print("best move evaluation", predictions[index])
    print(predictions[index])

    returnBoard = chess.Board(potentialPositions[index])
    boardsvg = chess.svg.board(board=returnBoard)
    outputfile = open('./cache/temp.svg', "w")
    outputfile.write(boardsvg)
    outputfile.close()

    file1 = open('./cache/FEN.txt',"w")

    brev = returnBoard.fen()
    brevSplits = re.split(" ", brev)
    # brevSplits[1] = nturn
    if nturn == "b":
        brevSplits[1] = brevSplits[1].replace("b", "w")
    elif nturn == "w":
        brevSplits[1] = brevSplits[1].replace("w", "b")
    else:
        print("lmao")
    brevJoined = " ".join(brevSplits)

    print(brevSplits[1])

    file1.write(brevJoined)
    file1.close()
    print('FENBOARD', brevJoined)