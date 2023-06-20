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
import sys
import chess.svg

print("sex")
n = int(sys.argv[1])
print(n+1)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels=3, padding=2, out_channels=64, kernel_size=3)
        self.fc_layer1 = nn.Linear(54 * 54 * 64, 128)
        self.fc_layer2 = nn.Linear(128, 512)
        self.fc_layer3 = nn.Linear(512, 512)
        self.fc_layer4 = nn.Linear(512, 13)

    def forward(self, x):
        # input: 32x1x28x28 -> output: 32x32x26x26
        x = self.conv_layer(x)
        x = F.relu(x)

        # input: 32x32x26x26 -> output: 32x(32*26*26)
        x = x.flatten(start_dim = 1)

        # input: 32x(32*26*26) -> output: 32x128
        x = self.fc_layer1(x)
        x = F.relu(x)

        x = self.fc_layer2(x)
        x = F.relu(x)

        x = self.fc_layer3(x)
        x = F.relu(x)

        # input: 32x128 -> output: 32x13 (32 images, 13 classes)
        x = self.fc_layer4(x)
        out = F.softmax(x, dim=1)
        return out
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleModel()
model = model.to(device)

labels = ['blackbishop', 'blackking', 'blackknight', 'blackpawn', 'blackqueen', 'blackrook', 'nothing', 'whitebishop', 'whiteking', 'whiteknight', 'whitepawn', 'whitequeen', 'whiterook']
checkpoint = torch.load('./model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
    ])

def predict_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_image = image.copy()
    orig_image = cv2.resize(orig_image, (260, 260))

    # cv2.imshow('Result', orig_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image = transform(image)

    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image.to(device))
    output_label = torch.topk(outputs, 1)
    pred_class = labels[int(output_label.indices)]

    return(pred_class)
    # print('prediction=', pred_class)


# image = cv2.imread('./testboards/wking-modified.jpg')
# predict_img(image)

def classify_lines(lines):
    # print(lines)
    h=[]
    v=[]
    for i in range(lines.shape[0]):
        if(abs(lines[i][0][0]-lines[i][0][2]) < 100):
            h.append(lines[i])
        else:
            v.append(lines[i])

    return h, v
        # print(lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3])

def get_lines(lines_in):
    if cv2.__version__ < '3.0':
        return lines_in[0]
    return [l[0] for l in lines_in]

def merge_lines_pipeline_2(lines):
    super_lines_final = []
    super_lines = []
    min_distance_to_merge = 30
    min_angle_to_merge = 30
    
    for line in lines:
        create_new_group = True
        group_updated = False

        for group in super_lines:
            for line2 in group:
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        #print("angles", orientation_i, orientation_j)
                        #print(int(abs(orientation_i - orientation_j)))
                        group.append(line)

                        create_new_group = False
                        group_updated = True
                        break
            
            if group_updated:
                break

        if (create_new_group):
            new_group = []
            new_group.append(line)

            for idx, line2 in enumerate(lines):
                # check the distance between lines
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        #print("angles", orientation_i, orientation_j)
                        #print(int(abs(orientation_i - orientation_j)))

                        new_group.append(line2)

                        # remove line from lines list
                        #lines[idx] = False
            # append new group
            super_lines.append(new_group)
        
    
    for group in super_lines:
        super_lines_final.append(merge_lines_segments1(group))
    
    return super_lines_final

def merge_lines_segments1(lines, use_log=False):
    if(len(lines) == 1):
        return lines[0]
    
    line_i = lines[0]
    
    # orientation
    orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
    
    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])
        
    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
        
        #sort by y
        points = sorted(points, key=lambda point: point[1])
        
        if use_log:
            print("use y")
    else:
        
        #sort by x
        points = sorted(points, key=lambda point: point[0])
        
        if use_log:
            print("use x")
    
    return [points[0], points[len(points)-1]]

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
# https://stackoverflow.com/questions/32702075/what-would-be-the-fastest-way-to-find-the-maximum-of-all-possible-distances-betw
def lines_close(line1, line2):
    dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
    dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
    dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
    dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])
    
    if (min(dist1,dist2,dist3,dist4) < 100):
        return True
    else:
        return False
    
def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude
 
#Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
# https://nodedangles.wordpress.com/2010/05/16/measuring-distance-from-a-point-to-a-line-segment/
# http://paulbourke.net/geometry/pointlineplane/
def DistancePointLine(px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)
 
    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine
 
    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)
 
    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)
 
    return DistancePointLine

def get_distance(line1, line2):
    dist1 = DistancePointLine(line1[0][0], line1[0][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = DistancePointLine(line1[1][0], line1[1][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = DistancePointLine(line2[0][0], line2[0][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = DistancePointLine(line2[1][0], line2[1][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    
    return min(dist1,dist2,dist3,dist4)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

# someimage = './2D-Chessboard-and-Chess-Pieces-4/valid/lichess3986__kPrQKQbk-BRR1q11B-bQk11BNk-NrBqbrrb-qBpQNn1K-qn1qqnqB-K1pbQnBP-pKpRKrbK_png.rf.764a007ff332038a44e9fbb2a9aadbcd.jpg'
# someimage = './1_r6UCJiab1dAUJwdjKXTtoA.png'
# someimage = './2D-Chessboard-and-Chess-Pieces-4/train/lichess1815__BkqrBB1r-BqnBbKQp-NbkRnPrp-pBqkRq1B-RbkrRPKP-pQpKNkpN-kbPnKPBK-bprPBpRQ_png.rf.694738e2c406fdc7bac8376a15521411.jpg'
# someimage = './2D-Chessboard-and-Chess-Pieces-4/train/lichess0482__NKqNPQqQ-rrPrkpqR-1RKKQBqQ-NnknBrRk-NNQbBN1n-qPqKNpQN-BNNpqQPB-n1Rb1PnK_png.rf.30d1a1ec195d181a341cc33d941da378.jpg'
# someimage = './2D-Chessboard-and-Chess-Pieces-4/test/lichess3442__NrrNpPQ1-q1BNRQpP-PRKqbqnn-nqQNBQbN-kBQKQnQp-RBbQqkQR-bQp1NNpK-k1nkqrkq_png.rf.a212fcb70316e88c3a4713657d68a1d5.jpg'
# someimage = './2D-Chessboard-and-Chess-Pieces-4/valid/lichess0289__RqrQkRQN-KQbnPPPn-BqBB1npP-RBPRRPbp-rpK1kBRq-QQBpNBP1-NPQrNRnq-nNNkQrbP_png.rf.dc03d6959667daf88a96b4c111c65050.jpg'
# someimage = './2D-Chessboard-and-Chess-Pieces-4/valid/lichess3360__qKBRBQNR-QBbprkpR-KBqpNpnq-BrNKK1np-bNRbrBQQ-KBkP1KrN-bn1qppnB-KQqnPbQP_png.rf.5ba6bbc2774d54308dbaa3a150a1e953.jpg'
# someimage = './2D-Chessboard-and-Chess-Pieces-4/valid/lichess3735__11RKNnpQ-Brq1KrkN-RNbKkPQQ-QrKnrRRn-pRRBnQnp-BbqRbQ1B-K1NRBPRk-QK11P1Kr_png.rf.8503fe07372da49be62817b6b2f08b49.jpg'
# someimage = './2D-Chessboard-and-Chess-Pieces-4/valid/lichess0262__kQnRNkRq-PBNKRpqk-pqPbBrpb-kKp1qrNp-kKqNrKkn-RKRQQPrK-QQKkPb1B-rRnRRp1q_png.rf.930e9010cdc9dfc83a19fd4438b3e22d.jpg'
# someimage = './2D-Chessboard-and-Chess-Pieces-4/valid/lichess1096__BQbqNNnK-1RPrRbpq-rrRbnrqQ-bnKKbQQP-BpPQPRkQ-BPrPBbQR-pKkpn1pP-KKKrNq1r_png.rf.c59778da5a19a56c65a1f0064f880f79.jpg'
# someimage = './2D-Chessboard-and-Chess-Pieces-4/test/lichess0325__kbnbPBNB-RKppQrrP-qrqkQQ1N-qpKpbQbP-nqnBqpNp-NKNRRqpr-NrnqRknp-QBrnKbPQ_png.rf.f72f267a36c622be75ff5cd9ba4b8abc.jpg'
# someimage = './2D-Chessboard-and-Chess-Pieces-4/train/lichess2734__1P1nb1KB-RkbBrbbQ-K1kqpBRQ-rBRNrNKB-BrrnPKnN-KKPbQBpR-PPQkPqqN-111rKPqq_png.rf.90d7c1955e4512391a600eed86703acb.jpg'
# someimage = './2D-Chessboard-and-Chess-Pieces-4/test/lichess3722__R1RqNKBk-rbkRPKBR-bn1RKkQN-nQBqPKBN-RbQbqPqN-NQbRbqBB-qqKbRbQq-pbkbQ1KB_png.rf.b08719fb4203d190932321c1c4273417.jpg'
# someimage = './2D-Chessboard-and-Chess-Pieces-4/train/lichess2434__rPQkBP1k-QBQqqnkQ-nNnPRQPn-ppKNKKBq-NKrBqqpr-kbPRkBNP-PPqKKQ1N-pqQqK1p1_png.rf.651d48ef27b5dc7a7915b20a651f9490.jpg'

someimage = './testboards/DkY6pMWXcAAhcPy.jpg'
# someimage = '.testboard/Tt1W59I.png'
# someimage = '.testboards/evz3bcddv7u21.png'
# randomdir = random.choice(os.listdir('./2D-Chessboard-and-Chess-Pieces-4/'))
# randomboard = random.choice(os.listdir('./2D-Chessboard-and-Chess-Pieces-4/' + randomdir))


# someimage = './2D-Chessboard-and-Chess-Pieces-4/' + randomdir + '/' + randomboard
# !!!!UNCOMMENT THE LINE ABOVE FOR A RANDOM IMAGE IN THE DATASET (might produce errors due to inaccurate line detection)

# print(someimage)

img_board = cv2.imread(someimage)
img_board = cv2.resize(img_board, (900, 900), interpolation=cv2.INTER_AREA)
# img_board = cv2.blur(img_board, (1,1))

img_board = cv2.copyMakeBorder(img_board, 25, 25, 25, 25, cv2.BORDER_WRAP)
img_board_gray = cv2.cvtColor(img_board, cv2.COLOR_BGR2GRAY)

s = 2
kernel = np.ones((s, s), np.uint8)
img_board_gray_grad = cv2.morphologyEx(img_board_gray, cv2.MORPH_GRADIENT, kernel)

thresh = 10
im_bw = cv2.threshold(img_board_gray_grad, thresh, 255, cv2.THRESH_BINARY)[1]

# edges = cv2.Canny(im_bw,40,150,apertureSize = 7)
kernel = np.ones((3,3),np.uint8)
edges = cv2.dilate(im_bw,kernel,iterations = 3)

lines = cv2.HoughLinesP(image=edges, rho = 2, theta = np.pi/90, threshold=200, lines = np.array([]), minLineLength=850, maxLineGap=5)

# for line in get_lines(lines):
#     leftx, boty, rightx, topy = line
#     cv2.line(img_board, (leftx, boty), (rightx,topy), (0,0,255), 2) 

_lines = []
for _line in get_lines(lines):
    _lines.append([(_line[0], _line[1]),(_line[2], _line[3])])
    
# sort
_lines_x = []
_lines_y = []
for line_i in _lines:
    orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
        _lines_y.append(line_i)
    else:
        _lines_x.append(line_i)
        
_lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
_lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])

merged_lines_x = merge_lines_pipeline_2(_lines_x)
merged_lines_y = merge_lines_pipeline_2(_lines_y)

merged_lines_all = []
merged_lines_all.extend(merged_lines_x)
merged_lines_all.extend(merged_lines_y)
# print("process groups lines", len(_lines), len(merged_lines_all))
# for line in merged_lines_all:
#     cv2.line(img_board, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 2)

intersections = []
boardcorners = []

for horizontal in merged_lines_x:
    for vertical in merged_lines_y:
        intersections.append(line_intersection(horizontal, vertical))


prunedintersections = []
for intersection in intersections:
    if(not(intersection[0]<15 or intersection[1]>935 or intersection[0]>935 or intersection[1]<15)):
        prunedintersections.append(intersection)

intersections = prunedintersections
# print(prunedintersections)

smallest = 2000
topleft = intersections[0]
for intersection in intersections:
    if(intersection[0]+intersection[1]<smallest):
        smallest = intersection[0] + intersection[1]
        topleft = intersection
boardcorners.append([topleft[0], topleft[1]])

largest = 0
bottomright = intersections[0]
for intersection in intersections:
    if(intersection[0]+intersection[1]>largest):
        largest = intersection[0] + intersection[1]
        bottomright = intersection
boardcorners.append([bottomright[0], bottomright[1]])

topright = intersections[0]
for intersection in intersections:
    if((intersection[0]>topright[0]*0.9) and (intersection[1]<topright[1]*1.1)):
        topright = intersection
boardcorners.append([topright[0], topright[1]])

bottomleft = intersections[0]
for intersection in intersections:
    if((intersection[0]<bottomleft[0]*1.1) and (intersection[1]>bottomleft[1]*0.9)):
        bottomleft = intersection
boardcorners.append([bottomleft[0], bottomleft[1]])

# print(boardcorners)

# for corner in boardcorners:
#     cv2.circle(img_board, (round(corner[0]), round(corner[1])), radius=4, color=(0,255,0), thickness=7)

m = cv2.getPerspectiveTransform(np.array([boardcorners[0], boardcorners[2], boardcorners[1], boardcorners[3]], np.float32), np.array([[0,0],[952,0],[952,952],[0,952]], np.float32))
img_board = cv2.warpPerspective(img_board, m, (952, 952))

# Show the result
# cv2.imshow('Result', img_board)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

piecesarray = []
xcoord = 0
ycoord = 0
increment = 119

for i in range(0, 8, 1):
    ycoord=0
    for j in range(0, 8, 1):
        temp = img_board[xcoord:xcoord+119, ycoord:ycoord+119]
        temp = cv2.resize(temp, (52, 52))
        piecesarray.append(predict_img(temp))
        ycoord+=119
    xcoord+=119

# print(piecesarray)

pos = ""

for i in range(0, 64, 1):
    if(piecesarray[i]=="blackrook"):
        pos+='r'
    elif(piecesarray[i]=="whiterook"):
        pos+='R'
    elif(piecesarray[i]=="blackbishop"):
        pos+='b'
    elif(piecesarray[i]=="whitebishop"):
        pos+='B'
    elif(piecesarray[i]=="blackknight"):
        pos+='n'
    elif(piecesarray[i]=="whiteknight"):
        pos+='N'
    elif(piecesarray[i]=="blackqueen"):
        pos+='q'
    elif(piecesarray[i]=="whitequeen"):
        pos+='Q'
    elif(piecesarray[i]=="blackking"):
        pos+='k'
    elif(piecesarray[i]=="whiteking"):
        pos+='K'
    elif(piecesarray[i]=="blackpawn"):
        pos+='p'
    elif(piecesarray[i]=="whitepawn"):
        pos+='P'
    elif(piecesarray[i]=="nothing"):
        pos+='1'
    if((i+1)%8==0 and i!=63):
        pos+='/'

FENpos = ""

counter=0
for i in pos:
    if(i=='1'):
        counter+=1
    else:
        # print(counter)
        if(counter!=0):
            FENpos+=str(counter)
        FENpos+=i
        counter=0
if(counter!=0):
    FENpos+=str(counter)

# print(FENpos)

board = chess.Board(FENpos)
# print(board.fen())
board

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
    if "b" in fstr:
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

temp = board.fen()

boardInfoMatrix = board_to_bits([temp,0])
# print(board_to_bits([temp,0]))
bitMatrix = np.array([boardInfoMatrix[4:]])
auxMatrix = np.array([boardInfoMatrix[:4]])

temp

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

index = predictions.index(max(predictions))
print("best move evaluation", predictions[index])
print(predictions[index])

returnBoard = chess.Board(potentialPositions[index])
boardsvg = chess.svg.board(board=returnBoard)
outputfile = open('./cache/temp.svg', "w")
outputfile.write(boardsvg)
outputfile.close()