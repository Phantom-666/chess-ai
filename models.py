import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import piece_map
import torch



class LinearPieceSelectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearPieceSelectionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


input_size = 8 * 8  * 12 + 2 + 8 * 8  * 12 + 2
hidden_size = 100
output_size = 1
learning_rate = 0.001


def encode_new_board_with_pieces(board):
    
    
    feature_vector = np.zeros((8, 8, 12))  # 8x8 board, 12 channels for piece types

    
    for i in range(8):
        for j in range(8):
            if board[i][j] != '':
                piece_index = piece_map[board[i][j]]

                feature_vector[i, j, piece_index] = 1


    return feature_vector


def create_next_move_board(board, i, j, next_i, next_j):

    newBoard = [["" for _ in range(8)] for _ in range(8)]

    board_size = 8

    for k in range(board_size):
        for y in range(board_size):
            piece = board[k][y]['piece']
            newBoard[k][y] = piece




    piece = newBoard[i][j]
    newBoard[i][j] = ''

    newBoard[next_i][next_j] = piece


    arr = encode_new_board_with_pieces(newBoard)

    return arr.flatten()


            


def encode_board_with_pieces(board, prev_board, turn):
    
    
    feature_vector = np.zeros((8, 8, 12))  # 8x8 board, 12 channels for piece types

    only_whites = np.zeros((8, 8))
    
    for i in range(8):
        for j in range(8):
            if board[i][j] != '':
                piece_index = piece_map[board[i][j]]
                feature_vector[i, j, piece_index] = 1
                
                if board[i][j].startswith(turn):
                    if (len(prev_board[i][j]['candidateMoves'])):
                        only_whites[i, j] = 1
    
    return feature_vector, only_whites


def encode_moves(board, turn):
    board_size = 8
    newBoard = [["" for _ in range(8)] for _ in range(8)]

    for i in range(board_size):
        for j in range(board_size):
            piece = board[i][j]['piece']
            newBoard[i][j] = piece
            

    encoded_board, only_whites = encode_board_with_pieces(newBoard, board, turn)

    
    return encoded_board.flatten(), only_whites



def process_white_model(data, model):
    board = data['board']
    turn = data['turn']

    encoded_board, only_whites = encode_moves(board, turn)

    highest_score = -100
    final_object = {}

    for i in range(only_whites.shape[0]):
        for j in range(only_whites.shape[1]):
            
            if only_whites[i,j] == 1 :

                # board_size = 8
                # for col in range(board_size):
                #     for row in range(board_size):
                        
                if 'candidateMoves' in board[i][j]:
                    candidateMoves = board[i][j]['candidateMoves']
                    if len(candidateMoves) > 0:
                        for moves in candidateMoves:
                            moves_board = create_next_move_board(board, i, j, moves[0], moves[1])
                            arr = np.append(encoded_board, i)
                            arr = np.append(arr, j)

                            # random_float = random.uniform(-1, 1)
                            # arr = np.append(arr, random_float)
                            result = np.concatenate((arr, moves_board))

                            result = np.append(result, moves[0])
                            result = np.append(result, moves[1])

                            
                            encoded_board_tensor = torch.tensor(result, dtype=torch.float).unsqueeze(0)
                            output = model(encoded_board_tensor)
                            value = output.item()
                            if (value > highest_score):
                                highest_score = value
                                final_object = { 'piece' : {'col' :i, 'row': j, 'piece': board[i][j]['piece']}, 'moveTo' : {'first' : moves[0], 'second': moves[1]}}

    return highest_score, final_object