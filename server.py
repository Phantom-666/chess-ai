from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import torch
from models import  LinearPieceSelectionModel
from utils import encode_board_with_pieces, encode_new_board_with_pieces
import json
import random
import string


app = Flask(__name__)
CORS(app)


def save_to_json_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


root_path = os.getcwd()
data_path = os.path.join(root_path, 'data') 


def encode_moves(board, turn):
    board_size = 8
    newBoard = [["" for _ in range(8)] for _ in range(8)]

    for i in range(board_size):
        for j in range(board_size):
            piece = board[i][j]['piece']
            newBoard[i][j] = piece
            

    encoded_board, only_whites = encode_board_with_pieces(newBoard, board, turn)

    
    return encoded_board.flatten(), only_whites


def generate_random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))



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


# save_to_json_file({'board': board, 'piece' : piece, 'dist': dist}, os.path.join(data_path, generate_random_string(10) + '.json'))
# white_model.load_state_dict(torch.load('linear_piece_selection_model.pth'))




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

                            random_float = random.uniform(-1, 1)
                            arr = np.append(arr, random_float)
                            result = np.concatenate((arr, moves_board))

                            
                            encoded_board_tensor = torch.tensor(result, dtype=torch.float).unsqueeze(0)
                            output = model(encoded_board_tensor)
                            value = output.item()
                            if (value > highest_score):
                                highest_score = value
                                final_object = { 'piece' : {'col' :i, 'row': j, 'piece': board[i][j]['piece']}, 'moveTo' : {'first' : moves[0], 'second': moves[1]}}

    return highest_score, final_object
    
from models import input_size, hidden_size, output_size


white_model = LinearPieceSelectionModel(input_size, hidden_size, output_size)
black_model = LinearPieceSelectionModel(input_size, hidden_size, output_size)


sessions_path = os.path.join(root_path, 'sessions')

@app.route('/calc_move', methods=['POST'])
def post_route():
    data = request.json
    turn = data['turn']

    session_name = data['session_name']

    s_path = os.path.join(sessions_path, session_name)

    if not os.path.exists(s_path):
        os.makedirs(s_path)

        os.makedirs(os.path.join(s_path, 'w'))
        os.makedirs(os.path.join(s_path, 'b'))


    print('turn', turn)


    if turn == 'w':
        highest_score, final_object = process_white_model(data, white_model)
        save_to_json_file({'board': data['board'], 'final_object' : final_object}, os.path.join(os.path.join(s_path, 'w'), generate_random_string(10) + '.json'))

    else:
        highest_score, final_object = process_white_model(data, black_model)
    
    save_to_json_file({'board': data['board'], 'final_object' : final_object}, os.path.join(os.path.join(s_path, turn), generate_random_string(10) + '.json'))
   
    # print('final_object', final_object)
    result = {'final_object' :final_object}

    return jsonify(result)

session_path = os.path.join(root_path, 'session')

@app.route('/checkmate', methods=['POST'])
def checkmate_route():
    data = request.json

    session_name = data['session_name']

    save_to_json_file({'session_name': session_name, 'turn' : data['turn']}, os.path.join(session_path, generate_random_string(10) + '.json'))

    return jsonify({})



if __name__ == '__main__':
    app.run(debug=True)  # Run the server in debug mode