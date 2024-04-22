import numpy as np
import os
from config import trained_models_path
import json
import torch
import string
import random

piece_map = {'wp': 0, 'wn': 1, 'wb': 2, 'wr': 3, 'wq': 4, 'wk': 5,
                 'bp': 6, 'bn': 7, 'bb': 8, 'br': 9, 'bq': 10, 'bk': 11}



def check_for_trained_model(model):

    models = os.listdir(trained_models_path)

    if len(models) > 0:
        print(f'loaded {models[-1]}')

        model.load_state_dict(torch.load(os.path.join(trained_models_path, models[-1])))

    return model


def save_to_json_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def generate_random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))




def parse_best_move(move):

    # e7e5
    ranks = {'a' : 0, 'b' : 1, 'c' : 2, 'd' : 3, 'e' : 4, 'f' : 5, 'g' : 6, 'h' : 7}

    file = ranks[move[0]]
    rank = int(move[1]) - 1


    y = ranks[move[2]]
    x = int(move[3]) - 1


    return {'rank' : rank, 'file' : file, 'x' : x, 'y' : y}



# save_to_json_file({'board': board, 'piece' : piece, 'dist': dist}, os.path.join(data_path, generate_random_string(10) + '.json'))
# white_model.load_state_dict(torch.load('linear_piece_selection_model.pth'))