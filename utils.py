import numpy as np

piece_map = {'wp': 0, 'wn': 1, 'wb': 2, 'wr': 3, 'wq': 4, 'wk': 5,
                 'bp': 6, 'bn': 7, 'bb': 8, 'br': 9, 'bq': 10, 'bk': 11}

def encode_new_board_with_pieces(board):
    
    
    feature_vector = np.zeros((8, 8, 12))  # 8x8 board, 12 channels for piece types

    
    for i in range(8):
        for j in range(8):
            if board[i][j] != '':
                piece_index = piece_map[board[i][j]]

                feature_vector[i, j, piece_index] = 1


    return feature_vector

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


