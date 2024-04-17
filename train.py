import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import LinearPieceSelectionModel
import numpy as np
import os 
import json
from server import encode_moves, create_next_move_board
import random
from models import input_size, hidden_size, output_size, learning_rate

root_path = os.getcwd()
sessions_path = os.path.join(root_path, 'sessions')
session_path = os.path.join(root_path, 'session')



if torch.cuda.is_available():
    print("CUDA (GPU) is available.")
else:
    print("CUDA (GPU) is not available. Running on CPU.")



def process_session(session_path, sessions_path):

    result = [] 

    files = os.listdir(session_path)

    for file_name in files:

        f = open(os.path.join(session_path, file_name))

        lines = f.readlines()

        f.close()

        js_obj = json.loads(' '.join(lines))

        session_name = js_obj['session_name']
        turn = js_obj['turn']

        result.append({'path' : os.path.join(sessions_path, session_name, turn), 'turn' :turn })


    return result




def process_folder(folder_path):
    arr = []
    out = []

    items = os.listdir(folder_path)

    for item in items:
        file = open(os.path.join(folder_path, item))
        lines = file.readlines()
        file.close()
        js_obj = json.loads(' '.join(lines))

        my_board = js_obj['board']
        final_object = js_obj['final_object']

        arr.append(my_board)
        out.append(final_object)


    return arr, out



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Assuming you have already prepared your data
def train():
    encoded_data = []
    train_folders = process_session(session_path, sessions_path)

    X = []
    y = []

    turns = []

    for folder_path in train_folders:
        temp_x_array, temp_y_array = process_folder(folder_path['path'])

        for tempX in temp_x_array:
            X.append(tempX)
            turns.append(folder_path['turn'])

        for tempY in temp_y_array:
            y.append(tempY)


    print('data loaded', len(X), len(y), len(turns))

    # for board_index in range(len(arr)):

    #     print('board_index', board_index)

    #     board = arr[board_index]

    #     print('arr{}'.format(board_index), board)

    # return 

    for board_index in range(len(X)):
        board = X[board_index]
        encoded_board, only_whites = encode_moves(board, turns[board_index])
        out_obj = y[board_index]
        correct_i = out_obj['piece']['col']
        correct_j = out_obj['piece']['row']
        move_to = out_obj['moveTo']

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

                                encoded_board_tensor = torch.tensor(result, dtype=torch.float32).unsqueeze(0).cuda()
                                label = torch.tensor([-1.0], dtype=torch.float32).cuda()

                                if i ==  correct_i and j == correct_j:
                                    if moves[0] == move_to['first'] and moves[1] == move_to['second']:
                                        label = torch.tensor([1.0], dtype=torch.float32)
                                        # print('correct', correct_i, correct_j, move_to['first'], move_to['second'])
                                        
                                encoded_data.append((encoded_board_tensor, label))

                                

    
    batch_size=32

    train_loader = torch.utils.data.DataLoader(encoded_data)
    val_loader = train_loader


    # train_dataset = TensorDataset(X_train, y_train)

    # Define your model
    model = LinearPieceSelectionModel(input_size, hidden_size, output_size).cuda()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 100

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:

            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

    # Evaluate model
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:

              
            inputs = inputs.cuda()
            labels = labels.cuda()


            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Validation Loss: {val_loss}")

    torch.save(model.state_dict(), "chess.pth")


if __name__=="__main__":
    train()