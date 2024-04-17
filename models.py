import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F


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


input_size = 8 * 8  * 12 + 2 + 8 * 8  * 12 + 1
hidden_size = 100
output_size = 1
learning_rate = 0.001