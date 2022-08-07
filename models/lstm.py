# Original model presented in: 
#   C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, M. Shah, 
#   Deep Learning Human Mind for Automated Visual Classification, 
#   CVPR 2017 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

torch.utils.backcompat.broadcast_warning.enabled = True

if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True

class Model(nn.Module):
    def __init__(self, input_size=128, lstm_size=128, lstm_layers=1, output_size=128):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size

        # Define internal modules
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.output = nn.Linear(lstm_size, output_size)
        self.classifier = nn.Linear(output_size, 40)
        
    def forward(self, x):
        # Prepare LSTM initiale state
        batch_size = x.size(0)
        
        device = x.device

        lstm_init = (
            torch.zeros(self.lstm_layers, batch_size, self.lstm_size, device=device), 
            torch.zeros(self.lstm_layers, batch_size, self.lstm_size, device=device)
        )

        # if x.is_cuda: 
        #     lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        # elif x.device.startswith("mps"):
        #     lstm_init = (lstm_init[0].to("mps"), lstm_init[0].to("mps"))
        # else:
        #     print("On CPU")
        
        lstm_init = (Variable(lstm_init[0]), Variable(lstm_init[1]))

        # Forward LSTM and get final state
        x = self.lstm(x, lstm_init)[0][:, -1, :]

        # Forward output
        x = F.relu(self.output(x))
        x = self.classifier((x))

        return x
