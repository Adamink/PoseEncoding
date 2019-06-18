import torch
import torch.nn as nn

import os
import numpy as np

class TimeDistributed(nn.Module):
    """
    A layer that could be nested to apply sub operation to every timestep of sequence input.
    """ 
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class LSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden, num_layers, 
      mean_after_fc = True, mask_empty_frame = False):
        super(LSTM, self).__init__()
        self.input_size = input_size 
        self.num_classes = num_classes
        self.hidden = hidden
        self.num_layers = num_layers
        self.mask_empty_frame= mask_empty_frame

        self.classifier = nn.LSTM(input_size = input_size, 
                    hidden_size = hidden, 
                    num_layers = num_layers, 
                    batch_first=True) # (batch, max_frame, input_size)
        self.fc = TimeDistributed(nn.Linear(hidden, num_classes))

    def _mask_empty_frame(self, x, frame_num):

        batch = x.size(0)
        time_step = x.size(1)
        num_classes = x.size(2)

        idx = torch.arange(0, time_step, 1).cuda().long().expand(batch, time_step)
        frame_num_expand = frame_num.view(batch,1).repeat(1,time_step)
        #(batch, time_step, num_classes)
        mask = (idx < frame_num_expand).float().view(batch, time_step, 1).repeat(1,1,num_classes) 
        x = x * mask
        return x
    def forward(self, x, target, frame_num):
        """
        x: (batch, time_step, input_size)
        frame_num: (batch, 1)
        """
        x, hn = self.classifier(x)

        # mean after fc
        x = self.fc(x) #(batch, time_step, num_classes)
        if self.mask_empty_frame:
            x = self._mask_empty_frame(x, frame_num)
            x = torch.sum(x, dim = 1)
            eps = 0.01 # to deal with 0 frame_num
            frame_num = frame_num.view(-1,1).float() + eps
            x = x / frame_num
        else:
            x = torch.mean(x, dim = 1)
        return x
    