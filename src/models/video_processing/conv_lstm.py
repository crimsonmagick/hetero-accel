from torch import nn
import torch
from src.models.video_processing.conv_lstm_cell import ConvLSTMCell

# -*- encoding: utf-8 -*-
'''
@File    :   convlstm.py
@Time    :   2023/10/15 22:55:01
@Author  :   Zhang Wenjie 
@Version :   3.9.13
@Contact :   1710269958@qq.com
'''


class ConvLSTM2d(nn.Module):
    """ConvLSTM2d Implementation
    Note:should be modified to implement conv3d,conv1d some day

    Input: x
    x:(B,C,S,H,W) (consistency with batchnorm3d)
    B:batch_size
    C:channels
    S:sequence_length
    H,W:height and width

    Output: (h_n, c_n) 
        h_n:the final hidden state:(nl,B,F,H,W) or (B,F,H,W)
        c_n:the final cell state:(nl,B,F,H,W) or (B,F,H,W)

        * nl:the num_layers,default = 1
            if nl==1,the dim 0 will be squeezed for consistency
        * if return_sequences=True : (nl,B,F,S,H,W) or (B,F,S,H,W) if nl==1
    """

    def __init__(self, in_channels, out_channels, kernel_size,batch_size,img_size,num_layers=1, return_sequences=True) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.cell_dict = nn.ModuleDict()
        self.return_sequences = return_sequences


        for i in range(num_layers):
            # in_channels->out_channels->……_>out_channels
            if i == 0:
                self.cell_dict[f'cell_{i}'] = ConvLSTMCell(
                    in_channels, out_channels, kernel_size,batch_size,img_size)
            else:
                self.cell_dict[f'cell_{i}'] = ConvLSTMCell(
                    out_channels, out_channels, kernel_size,batch_size,img_size)

    def forward(self, x):
        states_dict = {}
        for step in range(x.size(2)):
            # x.size(2) is S
            for i in range(self.num_layers):
                if f'cell_{i}' not in states_dict:
                    states_dict[f'cell_{i}'] = self.cell_dict[f'cell_{i}'].init_hiddens(device=x.device)
                if i == 0:
                    h, c = self.cell_dict[f'cell_{i}'](
                        x[:,:,step], *states_dict[f'cell_{i}'])  # h,c ->(B,F,H,W)
                else:
                    # input h_{t}^{l-1},h_{t-1}^{l},c_{t-1}^l
                    h, c = self.cell_dict[f'cell_{i}'](
                        states_dict[f'cell_{i-1}'][0], *states_dict[f'cell_{i}'])

                # update hidden_state and cell state
                states_dict[f'cell_{i}'] = [h, c]

            if self.return_sequences:
                if step == 0:
                    # initialize h_n and c_n {"cell_0":[h_0,c_0],"cell_1":……}

                    # stacking creates dim = 0 which is nl 
                    # (B,F,H,W)->(nl,B,F,1,H,W)
                    h_n = torch.stack([layer[0] for layer in states_dict.values()], dim=0).unsqueeze(dim=3) 
                    c_n = torch.stack([layer[1] for layer in states_dict.values()], dim=0).unsqueeze(dim=3)
                else:
                    h_n = torch.cat([h_n,
                                     torch.stack([layer[0] for layer in states_dict.values()], dim=0).unsqueeze(dim=3)  # (nl,B,F,H,W)->(nl,B,F,S,H,W) contatenate along dim 3
                                     ], dim=3)
                    c_n = torch.cat([c_n,
                                    torch.stack(
                                        [layer[1] for layer in states_dict.values()], dim=0).unsqueeze(dim=3)
                                     ], dim=3)

        if not self.return_sequences:
            # values->[(h,c),(h_c),……]
            h_n = torch.stack([layer[0]
                              for layer in states_dict.values()], dim=0)
            c_n = torch.stack([layer[1]
                              for layer in states_dict.values()], dim=0)

        if self.num_layers == 1:
            h_n.squeeze_(0) # (1,B,F,S,H,W) or (1,B,F,H,W) ->(B,F,S,H,W) or (B,F,H,W)
            c_n.squeeze_(0)
        return [h_n, c_n]


def conv_lstm(pretrained=True):
    model = ConvLSTM2d(in_channels=3,
                       out_channels=32,
                       kernel_size=5,
                       batch_size=3,
                       img_size=(64,64),
                       num_layers=1,
                       return_sequences=True)
    return model


if __name__ == '__main__':
    convlstm2d = ConvLSTM2d(in_channels=3,out_channels=32,kernel_size=5,batch_size=3,img_size=(64,64),num_layers=1, return_sequences=True)
    # batch_size:3
    # in_channels:3
    # out_channels:32
    # seq_len:16
    # height,width:64x64
    test = torch.randn(3, 3, 16, 64, 64)
    print(convlstm2d(test)[0].shape)
    convlstm2d = ConvLSTM2d(3, 32, 5,3,(64,64), num_layers=2, return_sequences=True)
    test = torch.randn(3, 3, 16, 64, 64)
    print(convlstm2d(test)[0].shape)
