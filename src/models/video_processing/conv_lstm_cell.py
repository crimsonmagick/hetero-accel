from torch import nn
import torch

# -*- encoding: utf-8 -*-
'''
@File    :   convlstmcell.py
@Time    :   2023/10/15 21:32:51
@Author  :   Zhang Wenjie 
@Version :   3.9.13
@Contact :   1710269958@qq.com
'''


class ConvLSTMCell(nn.Module):
    """ConvLSTMCell Implementation

    Input: x, (h_0, c_0)
        x:(B,C,H,W) Note that C is represented as P in the original paper
        h_0,c_0 should be initialized as same dimension tensor filled with 0

    Output: h_1, c_1 
        h_1:the output hidden state:(B,F,H,W) Note that F is filters,namely out_channels
        c_1:the output cell state:(B,F,H,W)
    """

    def __init__(self, in_channels, out_channels, kernel_size,batch_size,img_size) -> None:
        super().__init__()
        self.padding = int((kernel_size-1)/2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.c_h_shape = (batch_size, self.out_channels,*img_size)  # c or h shape,they are the same

        def get_params(): return [
            nn.Conv2d(self.in_channels, self.out_channels,
                      kernel_size, padding=self.padding, bias=True),
            nn.Conv2d(self.out_channels, self.out_channels,
                      kernel_size, padding=self.padding, bias=False),
            nn.Conv2d(self.out_channels, self.out_channels,
                      kernel_size, padding=self.padding, bias=False),
        ]
        # The reason why bias = False is that adding two bias is equivalent to adding one

        # The LSTM in paper is with peephole connection s
        # ---------------Input Gate----------------
        self.W_xi, self.W_hi, self.W_ci = get_params()
        # ---------------Forget Gate----------------
        self.W_xf, self.W_hf, self.W_cf = get_params()
        # ---------------Output Gate----------------
        self.W_xo, self.W_ho, self.W_co = get_params()
        # ---------------Memory Cell----------------
        self.W_xc, self.W_hc = get_params()[:2]

        self.W_ci = nn.Parameter(torch.zeros(self.c_h_shape))
        self.W_cf = nn.Parameter(torch.zeros(self.c_h_shape))
        self.W_co = nn.Parameter(torch.zeros(self.c_h_shape))

    def init_hiddens(self,device):
        return [torch.zeros(self.c_h_shape,device=device), torch.zeros(self.c_h_shape,device=device)]  # ->[h,c]

    def forward(self, x, h_0, c_0):
        # input x_{t},h_{t-1} and c_{t-1}
        I = torch.sigmoid(self.W_xi(x) + self.W_hi(h_0) + self.W_ci*c_0)
        F = torch.sigmoid(self.W_xf(x) + self.W_hf(h_0) + self.W_cf*c_0)
        O = torch.sigmoid(self.W_xo(x) + self.W_ho(h_0) + self.W_co*c_0)
        c_1 = F * c_0+I * torch.tanh(self.W_xc(x)+self.W_hc(h_0))
        h_1 = O * torch.tanh(c_1)
        return [h_1, c_1]


if __name__ == '__main__':
    convlstmCell = ConvLSTMCell(20, 64, 5,10,(128,128))  # 20 channels -> 64 channels
    # 10 batchs,each image is 20 channels,128x128
    test = torch.randn(10, 20, 128, 128)
    h_0, c_0 = convlstmCell.init_hiddens(device=test.device)
    result = convlstmCell(test, h_0, c_0)
    print('h_1:', result[0].shape, 'c_1', result[1].shape)