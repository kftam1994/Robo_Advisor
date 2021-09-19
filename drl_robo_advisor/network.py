import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    """

    Deep reinforcement learning network implementation in Pytorch

    Methods
    -------
    forward(x, last_w)

    """
    def __init__(self, feature_number, rows, columns, conv_layer_outputs, eiie_dense_out_channels):
        """

        Parameters
        ----------
        feature_number : int
            number of data features
        rows : int
            number of assets or stocks
        columns : int
            length of the window period of historical data
        conv_layer_outputs : list[int]
            A list of output channels of 2D convolutional layer. The lenght of this list means the number of 2D convolutional layer the network will have before EIIE layers
        eiie_dense_out_channels : int
            The number of output channels for 2D Convolutional layer in EIIE layers

        """
        super(Network, self).__init__()

        self._rows = rows
        self._columns = columns
        self.conv = nn.Sequential()
        for layer_id, each_layer_out_channels in enumerate(conv_layer_outputs):
            if layer_id==0:
                self.conv.add_module(f'conv_layer_{layer_id}',nn.Conv2d(in_channels=feature_number, out_channels=each_layer_out_channels, kernel_size=(1,2), stride=(1,1)))
            else:
                self.conv.add_module(f'conv_layer_{layer_id}',nn.Conv2d(in_channels=conv_layer_outputs[layer_id-1], out_channels=each_layer_out_channels,kernel_size=(1, 2), stride=(1, 1)))
            self.conv.add_module(f'conv_layer_{layer_id}_batchnorm',nn.BatchNorm2d(each_layer_out_channels))
            self.conv.add_module(f'conv_layer_{layer_id}_leakyrelu',nn.LeakyReLU())

        conv_shape = self._get_conv_out_shape(layer=self.conv,shape=[feature_number,rows,columns])

        self.eiie_dense = nn.Sequential(nn.Conv2d(in_channels=conv_layer_outputs[-1],out_channels=eiie_dense_out_channels,kernel_size=(1,conv_shape[3]), stride=(1,1)),#out_channels=10,
                                        nn.BatchNorm2d(eiie_dense_out_channels),
                                        nn.LeakyReLU())
        eiie_dense_width = self._get_conv_out_shape(layer=self.eiie_dense, shape=[conv_layer_outputs[-1], conv_shape[2], conv_shape[3]])[3]

        self.eiie_output_with_w = nn.Sequential(nn.Conv2d(in_channels=eiie_dense_out_channels+1,out_channels=1,kernel_size=(1,eiie_dense_width), stride=(1,1)),#in_channels=11,kernel_size=(1,1), stride=(1,1)),
                                                nn.BatchNorm2d(1),
                                                nn.LeakyReLU(),
                                                nn.Flatten())

        self.softmax_voting = nn.Sequential(nn.Linear(rows+1,rows+1),nn.Softmax(dim=1))

    def _get_conv_out_shape(self,layer,shape):
        """

        Mimic the input to retrieve the total output size of the sequential of layers

        Parameters
        ----------
        layer : torch.nn.Sequential()
            A sequential of convolutional layer
        shape : list[int]
            The input shape of the sequential

        Returns
        -------
        o_size : int
            The output size which is a product of output shapes

        """
        o = layer(torch.zeros(1,*shape))
        o_size = o.size()
        return o_size

    def forward(self,x, last_w):
        """

        Run the input through the network and generate prediction

        Parameters
        ----------
        x : torch.tensor
            A torch tensor array of batches of historical prices data with shape [number of batches, number of assets, lenght of window, number of features]
        last_w : torch.tensor
            A torch tensor array of batches of last weight predicted with shape [number of batches, number of assets]

        Returns
        -------
        final_out : torch.tensor
            A torch tensor array of predicted weight for each asset and cash (so there is +1) with shape [number of batches, number of assets + 1]

        """
        input_num = x.shape[0]
        conv_out = self.conv(x)
        eiie_dense_out = self.eiie_dense(conv_out)

        eiie_dense_height = eiie_dense_out.size()[2]

        w = torch.reshape(last_w, [-1, 1, int(eiie_dense_height), 1])

        eiie_dense_out_reshape_w = torch.cat([eiie_dense_out, w], dim=1)

        eiie_output_with_w_out = self.eiie_output_with_w(eiie_dense_out_reshape_w)

        cash_bias = torch.zeros([1,1], dtype=torch.float32).to(device)
        cash_bias = torch.tile(cash_bias, (input_num, 1))

        eiie_output_with_w_out_cash = torch.cat([cash_bias, eiie_output_with_w_out], dim=1)

        final_out = self.softmax_voting(eiie_output_with_w_out_cash)

        assert torch.all(torch.round(torch.sum(final_out, dim=1)) == 1), f'Some output weights are not equal to one {torch.round(torch.sum(final_out, dim=1)) == 1}'
        return final_out