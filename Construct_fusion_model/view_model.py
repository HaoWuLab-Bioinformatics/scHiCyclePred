import torch
from torch import nn


def com_linearsize(linear_size, Con_layer, kernel_size):
    for i in range(Con_layer):
        # indicates integer division, it can return the integer part of the quotient (rounded down)
        linear_size = int(((linear_size + 2 * 1 - kernel_size) / 1 + 1) // 2)
    if Con_layer == 0:
        linear_size = 0
    return linear_size


if __name__ == '__main__':
    # BCP, CDD and SICP parts of the network structure is actually the same, and BCP, CDD and SICP three parts are executed in parallel
    # Look at the BCP part first
    # Already x1 in the network, that is, the BCP part of each batch input data size is: [64,1,2660]
    images_BCP = torch.rand(size=(64, 1, 2660), dtype=torch.float32)
    Con_layer_BCP = 2
    Con_layer_CDD = 2
    Con_layer_SICP = 2
    linear_layer = 1
    kernel_size = 7
    cnn_feature = 32
    out_feature = 0
    dp = 0.3
    lr = 0.0001
    sum_acc = 0
    # The network structure that handles the BCP piece in isolation looks like this.
    net_BCP = nn.Sequential(
        # Calculation method n2=(n-k+2p+s)/s
        nn.Conv1d(
            in_channels=1,
            out_channels=cnn_feature,
            kernel_size=kernel_size,
            stride=1,
            padding=1
        ),
        nn.BatchNorm1d(num_features=cnn_feature),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Dropout(dp),
        nn.Conv1d(in_channels=cnn_feature, out_channels=cnn_feature, kernel_size=kernel_size,
                  stride=1, padding=1),
        nn.BatchNorm1d(num_features=cnn_feature),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Dropout(dp)
    )
    print("BCP部分")
    print("初始shape：", images_BCP.shape)
    for layer in net_BCP:
        images_BCP = layer(images_BCP)
        print(layer.__class__.__name__, 'output shape: \t', images_BCP.shape)
    linear_size_BCP_init = 2660
    linear_size_BCP = com_linearsize(linear_size_BCP_init, Con_layer_BCP, kernel_size)
    # linear_size_BCP is actually the size of the feature of the tensor after the above network structure, i.e., the size of the 2nd dimension
    # cnn_feature is the number of channels of the tensor after the above network structure
    # -1 is actually 64, which is the batch size
    images_BCP = images_BCP.view(-1, cnn_feature * linear_size_BCP)
    print("view以后的shape：", images_BCP.shape)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Output results.
    # Initial shape： torch.Size([64, 1, 2660])
    # Conv1d output shape: 	 torch.Size([64, 32, 2656])
    # BatchNorm1d output shape: 	 torch.Size([64, 32, 2656])
    # ReLU output shape: 	 torch.Size([64, 32, 2656])
    # MaxPool1d output shape: 	 torch.Size([64, 32, 1328])
    # Dropout output shape: 	 torch.Size([64, 32, 1328])
    # Conv1d output shape: 	 torch.Size([64, 32, 1324])
    # BatchNorm1d output shape: 	 torch.Size([64, 32, 1324])
    # ReLU output shape: 	 torch.Size([64, 32, 1324])
    # MaxPool1d output shape: 	 torch.Size([64, 32, 662])
    # Dropout output shape: 	 torch.Size([64, 32, 662])
    # The shape after view: torch.Size([64, 21184])


    # Next look at CDD
    images_CDD = torch.rand(size=(64, 1, 98), dtype=torch.float32)
    net_CDD = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=cnn_feature, kernel_size=kernel_size, stride=1, padding=1),
        nn.BatchNorm1d(num_features=cnn_feature),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Dropout(dp),
        nn.Conv1d(in_channels=cnn_feature, out_channels=cnn_feature,
                  kernel_size=kernel_size, stride=1, padding=1),
        nn.BatchNorm1d(num_features=cnn_feature),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Dropout(dp)
    )
    print("CDD部分")
    print("初始shape：", images_CDD.shape)
    for layer in net_CDD:
        images_CDD = layer(images_CDD)
        print(layer.__class__.__name__, 'output shape: \t', images_CDD.shape)
    linear_size_CDD_init = 98
    linear_size_CDD = com_linearsize(linear_size_CDD_init, Con_layer_CDD, kernel_size)
    images_CDD = images_CDD.view(-1, cnn_feature * linear_size_CDD)
    print("view以后的shape：", images_CDD.shape)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Output results:
    # CDD section
    # Initial shape: torch.Size([64, 1, 98])
    # Conv1d output shape: 	 torch.Size([64, 32, 94])
    # BatchNorm1d output shape: 	 torch.Size([64, 32, 94])
    # ReLU output shape: 	 torch.Size([64, 32, 94])
    # MaxPool1d output shape: 	 torch.Size([64, 32, 47])
    # Dropout output shape: 	 torch.Size([64, 32, 47])
    # Conv1d output shape: 	 torch.Size([64, 32, 43])
    # BatchNorm1d output shape: 	 torch.Size([64, 32, 43])
    # ReLU output shape: 	 torch.Size([64, 32, 43])
    # MaxPool1d output shape: 	 torch.Size([64, 32, 21])
    # Dropout output shape: 	 torch.Size([64, 32, 21])
    # The shape after view: torch.Size([64, 672])


    # Then look at the SICP:
    images_SICP = torch.rand(size=(64, 1, 2660), dtype=torch.float32)
    net_SICP = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=cnn_feature, kernel_size=kernel_size, stride=1, padding=1),
        nn.BatchNorm1d(num_features=cnn_feature),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Dropout(dp),
        nn.Conv1d(in_channels=cnn_feature, out_channels=cnn_feature,
                  kernel_size=kernel_size, stride=1, padding=1),
        nn.BatchNorm1d(num_features=cnn_feature),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
        nn.Dropout(dp)
    )
    print("SICP部分")
    print("初始shape：", images_SICP.shape)
    for layer in net_SICP:
        images_SICP = layer(images_SICP)
        print(layer.__class__.__name__, 'output shape: \t', images_SICP.shape)
    linear_size_SICP_init = 2660
    linear_size_SICP = com_linearsize(linear_size_SICP_init, Con_layer_SICP, kernel_size)
    images_SICP = images_SICP.view(-1, cnn_feature * linear_size_SICP)
    print("view以后的shape：", images_SICP.shape)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Output results:
    # SICP section
    # Initial shape: torch.Size([64, 1, 2660])
    # Conv1d output shape: 	 torch.Size([64, 32, 2656])
    # BatchNorm1d output shape: 	 torch.Size([64, 32, 2656])
    # ReLU output shape: 	 torch.Size([64, 32, 2656])
    # MaxPool1d output shape: 	 torch.Size([64, 32, 1328])
    # Dropout output shape: 	 torch.Size([64, 32, 1328])
    # Conv1d output shape: 	 torch.Size([64, 32, 1324])
    # BatchNorm1d output shape: 	 torch.Size([64, 32, 1324])
    # ReLU output shape: 	 torch.Size([64, 32, 1324])
    # MaxPool1d output shape: 	 torch.Size([64, 32, 662])
    # Dropout output shape: 	 torch.Size([64, 32, 662])
    # The shape after view: torch.Size([64, 21184])


    # BCP, CDD and SICP are executed in parallel
    # Then look at the processing after the three feature sets are merged.
    x = torch.cat((images_BCP, images_CDD, images_SICP), 1)
    # 21184+672+21184=43040
    # torch.Size([64, 43040])，At this point each cell is a row of data, and then there are a total of 64 cells in a batch, corresponding to 64 rows of data
    net_houxv = nn.Sequential(
        nn.Linear(in_features=cnn_feature * (linear_size_BCP + linear_size_CDD + linear_size_SICP), out_features=4)
    )
    print("三个特征集合并以后的部分")
    print("三个特征集合并以后的shape", x.shape)
    for layer in net_houxv:
        x = layer(x)
        print(layer.__class__.__name__, 'output shape: \t', x.shape)
    # Output results.
    # The part after the merging of the three feature sets
    # The shape after merging the three feature sets torch.Size([64, 43040])
    # Linear output shape: torch.Size([64, 4])


    # Final softmax layer
    # If you use softmax in four classes, then the four values generated are all in the interval [0,1] and the sum of the four values is 1.
    # If log_softmax is used, the four values generated are in the interval [-infinity, 0], and the sum of the four values is not a fixed value, but still the probability of which value is greater is greater.
    #The advantage of log_softmax is that the computation process is smoother and there is no overflow problem, while it is more convenient and faster than, for example, softmax first and then log twice in the operation.
    x = nn.functional.log_softmax(x, dim=1)
    print("softmax层以后，最后的输出规模为：", x.shape)
    print(x)
    # After the softmax layer, the final output size is: torch.Size([64, 4])
    # tensor([[-1.1984, -2.0972, -1.3254, -1.1717],
    #         [-1.9821, -1.1155, -1.0043, -1.7828],
    #          .................................. ,
    #         [-1.9711, -2.1480, -0.5472, -1.7991]], grad_fn=<LogSoftmaxBackward0>)
    outputs = x
    print(outputs.data)
    dd, prediction = torch.max(outputs.data, 1)
    print(dd)
    print(prediction)
    print(images_BCP.size(0))

