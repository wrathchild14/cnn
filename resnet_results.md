# ResNet18 implementation
Results of quick testing with my GPU: RTX 3060 12GB

(bs = batch size,
lr = learning rate)

## bs 64 lr 0.001

```
Epoch: 1/10: 116800img [02:22, 817.77img/s, loss=3.3579032]
Accuracy of the network on val images: 27 %
Test loss: 0.37848195791244504
Epoch: 2/10: 116800img [02:25, 802.28img/s, loss=1.8222764]
Epoch: 3/10: 116800img [02:29, 783.01img/s, loss=0.9606193]
Epoch: 4/10: 116800img [02:29, 780.68img/s, loss=1.0076089]
Epoch: 5/10: 116800img [02:26, 796.19img/s, loss=0.6762087]
Epoch: 6/10: 116800img [02:29, 779.77img/s, loss=0.30723077]
Accuracy of the network on val images: 85 %
Test loss: 0.07377233630418778
Epoch: 7/10: 116800img [02:20, 832.49img/s, loss=0.42402822]
Epoch: 8/10: 116800img [02:16, 854.13img/s, loss=0.24241826]
Epoch: 9/10: 116800img [02:23, 816.48img/s, loss=0.081934065]
Epoch: 10/10: 116800img [02:29, 783.85img/s, loss=0.31856334]
Accuracy of the network on val images: 86 %
Test loss: 0.07679324591159821
```
## bs 32 lr 0.001

```
Epoch: 1/10: 58400img [03:49, 254.44img/s, loss=3.5501328]                          
Epoch: 2/10: 58400img [04:22, 222.27img/s, loss=2.0432203]                          
Epoch: 3/10: 58400img [02:22, 410.72img/s, loss=1.6506236]
Epoch: 4/10: 58400img [02:18, 422.07img/s, loss=0.9078821]
Epoch: 5/10: 58400img [02:18, 420.37img/s, loss=0.6659681]
Accuracy of the network on val images: 80 %
Test loss: 0.09524237978458404
Epoch: 6/10: 58400img [02:16, 428.19img/s, loss=1.2105707]
Epoch: 7/10: 58400img [02:15, 429.61img/s, loss=1.1813657]
Epoch: 8/10: 58400img [02:15, 430.15img/s, loss=0.2881235]
Epoch: 9/10: 58400img [02:15, 429.56img/s, loss=0.059782557]
Epoch: 10/10: 58400img [02:15, 429.84img/s, loss=0.15580152]
Accuracy of the network on val images: 89 %
Test loss: 0.0511300667161122
```

## bs 32 lr 0.0001
```
Epoch: 0/10: 58400img [02:23, 406.13img/s, loss=3.4892979]
Accuracy of the network on val images: 37 %
Test loss: 0.34654673910140993
Epoch: 1/10: 58400img [02:19, 418.95img/s, loss=2.8399792]
Epoch: 2/10: 58400img [02:19, 418.50img/s, loss=1.5269742]
Epoch: 3/10: 58400img [02:19, 417.92img/s, loss=0.73510504]
Epoch: 4/10: 58400img [02:19, 418.92img/s, loss=0.8415941]
Epoch: 5/10: 58400img [02:16, 428.26img/s, loss=0.7340727]
Accuracy of the network on val images: 81 %
Test loss: 0.08508806946873665
Epoch: 6/10: 58400img [02:16, 428.93img/s, loss=0.20069921]
Epoch: 7/10: 58400img [02:14, 432.88img/s, loss=0.20231822]
Epoch: 8/10: 58400img [02:14, 432.92img/s, loss=0.18289861]
Epoch: 9/10: 58400img [02:16, 427.28img/s, loss=0.2368166]
Accuracy of the network on val images: 89 %
Test loss: 0.052206455644220114
```

## bs 64 lr 0.0001
```
Epoch: 0/10: 116800img [02:14, 866.09img/s, loss=0.39219618]
Accuracy of the network on val images: 86 %
Test loss: 0.06760005807876587
Epoch: 1/10: 116800img [02:14, 866.54img/s, loss=0.3007535]
Epoch: 2/10: 116800img [02:15, 864.67img/s, loss=0.08396077]
Epoch: 3/10: 116800img [02:14, 869.32img/s, loss=0.016562518]
Epoch: 4/10: 116800img [02:14, 870.90img/s, loss=0.087207586]
Epoch: 5/10: 116800img [02:14, 870.90img/s, loss=0.22110812]
Accuracy of the network on val images: 87 %
Test loss: 0.07419805264472962
Epoch: 6/10: 116800img [02:13, 871.89img/s, loss=0.33264256]
Epoch: 7/10: 116800img [02:15, 862.16img/s, loss=0.31310457]
Epoch: 8/10: 116800img [02:14, 871.26img/s, loss=0.05913632]
Epoch: 9/10: 116800img [02:14, 871.33img/s, loss=0.0032790988]
Accuracy of the network on val images: 92 %
Test loss: 0.04202882882021368
```

## print(ResNet)

```
ResNet18(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
```