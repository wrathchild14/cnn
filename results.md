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

```

## bs 64 lr 0.0001
```

```