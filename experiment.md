
需要很大的数据量来使模型收敛

但是4w张图片内存就吃紧了，5w张就直接内存不足

解决方案，采用 keras 的 ImageDataGenerator 类
但是此问题是多损失问题，不能直接使用，最好的方式是继承之后修改。
但是目前不知道怎么修改，所以只有直接 copy keras 的源码，然后修改
将其中一个函数的返回值 (batch_x, batch_y) 修改为： 
(batch_x,[batch_y, batch_y, batch_y])
还有一些其他问题，如某个函数确实，需要自己再找。

通过这种 batch 的方式，训练时内存仅仅占 6G 左右。

3/12/2019
epochs = 40
并未收敛
196/196 [==============================] - 999s 5s/step 
- loss: 0.4630 - output_loss: 0.2054 - auxiliary_output_1_loss: 0.5306 - auxiliary_output_2_loss: 0.3279 - output_acc: 0.9275 - auxiliary_output_1_acc: 0.8149 - auxiliary_output_2_acc: 0.8882 
- val_loss: 0.7020 - val_output_loss: 0.4127 - val_auxiliary_output_1_loss: 0.5257 - val_auxiliary_output_2_loss: 0.4388 - val_output_acc: 0.8712 - val_auxiliary_output_1_acc: 0.8179 - val_auxiliary_output_2_acc: 0.8560

#### Inception_v1_1
epochs = 200

earlystopping.patience = 10

训练了14个小时，val_acc有87.8
patience = 10 感觉收敛不是太明显，再稍微设置大一点


#### experiment 2
epochs = 200

earlystopping.patience = 15

增加bach_normalization

## new
Trainable params: 10,337,230

### experiment 1
batch size = 512 out of memory
1. batch size = 400
2. gpus =2
3. optimizer = adamax
