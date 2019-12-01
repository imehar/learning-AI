
#### Base Model Accuracy :    82.48


#### [DepthWise Model Accuracy](3rd_DNN.ipynb) : 84.17


#### Depthwise Separable Model definition

```
model = Sequential()

#layer                                                                      #ouput                #receptive feild
model.add(SeparableConv2D(64, (3, 3), input_shape=(32, 32, 3),activation='relu',use_bias=False))     #30     #3
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(SeparableConv2D(128, (3, 3),activation='relu',use_bias=False))      #28                    #5
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(SeparableConv2D(128, (3, 3),activation='relu',use_bias=False))      #26                    #7
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(AveragePooling2D((2,2)))                                            #13                    #8
model.add(Convolution2D(64,(1,1),activation='relu',use_bias=False))           #13                    #8
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(SeparableConv2D(64, (3, 3),activation='relu',use_bias=False))       #11                    #12
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(SeparableConv2D(128, (3, 3),activation='relu',use_bias=False))      #9                     #16
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(SeparableConv2D(128, (3, 3),activation='relu',use_bias=False))      #7                     #20
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Convolution2D(64,(1,1),activation='relu',use_bias=False))           #7                     #20
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(SeparableConv2D(64, (3, 3),activation='relu',use_bias=False))       #5                     #24
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(SeparableConv2D(64, (3, 3),activation='relu',use_bias=False))       #3                     #28
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Convolution2D(10,(1,1),activation='relu',use_bias=False))           #3                     #28
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(SeparableConv2D(10, (3, 3),activation='softmax'))                   #1                     #32

model.add(Flatten())

```

<br>

#### Model logs

```
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.03.
390/390 [==============================] - 69s 176ms/step - loss: 1.7860 - acc: 0.3327 - val_loss: 3.8963 - val_acc: 0.3373
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0227445034.
390/390 [==============================] - 60s 154ms/step - loss: 1.4224 - acc: 0.4863 - val_loss: 1.9521 - val_acc: 0.4769
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0183150183.
390/390 [==============================] - 60s 154ms/step - loss: 1.2515 - acc: 0.5526 - val_loss: 1.4627 - val_acc: 0.5324
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0153295861.
390/390 [==============================] - 60s 154ms/step - loss: 1.1218 - acc: 0.5996 - val_loss: 1.3430 - val_acc: 0.5746
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0131810193.
390/390 [==============================] - 60s 155ms/step - loss: 1.0216 - acc: 0.6369 - val_loss: 1.0545 - val_acc: 0.6558
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0115606936.
390/390 [==============================] - 60s 154ms/step - loss: 0.9350 - acc: 0.6727 - val_loss: 0.9512 - val_acc: 0.6762
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.010295127.
390/390 [==============================] - 60s 155ms/step - loss: 0.8689 - acc: 0.6936 - val_loss: 1.0138 - val_acc: 0.6653
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0092793071.
390/390 [==============================] - 61s 156ms/step - loss: 0.8174 - acc: 0.7134 - val_loss: 0.8607 - val_acc: 0.7144
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0084459459.
390/390 [==============================] - 61s 156ms/step - loss: 0.7782 - acc: 0.7277 - val_loss: 0.8570 - val_acc: 0.7176
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0077499354.
390/390 [==============================] - 61s 155ms/step - loss: 0.7438 - acc: 0.7391 - val_loss: 0.7628 - val_acc: 0.7391
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0071599045.
390/390 [==============================] - 60s 155ms/step - loss: 0.7146 - acc: 0.7512 - val_loss: 0.6856 - val_acc: 0.7677
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0066533599.
390/390 [==============================] - 61s 155ms/step - loss: 0.6912 - acc: 0.7611 - val_loss: 0.6966 - val_acc: 0.7604
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0062137531.
390/390 [==============================] - 61s 156ms/step - loss: 0.6745 - acc: 0.7675 - val_loss: 0.6959 - val_acc: 0.7628
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.005828638.
390/390 [==============================] - 60s 155ms/step - loss: 0.6540 - acc: 0.7722 - val_loss: 0.6677 - val_acc: 0.7783
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0054884742.
390/390 [==============================] - 60s 155ms/step - loss: 0.6365 - acc: 0.7811 - val_loss: 0.6417 - val_acc: 0.7787
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0051858254.
390/390 [==============================] - 60s 154ms/step - loss: 0.6213 - acc: 0.7845 - val_loss: 0.6973 - val_acc: 0.7710
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.00491481.
390/390 [==============================] - 61s 156ms/step - loss: 0.6076 - acc: 0.7904 - val_loss: 0.6187 - val_acc: 0.7944
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0046707146.
390/390 [==============================] - 61s 155ms/step - loss: 0.5955 - acc: 0.7944 - val_loss: 0.5953 - val_acc: 0.7980
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0044497182.
390/390 [==============================] - 61s 156ms/step - loss: 0.5880 - acc: 0.7972 - val_loss: 0.6436 - val_acc: 0.7864
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.00424869.
390/390 [==============================] - 61s 155ms/step - loss: 0.5783 - acc: 0.8006 - val_loss: 0.5738 - val_acc: 0.8057
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0040650407.
390/390 [==============================] - 61s 156ms/step - loss: 0.5670 - acc: 0.8046 - val_loss: 0.5610 - val_acc: 0.8109
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.0038966099.
390/390 [==============================] - 61s 156ms/step - loss: 0.5579 - acc: 0.8074 - val_loss: 0.6676 - val_acc: 0.7764
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0037415814.
390/390 [==============================] - 61s 155ms/step - loss: 0.5463 - acc: 0.8098 - val_loss: 0.5683 - val_acc: 0.8079
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0035984167.
390/390 [==============================] - 61s 156ms/step - loss: 0.5483 - acc: 0.8095 - val_loss: 0.5607 - val_acc: 0.8159
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0034658041.
390/390 [==============================] - 61s 155ms/step - loss: 0.5362 - acc: 0.8148 - val_loss: 0.5958 - val_acc: 0.8037
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0033426184.
390/390 [==============================] - 61s 155ms/step - loss: 0.5305 - acc: 0.8162 - val_loss: 0.6051 - val_acc: 0.8003
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.003227889.
390/390 [==============================] - 61s 155ms/step - loss: 0.5253 - acc: 0.8171 - val_loss: 0.5717 - val_acc: 0.8127
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.003120774.
390/390 [==============================] - 61s 156ms/step - loss: 0.5190 - acc: 0.8206 - val_loss: 0.5321 - val_acc: 0.8207
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0030205397.
390/390 [==============================] - 61s 156ms/step - loss: 0.5155 - acc: 0.8228 - val_loss: 0.5758 - val_acc: 0.8145
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0029265438.
390/390 [==============================] - 60s 155ms/step - loss: 0.5108 - acc: 0.8222 - val_loss: 0.5250 - val_acc: 0.8215
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0028382214.
390/390 [==============================] - 61s 155ms/step - loss: 0.5066 - acc: 0.8255 - val_loss: 0.5256 - val_acc: 0.8240
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0027550739.
390/390 [==============================] - 61s 155ms/step - loss: 0.5015 - acc: 0.8258 - val_loss: 0.5278 - val_acc: 0.8265
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0026766595.
390/390 [==============================] - 60s 155ms/step - loss: 0.4968 - acc: 0.8281 - val_loss: 0.5169 - val_acc: 0.8302
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0026025852.
390/390 [==============================] - 61s 156ms/step - loss: 0.4879 - acc: 0.8321 - val_loss: 0.5279 - val_acc: 0.8230
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0025325004.
390/390 [==============================] - 61s 155ms/step - loss: 0.4888 - acc: 0.8303 - val_loss: 0.5192 - val_acc: 0.8241
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0024660912.
390/390 [==============================] - 61s 155ms/step - loss: 0.4846 - acc: 0.8343 - val_loss: 0.5068 - val_acc: 0.8325
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0024030759.
390/390 [==============================] - 61s 156ms/step - loss: 0.4842 - acc: 0.8326 - val_loss: 0.4981 - val_acc: 0.8300
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0023432008.
390/390 [==============================] - 61s 156ms/step - loss: 0.4783 - acc: 0.8356 - val_loss: 0.5315 - val_acc: 0.8271
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0022862369.
390/390 [==============================] - 60s 155ms/step - loss: 0.4732 - acc: 0.8356 - val_loss: 0.5138 - val_acc: 0.8332
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0022319768.
390/390 [==============================] - 61s 155ms/step - loss: 0.4730 - acc: 0.8370 - val_loss: 0.5124 - val_acc: 0.8294
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0021802326.
390/390 [==============================] - 61s 156ms/step - loss: 0.4700 - acc: 0.8372 - val_loss: 0.4971 - val_acc: 0.8339
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0021308332.
390/390 [==============================] - 61s 156ms/step - loss: 0.4645 - acc: 0.8392 - val_loss: 0.4894 - val_acc: 0.8336
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0020836227.
390/390 [==============================] - 61s 156ms/step - loss: 0.4615 - acc: 0.8398 - val_loss: 0.4994 - val_acc: 0.8394
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0020384589.
390/390 [==============================] - 61s 155ms/step - loss: 0.4586 - acc: 0.8410 - val_loss: 0.4854 - val_acc: 0.8383
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0019952115.
390/390 [==============================] - 61s 155ms/step - loss: 0.4571 - acc: 0.8409 - val_loss: 0.5133 - val_acc: 0.8316
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.001953761.
390/390 [==============================] - 61s 155ms/step - loss: 0.4547 - acc: 0.8420 - val_loss: 0.4796 - val_acc: 0.8406
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0019139977.
390/390 [==============================] - 61s 155ms/step - loss: 0.4527 - acc: 0.8427 - val_loss: 0.4933 - val_acc: 0.8391
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0018758207.
390/390 [==============================] - 61s 156ms/step - loss: 0.4520 - acc: 0.8430 - val_loss: 0.4847 - val_acc: 0.8389
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0018391368.
390/390 [==============================] - 61s 155ms/step - loss: 0.4472 - acc: 0.8456 - val_loss: 0.4879 - val_acc: 0.8408
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0018038603.
390/390 [==============================] - 60s 155ms/step - loss: 0.4514 - acc: 0.8437 - val_loss: 0.4834 - val_acc: 0.8417
Model took 3038.67 seconds to train


```
