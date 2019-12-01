
#### Base model Accuracy :    82.48


#### [DepthWise Model Accuracy](3rd_DNN.ipynb) : 83.93


#### Depthwise Separable Model definition

```
model = Sequential()

#layer                                                          #ouput                #receptive feild
model.add(SeparableConv2D(64, (3, 3), input_shape=(32, 32, 3),activation='relu'))     #30     #3
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(SeparableConv2D(128, (3, 3),activation='relu'))      #28                    #5
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(SeparableConv2D(128, (3, 3),activation='relu'))      #26                    #7
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(AveragePooling2D((2,2)))                             #13                    #8
model.add(Convolution2D(64,(1,1),activation='relu'))           #13                    #8
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(SeparableConv2D(64, (3, 3),activation='relu'))       #11                    #12
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(SeparableConv2D(128, (3, 3),activation='relu'))      #9                     #16
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(SeparableConv2D(128, (3, 3),activation='relu'))      #7                     #20
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Convolution2D(64,(1,1),activation='relu'))           #7                     #20
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(SeparableConv2D(64, (3, 3),activation='relu'))       #5                     #24
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(SeparableConv2D(128, (3, 3),activation='relu'))      #3                     #28
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Convolution2D(10,(1,1),activation='relu'))           #3                     #28
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(SeparableConv2D(10, (3, 3),activation='softmax'))    #1                     #32

model.add(Flatten())
```

<br>

#### Model logs

```
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.03.
390/390 [==============================] - 51s 132ms/step - loss: 1.7198 - acc: 0.3643 - val_loss: 5.1467 - val_acc: 0.3503
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0227445034.
390/390 [==============================] - 37s 94ms/step - loss: 1.3063 - acc: 0.5310 - val_loss: 2.1110 - val_acc: 0.4530
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0183150183.
390/390 [==============================] - 36s 93ms/step - loss: 1.1464 - acc: 0.5926 - val_loss: 2.1538 - val_acc: 0.4663
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0153295861.
390/390 [==============================] - 36s 93ms/step - loss: 1.0246 - acc: 0.6398 - val_loss: 1.0036 - val_acc: 0.6633
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0131810193.
390/390 [==============================] - 36s 93ms/step - loss: 0.9329 - acc: 0.6736 - val_loss: 1.1021 - val_acc: 0.6392
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0115606936.
390/390 [==============================] - 36s 93ms/step - loss: 0.8695 - acc: 0.6953 - val_loss: 1.2422 - val_acc: 0.6212
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.010295127.
390/390 [==============================] - 36s 93ms/step - loss: 0.8075 - acc: 0.7194 - val_loss: 0.8674 - val_acc: 0.7072
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0092793071.
390/390 [==============================] - 36s 93ms/step - loss: 0.7646 - acc: 0.7328 - val_loss: 0.7752 - val_acc: 0.7360
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0084459459.
390/390 [==============================] - 36s 92ms/step - loss: 0.7222 - acc: 0.7495 - val_loss: 0.7523 - val_acc: 0.7465
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0077499354.
390/390 [==============================] - 35s 91ms/step - loss: 0.6910 - acc: 0.7604 - val_loss: 0.8374 - val_acc: 0.7296
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0071599045.
390/390 [==============================] - 35s 91ms/step - loss: 0.6661 - acc: 0.7669 - val_loss: 0.7171 - val_acc: 0.7586
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0066533599.
390/390 [==============================] - 36s 91ms/step - loss: 0.6446 - acc: 0.7768 - val_loss: 0.7093 - val_acc: 0.7590
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0062137531.
390/390 [==============================] - 35s 91ms/step - loss: 0.6232 - acc: 0.7829 - val_loss: 0.6732 - val_acc: 0.7770
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.005828638.
390/390 [==============================] - 35s 91ms/step - loss: 0.6061 - acc: 0.7891 - val_loss: 0.6656 - val_acc: 0.7807
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0054884742.
390/390 [==============================] - 35s 91ms/step - loss: 0.5981 - acc: 0.7911 - val_loss: 0.6002 - val_acc: 0.8054
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0051858254.
390/390 [==============================] - 35s 91ms/step - loss: 0.5747 - acc: 0.8008 - val_loss: 0.6911 - val_acc: 0.7742
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.00491481.
390/390 [==============================] - 35s 91ms/step - loss: 0.5618 - acc: 0.8043 - val_loss: 0.6120 - val_acc: 0.8025
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0046707146.
390/390 [==============================] - 36s 91ms/step - loss: 0.5538 - acc: 0.8093 - val_loss: 0.6574 - val_acc: 0.7872
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0044497182.
390/390 [==============================] - 35s 91ms/step - loss: 0.5419 - acc: 0.8112 - val_loss: 0.5981 - val_acc: 0.8016
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.00424869.
390/390 [==============================] - 36s 91ms/step - loss: 0.5324 - acc: 0.8169 - val_loss: 0.5910 - val_acc: 0.8042
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0040650407.
390/390 [==============================] - 36s 92ms/step - loss: 0.5261 - acc: 0.8186 - val_loss: 0.5802 - val_acc: 0.8117
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.0038966099.
390/390 [==============================] - 37s 94ms/step - loss: 0.5156 - acc: 0.8211 - val_loss: 0.5565 - val_acc: 0.8125
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0037415814.
390/390 [==============================] - 37s 94ms/step - loss: 0.5019 - acc: 0.8240 - val_loss: 0.5534 - val_acc: 0.8180
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0035984167.
390/390 [==============================] - 36s 93ms/step - loss: 0.4969 - acc: 0.8282 - val_loss: 0.5560 - val_acc: 0.8171
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0034658041.
390/390 [==============================] - 36s 91ms/step - loss: 0.4941 - acc: 0.8292 - val_loss: 0.5621 - val_acc: 0.8111
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0033426184.
390/390 [==============================] - 35s 91ms/step - loss: 0.4795 - acc: 0.8334 - val_loss: 0.5590 - val_acc: 0.8182
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.003227889.
390/390 [==============================] - 36s 92ms/step - loss: 0.4810 - acc: 0.8330 - val_loss: 0.5544 - val_acc: 0.8198
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.003120774.
390/390 [==============================] - 36s 91ms/step - loss: 0.4734 - acc: 0.8357 - val_loss: 0.5811 - val_acc: 0.8136
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0030205397.
390/390 [==============================] - 36s 91ms/step - loss: 0.4677 - acc: 0.8371 - val_loss: 0.5600 - val_acc: 0.8124
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0029265438.
390/390 [==============================] - 36s 91ms/step - loss: 0.4656 - acc: 0.8389 - val_loss: 0.5408 - val_acc: 0.8243
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0028382214.
390/390 [==============================] - 36s 91ms/step - loss: 0.4511 - acc: 0.8434 - val_loss: 0.5404 - val_acc: 0.8224
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0027550739.
390/390 [==============================] - 35s 91ms/step - loss: 0.4527 - acc: 0.8419 - val_loss: 0.5469 - val_acc: 0.8232
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0026766595.
390/390 [==============================] - 35s 91ms/step - loss: 0.4474 - acc: 0.8445 - val_loss: 0.5548 - val_acc: 0.8218
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0026025852.
390/390 [==============================] - 35s 91ms/step - loss: 0.4438 - acc: 0.8434 - val_loss: 0.5298 - val_acc: 0.8272
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0025325004.
390/390 [==============================] - 35s 90ms/step - loss: 0.4365 - acc: 0.8455 - val_loss: 0.5253 - val_acc: 0.8283
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0024660912.
390/390 [==============================] - 35s 90ms/step - loss: 0.4317 - acc: 0.8491 - val_loss: 0.5407 - val_acc: 0.8268
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0024030759.
390/390 [==============================] - 35s 90ms/step - loss: 0.4319 - acc: 0.8485 - val_loss: 0.5363 - val_acc: 0.8280
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0023432008.
390/390 [==============================] - 35s 91ms/step - loss: 0.4227 - acc: 0.8518 - val_loss: 0.5619 - val_acc: 0.8259
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0022862369.
390/390 [==============================] - 35s 91ms/step - loss: 0.4249 - acc: 0.8510 - val_loss: 0.5183 - val_acc: 0.8320
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0022319768.
390/390 [==============================] - 35s 90ms/step - loss: 0.4220 - acc: 0.8533 - val_loss: 0.5086 - val_acc: 0.8344
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0021802326.
390/390 [==============================] - 35s 90ms/step - loss: 0.4165 - acc: 0.8532 - val_loss: 0.5089 - val_acc: 0.8370
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0021308332.
390/390 [==============================] - 35s 90ms/step - loss: 0.4173 - acc: 0.8531 - val_loss: 0.5386 - val_acc: 0.8290
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0020836227.
390/390 [==============================] - 35s 91ms/step - loss: 0.4128 - acc: 0.8562 - val_loss: 0.5163 - val_acc: 0.8313
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0020384589.
390/390 [==============================] - 36s 92ms/step - loss: 0.4112 - acc: 0.8572 - val_loss: 0.5249 - val_acc: 0.8366
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0019952115.
390/390 [==============================] - 36s 93ms/step - loss: 0.4070 - acc: 0.8564 - val_loss: 0.5342 - val_acc: 0.8316
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.001953761.
390/390 [==============================] - 37s 94ms/step - loss: 0.4079 - acc: 0.8556 - val_loss: 0.5095 - val_acc: 0.8386
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0019139977.
390/390 [==============================] - 36s 93ms/step - loss: 0.3974 - acc: 0.8610 - val_loss: 0.5346 - val_acc: 0.8296
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0018758207.
390/390 [==============================] - 37s 94ms/step - loss: 0.4001 - acc: 0.8592 - val_loss: 0.5316 - val_acc: 0.8325
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0018391368.
390/390 [==============================] - 36s 93ms/step - loss: 0.3994 - acc: 0.8621 - val_loss: 0.5419 - val_acc: 0.8240
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0018038603.
390/390 [==============================] - 36s 93ms/step - loss: 0.3945 - acc: 0.8622 - val_loss: 0.5142 - val_acc: 0.8393
Model took 1807.00 seconds to train

```
