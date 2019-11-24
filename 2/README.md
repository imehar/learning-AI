### [Score](2nd_DNN.ipynb) 0.9949

### Process for model building

Modified the previous architecture, so that the model learns features with a block of three convolutions in a block.
Used 1 X 1 convolution after each block to reduce z-dimension and used this as input to next block.




### Logs

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.0035.
60000/60000 [==============================] - 6s 98us/step - loss: 0.0965 - acc: 0.9696 - val_loss: 0.0605 - val_acc: 0.9804
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0026535254.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0599 - acc: 0.9813 - val_loss: 0.0478 - val_acc: 0.9839
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0021367521.
60000/60000 [==============================] - 6s 95us/step - loss: 0.0486 - acc: 0.9848 - val_loss: 0.0311 - val_acc: 0.9894
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0017884517.
60000/60000 [==============================] - 6s 95us/step - loss: 0.0405 - acc: 0.9873 - val_loss: 0.0275 - val_acc: 0.9912
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0015377856.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0374 - acc: 0.9882 - val_loss: 0.0259 - val_acc: 0.9915
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0013487476.
60000/60000 [==============================] - 6s 94us/step - loss: 0.0355 - acc: 0.9889 - val_loss: 0.0274 - val_acc: 0.9907
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0012010981.
60000/60000 [==============================] - 6s 95us/step - loss: 0.0311 - acc: 0.9899 - val_loss: 0.0253 - val_acc: 0.9919
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0010825858.
60000/60000 [==============================] - 6s 94us/step - loss: 0.0299 - acc: 0.9906 - val_loss: 0.0220 - val_acc: 0.9933
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0009853604.
60000/60000 [==============================] - 6s 95us/step - loss: 0.0280 - acc: 0.9911 - val_loss: 0.0251 - val_acc: 0.9925
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0009041591.
60000/60000 [==============================] - 6s 94us/step - loss: 0.0272 - acc: 0.9914 - val_loss: 0.0189 - val_acc: 0.9936
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0008353222.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0248 - acc: 0.9923 - val_loss: 0.0190 - val_acc: 0.9942
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.0007762253.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0244 - acc: 0.9918 - val_loss: 0.0228 - val_acc: 0.9931
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0007249379.
60000/60000 [==============================] - 6s 95us/step - loss: 0.0234 - acc: 0.9927 - val_loss: 0.0203 - val_acc: 0.9938
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0006800078.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0227 - acc: 0.9927 - val_loss: 0.0215 - val_acc: 0.9927
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.000640322.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0221 - acc: 0.9925 - val_loss: 0.0173 - val_acc: 0.9947
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.000605013.
60000/60000 [==============================] - 6s 94us/step - loss: 0.0212 - acc: 0.9930 - val_loss: 0.0189 - val_acc: 0.9935
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.0005733945.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0205 - acc: 0.9931 - val_loss: 0.0172 - val_acc: 0.9945
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0005449167.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0201 - acc: 0.9931 - val_loss: 0.0204 - val_acc: 0.9936
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0005191338.
60000/60000 [==============================] - 6s 95us/step - loss: 0.0194 - acc: 0.9934 - val_loss: 0.0185 - val_acc: 0.9943
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.0004956805.
60000/60000 [==============================] - 6s 94us/step - loss: 0.0183 - acc: 0.9942 - val_loss: 0.0175 - val_acc: 0.9949
