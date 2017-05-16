#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

import os

import numpy as np
import theano

import lasagne
from lasagne import layers
from lasagne.updates import adam

import bj_loss
import deconv_3d

#load the data

data = np.load('../data-160-160-24.npy')
seg_data = np.load('../segdata-160-160-24.npy')

data = np.expand_dims(data, axis=1)
seg_data = np.expand_dims(seg_data, axis=1)

print(data.shape)
print(seg_data.shape)

train_data = data[:-2]
validation_data = data[-2:]
train_seg = seg_data[:-2]
validation_seg = seg_data[-2:]

train_data *= 1/np.max(train_data)

#display_numpy(train_data[1][0])
#display_numpy(train_seg[1][0])
#print(train_data[1][0][10][50][25:75])
#print(train_seg[1][0][10][50][25:75])

for a in range(8):
    for i in range(24):
        for j in range(160):
            for k in range(160):
                if (train_seg[a][0][i][j][k]>0.5):
                    train_seg[a][0][i][j][k] = 1
                    #print(str(i)+" "+str(j)+" "+str(k))
                else:
                    train_seg[a][0][i][j][k] = 0



input_var = theano.tensor.tensor5()
target_var = theano.tensor.tensor5()

layer = lasagne.layers.InputLayer((None, 1, 24, 160, 160), input_var)

#start
layer = layers.Conv3DLayer(layer, num_filters=8, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))

tmp1 = layer

#contracting_block 1
layer = layers.Conv3DLayer(layer, num_filters=8, filter_size=(2, 2, 2), stride=2, pad=0, W=lasagne.init.Normal(), nonlinearity=None)
tmp_layer1 = layer
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))
layer = layers.Conv3DLayer(layer, num_filters=16, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.ConcatLayer([layer, tmp_layer1])
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))
	
tmp2 = layer

#contracting_block 2
layer = layers.Conv3DLayer(layer, num_filters=32, filter_size=(2, 2, 2), stride=2, pad=0, W=lasagne.init.Normal(), nonlinearity=None)
tmp_layer2 = layer
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))
layer = layers.Conv3DLayer(layer, num_filters=32, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.ConcatLayer([layer, tmp_layer2])
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))

tmp3 = layer

#contracting_block 3
layer = layers.Conv3DLayer(layer, num_filters=64, filter_size=(2, 2, 2), stride=2, pad=0, W=lasagne.init.Normal(), nonlinearity=None)
tmp_layer3 = layer
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))
layer = layers.Conv3DLayer(layer, num_filters=64, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.ConcatLayer([layer, tmp_layer3])
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))

#expanding_block 1
layer = layers.Conv3DLayer(layer, num_filters=32, filter_size=(1, 1, 1), stride=1, pad=0, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))
#število filtrov pri dekonvoluciji?
layer = Conv3DLayerTransposed(layer, num_filters=32, filter_size=(2, 2, 2), stride=2, crop=0, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))
layer = layers.ConcatLayer([layer, tmp3])
#layer = layers.ElemwiseSumLayer([layer, tmp])
layer = layers.Conv3DLayer(layer, num_filters=64, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))
#tmp4 = layers.Conv3DLayer(layer, num_filters=1, filter_size=(1, 1, 1), stride=1, pad=0)

#expanding_block 2
layer = layers.Conv3DLayer(layer, num_filters=16, filter_size=(1, 1, 1), stride=1, pad=0, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))
#število filtrov pri dekonvoluciji?
layer = Conv3DLayerTransposed(layer, num_filters=16, filter_size=(2, 2, 2), stride=2, crop=0, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))
layer = layers.ConcatLayer([layer, tmp2])
#layer = layers.ElemwiseSumLayer([layer, tmp])
layer = layers.Conv3DLayer(layer, num_filters=32, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))
#tmp5 = layers.Conv3DLayer(layer, num_filters=1, filter_size=(1, 1, 1), stride=1, pad=0)

#expanding_block 3
layer = layers.Conv3DLayer(layer, num_filters=8, filter_size=(1, 1, 1), stride=1, pad=0, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))
#število filtrov pri dekonvoluciji?
layer = Conv3DLayerTransposed(layer, num_filters=8, filter_size=(2, 2, 2), stride=2, crop=0, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))
layer = layers.ConcatLayer([layer, tmp1])
#layer = layers.ElemwiseSumLayer([layer, tmp])
layer = layers.Conv3DLayer(layer, num_filters=16, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0))
layer = layers.Conv3DLayer(layer, num_filters=1, filter_size=(1, 1, 1), stride=1, pad=0)

#agregacija

layer = layers.NonlinearityLayer(layer, nonlinearity=lasagne.nonlinearities.sigmoid)
#tmp6 = layers.Conv3DLayer(layer, num_filters=1, filter_size=(1, 1, 1), stride=1, pad=0)

prediction = lasagne.layers.get_output(layer)

loss = binary_jaccard_index(prediction, target_var)
loss = loss.mean() 

params = lasagne.layers.get_all_params(layer, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=0.00005, beta1=0.1, beta2=0.001)

train_fn = theano.function([input_var, target_var], [loss, prediction], updates=updates)

for epoch in range(10):
    loss = train_fn(train_data, train_seg)
    #print(loss[1][1][0][10][50][25:75])
    #display_numpy(loss[1][0][0]) 
    print("Epoch %d: Loss %g" % (epoch + 1, loss[0]))