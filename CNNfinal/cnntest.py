#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

import os

import numpy as np
import theano

import lasagne
from lasagne import layers
from lasagne.updates import adam

from bj_loss import binary_jaccard_index
from deconv_3d import Conv3DLayerTransposed
from softmax import softmax

from lasagne import init
from lasagne import nonlinearities

#load the data

#load the data

data = np.load('../data-160-160-24.npy')
seg_data = np.load('../segdata-160-160-24.npy')

data = np.expand_dims(data, axis=1)
seg_data = np.expand_dims(seg_data, axis=1)

print(data.shape)
print(seg_data.shape)

train_data = data[:8]
test_data = data[-2:]
train_seg = seg_data[:8]
test_seg = seg_data[-2:]

#train_data /= np.max(train_data)
#print(np.max(train_data))

#prikaze eno 3d sliko
def display_numpy(picture):
    fig = plt.figure()
    for num,slice in enumerate(picture):
        y = fig.add_subplot(4,6,num+1)
        y.imshow(slice, cmap='gray')
    plt.show()


input_var = theano.tensor.tensor5()
target_var = theano.tensor.tensor5()


layer = lasagne.layers.InputLayer((None, 1, 24, 160, 160), input_var)

#start
layer = layers.Conv3DLayer(layer, num_filters=8, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))

tmp1 = layer

#contracting_block 1
layer = layers.Conv3DLayer(layer, num_filters=8, filter_size=(2, 2, 2), stride=2, pad=0, W=lasagne.init.Normal(), nonlinearity=None)
tmp_layer1 = layer
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))
layer = layers.Conv3DLayer(layer, num_filters=16, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.ConcatLayer([layer, tmp_layer1])
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))
	
tmp2 = layer

#contracting_block 2
layer = layers.Conv3DLayer(layer, num_filters=32, filter_size=(2, 2, 2), stride=2, pad=0, W=lasagne.init.Normal(), nonlinearity=None)
tmp_layer2 = layer
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))
layer = layers.Conv3DLayer(layer, num_filters=32, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.ConcatLayer([layer, tmp_layer2])
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))

tmp3 = layer

#contracting_block 3
layer = layers.Conv3DLayer(layer, num_filters=64, filter_size=(2, 2, 2), stride=2, pad=0, W=lasagne.init.Normal(), nonlinearity=None)
tmp_layer3 = layer
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))
layer = layers.Conv3DLayer(layer, num_filters=64, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.ConcatLayer([layer, tmp_layer3])
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))

#expanding_block 1
layer = layers.Conv3DLayer(layer, num_filters=32, filter_size=(1, 1, 1), stride=1, pad=0, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))
#stevilo filtrov pri dekonvoluciji?
layer = Conv3DLayerTransposed(layer, num_filters=32, filter_size=(2, 2, 2), stride=2, crop=0, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))
layer = layers.ConcatLayer([layer, tmp3])
#layer = layers.ElemwiseSumLayer([layer, tmp])
layer = layers.Conv3DLayer(layer, num_filters=64, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))
#tmp4 = layers.Conv3DLayer(layer, num_filters=1, filter_size=(1, 1, 1), stride=1, pad=0)

#expanding_block 2
layer = layers.Conv3DLayer(layer, num_filters=16, filter_size=(1, 1, 1), stride=1, pad=0, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))
#stevilo filtrov pri dekonvoluciji?
layer = Conv3DLayerTransposed(layer, num_filters=16, filter_size=(2, 2, 2), stride=2, crop=0, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))
layer = layers.ConcatLayer([layer, tmp2])
#layer = layers.ElemwiseSumLayer([layer, tmp])
layer = layers.Conv3DLayer(layer, num_filters=32, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))
#tmp5 = layers.Conv3DLayer(layer, num_filters=1, filter_size=(1, 1, 1), stride=1, pad=0)

#expanding_block 3
layer = layers.Conv3DLayer(layer, num_filters=8, filter_size=(1, 1, 1), stride=1, pad=0, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))
#stevilo filtrov pri dekonvoluciji?
layer = Conv3DLayerTransposed(layer, num_filters=8, filter_size=(2, 2, 2), stride=2, crop=0, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))
layer = layers.ConcatLayer([layer, tmp1])
#layer = layers.ElemwiseSumLayer([layer, tmp])
layer = layers.Conv3DLayer(layer, num_filters=16, filter_size=(3, 3, 3), stride=1, pad=1, W=lasagne.init.Normal(), nonlinearity=None)
layer = layers.BatchNormLayer(layer)
layer = layers.ParametricRectifierLayer(layer, alpha=init.Constant(0.1))
layer = layers.Conv3DLayer(layer, num_filters=2, filter_size=(1, 1, 1), stride=1, pad=0)

#agregacija

#layer = layers.NonlinearityLayer(layer, nonlinearity=lasagne.nonlinearities.sigmoid)
#tmp6 = layers.Conv3DLayer(layer, num_filters=1, filter_size=(1, 1, 1), stride=1, pad=0)

prediction = lasagne.layers.get_output(layer)
prediction = softmax(prediction, (8,2,24,160,160), 2, 1)

loss = binary_jaccard_index(prediction[:,1:2,:,:,:], target_var)
loss += binary_jaccard_index(prediction[:,0:1,:,:,:], target_var)
#loss = binary_jaccard_index(prediction, target_var)
loss = loss.mean() 

params = lasagne.layers.get_all_params(layer, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=0.001, beta1=0.1, beta2=0.001)#, rho=0.9, epsilon=1e-06) #

train_fn = theano.function([input_var, target_var], [loss, prediction], updates=updates)

for epoch in range(500):
    loss = train_fn(train_data, train_seg)
    #print(loss[1][1][0][10][50][25:75])
    #display_numpy(loss[1][0][0]) 
    print("Epoch %d: Loss %g" % (epoch + 1, loss[0]))
    

test_prediction = lasagne.layers.get_output(layer, deterministic=True)
test_prediction = softmax(test_prediction, (1,2,24,160,160), 2, 1)
predict_fn = theano.function([input_var], [theano.tensor.argmax(test_prediction, axis=1)])
p1 = predict_fn(test_data[0])
print(type(p1))
p1 = np.array(p1)
print(p1.shape)
display_numpy(p1[0][0])
print(binary_jaccard_index_test(p1, test_seg[0]))
print(((p1 - test_seg[0]) ** 2).mean(axis=None))