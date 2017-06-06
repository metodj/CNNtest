import theano.tensor as T

def softmax(inputToSoftmax, inputToSoftmaxShape, numberOfOutputClasses, softmaxTemperature):
    # The softmax function works on 2D tensors (matrices). It computes the softmax for each row. Rows are independent, eg different samples in the batch. Columns are the input features, eg class-scores.
    # Softmax's input 2D matrix should have shape like: [ datasamples, #Classess ]
    # My class-scores/class-FMs are a 5D tensor (batchSize, #Classes, r, c, z).
    # I need to reshape it to a 2D tensor.
    # The reshaped 2D Tensor will have dimensions: [ batchSize * r * c * z , #Classses ]
    # The order of the elements in the rows after the reshape should be :
     
    inputToSoftmaxReshaped = inputToSoftmax.dimshuffle(0, 2, 3, 4, 1) # [batchSize, r, c, z, #classes), the classes stay as the last dimension.
    inputToSoftmaxFlattened = inputToSoftmaxReshaped.flatten(1) 
    # flatten is "Row-major" 'C' style. ie, starts from index [0,0,0] and grabs elements in order such that last dim index increases first and first index increases last. (first row flattened, then second follows, etc)
    numberOfVoxelsDenselyClassified = inputToSoftmaxShape[2]*inputToSoftmaxShape[3]*inputToSoftmaxShape[4]
    firstDimOfInputToSoftmax2d = inputToSoftmaxShape[0]*numberOfVoxelsDenselyClassified # batchSize*r*c*z.
    inputToSoftmax2d = inputToSoftmaxFlattened.reshape((firstDimOfInputToSoftmax2d, numberOfOutputClasses)) # Reshape works in "Row-major", ie 'C' style too.
    # Predicted probability per class.
    p_y_given_x_2d = T.nnet.softmax(inputToSoftmax2d/softmaxTemperature)
    # Segmentation (EM) for each voxel
    y_pred_2d = T.argmax(p_y_given_x_2d, axis=1)
     
    p_y_given_x_classMinor = p_y_given_x_2d.reshape((inputToSoftmaxShape[0], inputToSoftmaxShape[2], inputToSoftmaxShape[3], inputToSoftmaxShape[4], inputToSoftmaxShape[1])) #Result: batchSize, R,C,Z, Classes.
 
    p_y_given_x = p_y_given_x_classMinor.dimshuffle(0,4,1,2,3) #Result: batchSize, Class, R, C, Z
    y_pred = y_pred_2d.reshape((inputToSoftmaxShape[0], inputToSoftmaxShape[2], inputToSoftmaxShape[3], inputToSoftmaxShape[4])) #Result: batchSize, R, C, Z
     
    #return also: p_y_given_x_2d_train = p_y_given_x_2d_train # For convenience in negativeLogLikelihood. Not sure how to implement more efficiently. It would get rid of this if I would.
    #return ( p_y_given_x, y_pred )
    return p_y_given_x