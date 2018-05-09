
# coding: utf-8

# In[1]:

from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Input, Masking, add, concatenate
from keras.layers import Embedding, GRU, LSTM, CuDNNGRU, CuDNNLSTM, TimeDistributed, Bidirectional
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.optimizers import Adam, Nadam, RMSprop, SGD
from keras.regularizers import l2, l1, l1_l2
from keras import backend as K


# # MLP network designs

# In[ ]:

def cs_setup_mlp(params, inshape=None, classes=None):
    
    # Parse network hyperparameters
    num_layer = int(params['num_layer'])
    units1 = int(params['layer1_units'])
    units2 = int(params['layer2_units'])
    units3 = int(params['layer3_units'])
    units4 = int(params['layer4_units'])
    units5 = int(params['layer5_units'])
    units6 = int(params['layer6_units'])
    relu_flag = str(params['relu_type'])
    dropval = float(params['dropval'])
    reg_flag = str(params['reg_type'])
    reg_val = 10**(-float(params['reg_val']))
    
    # Setup regularizer
    if reg_flag == "l1":
        reg = l1(reg_val)
        print("Regularizer "+reg_flag+" set at "+str(reg_val))
    elif reg_flag == "l2":
        reg = l2(reg_val)
        print("Regularizer "+reg_flag+" set at "+str(reg_val))
    elif reg_flag == "l1_l2":
        reg = l1_l2(reg_val)
        print("Regularizer "+reg_flag+" set at "+str(reg_val))
    else:
        reg = None
        print("NOTE: No regularizers used")
        
    # Setup neural network
    inlayer = Input(shape=[inshape])
    if num_layer >= 1:
        x = Dense(units1, kernel_regularizer=reg)(inlayer)
        if relu_flag == "relu":
            x = Activation("relu")(x)
        elif relu_flag == "elu":
            x = Activation("elu")(x)
        elif relu_flag == "prelu":
            x = PReLU()(x)
        elif relu_flag == "leakyrelu":
            x = LeakyReLU()(x)
        x = Dropout(dropval)(x)
    if num_layer >= 2:
        x = Dense(units2, kernel_regularizer=reg)(x)
        if relu_flag == "relu":
            x = Activation("relu")(x)
        elif relu_flag == "elu":
            x = Activation("elu")(x)
        elif relu_flag == "prelu":
            x = PReLU()(x)
        elif relu_flag == "leakyrelu":
            x = LeakyReLU()(x)
        x = Dropout(dropval)(x)
    if num_layer >= 3:
        x = Dense(units3, kernel_regularizer=reg)(x)
        if relu_flag == "relu":
            x = Activation("relu")(x)
        elif relu_flag == "elu":
            x = Activation("elu")(x)
        elif relu_flag == "prelu":
            x = PReLU()(x)
        elif relu_flag == "leakyrelu":
            x = LeakyReLU()(x)
        x = Dropout(dropval)(x)
    if num_layer >= 4:
        x = Dense(units4, kernel_regularizer=reg)(x)
        if relu_flag == "relu":
            x = Activation("relu")(x)
        elif relu_flag == "elu":
            x = Activation("elu")(x)
        elif relu_flag == "prelu":
            x = PReLU()(x)
        elif relu_flag == "leakyrelu":
            x = LeakyReLU()(x)
        x = Dropout(dropval)(x)
    if num_layer >= 5:
        x = Dense(units5, kernel_regularizer=reg)(x)
        if relu_flag == "relu":
            x = Activation("relu")(x)
        elif relu_flag == "elu":
            x = Activation("elu")(x)
        elif relu_flag == "prelu":
            x = PReLU()(x)
        elif relu_flag == "leakyrelu":
            x = LeakyReLU()(x)
        x = Dropout(dropval)(x)
    if num_layer >= 6:
        x = Dense(units6, kernel_regularizer=reg)(x)
        if relu_flag == "relu":
            x = Activation("relu")(x)
        elif relu_flag == "elu":
            x = Activation("elu")(x)
        elif relu_flag == "prelu":
            x = PReLU()(x)
        elif relu_flag == "leakyrelu":
            x = LeakyReLU()(x)
        x = Dropout(dropval)(x)
    
    # Specify output layer
    if classes == 1:
        label = Dense(classes, activation='linear', name='predictions')(x)
    elif classes >= 2:
        label = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        raise("ERROR in specifying tasktype")
        
    # Create base model
    model = Model(inputs=inlayer,outputs=label, name='MLP')
    
    # Create intermediate model
    submodel = Model(inputs=inlayer,outputs=x, name='MLP_truncated')
    
    # Specify training method
    if classes == 1:
        model.compile(optimizer="RMSprop", loss="mean_squared_error")
        submodel.compile(optimizer="RMSprop", loss="mean_squared_error")
    elif classes >= 2:
        model.compile(optimizer="RMSprop", loss="categorical_crossentropy")
        submodel.compile(optimizer="RMSprop", loss="categorical_crossentropy")
    else:
        raise("ERROR in specifying tasktype")
    
    return(model, submodel)


# # RNN network designs

# In[ ]:

def cs_setup_rnn(params, inshape=None, classes=None, char=None):
    
    # Parse network hyperparameters
    em_dim = int(params['em_dim']*10)
    kernel_size = 3
    filters = int(params['conv_units']*6)    
    num_layer = int(params['num_layer'])
    units1 = int(params['layer1_units']*6)
    units2 = int(params['layer2_units']*6)
    units3 = int(params['layer3_units']*6)
    relu_flag = str(params['relu_type'])
    dropval = float(params['dropval'])
    reg_flag = str(params['reg_type'])
    reg_val = 10**(-float(params['reg_val']))

    # Setup regularizer
    if reg_flag == "l1":
        reg = l1(reg_val)
        print("Regularizer "+reg_flag+" set at "+str(reg_val))
    elif reg_flag == "l2":
        reg = l2(reg_val)
        print("Regularizer "+reg_flag+" set at "+str(reg_val))
    elif reg_flag == "l1_l2":
        reg = l1_l2(reg_val)
        print("Regularizer "+reg_flag+" set at "+str(reg_val))
    else:
        reg = None
        print("NOTE: No regularizers used")
    
    # Setup neural network
    inlayer = Input(shape=[inshape])
    x = Embedding(input_dim=len(char)+1,output_dim=em_dim)(inlayer)   
    x = Conv1D(filters, kernel_size, strides=1, padding="same", kernel_regularizer=reg)(x)
    if relu_flag == "relu":
        x = Activation("relu")(x)
    elif relu_flag == "elu":
        x = Activation("elu")(x)
    elif relu_flag == "prelu":
        x = PReLU()(x)
    elif relu_flag == "leakyrelu":
        x = LeakyReLU()(x)
    if params['celltype'] == "GRU":
        if num_layer == 1:
            x = Bidirectional(CuDNNGRU(units1, return_sequences=False))(x)
            x = Dropout(dropval)(x)
        elif num_layer == 2:
            x = Bidirectional(CuDNNGRU(units1, return_sequences=True))(x)
            x = Dropout(dropval)(x)
            x = Bidirectional(CuDNNGRU(units2, return_sequences=False))(x)
            x = Dropout(dropval)(x)
        elif num_layer == 3:
            x = Bidirectional(CuDNNGRU(units1, return_sequences=True))(x)
            x = Dropout(dropval)(x)
            x = Bidirectional(CuDNNGRU(units2, return_sequences=True))(x)
            x = Dropout(dropval)(x)
            x = Bidirectional(CuDNNGRU(units3, return_sequences=False))(x)
            x = Dropout(dropval)(x)
    if params['celltype'] == "LSTM":
        if num_layer == 1:
            x = Bidirectional(CuDNNLSTM(units1, return_sequences=False))(x)
            x = Dropout(dropval)(x)
        elif num_layer == 2:
            x = Bidirectional(CuDNNLSTM(units1, return_sequences=True))(x)
            x = Dropout(dropval)(x)
            x = Bidirectional(CuDNNLSTM(units2, return_sequences=False))(x)
            x = Dropout(dropval)(x)
        elif num_layer == 3:
            x = Bidirectional(CuDNNLSTM(units1, return_sequences=True))(x)
            x = Dropout(dropval)(x)
            x = Bidirectional(CuDNNLSTM(units2, return_sequences=True))(x)
            x = Dropout(dropval)(x)
            x = Bidirectional(CuDNNLSTM(units3, return_sequences=False))(x)
            x = Dropout(dropval)(x)
    
    # Specify output layer
    if classes == 1:
        label = Dense(classes, activation='linear', name='predictions')(x)
    elif classes >= 2:
        label = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        raise("ERROR in specifying tasktype")
        
    # Create base model
    model = Model(inputs=inlayer,outputs=label, name='SMILES2vec')
    
    # Create intermediate model
    submodel = Model(inputs=inlayer,outputs=x, name='SMILES2vec_truncated')
    
    # Specify training method
    if classes == 1:
        model.compile(optimizer="RMSprop", loss="mean_squared_error")
        submodel.compile(optimizer="RMSprop", loss="mean_squared_error")
    elif classes >= 2:
        model.compile(optimizer="RMSprop", loss="categorical_crossentropy")
        submodel.compile(optimizer="RMSprop", loss="categorical_crossentropy")
    else:
        raise("ERROR in specifying tasktype")
    
    return(model, submodel)


# # CNN Network Designs

# In[ ]:

def conv2d_bn(x, nb_filter, kernel_size=4, padding='same', strides=2):

    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1
        
    x = Conv2D(nb_filter, kernel_size=(kernel_size,kernel_size), strides=(strides,strides), padding=padding)(x)
    x = Activation("relu")(x)
    
    return x


# In[ ]:

def inception_resnet_v2_A(input_tensor, nb_params, last_params, scale_residual=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input_tensor

    ir1 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)

    ir2 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    ir2 = Conv2D(nb_params, kernel_size=(3,3), activation='relu', padding='same')(ir2)

    ir3 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    ir3 = Conv2D(int(nb_params*1.5), kernel_size=(3,3), activation='relu', padding='same')(ir3)
    ir3 = Conv2D(int(nb_params*2.0), kernel_size=(3,3), activation='relu', padding='same')(ir3)

    ir_merge = concatenate([ir1, ir2, ir3], axis=channel_axis)

    ir_conv = Conv2D(last_params, kernel_size=(1,1), activation='linear', padding='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = Activation("relu")(out)
    
    return out


# In[ ]:

def inception_resnet_v2_B(input_tensor, nb_params, last_params, scale_residual=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input_tensor

    ir1 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)

    ir2 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    ir2 = Conv2D(int(nb_params*1.25), kernel_size=(1,7), activation='relu', padding='same')(ir2)
    ir2 = Conv2D(int(nb_params*1.50), kernel_size=(7,1), activation='relu', padding='same')(ir2)

    ir_merge = concatenate([ir1, ir2], axis=channel_axis)

    ir_conv = Conv2D(last_params, kernel_size=(1,1), activation='linear', padding='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = Activation("relu")(out)
    
    return out


# In[ ]:

def inception_resnet_v2_C(input_tensor, nb_params, last_params, scale_residual=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input_tensor

    ir1 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)

    ir2 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    ir2 = Conv2D(int(nb_params*1.1666666), kernel_size=(1,3), activation='relu', padding='same')(ir2)
    ir2 = Conv2D(int(nb_params*1.3333333), kernel_size=(3,1), activation='relu', padding='same')(ir2)

    ir_merge = concatenate([ir1, ir2], axis=channel_axis)

    ir_conv = Conv2D(last_params, kernel_size=(1,1), activation='linear', padding='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = add([init, ir_conv])
    out = Activation("relu")(out)
    
    return out


# In[ ]:

def reduction_A(input_tensor, nb_params):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = MaxPooling2D((3,3), padding='valid', strides=(2,2))(input_tensor)

    r2 = Conv2D(int(nb_params*1.5), kernel_size=(3,3), activation='relu', padding='valid', strides=(2,2))(input_tensor)

    r3 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    r3 = Conv2D(nb_params, kernel_size=(3,3), activation='relu', padding='same')(r3)
    r3 = Conv2D(int(nb_params*1.5), kernel_size=(3,3), activation='relu', padding='valid', strides=(2,2))(r3)

    m = concatenate([r1, r2, r3], axis=channel_axis)
    m = Activation('relu')(m)
    
    return m


# In[ ]:

def reduction_resnet_v2_B(input_tensor, nb_params):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = MaxPooling2D((3,3), padding='valid', strides=(2,2))(input_tensor)

    r2 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    r2 = Conv2D(int(nb_params*1.5), kernel_size=(3,3), activation='relu', padding='valid', strides=(2,2))(r2)

    r3 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    r3 = Conv2D(int(nb_params*1.125), kernel_size=(3,3), activation='relu', padding='valid', strides=(2, 2))(r3)

    r4 = Conv2D(nb_params, kernel_size=(1,1), activation='relu', padding='same')(input_tensor)
    r4 = Conv2D(int(nb_params*1.125), kernel_size=(3,3), activation='relu', padding='same')(r4)
    r4 = Conv2D(int(nb_params*1.25), kernel_size=(3,3), activation='relu', padding='valid', strides=(2, 2))(r4)
    
    m = concatenate([r1, r2, r3, r4], axis=channel_axis)
    m = Activation('relu')(m)
    
    return m


# In[ ]:

def end_block_droppool(input_tensor, dropval):
        
    x = GlobalAveragePooling2D(data_format="channels_last", name="final_pool")(input_tensor)
    x = Dropout(dropval, name="dropout_end")(x)
    
    return(input_tensor, x)


# In[ ]:

def end_block_pool(input_tensor):
    
    x = GlobalAveragePooling2D(data_format="channels_last", name="final_pool")(input_tensor)
    
    return(input_tensor, x)


# In[ ]:

def cs_setup_cnn(params, inshape=None, classes=None):
    """Instantiate the Inception v3 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `tf` dim ordering)
            or `(3, 299, 299)` (with `th` dim ordering).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.
    """
    
    #Clear GPU memory
    K.clear_session()
       
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
    else:
        channel_axis = -1
    print("Channel axis is "+str(channel_axis))
        
    # Assign image input
    inlayer = Input(inshape)
    x = conv2d_bn(inlayer, params['conv1_units'], kernel_size=4, strides=2)

    # Inception Resnet A
    for i in range(params['num_block1']):
        last_params = params['conv1_units']
        x = inception_resnet_v2_A(x, params['conv2_units'], last_params, scale_residual=False)

    # Reduction A
    x = reduction_A(x, params['conv3_units'])

    # Inception Resnet B
    for i in range(params['num_block2']):
        last_params = int(params['conv1_units']+(params['conv3_units']*3))
        x = inception_resnet_v2_B(x, params['conv4_units'], last_params, scale_residual=False)

    # Reduction Resnet B
    x = reduction_resnet_v2_B(x, params['conv5_units'])

    # Inception Resnet C
    for i in range(params['num_block3']):
        last_params = int(params['conv1_units']+(params['conv3_units']*3))+int(params['conv5_units']*3.875)
        x = inception_resnet_v2_C(x, params['conv6_units'], last_params, scale_residual=False)
            
    # Classification block
    before_pool, after_pool = end_block_droppool(x, params['dropval'])

    # Specify output layer
    if classes == 1:
        label = Dense(classes, activation='linear', name='predictions')(after_pool)
    elif classes >= 2:
        label = Dense(classes, activation='softmax', name='predictions')(after_pool)
    else:
        raise("ERROR in specifying tasktype")
        
    # Create base model
    model = Model(inputs=inlayer,outputs=label, name='Chemception')
    
    # Create intermediate model
    submodel = Model(inputs=inlayer,outputs=after_pool, name='Chemception_truncated')
    
    # Specify training method
    if classes == 1:
        model.compile(optimizer="RMSprop", loss="mean_squared_error")
        submodel.compile(optimizer="RMSprop", loss="mean_squared_error")
    elif classes >= 2:
        model.compile(optimizer="RMSprop", loss="categorical_crossentropy")
        submodel.compile(optimizer="RMSprop", loss="categorical_crossentropy")
    else:
        raise("ERROR in specifying tasktype")
    
    return(model, submodel)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



