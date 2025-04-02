from tensorflow.keras.layers import Activation, SpatialDropout3D


def aggregate(scale_list, name):
        X = tf.keras.layers.concatenate(scale_list, axis=-1)
        X = conv_block(X, base_filters * 5, stack_num_up)
        return X

def deep_sup(inputs, scale):
    size = 2 ** (scale-1)
    conv = layer_util.get_nd_layer('Conv', rank)
    upsamp = layer_util.get_nd_layer(upsamp_type, rank)
    if rank == 2:
        upsamp_config = dict(size=(size, size), interpolation='bilinear')
    else:
        upsamp_config = dict(size=(size, size, 1))

    X = inputs  
    X = conv(n_outputs, activation=None, **conv_config, name=f'deepsup_conv_{scale}')(X)
    if scale != 1:
        X = upsamp(**upsamp_config, name=f'deepsup_upsamp_{scale}')(X)
    return X

def full_scale(inputs, to_layer, from_layer):
    layer_diff = from_layer - to_layer  
    size = 2 ** abs(layer_diff)
    conv = layer_util.get_nd_layer('Conv', rank)
    maxpool = layer_util.get_nd_layer('MaxPool', rank)
    upsamp = layer_util.get_nd_layer(upsamp_type, rank)
    if rank == 2:
        upsamp_config = dict(size=(size, size), interpolation='bilinear')
    else:
        upsamp_config = dict(size=(size, size, 1))

    X = inputs        
    if to_layer < from_layer:
        X = upsamp(**upsamp_config, name=f'fullscale_{from_layer}_{to_layer}')(X)
    elif to_layer > from_layer:
        X = maxpool(pool_size=(size, size) if rank == 2 else (size, size, 1), name=f'fullscale_maxpool_{from_layer}_{to_layer}')(X)
    X = conv_block(X, base_filters, stack_num_up)
    return X


def conv_block(inputs, filters, num_stacks):
    conv = layer_util.get_nd_layer('Conv', rank)
    X = inputs
    for _ in range(num_stacks):
        X = conv(filters, **conv_config)(X)
        if batch_norm:
            X = tf.keras.layers.BatchNormalization(axis=-1)(X)
        X = tf.keras.layers.LeakyReLU()(X)
    return X


def encode(inputs, scale, num_stacks):
    maxpool = layer_util.get_nd_layer('MaxPool', rank)
    scale -= 1  # python index
    filters = base_filters * 2 ** scale
    filters = filters - 1 if scale == 4 else filters
    
    X = inputs
    if scale != 0:
        X = maxpool(pool_size=(2, 2) if rank == 2 else (2, 2, 1), name=f'encoding_{scale}_maxpool')(X)
    X = conv_block(X, filters, num_stacks)
    return X

def build_multiflowseg():
    rank = 3
    n_outputs = 6
    add_dropout = True
    dropout_rate = 0.3
    base_filters = 16
    kernel_size = 3
    stack_num_down = 3
    stack_num_up = 1
    batch_norm = 1
    CGM = True
    supervision = True
    upsamp_type = 'UpSampling'
    conv_config = dict(kernel_size=3, 
                       padding='same', 
                       kernel_initializer='he_normal')

    

    tf.keras.backend.clear_session()
    image_input = tf.keras.Input(shape=input_shape, name='image_input')    
    cgm_input = tf.keras.Input(shape=[6], name='cgm_input')
    one_hot_input = tf.keras.Input(shape=[6], name='one_hot_input')
    mask_input = tf.keras.Input(shape=output_shape, name='mask_input')

    conv = layer_util.get_nd_layer('Conv', rank=rank)
    
    T1 = one_hot_input
    bottle_neck_height_width = 8
    rank = 2
    
    T1 = tf.keras.layers.Dense((bottle_neck_height_width/2)**rank)(T1) 
    T1 = tf.keras.layers.Dense((bottle_neck_height_width)**rank)(T1)
    
    T1 = tf.reshape(T1, (-1, bottle_neck_height_width, bottle_neck_height_width, 1, 1))
    T1 = tf.tile(T1, [1, 1, 1, 32, 1]) # copied over 32 frames


    X = image_input
    XE1 = encode(X, scale=1, num_stacks=stack_num_down)
    XE2 = encode(XE1, scale=2, num_stacks=stack_num_down)
    XE3 = encode(XE2, scale=3, num_stacks=stack_num_down)
    XE4 = encode(XE3, scale=4, num_stacks=stack_num_down)
    XE5 = encode(XE4, scale=5, num_stacks=stack_num_down)
    XE5 = tf.concat([XE5, T1], axis=-1)


    # Classification Guided Module. Part 1
    if CGM:
        X_CGM = XE5
#         X_CGM = tf.keras.layers.Dropout(rate=0.2)(X_CGM)
        X_CGM = tf.keras.layers.SpatialDropout3D(rate=0.5)(X_CGM)
        X_CGM = conv(n_outputs, kernel_size=(1, 1, 1), padding="same", strides=(1, 1, 1), name='CGM_conv')(X_CGM)
        X_CGM = tf.keras.layers.GlobalMaxPooling3D()(X_CGM)
        if n_outputs == 1:
            X_CGM = tf.keras.activations.sigmoid(X_CGM)
            X_CGM = tf.keras.backend.max(X_CGM, axis=-1)
        else:
            X_CGM = tf.keras.activations.softmax(X_CGM, axis=-1)
            cgm_output = X_CGM
            vessel_probs = tf.gather(X_CGM, [1,2,3,4,5], axis=-1)
            max_vessel_probs = tf.reduce_max(vessel_probs, axis=-1, keepdims=True)
            max_vessel_indices = tf.argmax(vessel_probs, axis=-1, output_type=tf.int32)
            one_hot_mask = tf.one_hot(max_vessel_indices, depth=5, axis=-1)
            bkg = tf.ones_like(one_hot_mask)[:,:1]
            X_CGM = tf.concat([bkg, one_hot_mask], axis = -1)
            X_CGM = tf.reshape(X_CGM, (-1, 1, 1, 1, 6))

    XD5 = XE5
    XD4_from_XD5 = full_scale(XD5, 4, 5)
    XD4_from_XE4 = full_scale(XE4, 4, 4)
    XD4_from_XE3 = full_scale(XE3, 4, 3)
    XD4_from_XE2 = full_scale(XE2, 4, 2)
    XD4_from_XE1 = full_scale(XE1, 4, 1)
    XD4 = aggregate([XD4_from_XD5, XD4_from_XE4, XD4_from_XE3, XD4_from_XE2, XD4_from_XE1], name='agg_XD4')

    XD3_from_XD5 = full_scale(XD5, 3, 5)
    XD3_from_XD4 = full_scale(XD4, 3, 4)
    XD3_from_XE3 = full_scale(XE3, 3, 3)
    XD3_from_XE2 = full_scale(XE2, 3, 2)
    XD3_from_XE1 = full_scale(XE1, 3, 1)
    XD3 = aggregate([XD3_from_XD5, XD3_from_XD4, XD3_from_XE3, XD3_from_XE2, XD3_from_XE1], name='agg_XD3')

    XD2_from_XD5 = full_scale(XD5, 2, 5)
    XD2_from_XD4 = full_scale(XD4, 2, 4)
    XD2_from_XD3 = full_scale(XD3, 2, 3)
    XD2_from_XE2 = full_scale(XE2, 2, 2)
    XD2_from_XE1 = full_scale(XE1, 2, 1)
    XD2 = aggregate([XD2_from_XD5, XD2_from_XD4, XD2_from_XD3, XD2_from_XE2, XD2_from_XE1], name='agg_XD2')

    XD1_from_XD5 = full_scale(XD5, 1, 5)
    XD1_from_XD4 = full_scale(XD4, 1, 4)
    XD1_from_XD3 = full_scale(XD3, 1, 3)
    XD1_from_XD2 = full_scale(XD2, 1, 2)
    XD1_from_XE1 = full_scale(XE1, 1, 1)
    XD1 = aggregate([XD1_from_XD5, XD1_from_XD4, XD1_from_XD3, XD1_from_XD2, XD1_from_XE1], name='agg_XD1')

    if supervision:
        XD5 = deep_sup(XD5, 5)
        XD4 = deep_sup(XD4, 4)
        XD3 = deep_sup(XD3, 3)
        XD2 = deep_sup(XD2, 2)
    XD1 = deep_sup(XD1, 1)



    XD5 = tf.keras.layers.Activation(activation='sigmoid' if n_outputs == 1 else 'softmax', name='output5')(XD5)
    XD4 = tf.keras.layers.Activation(activation='sigmoid' if n_outputs == 1 else 'softmax', name='output4')(XD4)
    XD3 = tf.keras.layers.Activation(activation='sigmoid' if n_outputs == 1 else 'softmax', name='output3')(XD3)
    XD2 = tf.keras.layers.Activation(activation='sigmoid' if n_outputs == 1 else 'softmax', name='output2')(XD2)
    XD1 = tf.keras.layers.Activation(activation='sigmoid' if n_outputs == 1 else 'softmax', name='output1')(XD1)

    # Classification Guided Module. Part 2
    if CGM:
        XD5 *= X_CGM
        XD4 *= X_CGM
        XD3 *= X_CGM
        XD2 *= X_CGM
        XD1 *= X_CGM

    if supervision:
        outputs = [XD5, XD4, XD3, XD2, XD1]
    else:
        outputs = XD1

    model = tf.keras.Model(inputs=[image_input, cgm_input, one_hot_input, mask_input], outputs=outputs)

    # Compile model

    # Define focal_tversky_loss and add as metrics
    focal_loss = 0
    for i, output in enumerate(outputs):
        loss = focal_tversky_loss(mask_input, output)
        focal_loss += loss
        model.add_metric(loss, name=f'output{5-i}_loss', aggregation='mean')

    # Add CGM loss
    cgm_loss = categorical_crossentropy(cgm_input, cgm_output)
    model.add_metric(cgm_loss, name='cgm_loss', aggregation='mean')

    # Add total loss to model
    model.add_loss(cgm_loss * 0.25)
    model.add_loss(focal_loss)
    model.add_metric(focal_loss*cgm_loss, name='cgm_focal_loss', aggregation='mean')

model.compile(loss=None, optimizer=tf.keras.optimizers.Adam(), loss_weights = [0.25,0.25,0.25,0.25,1])
    # model.summary()


