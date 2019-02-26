def identity_Block_deep(input, filters, kernel_size, with_conv_shortcut=False,use_bias=True, mode='sum'):
    x = _conv_relu1D(filters=filters*2, kernel_size=1, subsample=1,use_bias=use_bias)(input)
    x = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(x)
    x = _conv_relu1D(filters=filters, kernel_size=1, subsample=1,use_bias=use_bias)(x)
    if with_conv_shortcut:
        shortcut = _conv_relu1D(filters=filters, kernel_size=kernel_size, subsample=1,use_bias=use_bias)(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        return x

def identity_Block_deep_2D(input, filters, nb_row, nb_col, with_conv_shortcut=False,use_bias=True, mode='sum'):
    x = _conv_relu2D(filters=filters*2, nb_row = 1, nb_col = 1, subsample=(1, 1), use_bias=use_bias)(input)
    x = _conv_relu2D(filters=filters, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1), use_bias=use_bias)(x)
    x = _conv_relu2D(filters=filters, nb_row=1, nb_col = 1, subsample=(1, 1))(x)
    if with_conv_shortcut:
        shortcut = _conv_relu2D(filters=filters, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        return x

def DeepResnet_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss, kernel_size=7, subsample=1, use_bias=use_bias)(DNCON4_1D_conv) 
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)
        DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)

        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss*2, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv) 

        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*2, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*2, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*2, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)
        DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)

        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss*4, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv) 

        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = identity_Block_deep(DNCON4_1D_conv, filters=filterss*4, kernel_size=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)
        DNCON4_1D_conv = BatchNormalization(axis=-1)(DNCON4_1D_conv)
        
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)     
        DNCON4_1D_convs.append(DNCON4_1D_conv)

    if len(filter_sizes) > 1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]
    ## convert 1D to 2D
    DNCON4_genP=generatePairwiseF(output_shape=(sequence_length,sequence_length,DNCON4_1D_out.shape.as_list()[2]*3),batch_size=batch_size)(DNCON4_1D_out)
    
    print(DNCON4_genP.shape)
    # load 2D contact features
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(sequence_length,sequence_length,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)

    ######################### now merge new data to new architecture
    
    DNCON4_2D_input = concatenate([DNCON4_genP,contact_input], axis=-1) 
    
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv_in = DNCON4_2D_input
        # DNCON4_2D_conv_bn = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        DNCON4_2D_conv = _conv_relu2D(filters=filterss, nb_row=7, nb_col=7, subsample=(1,1))(DNCON4_2D_conv_in)
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)

        DNCON4_2D_conv_a1 = _conv_relu2D(filters=filterss, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_in)
        DNCON4_2D_conv_b1 = add([DNCON4_2D_conv_a1, DNCON4_2D_conv])

        DNCON4_2D_conv = _conv_relu2D(filters=filterss*2, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv_b1)
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*2, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*2, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*2, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)

        DNCON4_2D_conv_a2 = _conv_relu2D(filters=filterss*2, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_in)
        DNCON4_2D_conv_b2 = _conv_relu2D(filters=filterss*2, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_b1)

        DNCON4_2D_conv_c2 = add([DNCON4_2D_conv_a2, DNCON4_2D_conv_b2, DNCON4_2D_conv])
        
        DNCON4_2D_conv = _conv_relu2D(filters=filterss*4, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv_c2)
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = identity_Block_deep_2D(DNCON4_2D_conv, filters=filterss*4, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum')
        DNCON4_2D_conv = Dropout(0.3)(DNCON4_2D_conv)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        
        DNCON4_2D_conv_a3 = _conv_relu2D(filters=filterss*4, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_in)
        DNCON4_2D_conv_b3 = _conv_relu2D(filters=filterss*4, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_b1)
        DNCON4_2D_conv_c3 = _conv_relu2D(filters=filterss*4, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_c2)

        DNCON4_2D_conv = add([DNCON4_2D_conv_a3, DNCON4_2D_conv_b3, DNCON4_2D_conv_c3, DNCON4_2D_conv])

        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)

    if len(filter_sizes) > 1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]

    
    DNCON4_flatten = Flatten()(DNCON4_2D_out)
    
    DNCON4_RES = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
    DNCON4_RES.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=opt)

    return DNCON4_RES



####2019/1/21  0.6110 resnet on cov
def identity_Block_sallow_2D(input, filters, nb_row, nb_col, with_conv_shortcut=False,use_bias=True, mode='sum', kernel_initializer='he_normal', dilation_rate=(1, 1)):
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(input)
    # x = BatchNormalization(axis=-1)(x)
    x = Activation("relu")(x)
    # x = Dropout(0.4)(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same", dilation_rate=dilation_rate)(x)
    if with_conv_shortcut:
        shortcut = Conv2D(filters=filters, kernel_size=(nb_row, nb_col), strides=(1,1),use_bias=use_bias, kernel_initializer=kernel_initializer, padding="same")(input)
        if mode=='sum':
            x = add([x, shortcut])
        elif mode=='concat':
            x = concatenate([x, shortcut], axis=-1)
        x = Activation("relu")(x)
        return x
    else:
        if mode=='sum':
            x = add([x, input])
        elif mode=='concat':
            x = concatenate([x, input], axis=-1)
        x = Activation("relu")(x)
        return x
def DeepResnet_with_paras_2D(win_array,feature_2D_num,use_bias,hidden_type,filterss,nb_layers,opt, initializer = "he_normal", loss_function = "binary_crossentropy", weight_p=1.0, weight_n=1.0):
    filter_sizes=win_array
    contact_feature_num_2D=feature_2D_num
    contact_input_shape=(None,None,contact_feature_num_2D)
    contact_input = Input(shape=contact_input_shape)

    ######################### now merge new data to new architecture
    
    DNCON4_2D_input = contact_input
    
    DNCON4_2D_convs = []
    for fsz in filter_sizes:
        DNCON4_2D_conv_in = DNCON4_2D_input
        # DNCON4_2D_conv_bn = BatchNormalization(axis=-1)(DNCON4_2D_conv)
        DNCON4_2D_conv_in = Dense(64)(DNCON4_2D_conv_in)
        # DNCON4_2D_conv_in = Conv2D(filters=128, kernel_size=(1, 1), strides=(1,1),use_bias=use_bias, kernel_initializer=initializer, padding="same")(DNCON4_2D_conv_in)
        DNCON4_2D_conv = MaxoutConv2D(kernel_size=(1,1), output_dim=64)(DNCON4_2D_conv_in)
        # DNCON4_2D_conv = _conv_relu2D(filters=filterss, nb_row=7, nb_col=7, subsample=(1,1), kernel_initializer=initializer)(DNCON4_2D_conv_in)
        for idx in range(nb_layers):
            DNCON4_2D_conv = identity_Block_sallow_2D(DNCON4_2D_conv, filters=filterss, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum', kernel_initializer=initializer)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)

        DNCON4_2D_conv_a1 = _conv_relu2D(filters=filterss, nb_row=1, nb_col=1, subsample=(1,1), kernel_initializer=initializer)(DNCON4_2D_conv_in)
        DNCON4_2D_conv_b1 = add([DNCON4_2D_conv_a1, DNCON4_2D_conv])
        
        # DNCON4_2D_conv = identity_Block_sallow_2D(DNCON4_2D_conv, filters=filterss, nb_row=fsz,nb_col=fsz,with_conv_shortcut=True,use_bias=True, mode='concat', kernel_initializer=initializer, dilation_rate=(4,4))
        DNCON4_2D_conv = _conv_relu2D(filters=filterss*2, nb_row=1, nb_col=1, subsample=(1,1), kernel_initializer=initializer)(DNCON4_2D_conv)
        for idx in range(nb_layers+2):
            DNCON4_2D_conv = identity_Block_sallow_2D(DNCON4_2D_conv, filters=filterss*2, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum', kernel_initializer=initializer)
        DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)

        DNCON4_2D_conv_a2 = _conv_relu2D(filters=filterss*2, nb_row=1, nb_col=1, subsample=(1,1), kernel_initializer=initializer)(DNCON4_2D_conv_in)
        DNCON4_2D_conv_b2 = _conv_relu2D(filters=filterss*2, nb_row=1, nb_col=1, subsample=(1,1), kernel_initializer=initializer)(DNCON4_2D_conv_b1)
        DNCON4_2D_conv = add([DNCON4_2D_conv_a2, DNCON4_2D_conv_b2, DNCON4_2D_conv])
        
        # DNCON4_2D_conv = identity_Block_sallow_2D(DNCON4_2D_conv_c2, filters=filterss, nb_row=fsz,nb_col=fsz,with_conv_shortcut=True,use_bias=True, mode='concat', kernel_initializer=initializer, dilation_rate=(4,4))
        # # DNCON4_2D_conv = _conv_relu2D(filters=filterss*2, nb_row=1, nb_col=1, subsample=(1,1), kernel_initializer=initializer)(DNCON4_2D_conv_c2)
        # for idx in range(nb_layers+2):
        #     DNCON4_2D_conv = identity_Block_sallow_2D(DNCON4_2D_conv, filters=filterss*2, nb_row=fsz,nb_col=fsz,with_conv_shortcut=False,use_bias=True, mode='sum', kernel_initializer=initializer, dilation_rate=(8,8))
        # DNCON4_2D_conv = BatchNormalization(axis=-1)(DNCON4_2D_conv)
    
        # DNCON4_2D_conv_a3 = _conv_relu2D(filters=filterss*2, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_in)
        # DNCON4_2D_conv_b3 = _conv_relu2D(filters=filterss*2, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_b1)
        # DNCON4_2D_conv_c3 = _conv_relu2D(filters=filterss*2, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv_c2)

        # DNCON4_2D_conv = add([DNCON4_2D_conv_a3, DNCON4_2D_conv_b3, DNCON4_2D_conv_c3, DNCON4_2D_conv])
        # DNCON4_2D_conv = Conv2D(filters=filterss, kernel_size=(fsz, fsz), strides=(1, 1),use_bias=use_bias, dilation_rate=(4, 4), kernel_initializer=initializer, padding="same")(DNCON4_2D_conv)
        # DNCON4_2D_conv = _conv_relu2D(filters=filterss*2, nb_row=3, nb_col=3, subsample=(1,1), kernel_initializer=initializer, dilation_rate=(1, 1))(DNCON4_2D_conv)
        # DNCON4_2D_conv = _conv_relu2D(filters=filterss*2, nb_row=3, nb_col=3, subsample=(1,1), kernel_initializer=initializer, dilation_rate=(2, 2))(DNCON4_2D_conv)
        # DNCON4_2D_conv = _conv_relu2D(filters=filterss*2, nb_row=3, nb_col=3, subsample=(1,1), kernel_initializer=initializer, dilation_rate=(5, 5))(DNCON4_2D_conv)
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1), kernel_initializer=initializer, dilation_rate=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)

    if len(filter_sizes) > 1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]

    if loss_function == 'weighted_crossentropy':
        loss = _weighted_binary_crossentropy(weight_p, weight_n)
    else:
        loss = loss_function
    DNCON4_RES = Model(inputs=contact_input, outputs=DNCON4_2D_out)
    # categorical_crossentropy
    DNCON4_RES.compile(loss=loss, metrics=['accuracy'], optimizer=opt)
    DNCON4_RES.summary()
    return DNCON4_RES