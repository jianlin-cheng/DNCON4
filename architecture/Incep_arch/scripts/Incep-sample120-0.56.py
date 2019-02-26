def block_inception_a(inputs,filterss,kernel_size,use_bias=True):
    inputs_feanum = inputs.shape.as_list()[2]

    branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(inputs)

    branch_1 = _conv_bn_relu1D(filters=filterss+8, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(inputs)

    branch_2 = _conv_bn_relu1D(filters=filterss+16, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(inputs)

    branch_3 = _conv_bn_relu1D(filters=filterss+24, kernel_size=kernel_size, subsample=1, use_bias=use_bias)(inputs)

    #x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    net = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
    return net

def block_inception_a_2D(inputs,filters, nb_row, nb_col,subsample=(1, 1)):
    inputs_feanum = inputs.shape.as_list()[3]

    branch_0 = _conv_bn_relu2D(filters=filters, nb_row = 1, nb_col = 1, subsample=(1, 1))(inputs)

    branch_1 = _conv_bn_relu2D(filters=filters, nb_row = 1, nb_col = 1, subsample=(1, 1))(inputs)
    branch_1 = _conv_bn_relu2D(filters=filters+8, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(branch_1)

    branch_2 = _conv_bn_relu2D(filters=filters, nb_row = 1, nb_col = 1, subsample=(1, 1))(inputs)
    branch_2 = _conv_bn_relu2D(filters=filters+16, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(branch_2)
    branch_2 = _conv_bn_relu2D(filters=filters+32, nb_row = nb_row, nb_col = nb_col, subsample=(1, 1))(branch_2)

    net = concatenate([branch_0, branch_1, branch_2], axis=-1)
    merge_branch = _conv_bn_relu2D(filters=inputs_feanum, nb_row = 1, nb_col = 1, subsample=(1, 1))(net)
    outputs = add([inputs, merge_branch])
    return outputs

def DeepInception_with_paras(win_array,feature_1D_num,feature_2D_num,sequence_length,use_bias,hidden_type,filterss,nb_layers,opt,batch_size):
    DNCON4_1D_input_shape =(sequence_length,feature_1D_num)
    filter_sizes=win_array
    DNCON4_1D_input = Input(shape=DNCON4_1D_input_shape)
    DNCON4_1D_convs = []
    for fsz in filter_sizes:
        DNCON4_1D_conv = DNCON4_1D_input
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)

        ## start inception 1
        branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_1 = _conv_bn_relu1D(filters=filterss+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        DNCON4_1D_conv = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)#0.3

         ## start inception 2
        branch_0 = _conv_bn_relu1D(filters=filterss, kernel_size=fsz, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_0 = _conv_bn_relu1D(filters=filterss+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_0)

        branch_1 = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)
        branch_1 = _conv_bn_relu1D(filters=filterss+16, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_1)
        branch_1 = _conv_bn_relu1D(filters=filterss+32, kernel_size=fsz, subsample=1, use_bias=use_bias)(branch_1)

        DNCON4_1D_conv = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)#0.3

        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(nb_layers):
            DNCON4_1D_conv = block_inception_a(DNCON4_1D_conv,filterss,fsz)
            DNCON4_1D_conv = Dropout(0.2)(DNCON4_1D_conv)#0.3
        
        DNCON4_1D_conv = _conv_bn_relu1D(filters=filterss, kernel_size=1, subsample=1, use_bias=use_bias)(DNCON4_1D_conv)       
        DNCON4_1D_convs.append(DNCON4_1D_conv) 
    if len(filter_sizes) > 1:
        DNCON4_1D_out = Average(DNCON4_1D_convs)
    else:
        DNCON4_1D_out = DNCON4_1D_convs[0]

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
        DNCON4_2D_conv = DNCON4_2D_input
        DNCON4_2D_conv1 = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv)
        shape1 = DNCON4_2D_conv1.shape.as_list()[3]
        ## start inception 1
        branch_0 = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv1)
        branch_1 = _conv_bn_relu2D(filters=filterss*2, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv1)
        DNCON4_2D_conv2 = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_2D_conv2 = Dropout(0.2)(DNCON4_2D_conv2)#0.3
        DNCON4_2D_conv2 = _conv_bn_relu2D(filters=shape1, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv2)
        DNCON4_2D_conv2 = add([DNCON4_2D_conv1, DNCON4_2D_conv2])

        shape2 = DNCON4_2D_conv2.shape.as_list()[3]
        ## start inception 2
        branch_0 = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv2)
        branch_0 = _conv_bn_relu2D(filters=filterss*2, nb_row=fsz, nb_col=fsz, subsample=(1,1))(branch_0)

        branch_1 = _conv_bn_relu2D(filters=filterss, nb_row=fsz, nb_col=fsz, subsample=(1,1))(DNCON4_2D_conv2)
        branch_1 = _conv_bn_relu2D(filters=filterss*2, nb_row=fsz, nb_col=fsz, subsample=(1,1))(branch_1)
        branch_1 = _conv_bn_relu2D(filters=filterss*4, nb_row=fsz, nb_col=fsz, subsample=(1,1))(branch_1)

        DNCON4_2D_conv3 = concatenate([branch_0, branch_1], axis=-1)
        DNCON4_2D_conv3 = Dropout(0.2)(DNCON4_2D_conv3)#0.3
        DNCON4_2D_conv3 = _conv_bn_relu2D(filters=shape2, nb_row=1, nb_col=1, subsample=(1,1))(DNCON4_2D_conv)
        DNCON4_2D_conv3 = add([DNCON4_2D_conv2, DNCON4_2D_conv3])
        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(nb_layers):
            DNCON4_2D_conv = block_inception_a_2D(DNCON4_2D_conv,filterss,nb_row=fsz, nb_col=fsz)
            DNCON4_2D_conv = Dropout(0.2)(DNCON4_2D_conv)#0.3
            
        DNCON4_2D_conv = _conv_bn_sigmoid2D(filters=1, nb_row=fsz, nb_col=fsz, subsample=(1, 1))(DNCON4_2D_conv)
        DNCON4_2D_convs.append(DNCON4_2D_conv)

    if len(filter_sizes) > 1:
        DNCON4_2D_out = Average(DNCON4_2D_convs)
    else:
        DNCON4_2D_out = DNCON4_2D_convs[0]

    DNCON4_flatten = Flatten()(DNCON4_2D_out)
    DNCON4_INCEP = Model(inputs=[DNCON4_1D_input,contact_input], outputs=DNCON4_flatten)
    DNCON4_INCEP.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)
    # DNCON4_INCEP.summary()
    return DNCON4_INCEP