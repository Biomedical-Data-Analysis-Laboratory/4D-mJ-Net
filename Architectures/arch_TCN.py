from Model.constants import *
from Utils import model_utils, general_utils
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
from tcn.tcn.tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dropout, Concatenate, Conv2D, Conv3D, TimeDistributed, Permute


################################################################################
# model with Temporal Convolutional Network (TCN) 2.5D with single/multi encoders
def TCNet(params, batch=True, drop=True, leaky=True, MCD=True, single_enc=False, timedistr=False):
    T = getNUMBER_OF_IMAGE_PER_SECTION()+2
    vars = model_utils.get_init_params_2D(params,leaky)
    size_two = vars["size_two"]
    kernel_size_3 = vars["kernel_size_3"]
    kernel_size_2 = vars["kernel_size_2"]
    kernel_skip = (3,3,3)
    pool_skip = (1,1,T) if is_timelast() else (T,1,1)
    activ_func = vars["activ_func"]
    l1_l2_reg = vars["l1_l2_reg"]
    input_shape = (get_m(),get_n(),1)
    kernel_init = vars["kernel_init"]
    kernel_constraint = vars["kernel_constraint"]
    bias_constraint = vars["bias_constraint"]
    limchan = 2
    min_size = 8
    last_i = -1

    inputs, inputs_touse, latent_space, skip_conn = [], [], [], {}
    for _ in range(T):
        inp = layers.Input(shape=input_shape, sparse=False)
        inputs.append(inp)
        if timedistr: inp = layers.Reshape(input_shape+(1,))(inp)
        inputs_touse.append(inp)

    if timedistr:
        inputs_touse = Concatenate(-2)(inputs_touse)  # Concatenate together the inputs if we are using TimeDistributed layer
        inputs_touse = Permute((3,1,2,4))(inputs_touse)
        general_utils.print_int_shape(inputs_touse)
        inputs_touse = [inputs_touse]

    dict_layers = {}
    for inp in inputs_touse:
        i,idx,conv_level = 4,0,1
        if not single_enc: dict_layers = {}
        while (get_m()/conv_level)>min_size and (get_n()/conv_level)>min_size:
            key = str(int(get_m()/conv_level))
            i += 1
            if key not in skip_conn.keys(): skip_conn[key] = []
            channels = [int(2**i/limchan),int(2**i/limchan),int(2**i/limchan)]
            strides = [1,1,size_two]
            reshape = [False,False,True]
            for x in range(len(channels)):
                if reshape[x]:
                    howmany = 1 if not timedistr else T
                    shape = (int(get_m()/conv_level), int(get_n()/conv_level), howmany, int(2 ** i / limchan)) if is_timelast() \
                        else (howmany, int(get_m()/conv_level), int(get_n()/conv_level), int(2 ** i / limchan))
                    if key+"_"+str(idx)+"_reshape" not in dict_layers.keys(): dict_layers[key+"_"+str(idx)+"_reshape"] = layers.Reshape(shape)
                    skip_conn[key].append(dict_layers[key+"_"+str(idx)+"_reshape"](inp))

                if key+"_"+str(idx) not in dict_layers.keys():
                    ks = kernel_size_3 if strides[x]==1 else kernel_size_2
                    conv = Conv2D(channels[x], kernel_size=ks, activation=activ_func, kernel_regularizer=l1_l2_reg,
                                  kernel_initializer=kernel_init, padding="same", kernel_constraint=kernel_constraint,
                                  strides=strides[x], bias_constraint=bias_constraint)
                    if timedistr: conv = TimeDistributed(conv)
                    dict_layers[key + "_" + str(idx)] = conv
                inp = dict_layers[key + "_" + str(idx)](inp)
                general_utils.print_int_shape(inp)
                if leaky:
                    leakykey = key+"_"+str(idx)+"_leaky"
                    if leakykey not in dict_layers.keys(): dict_layers[leakykey] = layers.LeakyReLU(alpha=0.33)
                    inp = dict_layers[leakykey](inp)
                if batch:
                    batchkey = key+"_"+str(idx)+"_batch"
                    if batchkey not in dict_layers.keys(): dict_layers[batchkey] = layers.BatchNormalization()
                    inp = dict_layers[batchkey](inp)
                idx+=1
            conv_level*=2

        if drop and MCD: inp = model_utils.MonteCarloDropout(params["dropout"]["1"])(inp)
        elif drop and not MCD: inp = Dropout(params["dropout"]["1"])(inp)
        howmany = 1 if not timedistr else T
        if "LS_flat" not in dict_layers.keys(): dict_layers["LS_flat"] = layers.Reshape((1,min_size*min_size*int(2**i/limchan)*howmany))
        latent_space.append(dict_layers["LS_flat"](inp))
        last_i = i

    conc_latent = Concatenate(1)(latent_space) if not timedistr else latent_space[0]
    general_utils.print_int_shape(conc_latent)
    # With return_sequences=False, the output shape is: (batch_size, nb_filters)
    tcn_layer = TCN(nb_filters=min_size*min_size,
                    kernel_size=2,
                    dilations=(1,2,4,8,16),
                    padding='causal',
                    return_sequences=False,
                    dropout_rate=params["dropout"]["tcn"],
                    mcd=MCD)(conc_latent)
    general_utils.print_int_shape(tcn_layer)

    conv_up_dim = min_size
    inp = layers.Reshape((conv_up_dim,conv_up_dim,1))(tcn_layer)

    i = last_i
    while conv_up_dim<get_m() and conv_up_dim<get_n():
        key = str(conv_up_dim * 2)
        where = -1 if is_timelast() else 1
        if not timedistr:
            # Concatenate layers for the skip connections if we are not using the timedistr
            block = Concatenate(where)(skip_conn[key])
            general_utils.print_int_shape(block)
            block = model_utils.convolution_layer(block, int(2 ** i / limchan), kernel_skip, activ_func, l1_l2_reg, kernel_init, 'same', kernel_constraint, bias_constraint, leaky=leaky)
            general_utils.print_int_shape(block)
            if batch: block = layers.BatchNormalization()(block)
        else: block = skip_conn[key][0]

        # Reduce the third dimension (time)
        ## block = layers.AveragePooling3D(pool_skip)(block)
        for x,k in enumerate(params["stride"].keys()):
            if x>0:
                block = model_utils.convolution_layer(block,int(2 ** i / limchan),kernel_skip,activ_func,l1_l2_reg,
                                                      kernel_init,"same",kernel_constraint,bias_constraint,leaky=leaky,is2D=False)
                block = model_utils.convolution_layer(block,int(2 ** i / limchan),kernel_skip,activ_func,l1_l2_reg,
                                                      kernel_init,"same",kernel_constraint,bias_constraint,leaky=leaky,is2D=False)
            block = model_utils.convolution_layer(block,int(2 ** i / limchan),kernel_skip,activ_func,l1_l2_reg, kernel_init,
                                                  "same",kernel_constraint,bias_constraint,strides=(params["stride"][k],1,1),leaky=leaky,is2D=False)
        general_utils.print_int_shape(block)
        block = layers.Reshape((conv_up_dim*2, conv_up_dim*2, int(2 ** i / limchan)))(block)
        general_utils.print_int_shape(block)
        up = model_utils.up_layers(inp, [block],
                                   [int(2 ** i / limchan), int(2 ** i / limchan), int(2 ** i / limchan)], kernel_size_2,
                                   size_two, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint,
                                   params, leaky=leaky, is2D=True, batch=batch)
        i-=1
        conv_up_dim*=2
        inp = up
        general_utils.print_int_shape(up)

    # set the softmax activation function if the flag is set
    act_name = "softmax" if is_TO_CATEG() else "sigmoid"
    n_chann = len(get_labels()) if is_TO_CATEG() else 1

    y = model_utils.convolution_layer(inp, n_chann, (1, 1), act_name, l1_l2_reg, kernel_init, 'same', kernel_constraint,
                                      bias_constraint, is2D=True)
    general_utils.print_int_shape(y)

    model = models.Model(inputs=inputs, outputs=[y])
    return model


################################################################################
# model with Temporal Convolutional Network (TCN) 3.5D with single/multi encoders
def TCNet_3dot5D(params, batch=True, drop=True, leaky=True, MCD=True, single_enc=False):
    vars_3D = model_utils.get_init_params_3D(params, leaky)
    vars_3D["kernel_size"] = (3,3,3)
    vars_2D = model_utils.get_init_params_2D(params, leaky)
    vars_3D["input_shape_3D"] = (vars_3D["n_slices"], get_m(), get_n(), 1)
    k_skip = (3,3,getNUMBER_OF_IMAGE_PER_SECTION()) if is_timelast() else (getNUMBER_OF_IMAGE_PER_SECTION(), 3, 3)
    pool_skip = (1,1,getNUMBER_OF_IMAGE_PER_SECTION()) if is_timelast() else (getNUMBER_OF_IMAGE_PER_SECTION(), 1, 1)
    limchan = 4
    min_size = 4
    last_i = -1

    inputs, latent_space, skip_conn = [], [], {}
    for _ in range(getNUMBER_OF_IMAGE_PER_SECTION()): inputs.append(layers.Input(shape=vars_3D["input_shape_3D"], sparse=False))

    dict_layers = {}
    for inp in inputs:
        i,idx = 2,0
        if not single_enc: dict_layers = {}
        while K.int_shape(inp)[-3]>min_size and K.int_shape(inp)[-2]>min_size:
            key = str(K.int_shape(inp)[-3])
            i += 1
            if key not in skip_conn.keys(): skip_conn[key] = []
            channels = [int(2**i/limchan),int(2**i/limchan),int(2**i/limchan)]
            strides = [1, 1, vars_3D["size_two"]]
            reshape_and_skip = [False, False, True]
            for x in range(len(channels)):
                if reshape_and_skip[x]:
                    inp_tmp = inp
                    keyskip = key+"_"+str(idx)+"_skip"
                    if keyskip not in dict_layers.keys():
                        dict_layers[keyskip] = Conv3D(channels[x], kernel_size=vars_3D["kernel_size"],
                                                      activation=vars_3D["activ_func"], padding="same",
                                                      kernel_regularizer=vars_3D["l1_l2_reg"],
                                                      kernel_initializer=vars_3D["kernel_init"],
                                                      kernel_constraint=vars_3D["kernel_constraint"],
                                                      strides=(vars_3D["n_slices"],1,1),
                                                      bias_constraint=vars_3D["bias_constraint"])
                    inp_tmp = dict_layers[keyskip](inp_tmp)
                    if leaky:
                        leakykey = keyskip+"_leaky"
                        if leakykey not in dict_layers.keys(): dict_layers[leakykey] = layers.LeakyReLU(alpha=0.33)
                        inp = dict_layers[leakykey](inp_tmp)
                    if batch:
                        batchkey = keyskip+"_batch"
                        if batchkey not in dict_layers.keys(): dict_layers[batchkey] = layers.BatchNormalization()
                        inp = dict_layers[batchkey](inp_tmp)
                    shape = (1, K.int_shape(inp_tmp)[-2], K.int_shape(inp_tmp)[-3], int(2**i/limchan))
                    if key+"_"+str(idx)+"_reshape" not in dict_layers.keys(): dict_layers[key+"_"+str(idx)+"_reshape"] = layers.Reshape(shape)
                    skip_conn[key].append(dict_layers[key+"_"+str(idx)+"_reshape"](inp_tmp))

                if key+"_"+str(idx) not in dict_layers.keys():
                    dict_layers[key+"_"+str(idx)] = Conv3D(channels[x], kernel_size=vars_3D["kernel_size"],
                                                           activation=vars_3D["activ_func"], padding="same",
                                                           kernel_regularizer=vars_3D["l1_l2_reg"],
                                                           kernel_initializer=vars_3D["kernel_init"],
                                                           kernel_constraint=vars_3D["kernel_constraint"],
                                                           strides=strides[x],
                                                           bias_constraint=vars_3D["bias_constraint"])
                inp = dict_layers[key + "_" + str(idx)](inp)
                if leaky:
                    leakykey = key+"_"+str(idx)+"_leaky"
                    if leakykey not in dict_layers.keys(): dict_layers[leakykey] = layers.LeakyReLU(alpha=0.33)
                    inp = dict_layers[leakykey](inp)
                if batch:
                    batchkey = key+"_"+str(idx)+"_batch"
                    if batchkey not in dict_layers.keys(): dict_layers[batchkey] = layers.BatchNormalization()
                    inp = dict_layers[batchkey](inp)
                idx+=1

        if drop and MCD: inp = model_utils.MonteCarloDropout(params["dropout"]["1"])(inp)
        elif drop and not MCD: inp = Dropout(params["dropout"]["1"])(inp)

        i+=1
        channels = [int(2**i/limchan),int(2**i/limchan),int(2**i/limchan)]
        strides = [1, 1, (vars_3D["n_slices"],1,1)]
        for x in range(len(channels)):
            if "conv3D_"+str(x) not in dict_layers.keys():
                dict_layers["conv3D_"+str(x)] = Conv3D(channels[x], kernel_size=vars_3D["kernel_size"],
                                                       activation=vars_3D["activ_func"], padding="same",
                                                       kernel_regularizer=vars_3D["l1_l2_reg"],
                                                       kernel_initializer=vars_3D["kernel_init"],
                                                       kernel_constraint=vars_3D["kernel_constraint"],
                                                       strides=strides[x],
                                                       bias_constraint=vars_3D["bias_constraint"])
            inp = dict_layers["conv3D_"+str(x)](inp)
            if leaky:
                leakykey = "conv3D_"+str(x) + "_leaky"
                if leakykey not in dict_layers.keys(): dict_layers[leakykey] = layers.LeakyReLU(alpha=0.33)
                inp = dict_layers[leakykey](inp)
            if batch:
                batchkey = "conv3D_"+str(x) + "_batch"
                if batchkey not in dict_layers.keys(): dict_layers[batchkey] = layers.BatchNormalization()
                inp = dict_layers[batchkey](inp)

        if "LS_flat" not in dict_layers.keys(): dict_layers["LS_flat"] = layers.Reshape((1,min_size*min_size*int(2**i/limchan)))
        latent_space.append(dict_layers["LS_flat"](inp))
        last_i = i

    conc_latent = Concatenate(1)(latent_space)
    general_utils.print_int_shape(conc_latent)
    # With return_sequences=False, the output shape is: (batch_size, nb_filters)
    tcn_layer = TCN(nb_filters=64,
                    kernel_size=2,
                    dilations=(1, 2, 4, 8, 16),
                    padding='causal',
                    return_sequences=False,
                    dropout_rate=params["dropout"]["tcn"],
                    mcd=MCD)(conc_latent)
    general_utils.print_int_shape(tcn_layer)

    conv_up_dim = 8
    inp = layers.Reshape((conv_up_dim,conv_up_dim,1))(tcn_layer)

    i = last_i
    while K.int_shape(inp)[-3]<get_m() and K.int_shape(inp)[-2]<get_n():
        key = str(K.int_shape(inp)[-3] * 2)
        where = -1 if is_timelast() else 1
        # Concatenate layers for the skip connectio
        block = Concatenate(where)(skip_conn[key])
        general_utils.print_int_shape(block)
        block = model_utils.convolution_layer(block, int(2**i/limchan), k_skip, vars_2D["activ_func"],
                                              vars_2D["l1_l2_reg"], vars_2D["kernel_init"], 'same',
                                              vars_2D["kernel_constraint"], vars_2D["bias_constraint"], leaky=leaky)
        general_utils.print_int_shape(block)
        if batch: block = layers.BatchNormalization()(block)
        # Reduce the third dimension (time)
        block = layers.AveragePooling3D(pool_skip)(block)
        general_utils.print_int_shape(block)
        block = layers.Reshape((K.int_shape(inp)[-2] * 2, K.int_shape(inp)[-3] * 2, int(2 ** i / limchan)))(block)
        general_utils.print_int_shape(block)
        up = model_utils.up_layers(inp, [block], [int(2**i/limchan),int(2**i/limchan),int(2**i/limchan)],
                                   vars_2D["kernel_size_3"], vars_2D["size_two"], vars_2D["activ_func"],
                                   vars_2D["l1_l2_reg"], vars_2D["kernel_init"], vars_2D["kernel_constraint"],
                                   vars_2D["bias_constraint"], params, leaky=leaky, is2D=True, batch=batch)
        i-=1
        inp = up
        general_utils.print_int_shape(up)

    # set the softmax activation function if the flag is set
    act_name = "softmax" if is_TO_CATEG() else "sigmoid"
    n_chann = len(get_labels()) if is_TO_CATEG() else 1

    y = model_utils.convolution_layer(inp, n_chann, (1,1), act_name, vars_2D["l1_l2_reg"], vars_2D["kernel_init"],
                                      'same', vars_2D["kernel_constraint"], vars_2D["bias_constraint"], is2D=True)
    general_utils.print_int_shape(y)

    model = models.Model(inputs=inputs, outputs=[y])
    return model

