import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Reshape


def Conv4D(
        inp,
        filters,
        kernel_size,
        strides=(1,1,1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1,1,1,1),
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reduce_dim=2):
    """
    Performs a 4D convolution of the (t,z,x,y) dimension of a tensor with shape: (b,c,t,z,x,y) with k filters.
    This operator realizes a 4D convolution by performing several 3D convolutions.

    Returns: The output tensor will be of shape: (b,k,t',z',x',y').
    """

    assert len(inp.get_shape().as_list()) == 6, "Tensor of shape (b,t,z,x,y,c) expected"
    assert len(kernel_size) == 4, "4D kernel size expected"
    assert padding == 'same', "Padding format other than same not implemented"
    assert data_format == 'channels_last', "Data format other than 'channels_last' not yet implemented"
    assert dilation_rate == (1,1,1,1), "Dilation rate other than 1 not yet implemented"
    assert reduce_dim in [1,2], "reduce_dim can only be 1 or 2"

    if not name: name = 'conv4D'

    # input, kernel, and output sizes
    out_axis = 1 if reduce_dim == 2 else 2
    reduce_i = inp.get_shape().as_list()[out_axis]
    keep_i = inp.get_shape().as_list()[reduce_dim]
    (b, z_i, t_i, x_i, y_i, c_i) = tuple(inp.get_shape().as_list())
    reduce_k = list(kernel_size)[out_axis-1]
    keep_k = list(kernel_size)[reduce_dim-1]
    (z_k, t_k, x_k, y_k) = kernel_size

    # output size for 'same' convolution
    # (z_o, t_o, x_o, y_o) = (z_i, t_i, x_i, y_i)
    reduce_o = reduce_i

    # output tensors for each 3D frame
    frame_results = [None] * reduce_o
    # convolve each kernel frame i with each input frame j
    for i in range(reduce_k):
        frame_conv3D = None
        for j in range(reduce_i):
            # add results to this output frame
            out_frame = j - (i - reduce_k // 2) - (reduce_i - reduce_o) // 2
            if out_frame < 0 or out_frame >= reduce_o: continue
            # convolve input frame j with kernel frame i
            reshape_shape = (keep_i, x_i, y_i, c_i)
            if reduce_dim==2: inp_reduced = inp[:, j, :, :, :, :]  # reduce time using the z-dim
            elif reduce_dim==1: inp_reduced = inp[:, :, j, :, :, :]  # reduce z using time-dim

            input_layer = Reshape(reshape_shape)(inp_reduced)
            if frame_conv3D is None:
                frame_conv3D = Conv3D(filters, kernel_size=(keep_k, x_k, y_k), padding=padding, strides=strides,
                                      activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                                      kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      trainable=trainable)

            if frame_results[out_frame] is None: frame_results[out_frame] = frame_conv3D(input_layer)
            else: frame_results[out_frame] += frame_conv3D(input_layer)

    output = tf.stack(frame_results, axis=out_axis)
    if activation: output = activation(output)
    return output
