import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Reshape


def Conv4D(
        input,
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
        name=None):
    """
    Performs a 4D convolution of the (t,z,x,y) dimension of a tensor with shape: (b,c,t,z,x,y) with k filters.
    This operator realizes a 4D convolution by performing several 3D convolutions.

    Returns: The output tensor will be of shape: (b,k,t',z',x',y').
    """

    assert len(input.get_shape().as_list()) == 6, "Tensor of shape (b,t,z,x,y,c) expected"
    assert len(kernel_size) == 4, "4D kernel size expected"
    assert data_format == 'channels_last', "Data format other than 'channels_last' not yet implemented"
    assert dilation_rate == (1,1,1,1), "Dilation rate other than 1 not yet implemented"

    if not name: name = 'conv4D'

    # input, kernel, and output sizes
    (b, z_i, t_i, x_i, y_i, c_i) = tuple(input.get_shape().as_list())
    (z_k, t_k, x_k, y_k) = kernel_size

    # output size for 'valid' convolution
    if padding == 'valid': (z_o, t_o, x_o, y_o) = (z_i - z_k + 1, t_i - t_k + 1, x_i - x_k + 1, y_i - y_k + 1)
    else: (z_o, t_o, x_o, y_o) = (z_i, t_i, x_i, y_i)

    # output tensors for each 3D frame
    frame_results = [None] * z_o
    # convolve each kernel frame i with each input frame j
    for i in range(z_k):
        frame_conv3D = None
        for j in range(z_i):
            # add results to this output frame
            out_frame = j - (i - z_k // 2) - (z_i - z_o) // 2
            if out_frame < 0 or out_frame >= z_o: continue
            # convolve input frame j with kernel frame i
            input_layer = Reshape((t_i, x_i, y_i, c_i))(input[:, j, :, :])
            if frame_conv3D is None:
                frame_conv3D = Conv3D(filters, kernel_size=(t_k, x_k, y_k), padding=padding, strides=strides,
                                      activation=None, use_bias=use_bias, kernel_initializer=kernel_initializer,
                                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                                      bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                                      kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                      trainable=trainable)

            if frame_results[out_frame] is None: frame_results[out_frame] = frame_conv3D(input_layer)
            else: frame_results[out_frame] += frame_conv3D(input_layer)

    output = tf.stack(frame_results, axis=1)
    if activation: output = activation(output)
    return output
