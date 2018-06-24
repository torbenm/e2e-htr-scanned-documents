import tensorflow as tf

# This file contains layer functions from tf.layers extended by summary functions


def conv2d(inputs,
           filters,
           kernel_size,
           strides=(1, 1),
           padding='valid',
           data_format='channels_last',
           dilation_rate=(1, 1),
           activation=None,
           use_bias=True,
           kernel_initializer=None,
           bias_initializer=init_ops.zeros_initializer(),
           kernel_regularizer=None,
           bias_regularizer=None,
           activity_regularizer=None,
           kernel_constraint=None,
           bias_constraint=None,
           trainable=True,
           name=None,
           reuse=None):
    """Functional interface for the 2D convolution layer.

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
      inputs: Tensor input.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, height, width)`.

      dilation_rate: An integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
          kernel after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: A string, the name of the layer.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      Output tensor.

    Raises:
      ValueError: if eager execution is enabled.
    """
    layer = tf.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _reuse=reuse,
        _scope=name)
    activation = layer.apply(inputs)
    tf.summary.histogram("weights", layer.kernel)
    if use_bias:
        tf.summary.histogram("bias", layer.bias)
    tf.summary.histogram("activations", activation)
    return activation


def batch_normalization(inputs,
                        axis=-1,
                        momentum=0.99,
                        epsilon=1e-3,
                        center=True,
                        scale=True,
                        beta_initializer=init_ops.zeros_initializer(),
                        gamma_initializer=init_ops.ones_initializer(),
                        moving_mean_initializer=init_ops.zeros_initializer(),
                        moving_variance_initializer=init_ops.ones_initializer(),
                        beta_regularizer=None,
                        gamma_regularizer=None,
                        beta_constraint=None,
                        gamma_constraint=None,
                        training=False,
                        trainable=True,
                        name=None,
                        reuse=None,
                        renorm=False,
                        renorm_clipping=None,
                        renorm_momentum=0.99,
                        fused=None,
                        virtual_batch_size=None,
                        adjustment=None):
    """Functional interface for the batch normalization layer.

    Reference: http://arxiv.org/abs/1502.03167

    "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift"

    Sergey Ioffe, Christian Szegedy

    Note: when training, the moving_mean and moving_variance need to be updated.
    By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
    need to be added as a dependency to the `train_op`. Also, be sure to add
    any batch_normalization ops before getting the update_ops collection.
    Otherwise, update_ops will be empty, and training/inference will not work
    properly. For example:

    ```python
      x_norm = tf.layers.batch_normalization(x, training=training)

      # ...

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    ```

    Arguments:
      inputs: Tensor input.
      axis: An `int`, the axis that should be normalized (typically the features
        axis). For instance, after a `Convolution2D` layer with
        `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
      momentum: Momentum for the moving average.
      epsilon: Small float added to variance to avoid dividing by zero.
      center: If True, add offset of `beta` to normalized tensor. If False, `beta`
        is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.
      beta_initializer: Initializer for the beta weight.
      gamma_initializer: Initializer for the gamma weight.
      moving_mean_initializer: Initializer for the moving mean.
      moving_variance_initializer: Initializer for the moving variance.
      beta_regularizer: Optional regularizer for the beta weight.
      gamma_regularizer: Optional regularizer for the gamma weight.
      beta_constraint: An optional projection function to be applied to the `beta`
          weight after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      gamma_constraint: An optional projection function to be applied to the
          `gamma` weight after being updated by an `Optimizer`.
      training: Either a Python boolean, or a TensorFlow boolean scalar tensor
        (e.g. a placeholder). Whether to return the output in training mode
        (normalized with statistics of the current batch) or in inference mode
        (normalized with moving statistics). **NOTE**: make sure to set this
        parameter correctly, or else your training/inference will not work
        properly.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      name: String, the name of the layer.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      renorm: Whether to use Batch Renormalization
        (https://arxiv.org/abs/1702.03275). This adds extra variables during
        training. The inference is the same for either value of this parameter.
      renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
        scalar `Tensors` used to clip the renorm correction. The correction
        `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
        `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
        dmax are set to inf, 0, inf, respectively.
      renorm_momentum: Momentum used to update the moving means and standard
        deviations with renorm. Unlike `momentum`, this affects training
        and should be neither too small (which would add noise) nor too large
        (which would give stale estimates). Note that `momentum` is still applied
        to get the means and variances for inference.
      fused: if `None` or `True`, use a faster, fused implementation if possible.
        If `False`, use the system recommended implementation.
      virtual_batch_size: An `int`. By default, `virtual_batch_size` is `None`,
        which means batch normalization is performed across the whole batch. When
        `virtual_batch_size` is not `None`, instead perform "Ghost Batch
        Normalization", which creates virtual sub-batches which are each
        normalized separately (with shared gamma, beta, and moving statistics).
        Must divide the actual batch size during execution.
      adjustment: A function taking the `Tensor` containing the (dynamic) shape of
        the input tensor and returning a pair (scale, bias) to apply to the
        normalized values (before gamma and beta), only during training. For
        example, if axis==-1,
          `adjustment = lambda shape: (
            tf.random_uniform(shape[-1:], 0.93, 1.07),
            tf.random_uniform(shape[-1:], -0.1, 0.1))`
        will scale the normalized value by up to 7% up or down, then shift the
        result by up to 0.1 (with independent scaling and bias for each feature
        but shared across all examples), and finally apply gamma and/or beta. If
        `None`, no adjustment is applied. Cannot be specified if
        virtual_batch_size is specified.

    Returns:
      Output tensor.

    Raises:
      ValueError: if eager execution is enabled.
    """
    layer = tf.layers.BatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
        fused=fused,
        trainable=trainable,
        virtual_batch_size=virtual_batch_size,
        adjustment=adjustment,
        name=name,
        dtype=inputs.dtype.base_dtype,
        _reuse=reuse,
        _scope=name)
        activation = layer.apply(inputs, training=training)
        if scale:
            tf.summary.histogram('gamma', layer.gamma)
        if center:
            tf.summary.histogram('beta', layer.beta)
        tf.summary.histogram('moving_mean', layer.moving_mean)
        tf.summary.histogram('moving_variance', layer.moving_variance)

        tf.summary.histogram('activation', activation)
        return activation


def dense(
        inputs, units,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=init_ops.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None):
    """Functional interface for the densely-connected layer.

    This layer implements the operation:
    `outputs = activation(inputs.kernel + bias)`
    Where `activation` is the activation function passed as the `activation`
    argument (if not `None`), `kernel` is a weights matrix created by the layer,
    and `bias` is a bias vector created by the layer
    (only if `use_bias` is `True`).

    Arguments:
      inputs: Tensor input.
      units: Integer or Long, dimensionality of the output space.
      activation: Activation function (callable). Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: Initializer function for the weight matrix.
        If `None` (default), weights are initialized using the default
        initializer used by `tf.get_variable`.
      bias_initializer: Initializer function for the bias.
      kernel_regularizer: Regularizer function for the weight matrix.
      bias_regularizer: Regularizer function for the bias.
      activity_regularizer: Regularizer function for the output.
      kernel_constraint: An optional projection function to be applied to the
          kernel after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      bias_constraint: An optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: String, the name of the layer.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      Output tensor the same shape as `inputs` except the last dimension is of
      size `units`.

    Raises:
      ValueError: if eager execution is enabled.
    """
    layer = tf.layers.Dense(units,
                            activation=activation,
                            use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            activity_regularizer=activity_regularizer,
                            kernel_constraint=kernel_constraint,
                            bias_constraint=bias_constraint,
                            trainable=trainable,
                            name=name,
                            dtype=inputs.dtype.base_dtype,
                            _scope=name,
                            _reuse=reuse)
    activation = layer.apply(inputs)
    tf.summary.histogram("kernel", layer.kernel)
    tf.summary.histogram("bias", layer.bias)
    tf.summary.histogram("activation", activation)
    return activation
