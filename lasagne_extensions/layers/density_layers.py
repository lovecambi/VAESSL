import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.base import Layer
from lasagne import init
import math


class GaussianEntropyLayer(lasagne.layers.MergeLayer):
    def __init__(self, var, **kwargs):
        self.var = None
        if not isinstance(var, Layer):
            self.var, var = var, None
        input_lst = [i for i in [var] if not var is None]
        super(GaussianEntropyLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        logvar = self.var if self.var is not None else input.pop(0)
	
	c = 0.5 * (1 + math.log(2 * math.pi))
        entropy = c + 0.5 * logvar
	if var.ndim == 2:
	    return T.sum(entropy, axis=-1, keepdims=True)
	else:
            return T.mean(T.sum(entropy, axis=-1, keepdims=True), axis=(1, 2), keepdims=True)


class SimpleGaussianKLLayer(lasagne.layers.MergeLayer):
    """
    Note this KL divergence is between N(mu, diag(sigma^2)) and standard normal N(0,I).
    """
    def __init__(self, mu, var, **kwargs):
        self.mu, self.var = None, None
        if not isinstance(mu, Layer):
            self.mu, mu = mu, None
        if not isinstance(var, Layer):
            self.var, var = var, None
        input_lst = [i for i in [mu, var] if not i is None]
        super(SimpleGaussianKLLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        mu = self.mu if self.mu is not None else input.pop(0)
        logvar = self.var if self.var is not None else input.pop(0)

        kl = 0.5 * (T.sqr(mu) + T.exp(logvar) - logvar - 1)
	if mu.ndim == 2:
	    return T.sum(kl, axis=-1, keepdims=True)
	else:
            return T.mean(T.sum(kl, axis=-1, keepdims=True), axis=(1, 2), keepdims=True)


class GaussianKLLayer(lasagne.layers.MergeLayer):
    """
    Note this KL divergence is between two N(mu_i, diag(sigma_i^2)), i=1,2.
    """
    def __init__(self, mu1, var1, mu2, var2, **kwargs):
        self.mu1, self.var1, self.mu2, self.var2 = None, None, None, None
        if not isinstance(mu1, Layer):
            self.mu1, mu1 = mu1, None
        if not isinstance(var1, Layer):
            self.var1, var1 = var1, None
	if not isinstance(mu2, Layer):
            self.mu2, mu2 = mu2, None
        if not isinstance(var2, Layer):
            self.var2, var2 = var2, None
        input_lst = [i for i in [mu1, var1, mu2, var2] if not i is None]
        super(GaussianKLLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        mu1 = self.mu1 if self.mu1 is not None else input.pop(0)
        logvar1 = self.var1 if self.var1 is not None else input.pop(0)
	mu2 = self.mu2 if self.mu2 is not None else input.pop(0)
        logvar2 = self.var2 if self.var2 is not None else input.pop(0)

        kl = 0.5 * (T.sqr(mu1-mu2) / T.exp(logvar2) + T.exp(logvar1-logvar2) + logvar2  - logvar1 - 1)
	if mu.ndim == 2:
	    return T.sum(kl, axis=-1, keepdims=True)
	else:
            return T.mean(T.sum(kl, axis=-1, keepdims=True), axis=(1, 2), keepdims=True)


class StandardNormalLogDensityLayer(lasagne.layers.MergeLayer):
    def __init__(self, x, **kwargs):
        input_lst = [x]
        super(StandardNormalLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x = input.pop(0)
        c = - 0.5 * math.log(2 * math.pi)
        density = c - T.sqr(x) / 2
	if x.ndim == 2:
	    return T.sum(density, axis=-1, keepdims=True)
	else:
            return T.mean(T.sum(density, axis=-1, keepdims=True), axis=(1, 2), keepdims=True)


class GaussianLogDensityLayer(lasagne.layers.MergeLayer):
    def __init__(self, x, mu, var, **kwargs):
        self.x, self.mu, self.var = None, None, None
        if not isinstance(x, Layer):
            self.x, x = x, None
        if not isinstance(mu, Layer):
            self.mu, mu = mu, None
        if not isinstance(var, Layer):
            self.var, var = var, None
        input_lst = [i for i in [x, mu, var] if not i is None]
        super(GaussianLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x = self.x if self.x is not None else input.pop(0)
        mu = self.mu if self.mu is not None else input.pop(0)
        logvar = self.var if self.var is not None else input.pop(0)

        if mu.ndim > x.ndim:  # Check for sample dimensions.
            x = x.dimshuffle((0, 'x', 'x', 1))

        c = - 0.5 * math.log(2 * math.pi)
        density = c - logvar / 2 - T.sqr(x - mu) / (2 * T.exp(logvar))
	if mu.ndim == 2:
	    return T.sum(density, axis=-1, keepdims=True)
	else:
            return T.mean(T.sum(density, axis=-1, keepdims=True), axis=(1, 2), keepdims=True)


class BernoulliLogDensityLayer(lasagne.layers.MergeLayer):
    def __init__(self, x_mu, x, eps=1e-6, **kwargs):
        input_lst = [x_mu]
        self.eps = eps
        self.x = None

        if not isinstance(x, Layer):
            self.x, x = x, None
        else:
            input_lst += [x]
        super(BernoulliLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x_mu = input.pop(0)
        x = self.x if self.x is not None else input.pop(0)

        if x_mu.ndim > x.ndim:  # Check for sample dimensions.
            x = x.dimshuffle((0, 'x', 'x', 1))

        x_mu = T.clip(x_mu, self.eps, 1 - self.eps)
	if x.ndim == 2:
	    density = T.sum(-T.nnet.binary_crossentropy(x_mu, x), axis=-1, keepdims=True)
	else:
            density = T.mean(T.sum(-T.nnet.binary_crossentropy(x_mu, x), axis=-1, keepdims=True), axis=(1, 2),
                         keepdims=True)
        return density


class BernoulliLogDensityLayerWithLogits(lasagne.layers.MergeLayer):
    """
    logits is the pre-activation before sigmoid. This computation is more numerical stable.
    """
    def __init__(self, x_lg, x, **kwargs):
        input_lst = [x_lg]
        self.x = None

        if not isinstance(x, Layer):
            self.x, x = x, None
        else:
            input_lst += [x]
        super(BernoulliLogDensityLayerWithLogits, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x_lg = input.pop(0)
        x = self.x if self.x is not None else input.pop(0)

        if x_lg.ndim > x.ndim:  # Check for sample dimensions.
            x = x.dimshuffle((0, 'x', 'x', 1))

        density = - T.maximum(x_lg, 0) + x * x_lg - T.log(1+T.exp(-T.abs_(x_lg)))
	if x.ndim == 2:
	    density = T.sum(density, axis=-1, keepdims=True)
	else:
            density = T.mean(T.sum(density, axis=-1, keepdims=True), axis=(1, 2),
                         keepdims=True)
        return density


class BinomialLogDensityLayer(lasagne.layers.MergeLayer):
    """
    This computation omits the constant term log(n_choose_x) which contributes nothing to gradient.
    """
    def __init__(self, x_mu, x, n=2, eps=1e-6, **kwargs):
        input_lst = [x_mu]
	self.n = n
        self.eps = eps
        self.x = None

        if not isinstance(x, Layer):
            self.x, x = x, None
        else:
            input_lst += [x]
        super(BinomialLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x_mu = input.pop(0)
        x = self.x if self.x is not None else input.pop(0)

        if x_mu.ndim > x.ndim:  # Check for sample dimensions.
            x = x.dimshuffle((0, 'x', 'x', 1))

        x_mu = T.clip(x_mu, self.eps, 1 - self.eps)
	density = x * T.log(x_mu) + (self.n - x) * T.log(1 - x_mu)
	if x.ndim == 2:
	    density = T.sum(density, axis=-1, keepdims=True)
	else:
            density = T.mean(T.sum(density, axis=-1, keepdims=True), axis=(1, 2),
                         keepdims=True)
        return density


class BinomialLogDensityLayerWithLogits(lasagne.layers.MergeLayer):
    """
    logits is the pre-activation before sigmoid. This computation is more numerical stable.
    """
    def __init__(self, x_lg, x, n=2, **kwargs):
        input_lst = [x_lg]
        self.x = None
	self.n = n

        if not isinstance(x, Layer):
            self.x, x = x, None
        else:
            input_lst += [x]
        super(BinomialLogDensityLayerWithLogits, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x_lg = input.pop(0)
        x = self.x if self.x is not None else input.pop(0)

        if x_lg.ndim > x.ndim:  # Check for sample dimensions.
            x = x.dimshuffle((0, 'x', 'x', 1))

        density = - self.n * T.maximum(x_lg, 0) + x * x_lg - self.n * T.log(1+T.exp(-T.abs_(x_lg)))
	if x.ndim == 2:
	    density = T.sum(density, axis=-1, keepdims=True)
	else:
            density = T.mean(T.sum(density, axis=-1, keepdims=True), axis=(1, 2),
                         keepdims=True)
        return density


class MultinomialLogDensityLayer(lasagne.layers.MergeLayer):
    def __init__(self, x_mu, x, eps=1e-8, **kwargs):
        input_lst = [x_mu]
        self.eps = eps
        self.x = None
        if not isinstance(x, Layer):
            self.x, x = x, None
        else:
            input_lst += [x]
        super(MultinomialLogDensityLayer, self).__init__(input_lst, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, input, **kwargs):
        x_mu = input.pop(0)
        x = self.x if self.x is not None else input.pop(0)

        # Avoid Nans
        x_mu += self.eps

        if x_mu.ndim > x.ndim:  # Check for sample dimensions.
            x = x.dimshuffle((0, 'x', 'x', 1))
            # mean over the softmax outputs inside the log domain.
            x_mu = T.mean(x_mu, axis=(1, 2), keepdims=True)
	
        density = -T.sum(x * T.log(x_mu), axis=-1, keepdims=True)
        return density


class MyBatchNormLayer(Layer):

    def __init__(self, incoming, axes='auto', epsilon=1e-6, alpha=1e-2,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), std=init.Constant(1), **kwargs):
        super(MyBatchNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=True)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.std = self.add_param(std, shape, 'std',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        input_mean = input.mean(self.axes)
        input_std = T.sqrt(input.var(self.axes) + self.epsilon)

        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if use_averages:
            mean = self.mean
            std = self.std
        else:
            mean = input_mean
            std = input_std

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean + self.alpha * input_mean)
            running_std.default_update = ((1 - self.alpha) * running_std + self.alpha * input_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            std += 0 * running_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        std = std.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * (gamma / std) + beta
        return normalized

