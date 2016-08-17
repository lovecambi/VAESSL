# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:10:08 2016

@author: t-kafa
"""

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.compile.nanguardmode import NanGuardMode
#import numpy as np
import math

from base import Model

from lasagne import init
from lasagne.nonlinearities import rectify, softplus, sigmoid, softmax
from lasagne.objectives import categorical_crossentropy, categorical_accuracy
from lasagne.layers import (BatchNormLayer, DenseLayer, NonlinearityLayer, 
                            InputLayer, ReshapeLayer, DimshuffleLayer, get_all_params, 
                            get_output, ElemwiseSumLayer)
from lasagne_extensions.layers import (SampleLayer, GaussianLogDensityLayer, SimpleBernoulliSampleLayer,
#                                       SimpleSampleLayer, StandardNormalLogDensityLayer, 
                                       #MyBatchNormLayer,
                                       BernoulliLogDensityLayer, MultinomialLogDensityLayer, 
				       BinomialLogDensityLayerWithLogits, SimpleGaussianKLLayer)
from lasagne.updates import total_norm_constraint, adam
from lasagne.regularization import l1, l2, regularize_layer_params

class VAE_YZ_X(Model):
    
    def __init__(self, n_x, n_z, n_y, qz_hid, qy_hid, px_hid, x_dist='bernoulli', 
                 nonlinearity=rectify, px_nonlinearity=None, batchnorm=True, seed=1234, n_trial=2):
        
        super(VAE_YZ_X, self).__init__(n_x, px_hid, n_z, nonlinearity)
        
        self.x_dist = x_dist
        self.n_x = n_x
        self.n_z = n_z
        self.n_y = n_y
        self.batchnorm = batchnorm
        self._srng = RandomStreams(seed)
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        
        # Decide Glorot initializaiton of weights.
        init_w = 1e-3
        hid_w = ""
        if nonlinearity == rectify or nonlinearity == softplus:
            hid_w = "relu"
	
	self.sym_kl_wu = T.scalar('kl_warmup')  # warm-up training weight
        self.sym_beta = T.scalar('beta')  # scaling constant beta            
        self.sym_x_l = T.matrix('x')  # labeled inputs
        self.sym_t_l = T.matrix('t')  # labeled targets
        self.sym_x_u = T.matrix('x')  # unlabeled inputs
        self.sym_bs_l = T.iscalar('bs_l')  # number of labeled data
        self.sym_samples = T.iscalar('samples')  # MC samples
        self.sym_z = T.matrix('z')  # latent variable z
        
        # Assist methods for collecting the layers
        def dense_layer(layer_in, n, dist_w=init.GlorotNormal, dist_b=init.Normal):
            dense = DenseLayer(layer_in, n, dist_w(hid_w), dist_b(init_w), None)
            if batchnorm:
                dense = BatchNormLayer(dense)
            return NonlinearityLayer(dense, self.transf)

        def stochastic_layer(layer_in, n, samples, nonlin=None):
            mu = DenseLayer(layer_in, n, init.Normal(init_w), init.Normal(init_w), nonlin)
            logvar = DenseLayer(layer_in, n, init.Normal(init_w), init.Normal(init_w), nonlin)
            return SampleLayer(mu, logvar, eq_samples=samples, iw_samples=1), mu, logvar
        
        # Input layers
        l_x_in = InputLayer((None, n_x))
        l_y_in = InputLayer((None, n_y))

        # classifier q(y|x)
        l_qy_x = l_x_in
        for hid in qy_hid:
                l_qy_x = dense_layer(l_qy_x, hid)
        l_qy_x = DenseLayer(l_qy_x, n_y, init.GlorotNormal(), init.Normal(init_w), softmax)        
        # The shape of qy is (BatchSize, n_y)

        # Recognition Model: q(z|x, y)
        l_x_to_qz = DenseLayer(l_x_in, qz_hid[0], init.GlorotNormal(hid_w), init.Normal(init_w), None)
        l_y_to_qz = DenseLayer(l_y_in, qz_hid[0], init.GlorotNormal(hid_w), None, None) # bias is not necessary for both x and y
        l_qz_xy = ElemwiseSumLayer([l_x_to_qz, l_y_to_qz])
	if batchnorm:
	    l_qz_xy = BatchNormLayer(l_qz_xy)
	l_qz_xy = NonlinearityLayer(l_qz_xy, self.transf)
	if len(qz_hid) > 1:
            for hid in qz_hid[1:]:
                l_qz_xy = dense_layer(l_qz_xy, hid)
        l_qz_xy, l_qz_xy_mu, l_qz_xy_logvar = stochastic_layer(l_qz_xy, n_z, self.sym_samples)
        # The shape of mu, logvar is (BatchSize, n_z)
        # The shape of z is (BatchSize*self.sym_samples, n_z)   
        
        # Generative Model: p(x|y, z)
        l_y_to_px = DenseLayer(l_y_in, px_hid[0], init.GlorotNormal(hid_w), None, None) # bias is not necessary for both z and y
        l_y_to_px = DimshuffleLayer(l_y_to_px, (0, 'x', 'x', 1))
        l_qz_to_px = DenseLayer(l_qz_xy, px_hid[0], init.GlorotNormal(hid_w), init.Normal(init_w), None)
        l_qz_to_px = ReshapeLayer(l_qz_to_px, (-1, self.sym_samples, 1, px_hid[0]))
        l_px_zy = ReshapeLayer(ElemwiseSumLayer([l_qz_to_px, l_y_to_px]), [-1, px_hid[0]])        
        if batchnorm:
            l_px_zy = BatchNormLayer(l_px_zy)
        l_px_zy = NonlinearityLayer(l_px_zy, self.transf)
        if len(px_hid) > 1:
            for hid in px_hid[1:]:
                l_px_zy = dense_layer(l_px_zy, hid)
                
        if x_dist == 'bernoulli':
            l_px_zy = DenseLayer(l_px_zy, n_x, init.GlorotNormal(), init.Normal(init_w), sigmoid)
	elif x_dist == 'binomial':
	    self.n_trial = n_trial
	    #l_px_zy = DenseLayer(l_px_azy, n_x, init.GlorotNormal(), init.Normal(init_w), sigmoid)
	    #instead of above non-linear transform, we use logits for stable numerial computation
	    l_px_zy = DenseLayer(l_px_zy, n_x, init.GlorotNormal(), init.Normal(init_w), None)
        elif x_dist == 'multinomial':
            l_px_zy = DenseLayer(l_px_zy, n_x, init.GlorotNormal(), init.Normal(init_w), softmax)
        elif x_dist == 'gaussian':
            l_px_zy, l_px_zy_mu, l_px_zy_logvar = stochastic_layer(l_px_zy, n_x, 1, px_nonlinearity)
            
        # Reshape all the model layers to have the same size
        self.l_x_in = l_x_in
        self.l_y_in = l_y_in
        
        self.l_qy = DimshuffleLayer(l_qy_x, (0, 'x', 'x', 1))
        
        self.l_qz = ReshapeLayer(l_qz_xy, (-1, self.sym_samples, 1, n_z))
        self.l_qz_mu = DimshuffleLayer(l_qz_xy_mu, (0, 'x', 'x', 1))
        self.l_qz_logvar = DimshuffleLayer(l_qz_xy_logvar, (0, 'x', 'x', 1))
        
        self.l_px = ReshapeLayer(l_px_zy, (-1, self.sym_samples, 1, n_x))
        self.l_px_mu = ReshapeLayer(l_px_zy_mu, (-1, self.sym_samples, 1, n_x)) if x_dist == "gaussian" else None
        self.l_px_logvar = ReshapeLayer(l_px_zy_logvar, (-1, self.sym_samples, 1, n_x)) if x_dist == "gaussian" else None
        
        # Predefined functions
        inputs = [self.sym_x_l]
        outputs = get_output(self.l_qy, self.sym_x_l, deterministic=True).mean(axis=(1, 2))
        self.f_qy = theano.function(inputs, outputs)
        
#        inputs = [self.sym_x_l, self.sym_samples]
#        outputs = get_output(self.l_qz, self.sym_x_l, deterministic=True).mean(axis=(1, 2))
#        self.f_qz = theano.function(inputs, outputs)
#        
#        inputs = {l_qz_xy: self.sym_z, l_y_in: self.sym_t_l}
#        outputs = get_output(self.l_px, self.sym_z, deterministic=True)
#        self.f_px = theano.function([self.sym_z, self.sym_t_l, self.sym_samples], outputs)

        # Define model parameters
        self.model_params = get_all_params([self.l_qy, self.l_px])
        self.trainable_model_params = get_all_params([self.l_qy, self.l_px], trainable=True)
        
    def build_model(self, n_train, n_l):
        
        super(VAE_YZ_X, self).build_model(n_train)
                
        self.sym_n_l = n_l # no. of labeled data points
        
        # Define the layers for the density estimation used in the lower bound.
        #l_log_qz = GaussianLogDensityLayer(self.l_qz, self.l_qz_mu, self.l_qz_logvar)
        #l_log_pz = StandardNormalLogDensityLayer(self.l_qz)
        l_kl_z = SimpleGaussianKLLayer(self.l_qz_mu, self.l_qz_logvar) # Integration with Gaussian variable can be analytically computed.
        
        l_log_qy = MultinomialLogDensityLayer(self.l_qy, self.l_y_in, eps=1e-8)
        
        if self.x_dist == 'bernoulli':
            l_log_px = BernoulliLogDensityLayer(self.l_px, self.l_x_in) # return xlog(p) + (1-x)log(1-p)
	elif self.x_dist == 'binomial':
	    #l_log_px = BinomialLogDensityLayer(self.l_px, self.l_x_in, n=self.n_trial)
	    l_log_px = BinomialLogDensityLayerWithLogits(self.l_px, self.l_x_in, n=self.n_trial)
        elif self.x_dist == 'multinomial':
            l_log_px = MultinomialLogDensityLayer(self.l_px, self.l_x_in) # return -x*log(p)
        elif self.x_dist == 'gaussian':
            l_log_px = GaussianLogDensityLayer(self.l_x_in, self.l_px_mu, self.l_px_logvar)

        def lower_bound(kl_z, log_py, log_px, wu=1.):
            lb = log_px + log_py - wu * kl_z
            return lb
        
        # Lower bound for labeled data
        out_layers = [l_kl_z, l_log_qy, l_log_px]
        inputs = {self.l_x_in: self.sym_x_l, self.l_y_in: self.sym_t_l}
        out = get_output(out_layers, inputs, batch_norm_update_averages=False, batch_norm_use_averages=False)
        kl_z_l, log_qy_l, log_px_l = out
	sl_loss = - log_qy_l.mean() # used for performance observation
        # Prior p(y) expecting that all classes are evenly distributed
        py_l = softmax(T.zeros((self.sym_x_l.shape[0], self.n_y)))
        log_py_l = -categorical_crossentropy(py_l, self.sym_t_l).reshape((-1, 1)).dimshuffle((0, 'x', 'x', 1))
        lb_l = lower_bound(kl_z_l, log_py_l, log_px_l)
        lb_l = lb_l.mean(axis=(1, 2))  # Mean over the sampling dimensions
        log_qy_l *= (self.sym_beta * (self.sym_n_train / self.sym_n_l))  # Scale the supervised cross entropy with the alpha constant
        lb_l -= log_qy_l.mean(axis=(1,2))  # Collect the lower bound term and mean over sampling dimensions
        
        # Lower bound for unlabeled data
        bs_u = self.sym_x_u.shape[0]
        t_eye = T.eye(self.n_y, k=0)
        t_u = t_eye.reshape((self.n_y, 1, self.n_y)).repeat(bs_u, axis=1).reshape((-1, self.n_y))
        x_u = self.sym_x_u.reshape((1, bs_u, self.n_x)).repeat(self.n_y, axis=0).reshape((-1, self.n_x))
        
        out_layers = [l_kl_z, l_log_px]
        inputs = {self.l_x_in: x_u, self.l_y_in: t_u}
        out = get_output(out_layers, inputs, batch_norm_update_averages=False, batch_norm_use_averages=False)
        kl_z_u, log_px_u = out
        # Prior p(y) expecting that all classes are evenly distributed
        py_u = softmax(T.zeros((bs_u * self.n_y, self.n_y)))
        log_py_u = -categorical_crossentropy(py_u, t_u).reshape((-1, 1)).dimshuffle((0, 'x', 'x', 1))
        lb_u = lower_bound(kl_z_u, log_py_u, log_px_u)
        lb_u = lb_u.reshape((self.n_y, 1, 1, bs_u)).transpose(3, 1, 2, 0).mean(axis=(1, 2))
        inputs = {self.l_x_in: self.sym_x_u}
        y_u = get_output(self.l_qy, inputs, batch_norm_update_averages=True, batch_norm_use_averages=False).mean(axis=(1, 2))
        y_u += 1e-8  # Ensure that we get no NANs when calculating the entropy
        y_u /= T.sum(y_u, axis=1, keepdims=True)
        lb_u = (y_u * (lb_u - T.log(y_u))).sum(axis=1)
        
        # Regularizing with weight priors p(theta|N(0,1)), collecting and clipping gradients
        def log_normal(x, mean, std, eps=0.0):
            c = - 0.5 * math.log(2*math.pi)
            std += eps
            #return c - T.log(T.abs_(std)) - (x - mean)**2 / (2 * std**2)
            return - (x - mean)**2 / (2 * std**2) # this is only term contributing to gradient

        weight_priors = 0.0
        for p in self.trainable_model_params:
            if 'W' not in str(p):
                continue
            weight_priors +=  log_normal(p, 0, 1).sum() 

        # Compute loss function
        lb_labeled = lb_l.mean()
        lb_unlabeled = lb_u.mean()
        loss = - lb_labeled - lb_unlabeled  - weight_priors / self.sym_n_train + 1e-4 * l1(self.trainable_model_params[0])
        #loss = - lb_labeled - lb_unlabeled  - 5e-4 * weight_priors + 1e-4 * l1(self.trainable_model_params[0])

        grads_collect = T.grad(loss, self.trainable_model_params)
        params_collect = self.trainable_model_params       
        sym_beta1 = T.scalar('beta1')
        sym_beta2 = T.scalar('beta2')
        clip_grad, max_norm = 1, 5
        mgrads = total_norm_constraint(grads_collect, max_norm=max_norm)
        mgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]
        updates = adam(mgrads, params_collect, self.sym_lr, sym_beta1, sym_beta2)
        
        # Training function
        inputs = [self.sym_x_u, self.sym_x_l, self.sym_t_l, 
                  self.sym_beta, self.sym_lr, sym_beta1, sym_beta2, self.sym_samples]
        outputs = [loss, lb_labeled, lb_unlabeled, sl_loss]
        f_train = theano.function(inputs=inputs, outputs=outputs, updates=updates)

        # Default training args. Note that these can be changed during or prior to training.
        self.train_args['inputs']['beta'] = 0.1
        self.train_args['inputs']['learningrate'] = 3e-4
        self.train_args['inputs']['beta1'] = 0.9
        self.train_args['inputs']['beta2'] = 0.999
        self.train_args['inputs']['samples'] = 1
        self.train_args['outputs']['loss'] = '%0.4f'
        self.train_args['outputs']['lb-labeled'] = '%0.4f'
        self.train_args['outputs']['lb-unlabeled'] = '%0.4f'
	self.train_args['outputs']['sl_loss'] = '%0.4f'
        
        # Validation or Test function
        y = get_output(self.l_qy, self.sym_x_l, deterministic=True).mean(axis=(1, 2))
        class_err = (1. - categorical_accuracy(y, self.sym_t_l).mean()) * 100
        
        inputs = [self.sym_x_l, self.sym_t_l]
        f_validate = theano.function(inputs=inputs, outputs=[sl_loss, class_err])

        # Validation args.  Note that these can be changed during or prior to training.
        self.validate_args['outputs']['valid_sl_loss'] = '%0.4f'
        self.validate_args['outputs']['valid_err'] = '%0.2f%%'

        return f_train, f_validate, self.train_args, self.validate_args
        
    def get_output(self, x):
	# usually get prediction
        return self.f_qy(x)

    def model_info(self):
        qy_shapes = self.get_model_shape(get_all_params(self.l_qy))
        qz_shapes = self.get_model_shape(get_all_params(self.l_qz))
        px_shapes = self.get_model_shape(get_all_params(self.l_px))[(len(qz_shapes) - 1):]
        s = ""
        s += 'batch norm: %s.\n' % (str(self.batchnorm))
        s += 'x distribution: %s.\n' % (str(self.x_dist))
        s += 'model q(y|x): %s.\n' % str(qy_shapes)[1:-1]
        s += 'model q(z|x,y): %s.\n' % str(qz_shapes)[1:-1]
        s += 'model p(x|z,y): %s.\n' % str(px_shapes)[1:-1]
        return s

