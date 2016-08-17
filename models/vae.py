# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 18:06:48 2016

@author: t-kafa
"""

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.compile.nanguardmode import NanGuardMode

import math

from base import Model

from lasagne import init
from lasagne.nonlinearities import rectify, softplus, sigmoid, softmax
from lasagne.layers import (DenseLayer, NonlinearityLayer, 
                            InputLayer, ReshapeLayer, DimshuffleLayer, 
#                            BatchNormLayer, 
                            get_all_params, get_output)
from lasagne_extensions.layers import (SampleLayer, SimpleSampleLayer, GaussianLogDensityLayer,
#                                       StandardNormalLogDensityLayer, 
                                       MyBatchNormLayer, BernoulliLogDensityLayerWithLogits, 
                                       MultinomialLogDensityLayer, SimpleGaussianKLLayer)
from lasagne.updates import total_norm_constraint, adam

class VAE_Z_X(Model):
    
    def __init__(self, n_x, n_z, qz_hid, px_hid, x_dist='bernoulli', 
                 nonlinearity=rectify, px_nonlinearity=None, 
                 batchnorm=True, seed=1234, simple_mode=True):
        
        super(VAE_Z_X, self).__init__(n_x, px_hid, n_z, nonlinearity)
        
        self.x_dist = x_dist
        self.n_x = n_x
        self.n_z = n_z
        self.batchnorm = batchnorm
        self._srng = RandomStreams(seed)
        self.srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        self.simple_mode = simple_mode
        
        # Decide Glorot initializaiton of weights.
        init_w = 1e-3
        hid_w = 1.0
        init_b = 1e-3 #0.
        if nonlinearity == rectify or nonlinearity == softplus:
            hid_w = "relu"
        
        self.sym_kl_wu = T.scalar('kl_warmup')  # warm-up training weight
        self.sym_x = T.matrix('x')  # input variable
        self.sym_z = T.matrix('z')  # latent variable
        self.sym_samples = T.iscalar('samples')  # MC samples
        
        # Assist methods for collecting the layers
        def dense_layer(layer_in, n, dist_w=init.GlorotNormal, dist_b=init.Normal): #dist_w=init.GlorotUniform, dist_b=init.Constant):
            if batchnorm:
                dense = DenseLayer(layer_in, n, dist_w(hid_w), None, None) # batchnorm means no bias
                dense = MyBatchNormLayer(dense) # BatchNormLayer(dense)
            else:
                dense = DenseLayer(layer_in, n, dist_w(hid_w), dist_b(init_b), None)
            return NonlinearityLayer(dense, self.transf)

        def stochastic_layer(layer_in, n, samples, nonlin=None):
            mu = DenseLayer(layer_in, n, init.Normal(init_w), init.Normal(init_b), nonlin) #init.GlorotUniform(), init.Constant(init_b), nonlin)
            logvar = DenseLayer(layer_in, n, init.Normal(init_w), init.Normal(init_b), nonlin) #init.GlorotUniform(), init.Constant(init_b), nonlin)
            if self.simple_mode:
                return mu, mu, logvar #SimpleSampleLayer(mu, logvar), mu, logvar
            else:
                return SampleLayer(mu, logvar, eq_samples=samples, iw_samples=1), mu, logvar
        
        l_x_in = InputLayer((None, n_x))
        
        # Recognition Model: q(z|x)
        l_qz_x = l_x_in
        for hid in qz_hid:
            l_qz_x = dense_layer(l_qz_x, hid)
        l_qz_x, l_qz_x_mu, l_qz_x_logvar = stochastic_layer(l_qz_x, n_z, self.sym_samples)
        # The shape of l_qz_x_mu, l_qz_x_logvar is (BatchSize, nz)
        # The shape of l_qz_x is (BatchSize*self.sym_samples, n_z)   
        
        # Generative Model: p(x|z)
#        l_px_z = DenseLayer(l_qz_x, px_hid[0], init.GlorotUniform(), init.Constant(init_b), self.transf)
#        if len(px_hid) > 1:
#            for hid in px_hid[1:]:
#                l_px_z = dense_layer(l_px_z, hid) 
        l_px_z = l_qz_x
        for hid in px_hid:
            l_px_z = dense_layer(l_px_z, hid)
        
        if x_dist == 'gaussian':
            l_px_z, l_px_z_mu, l_px_z_logvar = stochastic_layer(l_px_z, n_x, 1, px_nonlinearity)
        else:
#            if batchnorm:
#                l_px_z = DenseLayer(l_px_z, n_x, init.GlorotUniform(), None, None)
#                l_px_z = BatchNormLayer(l_px_z, epsilon=1e-6, alpha=1e-2)
#            else:
#                l_px_z = DenseLayer(l_px_z, n_x, init.GlorotUniform(), init.Constant(init_b), None)            
            l_px_z = DenseLayer(l_px_z, n_x, init.GlorotNormal(), init.Normal(init_w), None)
            
            if x_dist == 'multinomial':
                l_px_z = NonlinearityLayer(l_px_z, softmax)
#            elif x_dist == 'bernoulli':
#                l_px_z = NonlinearityLayer(l_px_z, sigmoid)
                
        # Reshape all the model layers to have the same size
        self.l_x_in = l_x_in       
         
        if self.simple_mode:
            self.l_qz = l_qz_x
            self.l_qz_mu = l_qz_x_mu 
            self.l_qz_logvar = l_qz_x_logvar 
            
            self.l_px = l_px_z
            self.l_px_mu = l_px_z_mu if x_dist == "gaussian" else None
            self.l_px_logvar = l_px_z_logvar if x_dist == "gaussian" else None
        else:
            self.l_qz = ReshapeLayer(l_qz_x, (-1, self.sym_samples, 1, n_z)) 
            self.l_qz_mu = DimshuffleLayer(l_qz_x_mu, (0, 'x', 'x', 1)) 
            self.l_qz_logvar = DimshuffleLayer(l_qz_x_logvar, (0, 'x', 'x', 1)) 
            
            self.l_px = ReshapeLayer(l_px_z, (-1, self.sym_samples, 1, n_x))
            self.l_px_mu = ReshapeLayer(l_px_z_mu, (-1, self.sym_samples, 1, n_x)) if x_dist == "gaussian" else None
            self.l_px_logvar = ReshapeLayer(l_px_z_logvar, (-1, self.sym_samples, 1, n_x)) if x_dist == "gaussian" else None
        
#        # Predefined functions
#        inputs = [self.sym_x, self.sym_samples]
#        if self.simple_mode:            
#            outputs = get_output(self.l_qz, self.sym_x, deterministic=True)
#        else:
#            outputs = get_output(self.l_qz, self.sym_x, deterministic=True).mean(axis=(1, 2))
#        self.f_qz = theano.function(inputs, outputs)
#            
#        inputs = [self.sym_z, self.sym_samples]
#        outputs = get_output(self.l_px, self.sym_z, deterministic=True)
#        self.f_px = theano.function(inputs, outputs)

        # Define model parameters
        self.model_params = get_all_params([self.l_px])
        self.trainable_model_params = get_all_params([self.l_px], trainable=True)
        
    def build_model(self, n_train):
        
        super(VAE_Z_X, self).build_model(n_train)
        
        # Define the layers for the density estimation used in the lower bound.
        #l_log_qz = GaussianLogDensityLayer(self.l_qz, self.l_qz_mu, self.l_qz_logvar)
        #l_log_pz = StandardNormalLogDensityLayer(self.l_qz)
        l_kl_z = SimpleGaussianKLLayer(self.l_qz_mu, self.l_qz_logvar) # Integration with Gaussian variable can be analytically computed.
        
        if self.x_dist == 'bernoulli':
            l_log_px = BernoulliLogDensityLayerWithLogits(self.l_px, self.l_x_in) # use this layer means no activation with sigmoid
        elif self.x_dist == 'multinomial':
            l_log_px = MultinomialLogDensityLayer(self.l_px, self.l_x_in)
        elif self.x_dist == 'gaussian':
            l_log_px = GaussianLogDensityLayer(self.l_x_in, self.l_px_mu, self.l_px_logvar)
            
        out_layers = [l_kl_z, l_log_px]
        inputs = {self.l_x_in: self.sym_x}
        out = get_output(out_layers, inputs, deterministic=False) #batch_norm_update_averages=False, batch_norm_use_averages=False) #
        kl_z_, log_px_ = out
        
        # Regularizing with weight priors p(theta|N(0,1)), collecting and clipping gradients
        def log_normal(x, mean, std, eps=0.0):
            c = - 0.5 * math.log(2*math.pi)
            std += eps
            return c - T.log(T.abs_(std)) - (x - mean)**2 / (2 * std**2)
            
        weight_priors = 0.0
        for p in self.trainable_model_params:
            if 'W' not in str(p):
                continue
            weight_priors += log_normal(p, 0, 1).sum()
#            weight_priors += T.sqr(p).sum()
#        weight_priors *= -0.001
        
        # Compute lower bound and loss function
        lb = (log_px_ - kl_z_).mean()
        loss = - (log_px_ - self.sym_kl_wu * kl_z_).mean() - weight_priors / self.sym_n_train
                
        grads_collect = T.grad(loss, self.trainable_model_params)
        params_collect = self.trainable_model_params
        
        sym_beta1 = T.scalar('beta1')
        sym_beta2 = T.scalar('beta2')
#        clip_grad, max_norm = 1, 5
#        mgrads = total_norm_constraint(grads_collect, max_norm=max_norm)
#        mgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]
        mgrads = grads_collect
        updates = adam(mgrads, params_collect, self.sym_lr, sym_beta1, sym_beta2)
        
        # Training function
        inputs = [self.sym_x, self.sym_kl_wu, self.sym_lr, sym_beta1, sym_beta2]
        inputs = inputs + [self.sym_samples] if not self.simple_mode else inputs
        outputs = [loss, lb]
        f_train = theano.function(inputs=inputs, outputs=outputs, updates=updates)#, mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

        # Default training args. Note that these can be changed during or prior to training.
        self.train_args['inputs']['kl_warmup'] = 1
        self.train_args['inputs']['learningrate'] = 3e-4
        self.train_args['inputs']['beta1'] = 0.9
        self.train_args['inputs']['beta2'] = 0.999
        if not self.simple_mode:
            self.train_args['inputs']['samples'] = 1
        self.train_args['outputs']['loss'] = '%0.4f'
        self.train_args['outputs']['lb'] = '%0.4f'
        
        # Validation and test function
        if self.simple_mode:
            x_logits = get_output(self.l_px, self.sym_x, deterministic=True)
        else:
            x_logits = get_output(self.l_px, self.sym_x, deterministic=True).mean(axis=(1,2))
        out = get_output(out_layers, self.sym_x, deterministic=True)
        kl_z_, log_px_ = out
        lb = (log_px_ - kl_z_).mean()
        recon_err = T.square(T.nnet.sigmoid(x_logits) - self.sym_x).mean()
        
        inputs=[self.sym_x]
        inputs = inputs + [self.sym_samples] if not self.simple_mode else inputs
        f_validate = theano.function(inputs=inputs, outputs=[lb, recon_err])
        # Default validation args. Note that these can be changed during or prior to training.
        if not self.simple_mode:
            self.validate_args['inputs']['samples'] = 1
        self.validate_args['outputs']['valid_lb'] = '%0.4f'
        self.validate_args['outputs']['valid_recon_err'] = '%0.4f'        

        return f_train, f_validate, self.train_args, self.validate_args
        
    def model_info(self):
        qz_shapes = self.get_model_shape(get_all_params(self.l_qz))
        px_shapes = self.get_model_shape(get_all_params(self.l_px))[(len(qz_shapes) - 1):]
        s = ""
        s += 'batch norm: %s.\n' % (str(self.batchnorm))
        s += 'x distribution: %s.\n' % (str(self.x_dist))
        s += 'model q(z|x): %s.\n' % str(qz_shapes)[1:-1]
        s += 'model p(x|z): %s.\n' % str(px_shapes)[1:-1]
        return s
        
