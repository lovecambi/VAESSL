# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 21:22:37 2016

@author: fankai
"""

import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import theano
from training.trainssl import TrainModel, TrainM2onM1
from lasagne_extensions.nonlinearities import rectify
from data_loaders import mnist
from models.sdgmssl import SDGMSSL
from models.vae import VAE_Z_X
from models.vaessl import VAE_YZ_X
import numpy as np

import sys

seed = np.random.randint(1, 2147462579)

def run_vae_mnist(n_epochs=10):
    """
    Train a standard VAE model (M1) on the mnist dataset.
    """
    train_dataset, test_dataset, validation_dataset = mnist._download()
    
    n, n_x = train_dataset[0].shape    # Datapoints in the dataset, input features.
    bs = 100                           # The batchsize.
    n_ts_batches = test_dataset[0].shape[0]/bs
    n_vl_batches = validation_dataset[0].shape[0]/bs
    init_kl_wu = 0.
    
    model = VAE_Z_X(n_x=n_x, n_z=64, qz_hid=[512, 512], px_hid=[512, 512],
                    nonlinearity=rectify, batchnorm=True, x_dist='bernoulli', simple_mode=True)

    # Get the training functions.
    f_train, f_validate, train_args, validate_args = model.build_model(n)
    # Update the default function arguments.
    train_args['inputs']['kl_warmup'] = init_kl_wu
    train_args['inputs']['learningrate'] = 3e-3
#    # samples > 1 means simple_mode should be set Flase, default is True.
#    train_args['inputs']['samples'] = 5
#    validate_args['inputs']['samples'] = 5

    # Define training loop. Output training evaluations every 1 epoch
    train = TrainModel(model=model, output_freq=1)
    train.add_initial_training_notes("Training the VAE model with bn %s. seed %i." % (str(model.batchnorm), seed))
    train.train_model(f_train, train_args, f_validate, validate_args,
                      train_dataset[0], validation_dataset[0], test_dataset[0],
                      train_labeled=None, # labeled training data if appyling Semi-supervised training
                      batch_size=bs, 
                      n_valid_batches=n_vl_batches, 
                      n_test_batches=n_ts_batches, 
                      n_epochs=n_epochs,
                      anneal=[("learningrate", 1, 0.95, 3e-5)], # exp annealing training with a tuple of (var_name, every, scale constant, minimum value).
                      warmup=[("kl_warmup", init_kl_wu, 1., 100)] # if using linear warm-up training, parameter warmup should be initilized as 0                  
                      )


def run_vaessl_mnist(n_epochs=20):
    """
    Train a semi-supervised VAE model (M2) on the mnist dataset with 100 evenly distributed labels.
    """
    n_labeled = 100  # The total number of labeled data points.
    train_dataset, train_labeled, test_dataset, validation_dataset = mnist.load_semi_supervised(n_labeled=n_labeled, filter_std=0.0, seed=seed, train_valid_combine=True)

    n, n_x = train_dataset[0].shape
    bs = 100
#    n_ts_batches = test_dataset[0].shape[0]/bs
    n_vl_batches = validation_dataset[0].shape[0]/bs
#    init_kl_wu = 1.

    # Initialize the semi deep generative model.
    model = VAE_YZ_X(n_x=n_x, n_z=100, n_y=10, qz_hid=[500, 500], qy_hid=[500, 500], px_hid=[500, 500],
                    nonlinearity=rectify, batchnorm=True, x_dist='bernoulli')

    # Get the training functions.
    f_train, f_validate, train_args, validate_args = model.build_model(n, n_labeled)
    # Update the default function arguments.
    train_args['inputs']['beta'] = .1
    train_args['inputs']['learningrate'] = 3e-4
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 0.999
    train_args['inputs']['samples'] = 1

    # Define training loop. Output training evaluations every 1 epoch
    # and the custom evaluation method every 10 epochs.
    train = TrainModel(model=model, output_freq=1)
    train.add_initial_training_notes("Training the semi-supervised VAE with %i labels. bn %s. seed %i." % (
                                    n_labeled, str(model.batchnorm), seed))
    train.train_model(f_train, train_args, f_validate, validate_args,
                      train_dataset[0], validation_dataset, None,
		      train_labeled, # labeled training data if appyling Semi-supervised training
                      batch_size=100, 
                      n_valid_batches=n_vl_batches, 
                      n_test_batches=1, 
                      n_epochs=n_epochs,
                      anneal=[("learningrate", 200, 0.75, 3e-5)],
#                      warmup=[("kl_warmup", init_kl_wu, 1., 100)]
                      )

def run_M1M2_mnist(n_epochs=20):
    """
    M1: VAE_Z_X
    M2: VAE_YZ_X
    This model requires unsupervised VAE training first to obtain latent features, mean and logvar.
    Thus, the data feed to the VAESSL has the format: (mean, logvar, target)
    """

    # TODO, load data from M1 results

    
    # Initialize the semi deep generative model.
    model = VAE_YZ_X(n_x=n_x, n_z=100, n_y=10, qz_hid=[500, 500], qy_hid=[500, 500], px_hid=[500, 500],
                    nonlinearity=rectify, batchnorm=True, x_dist='gaussian')

    # Get the training functions.
    f_train, f_validate, train_args, validate_args = model.build_model(n, n_labeled)
    # Update the default function arguments.
    train_args['inputs']['beta'] = .1
    train_args['inputs']['learningrate'] = 3e-4
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 0.999
    train_args['inputs']['samples'] = 1
    
    # Evaluate the approximated classification error with K MC samples for a good estimate
    def M1M2_custom_evaluation(model, path, K=10):
    	mean_evals = []
    	for k in xrange(K):
	    eps = np.random.standard_normal(size=test_dataset[0].shape)
            cur_data = test_dataset[0] + np.exp(0.5*test_dataset[1]) * eps  
            mean_eval = model.get_output(cur_data)
            mean_evals.append(mean_eval)
        results = np.mean(np.asarray(mean_evals), axis=0)
    	t_class = np.argmax(test_dataset[2], axis=1)
    	y_class = np.argmax(results, axis=1)
    	missclass = (np.sum(y_class != t_class, dtype='float32') / len(y_class)) * 100.
    	train.write_to_logger("test %d-samples: %0.2f%%." % (K, missclass))

    # Define training loop. Output training evaluations every 1 epoch
    # and the custom evaluation method every 10 epochs.
    train = TrainModel(model=model, output_freq=1)
    train.add_initial_training_notes("Training the semi-supervised VAE with %i labels. bn %s. seed %i." % (
                                    n_labeled, str(model.batchnorm), seed))
    train.train_model(f_train, train_args, f_validate, validate_args,
                      train_dataset[0:2], validation_dataset, None,
		      train_labeled, # labeled training data if appyling Semi-supervised training
                      batch_size=100, 
                      n_valid_batches=n_vl_batches, 
                      n_test_batches=1, 
                      n_epochs=n_epochs,
                      anneal=[("learningrate", 200, 0.75, 3e-5)],
#                      warmup=[("kl_warmup", init_kl_wu, 1., 100)]
                      )


def run_sdgmssl_mnist(n_epochs=3):
    """
    Train a semi-supervised skip deep generative model on the mnist dataset with several evenly distributed labels.
    """
    n_labeled = 100  # The total number of labeled data points.
    train_dataset, train_labeled, test_dataset, validation_dataset = mnist.load_semi_supervised(n_labeled=n_labeled, filter_std=0.0, seed=seed, train_valid_combine=True)

    n, n_x = train_dataset[0].shape
    bs = 100
    n_ts_batches = test_dataset[0].shape[0]/bs
    n_vl_batches = validation_dataset[0].shape[0]/bs
#    init_kl_wu = 1.

    # Initialize the semi deep generative model.
    model = SDGMSSL(n_x=n_x, n_a=100, n_z=100, n_y=10, qa_hid=[500, 500], qz_hid=[500, 500], 
                    qy_hid=[500, 500], px_hid=[500, 500], pa_hid=[500, 500], 
                    nonlinearity=rectify, batchnorm=True, x_dist='bernoulli')

    # Get the training functions.
    f_train, f_validate, train_args, validate_args = model.build_model(n, n_labeled)
    # Update the default function arguments.
    train_args['inputs']['beta'] = .1
    train_args['inputs']['learningrate'] = 3e-4
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 0.999
    train_args['inputs']['samples'] = 5
    validate_args['inputs']['samples'] = 5

    # Define training loop. Output training evaluations every 1 epoch
    # and the custom evaluation method every 10 epochs.
    train = TrainModel(model=model, output_freq=1)
    train.add_initial_training_notes("Training the semi-supervised VAE with %i labels. bn %s. seed %i." % (
                                    n_labeled, str(model.batchnorm), seed))
    train.train_model(f_train, train_args, f_validate, validate_args,
                      train_dataset[0], validation_dataset, test_dataset,
		      train_labeled, # labeled training data if appyling Semi-supervised training
                      batch_size=100, 
                      n_valid_batches=n_vl_batches, 
                      n_test_batches=1, 
                      n_epochs=n_epochs,
                      anneal=[("learningrate", 200, 0.75, 3e-5)],
#                      warmup=[("kl_warmup", init_kl_wu, 1., 100)]
                      )

if __name__ == "__main__":
    run_vae_mnist()
    #run_vaessl_mnist()

    # TODO: test M2_custom_evaluation
    #run_M1M2_mnist() # need test
    #run_sdgmssl_mnist()

    
