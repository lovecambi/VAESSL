# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:21:17 2016

@author: t-kafa
"""

import sys, os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import theano
from training.trainssl import TrainModel
from lasagne_extensions.nonlinearities import rectify
from models.sdgmssl import SDGMSSL
from models.vae import VAE_Z_X
from models.vaessl import VAE_YZ_X
import numpy as np

import pandas
import sklearn.cross_validation
import sklearn.metrics

seed = np.random.randint(1, 2147462579)

def simulate(X, sigma_g=0.5, perc_causal=0.1):
    """
    Genearte binary labels
    """
    prng = np.random.RandomState(42)
    causal = prng.permutation(X.shape[1])[:int(perc_causal*X.shape[1])]
    print('mean MAF: %.4f' % np.mean((X[:, causal]==1).sum(axis=0)/X.shape[0]))
    nc = len(causal)
    scaled_sigma_g = (sigma_g/nc)/X[:, causal].var(axis=0).mean()
    betas = prng.randn(nc, 1)*np.sqrt(scaled_sigma_g)
    noise = prng.randn(X.shape[0], 1)*np.sqrt(1-sigma_g)
    y = np.dot(X[:, causal], betas)
    y += noise
    mu = y.mean()
    Y_bin = np.zeros(shape=(y.shape[0],2), dtype=theano.config.floatX)
    Y_bin[:,:1] = y < mu    # column idx 0 
    Y_bin[:,1:] = y >= mu   # column idx 1
    return Y_bin

def PCA_processing(X, reduced_dim=2, normalize=False):
    # X is N by dim matrix
    N = X.shape[0]
    if normalize:
	X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    C = np.dot(X.T, X)/(N - 1)
    evals, evec = np.linalg.eigh(C)
    W = evec[:,::-1]
    return W[:,:reduced_dim].astype(dtype=theano.config.floatX)
    
def run_vaessl_chr(data, y, p=0.1, n_epochs=20, data_dist='binomial'):  
        
    samples = data.shape[0]
    indices = np.arange(samples)
    np.random.shuffle(indices)
    
    n = samples * 4/5 # training data size, 4/5 of total as training
    tr_idx_ar = indices[:n] # training data index array
    train_dataset = (data[tr_idx_ar], y[tr_idx_ar])
    vl_idx_ar = indices[n:]
    validation_dataset = (data[vl_idx_ar], y[vl_idx_ar])

    # Set labeled training data   
    percent = p
    n_labeled = int(n * percent)
    train_labeled = (train_dataset[0][:n_labeled], train_dataset[1][:n_labeled])
    
    n_x = train_dataset[0].shape[1]
    bs = 100
#    init_kl_wu = 1.

    model = VAE_YZ_X(n_x=n_x, n_z=128, n_y=2, qz_hid=[512, 512], qy_hid=[512, 512], px_hid=[512, 512],
                    nonlinearity=rectify, batchnorm=True, x_dist=data_dist, n_trial=2)
                
    # Get the training functions.
    f_train, f_validate, train_args, validate_args = model.build_model(n, n_labeled)
    # Update the default function arguments.
    train_args['inputs']['beta'] = 1.5
    train_args['inputs']['learningrate'] = 3e-4
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 0.999
    train_args['inputs']['samples'] = 1
    
    def custom_evaluation(model, path):
        mean_evals = model.get_output(validation_dataset[0])
        y_true = validation_dataset[1][:,1]
        y_pred = mean_evals[:,1]
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_pred)
        auc = sklearn.metrics.auc(fpr, tpr)
        train.write_to_logger("Held-out AUC: %0.6f." % auc)

    # Define training loop. Output training evaluations every epoch
    # and the custom evaluation method every several epochs.
    train = TrainModel(model=model, output_freq=1, pickle_f_custom_freq=1, f_custom_eval=custom_evaluation)
    train.add_initial_training_notes("Training the semi-supervised VAE with %i labels. bn %s. seed %i." % (
                                    n_labeled, str(model.batchnorm), seed))
    train_mode = 'SSL'
    train.train_model(f_train, train_args, f_validate, validate_args, train_mode,
                      train_dataset[0], validation_dataset, None,
                      train_labeled, # labeled training data if appyling Semi-supervised training
                      batch_size=bs,
                      eval_valid_batch=True,
                      eval_test_batch=True,
                      n_epochs=n_epochs,
                      anneal=[("learningrate", 1, 0.95, 3e-5)],
#                      warmup=[("kl_warmup", init_kl_wu, 1., 100)]
                      )


def run_sdgmssl_chr(data, y, p=0.05, n_epochs=5, data_dist='binomial'):
    
    samples = data.shape[0]
    indices = np.arange(samples)
    np.random.shuffle(indices)
    
    n = samples * 4/5 # training data size, 4/5 of total as training
    tr_idx_ar = indices[:n] # training data index array
    train_dataset = (data[tr_idx_ar], y[tr_idx_ar])
    vl_idx_ar = indices[n:]
    validation_dataset = (data[vl_idx_ar], y[vl_idx_ar])
    
    percent = p
    n_labeled = int(n * percent)
    train_labeled = (train_dataset[0][:n_labeled], train_dataset[1][:n_labeled])
    
    n_x = train_dataset[0].shape[1]
    bs = 100
#    init_kl_wu = 1.

    model = SDGMSSL(n_x=n_x, n_a=128, n_z=128, n_y=2, qa_hid=[512,512], qz_hid=[512, 512], 
                    qy_hid=[512, 512], px_hid=[512, 512], pa_hid=[512, 512], 
                    nonlinearity=rectify, batchnorm=True, x_dist=data_dist, n_trial=2)
                
    # Get the training functions.
    f_train, f_validate, train_args, validate_args = model.build_model(n, n_labeled)
    # Update the default function arguments.
    train_args['inputs']['beta'] = 10.0
    train_args['inputs']['learningrate'] = 3e-4
    train_args['inputs']['beta1'] = 0.9
    train_args['inputs']['beta2'] = 0.999
    train_args['inputs']['samples'] = 5
    validate_args['inputs']['samples'] = 5

    # Define training loop. Output training evaluations every epoch
    # and the custom evaluation method every several epochs.
    def custom_evaluation(model, path):
        mean_evals = model.get_output(validation_dataset[0], 10)
        y_true = validation_dataset[1][:,1]
        y_pred = mean_evals[:,1]
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_pred)
        auc = sklearn.metrics.auc(fpr, tpr)
        train.write_to_logger("Held-out AUC: %0.6f." % auc)

    train = TrainModel(model=model, output_freq=1, pickle_f_custom_freq=1, f_custom_eval=custom_evaluation)
    train.add_initial_training_notes("Training the semi-supervised VAE with %i labels. bn %s. seed %i." % (
                                    n_labeled, str(model.batchnorm), seed))
    train_mode = 'SSL'
    train.train_model(f_train, train_args, f_validate, validate_args, train_mode, 
                      train_dataset[0], validation_dataset, None,
                      train_labeled, # labeled training data if appyling Semi-supervised training
                      batch_size=bs,
                      eval_valid_batch=True,
                      eval_test_batch=True,
                      n_epochs=n_epochs,
                      anneal=[("learningrate", 1, 0.95, 3e-5)],
#                      warmup=[("kl_warmup", init_kl_wu, 1., 100)]
                      )


if __name__ == "__main__":

    samples = [250000]
    
    data = pandas.read_csv('/home/t-kafa/chr20.dat', delimiter=' ', skiprows=499999-samples[-1], dtype=np.int8, index_col=None, names=None).values
    y = simulate(data, sigma_g=0.5, perc_causal=0.2)
    print "Finish Data Loading and Label Generation: Binomial Distributed Bino(2,p)."
    #run_vaessl_chr(data, y)
    run_sdgmssl_chr(data, y, n_epochs=2)
    #run_sdgmssl_chr(p=0.5)
"""
    # Data normalization
    PCA = True
    data = (data - data.mean(0)) / (data.std(0) + 1e-8)
    data = data.astype(dtype=theano.config.floatX)
    print "Finish Data Normalization: Gaussian distributed N(0,I)."
    run_sdgmssl_chr(data, y, n_epochs=2, data_dist='gaussian')

    # PCA
    PCA = True
    if PCA:
        dims = [4096]
        W = PCA_processing(data, dims[-1])
        data = np.dot(data, W)
        print "Finish Data PCA Processing."
    
    run_sdgmssl_chr(data, y, data_dist='gaussian')
"""    
