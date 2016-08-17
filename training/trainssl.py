# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 22:56:09 2016

@author: fankai
"""

import numpy as np
import theano
from utils import env_paths as paths
from base import Train
import time


class TrainModel(Train):
    """
    The class of train model supports basic unsupervised(USL), semi-supervised(SSL) and supervised(SL training.
    Data format for train and test
    USL: (x) and (x)
    SSL: (x_u) (x_l, t_l) and (x, t)
    SL:  (x, t) and (x, t)
    """


    def __init__(self, model, output_freq=1, pickle_f_custom_freq=None,
                 f_custom_eval=None):
        super(TrainModel, self).__init__(model, pickle_f_custom_freq, f_custom_eval)
        self.output_freq = output_freq

    def train_model(self, f_train, train_args, f_validate, validation_args, train_mode, 
                    train_dataset, validation_dataset, test_dataset=None, train_labeled=None,
                    batch_size=100, eval_valid_batch=False, eval_test_batch=False, n_epochs=100, 
                    anneal=None, warmup=None):
        """
        This function is to train defined model. 
        :f_train: training function of model
        :train_args: training arguments of model
        :f_validate: validate function of model
        :validate_args: validate arguments of model
        :train_mode: 'USL', 'SSL', 'SL', determine how to fetch mini-batch data.
        :train_dataset: 
        :validation_dataset:
        :test_dataset:
        :train_labeled: only not None for semi-supervised training, indicating train_dataset includes no labels.
        :batch_size: number of training data points in every iteration
        :eval_valid_batch: boolean, indicating validation dataset is evaluated in batches or not.
        :eval_test_batch: same as above.
        :n_epochs: number of total training epochs
        :anneal: list of exponential annealing training parameters, ('key', every, decay_rate, min_val)
            e.g., [('learningrate', 10, 0.95, 1e-5)]
        :warmup: list of linear warm up training parameters, ('key', initial_val, final_val, epochs)
            e.g., [('kl_warmup', 0, 1, 100)]
        """
        
        self.write_to_logger("### MODEL PARAMS ###")
        self.write_to_logger(self.model.model_info())
        self.write_to_logger("### TRAINING PARAMS ###")
        self.write_to_logger(
            "Train -> %s: %s" % (";".join(train_args['inputs'].keys()), str(train_args['inputs'].values())))
        self.write_to_logger(
            "Valid -> %s: %s" % (";".join(validation_args['inputs'].keys()), str(validation_args['inputs'].values())))
        
        if anneal is not None:
            for t in anneal:
                key, freq, rate, min_val = t
                self.write_to_logger(
                    "Exp. Annealing %s %0.4f after every %i epochs with minimum value %f." % (key, rate, int(freq), min_val))

        if warmup is not None:
            for t in warmup:
                key, s, e, wu_epochs = t
                self.write_to_logger(
                    "Linear increasing %s weight from %0.2f. to %0.2f in the first %d epochs." % (key, s, e, wu_epochs))


        self.write_to_logger("### TRAINING MODEL ###")

        if self.custom_eval_func is not None:
            self.custom_eval_func(self.model, paths.get_custom_eval_path(0, self.model.root_path))
        
        self.train_mode = train_mode
        self._srng = np.random.RandomState(np.random.randint(1,2147462579))
        self.max_iteration_u = self.model.sym_n_train / batch_size
        
        if train_labeled is not None:
            self.max_iteration_l = self.model.sym_n_l / batch_size
        
        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            start_time = time.time()
            train_outputs = []
            
            if self.train_mode == 'USL':
                for batch in self.iterate_minibatches_u(train_dataset, batch_size, shuffle=True):
                    train_output = f_train(batch, *train_args['inputs'].values())
                    train_outputs.append(train_output)
            elif self.train_mode == 'SSL':
                for batch in self.iterate_minibatches_ssl(train_dataset, train_labeled, batch_size, shuffle=True):
                    batch_x_u, batch_x_l, batch_t_l = batch
                    train_output = f_train(batch_x_u, batch_x_l, batch_t_l, *train_args['inputs'].values())
                    train_outputs.append(train_output)
            elif self.train_mode == 'SL':
                for batch in self.iterate_minibatches_u(train_dataset, batch_size, shuffle=True):
                    batch_x, batch_t = batch
                    train_output = f_train(batch_x, batch_t, *train_args['inputs'].values())
                    train_outputs.append(train_output)
            else:
                raise NotImplementedError
                    
            self.eval_train[epoch] = np.mean(np.array(train_outputs), axis=0)
                
            self.model.after_epoch()
            
            end_time = time.time() - start_time

            if anneal is not None:
                for t in anneal:
                    key, freq, rate, min_val = t
                    new_val = train_args['inputs'][key] * rate
                    if new_val < min_val:
                        train_args['inputs'][key] = min_val
                    elif epoch % freq == 0:
                        train_args['inputs'][key] = new_val

            if warmup is not None:
                for t in warmup:
                    key, s, e, wu_epochs = t
                    if epoch < wu_epochs:
                        train_args['inputs'][key] += (e - s) / wu_epochs
                    else:
                        train_args['inputs'][key] = e
                        
            if epoch % self.output_freq == 0:
                # evaluation on validation dataset
                if eval_valid_batch:
                    valid_outputs = []
                    if self.train_mode == 'USL':
                        for batch in self.iterate_minibatches_u(validation_dataset, batch_size, shuffle=False):
                            valid_output = f_validate(batch, *validation_args['inputs'].values())
                            valid_outputs.append(valid_output)
                    elif self.train_mode == 'SSL' or 'SL':
                        for batch in self.iterate_minibatches_l(validation_dataset, batch_size, shuffle=False):
                            valid_output = f_validate(batch[0], batch[1], *validation_args['inputs'].values())
                            valid_outputs.append(valid_output)
                    else:
                        raise NotImplementedError
                    
                    self.eval_validation[epoch] = np.mean(np.array(valid_outputs), axis=0)
                else:
                    if self.train_mode == 'USL':
                        f_validate(validation_dataset, *validation_args['inputs'].values())
                    elif self.train_mode == 'SSL' or 'SL':
                        self.eval_validation[epoch] = f_validate(validation_dataset[0], validation_dataset[1], *validation_args['inputs'].values())
                    else:
                        raise NotImplementedError
                    
                # evaluation on test dataset
                if test_dataset is not None:
                    if eval_valid_batch:
                        test_outputs = []
                        if self.train_mode == 'USL':
                            for batch in self.iterate_minibatches_u(test_dataset, batch_size, shuffle=False):
                                test_output = f_validate(batch, *validation_args['inputs'].values())
                                test_outputs.append(test_output)
                        elif self.train_mode == 'SSL' or 'SL':
                            for batch in self.iterate_minibatches_l(test_dataset, batch_size, shuffle=False):
                                test_output = f_validate(batch[0], batch[1], *validation_args['inputs'].values())
                                test_outputs.append(test_output)
                        else:
                            raise NotImplementedError
                            
                        self.eval_test[epoch] = np.mean(np.array(test_outputs), axis=0)
                    else:
                        if self.train_mode == 'USL':
                            f_validate(test_dataset, *validation_args['inputs'].values())
                        elif self.train_mode == 'SSL' or 'SL':
                            self.eval_test[epoch] = f_validate(test_dataset[0], test_dataset[1], *validation_args['inputs'].values())
                        else:
                            raise NotImplementedError
                                                
                else:
                    self.eval_test[epoch] = [0.] * len(validation_args['outputs'].keys())

                # Formatting the output string from the generic and the user-defined values.
                output_str = "epoch=%0" + str(len(str(n_epochs))) + "i; time=%0.2f;"
                output_str %= (epoch, end_time)

                def concatenate_output_str(out_str, d, test=False):
                    for k, v in zip(d.keys(), d.values()):
                        out_str += " %s=%s;" % (k.replace("valid", "test") if test else k, v)
                    return out_str

                output_str = concatenate_output_str(output_str, train_args['outputs'])
                output_str = concatenate_output_str(output_str, validation_args['outputs'])
                output_str = concatenate_output_str(output_str, validation_args['outputs'], test=True)
                
                outputs = [float(o) for o in self.eval_train[epoch]]
                outputs += [float(o) for o in self.eval_validation[epoch]]
                outputs += [float(o) for o in self.eval_test[epoch]]

                output_str %= tuple(outputs)
                self.write_to_logger(output_str)

            if self.pickle_f_custom_freq is not None and epoch % self.pickle_f_custom_freq == 0:
                if self.custom_eval_func is not None:
                    self.custom_eval_func(self.model, paths.get_custom_eval_path(epoch, self.model.root_path))
                self.plot_eval(self.eval_train, train_args['outputs'].keys(), "_train")
                self.plot_eval(self.eval_validation, validation_args['outputs'].keys(), "_validation")
                self.plot_eval(self.eval_test, validation_args['outputs'].keys(), "_test")
                self.dump_dicts()
                self.model.dump_model()
                
        if self.pickle_f_custom_freq is not None:
            self.model.dump_model()
            
    def iterate_minibatches_u(self, dataset, batchsize, shuffle=False):
        """
        This function tries to iterate unlabeled data in mini-batch
        """
        if shuffle:
            indices = np.arange(len(dataset))
            self._srng.shuffle(indices)
        for start_idx in xrange(0, len(dataset) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield dataset[excerpt]

    def iterate_minibatches_l(self, dataset, batchsize, shuffle=False):
        """
        This function tries to iterate labeled data in mini-batch.
        """
        if shuffle:
            indices = np.arange(len(dataset[0]))
            self._srng.shuffle(indices)
        for start_idx in xrange(0, len(dataset[0]) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield dataset[0][excerpt], dataset[1][excerpt]
    
    def iterate_minibatches_ssl(self, dataset_u, dataset_l, batchsize, shuffle=False):
        """
        This function tries to iterate unlabeled and labeled data in mini-batch simultaneously. 
        It is usually used for training.
        """
        if shuffle:
            indices_u = np.arange(len(dataset_u))
            self._srng.shuffle(indices_u)
        for idx_u in xrange(0, self.max_iteration_u):
            idx_l = idx_u % self.max_iteration_l
            if shuffle:
                if idx_l == 0:
                    indices_l = np.arange(len(dataset_l[0]))
                    self._srng.shuffle(indices_l)
                excerpt_u = indices_u[idx_u * batchsize: (idx_u + 1) * batchsize]
                excerpt_l = indices_l[idx_l * batchsize: (idx_l + 1) * batchsize]
            else:
                excerpt_u = slice(idx_u * batchsize, (idx_u + 1) * batchsize)
                excerpt_l = slice(idx_l * batchsize, (idx_l + 1) * batchsize)
            yield dataset_u[excerpt_u], dataset_l[0][excerpt_l], dataset_l[1][excerpt_l]
        
    def next_labeled_batch(self, dataset, batchsize_l):
        """
        This function randomly selects a minibatch of labeled dataset,
        not thoroughly iterates.
        """
        train_x_l = dataset[0]
        train_t_l = dataset[1]
        indices = self._srng.choice(size=[batchsize_l], a=train_x_l.shape[0], replace=False)
        return train_x_l[indices], train_t_l[indices]

    def bernoulli_sampling(self, batch_data):
        """
        This function can be used for sampling bernoulli input.
        """
        return self._srng.binomial(size=batch_data.shape, n=1, p=batch_data).astype(dtype=theano.config.floatX)


class TrainGaussian(TrainModel):
    """
    This class of training model assumes the each data point follows Gaussian distribution with the format: 
        (mean, var, label).
    Every time for fetch mini-batch data, sample data point ~ N(mean, var).
    """
    
        
    def __init__(self, model, output_freq=1, pickle_f_custom_freq=None,
                 f_custom_eval=None):
        super(TrainGaussian, self).__init__(model, output_freq, pickle_f_custom_freq, f_custom_eval)
        self.n_x = self.model.n_x # input dim                                                                               

    def iterate_minibatches_u(self, dataset, batchsize, shuffle=False):
        if shuffle:
            indices = np.arange(len(dataset[0]))
            self._srng.shuffle(indices)
        for idx_u in xrange(0, self.max_iteration_u):
            if shuffle:
                excerpt = indices[idx_u * batchsize: (idx_u + 1) * batchsize]
            else:
                excerpt = slice(idx_u * batchsize, (idx_u + 1) * batchsize)
            eps = self._srng.standard_normal(size=(batchsize, self.n_x))
            yield dataset[0][excerpt] + dataset[1][excerpt] * eps

    def iterate_minibatches_l(self, dataset, batchsize, shuffle=False):
        if shuffle:
            indices = np.arange(len(dataset[0]))
            self._srng.shuffle(indices)
        for idx_u in xrange(0, self.max_iteration_u):
            if shuffle:
                excerpt = indices[idx_u * batchsize: (idx_u + 1) * batchsize]
            else:
                excerpt = slice(idx_u * batchsize, (idx_u + 1) * batchsize)
            eps = self._srng.standard_normal(size=(batchsize, self.n_x)).astype(dtype=theano.config.floatX)
            yield dataset[0][excerpt] + dataset[1][excerpt] * eps, dataset[2][excerpt]

    def iterate_minibatches_ssl(self, dataset_u, dataset_l, batchsize, shuffle=False):
        # this function tries to iterate unlabeled and labeled data together, used for training usually
        if shuffle:
            indices_u = np.arange(len(dataset_u[0]))
            self._srng.shuffle(indices_u)
        for idx_u in xrange(0, self.max_iteration_u):
            idx_l = idx_u % self.max_iteration_l
            if shuffle:
                if idx_l == 0:
                    indices_l = np.arange(len(dataset_l[0]))
                    self._srng.shuffle(indices_l)
                excerpt_u = indices_u[idx_u * batchsize: (idx_u + 1) * batchsize]
                excerpt_l = indices_l[idx_l * batchsize: (idx_l + 1) * batchsize]
            else:
                excerpt_u = slice(idx_u * batchsize, (idx_u + 1) * batchsize)
                excerpt_l = slice(idx_l * batchsize, (idx_l + 1) * batchsize)
            eps_u = self._srng.standard_normal(size=(batchsize, self.n_x)).astype(dtype=theano.config.floatX)
            eps_l = self._srng.standard_normal(size=(batchsize, self.n_x)).astype(dtype=theano.config.floatX)
            yield dataset_u[0][excerpt_u] + dataset_u[1][excerpt_u] * eps_u, dataset_l[0][excerpt_l] + dataset_l[1][excerpt_l] * eps_l, dataset_l[2][excerpt_l]

    def next_labeled_batch(self, dataset_l, batchsize_l):
        # this function randomly selects a minibatch of labeled dataset.
        train_mu_l = dataset_l[0]
        train_var_l = dataset_l[1]
        train_t_l = dataset_l[2]
        indices = self._srng.choice(size=[batchsize_l], a=train_mu_l.shape[0], replace=False)
        eps_l = self._srng.standard_normal(size=(batchsize_l, self.n_x)).astype(dtype=theano.config.floatX)
        return train_mu_l[indices] + train_var_l[indices] * eps_l, train_t_l[indices]

