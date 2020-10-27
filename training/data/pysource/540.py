

import numpy as np

import threading

import time

from distkeras.parameter_servers import ADAGParameterServer
from distkeras.parameter_servers import DeltaParameterServer
from distkeras.parameter_servers import DynSGDParameterServer
from distkeras.parameter_servers import ExperimentalParameterServer

from distkeras.utils import deserialize_keras_model
from distkeras.utils import history_executor
from distkeras.utils import history_executors_average
from distkeras.utils import pickle_object
from distkeras.utils import serialize_keras_model
from distkeras.utils import set_keras_base_directory
from distkeras.utils import unpickle_object

from distkeras.networking import determine_host_address

from distkeras.workers import ADAGWorker
from distkeras.workers import AEASGDWorker
from distkeras.workers import DOWNPOURWorker
from distkeras.workers import DynSGDWorker
from distkeras.workers import ExperimentalWorker
from distkeras.workers import EAMSGDWorker
from distkeras.workers import SequentialWorker

from keras import backend as K

class Trainer(object):

    def __init__(self, keras_model, loss, worker_optimizer, metrics=["accuracy"], loss_weights=None):
        set_keras_base_directory()
        self.master_model = serialize_keras_model(keras_model)
        self.loss = loss
        self.loss_weights = loss_weights
        self.worker_optimizer = worker_optimizer
        self.metrics = metrics
        self.history = []
        self.training_time_start = 0
        self.training_time_end = 0
        self.training_time = 0
        self.max_mini_batches_prefetch = 100

    def set_max_prefetch(self, max_mini_batches):
        self.max_mini_batches_prefetch = max_mini_batches

    def set_model(self, model):
        self.master_model = serialize_keras_model(model)

    def record_training_start(self):
        self.training_time = 0
        self.training_time_start = time.time()

    def record_training_end(self):
        self.training_time_end = time.time()
        self.training_time = self.training_time_end - self.training_time_start

    def get_training_time(self):
        return self.training_time

    def get_history(self):
        return self.history

    def get_averaged_history(self):
        return history_executors_average(self.history)

    def get_executor_history(self, executor_id):
        return history_executor(self.history, executor_id)

    def train(self, dataframe, shuffle=False):
        raise NotImplementedError

    def serialize(self):
        return pickle_object(self)

class SingleTrainer(Trainer):

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], features_col="features",
                 label_col="label", num_epoch=1, batch_size=32, loss_weights=None):
        super(SingleTrainer, self).__init__(keras_model, loss, worker_optimizer, metrics, loss_weights)
        self.features_column = features_col
        self.label_column = label_col
        self.num_epoch = num_epoch
        self.batch_size = batch_size

    def allocate_worker(self):
        worker = SequentialWorker(model=self.master_model, features_col=self.features_column,
                                  label_col=self.label_column, batch_size=self.batch_size, num_epoch = self.num_epoch,
                                  optimizer=self.worker_optimizer, loss=self.loss, loss_weights=self.loss_weights, 
                                  metrics = self.metrics)

        return worker

    def train(self, dataframe, shuffle=False):
        if shuffle:
            dataframe = shuffle(dataframe)
        dataframe = dataframe.coalesce(1)
        dataframe.cache()
        worker = self.allocate_worker()
        worker.set_max_prefetch(self.max_mini_batches_prefetch)
        self.record_training_start()
        self.master_model = dataframe.rdd.mapPartitionsWithIndex(worker.train).collect()[0]
        self.record_training_end()

        return deserialize_keras_model(self.master_model)

class AveragingTrainer(Trainer):

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], features_col="features",
                 label_col="label", num_epoch=1, batch_size=32, num_workers=2, loss_weights=None):
        super(AveragingTrainer, self).__init__(keras_model, loss, worker_optimizer, metrics, loss_weights)
        self.features_column = features_col
        self.label_column = label_col
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.parameter_buffer = np.asarray(keras_model.get_weights())
        self.parameter_buffer.fill(0.0)

    def average_models(self, models):
        num_models = len(models)
        for i in range(0, num_models):
            weights = np.asarray(deserialize_keras_model(models[i]).get_weights())
            self.parameter_buffer += weights
        self.parameter_buffer /= num_models
        temp_model = deserialize_keras_model(self.master_model)
        temp_model.set_weights(self.parameter_buffer)
        self.master_model = serialize_keras_model(temp_model)

    def allocate_worker(self):
        worker = SequentialWorker(model=self.master_model, features_col=self.features_column,
                                  label_col=self.label_column, batch_size=self.batch_size, num_epoch = 1,
                                  optimizer=self.worker_optimizer, loss=self.loss, loss_weights=self.loss_weights, metrics = self.metrics)

        return worker

    def train(self, dataframe, shuffle=False):
        num_partitions = dataframe.rdd.getNumPartitions()
        if shuffle:
            dataframe = shuffle(dataframe)
        if num_partitions >= self.num_workers:
            dataframe = dataframe.coalesce(self.num_workers)
        else:
            dataframe = dataframe.repartition(self.num_workers)
        self.record_training_start()
        for i in range(0, self.num_epoch):
            worker = self.allocate_worker()
            worker.set_max_prefetch(self.max_mini_batches_prefetch)
            models = dataframe.rdd.mapPartitionsWithIndex(worker.train).collect()
            self.average_models(models)
        self.record_training_end()

        return deserialize_keras_model(self.master_model)

class EnsembleTrainer(Trainer):

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], features_col="features",
                 label_col="label", batch_size=32, num_ensembles=2, loss_weights=None):
        super(EnsembleTrainer, self).__init__(keras_model, loss, worker_optimizer, metrics, loss_weights)
        self.features_column = features_col
        self.label_column = label_col
        self.batch_size = batch_size
        self.num_ensembles = num_ensembles

    def allocate_worker(self):
        worker = SequentialWorker(model=self.master_model, features_col=self.features_column,
                                  label_col=self.label_column, batch_size=self.batch_size, num_epoch = self.num_epoch,
                                  optimizer=self.worker_optimizer, loss=self.loss, loss_weights=self.loss_weights, metrics=self.metrics)

        return worker

    def train(self, dataframe, shuffle=False):
        worker = self.allocate_worker()
        worker.set_max_prefetch(self.max_mini_batches_prefetch)
        num_partitions = dataframe.rdd.getNumPartitions()
        if shuffle:
            dataframe = shuffle(dataframe)
        if num_partitions >= self.num_workers:
            dataframe = dataframe.coalesce(self.num_workers)
        else:
            dataframe = dataframe.repartition(self.num_workers)
        self.record_training_start()
        models = dataframe.rdd.mapPartitionsWithIndex(worker.train).collect()
        self.record_training_end()

        return models

class DistributedTrainer(Trainer):

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, master_port=5000, loss_weights=None):
        super(DistributedTrainer, self).__init__(keras_model, loss, worker_optimizer, metrics, loss_weights)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.features_column = features_col
        self.label_column = label_col
        self.num_epoch = num_epoch
        self.parameter_server = None
        self.parameter_server_thread = None
        self.master_host = determine_host_address()
        self.master_port = master_port
        self.learning_rate = 1.0

    def set_minibatch_size(self, size):
        self.batch_size = size

    def get_minibatch_size(self):
        return self.batch_size

    def get_features_column(self):
        return self.features_column

    def get_label_column(self):
        return self.label_column

    def get_learning_rate(self):
        return self.learning_rate

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_num_epoch(self, num_epoch):
        self.num_epoch = num_epoch

    def get_num_epoch(self):
        return self.num_epoch

    def allocate_worker(self):
        raise NotImplementedError

    def set_master(self, master):
        self.master_host = master

    def determine_new_master(self):
        self.master_host = determine_host_address()

    def allocate_parameter_server(self):
        parameter_server = DeltaParameterServer(self.master_model, self.master_port)

        return parameter_server

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    def get_num_workers(self):
        return self.num_workers

    def num_updates(self):
        return self.parameter_server.num_updates()

    def service(self):
        self.parameter_server.start()
        self.parameter_server.initialize()
        self.parameter_server.run()

    def stop_service(self):
        self.parameter_server.stop()
        self.parameter_server_thread.join()
        self.parameter_server_thread = None

    def start_service(self):
        if not self.parameter_server_thread is None:
            self.stop_service()
        self.parameter_server_thread = threading.Thread(target=self.service)
        self.parameter_server_thread.start()

    def train(self, dataframe, shuffle=False):
        if self.parameter_server is not None:
            self.parameter_server.stop()
            self.parameter_server = None
        self.parameter_server = self.allocate_parameter_server()
        self.start_service()
        worker = self.allocate_worker()
        worker.set_max_prefetch(self.max_mini_batches_prefetch)
        num_partitions = dataframe.rdd.getNumPartitions()
        if shuffle:
            dataframe = shuffle(dataframe)
        if num_partitions >= self.num_workers:
            dataframe = dataframe.coalesce(self.num_workers)
        else:
            dataframe = dataframe.repartition(self.num_workers)
        dataframe.cache()
        self.record_training_start()
        self.history = dataframe.rdd.mapPartitionsWithIndex(worker.train).collect()
        self.record_training_end()
        self.stop_service()

        return self.parameter_server.get_model()

class AsynchronousDistributedTrainer(DistributedTrainer):

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, master_port=5000, loss_weights=None):
        super(AsynchronousDistributedTrainer, self).__init__(keras_model, worker_optimizer, loss, metrics, 
                                                             num_workers, batch_size, features_col,
                                                             label_col, num_epoch, master_port, loss_weights)
        self.parallelism_factor = 1

    def allocate_worker(self):
        raise NotImplementedError

    def set_parallelism_factor(self, factor):
        self.parallelism_factor = factor

    def get_parallelism_factor(self):
        return self.parallelism_factor

    def train(self, dataframe, shuffle=False):
        if self.parameter_server is not None:
            self.parameter_server.stop()
            self.parameter_server = None
        self.parameter_server = self.allocate_parameter_server()
        self.start_service()
        worker = self.allocate_worker()
        worker.set_max_prefetch(self.max_mini_batches_prefetch)
        num_partitions = dataframe.rdd.getNumPartitions()
        if shuffle:
            dataframe = shuffle(dataframe)
        parallelism = self.parallelism_factor * self.num_workers
        if num_partitions >= parallelism:
            dataframe = dataframe.coalesce(parallelism)
        else:
            dataframe = dataframe.repartition(parallelism)
        self.record_training_start()
        self.history = dataframe.rdd.mapPartitionsWithIndex(worker.train).collect()
        self.record_training_end()
        self.stop_service()

        return self.parameter_server.get_model()

class AEASGD(AsynchronousDistributedTrainer):

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, communication_window=32,
                 rho=5.0, learning_rate=0.1, master_port=5000, loss_weights=None):
        super(AEASGD, self).__init__(keras_model, worker_optimizer, loss, metrics, num_workers,
                                     batch_size, features_col, label_col, num_epoch, master_port, loss_weights)
        self.communication_window = communication_window
        self.rho = rho
        self.learning_rate = learning_rate

    def allocate_worker(self):
        worker = AEASGDWorker(self.master_model, self.worker_optimizer, self.loss, self.loss_weights, self.metrics,
                              self.features_column, self.label_column, self.batch_size, self.num_epoch,
                              self.master_host, self.master_port, self.rho, self.learning_rate,
                              self.communication_window)

        return worker

class DOWNPOUR(AsynchronousDistributedTrainer):

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, communication_window=5, master_port=5000, loss_weights=None):
        super(DOWNPOUR, self).__init__(keras_model, worker_optimizer, loss, metrics, num_workers,
                                       batch_size, features_col, label_col, num_epoch, master_port, loss_weights)
        self.communication_window = communication_window

    def allocate_worker(self):
        worker = DOWNPOURWorker(self.master_model, self.worker_optimizer, self.loss, self.loss_weights, self.metrics,
                                self.features_column, self.label_column, self.batch_size, self.num_epoch,
                                self.master_host, self.master_port, self.communication_window)

        return worker

class EAMSGD(AsynchronousDistributedTrainer):

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, communication_window=32,
                 rho=5.0, learning_rate=0.1, momentum=0.9, master_port=5000, loss_weights=None):
        super(EAMSGD, self).__init__(keras_model, worker_optimizer, loss, metrics, num_workers,
                                     batch_size, features_col, label_col, num_epoch, master_port, loss_weights)
        self.communication_window = communication_window
        self.rho = rho
        self.learning_rate = learning_rate
        self.momentum = momentum

    def allocate_worker(self):
        worker = EAMSGDWorker(self.master_model, self.worker_optimizer, self.loss, self.loss_weights, self.metrics,
                              self.features_column, self.label_column, self.batch_size, self.num_epoch,
                              self.master_host, self.master_port, self.rho, self.learning_rate,
                              self.momentum, self.communication_window)

        return worker

class ADAG(AsynchronousDistributedTrainer):

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, communication_window=12, master_port=5000, loss_weights=None):
        super(ADAG, self).__init__(keras_model, worker_optimizer, loss, metrics, num_workers,
                                   batch_size, features_col, label_col, num_epoch, master_port, loss_weights)
        self.communication_window = communication_window

    def allocate_worker(self):
        worker = ADAGWorker(self.master_model, self.worker_optimizer, self.loss, self.loss_weights, self.metrics,
                            self.features_column, self.label_column, self.batch_size, self.num_epoch,
                            self.master_host, self.master_port, self.communication_window)

        return worker

    def allocate_parameter_server(self):
        parameter_server = ADAGParameterServer(self.master_model, self.master_port)

        return parameter_server

class DynSGD(AsynchronousDistributedTrainer):

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, communication_window=5, master_port=5000, loss_weights=None):
        super(DynSGD, self).__init__(keras_model, worker_optimizer, loss, metrics, num_workers,
                                     batch_size, features_col, label_col, num_epoch, master_port, loss_weights)
        self.communication_window = communication_window

    def allocate_worker(self):
        worker = DynSGDWorker(self.master_model, self.worker_optimizer, self.loss, self.loss_weights, self.metrics,
                              self.features_column, self.label_column, self.batch_size, self.num_epoch,
                              self.master_host, self.master_port, self.communication_window)

        return worker

    def allocate_parameter_server(self):
        parameter_server = DynSGDParameterServer(self.master_model, self.master_port)

        return parameter_server

class Experimental(AsynchronousDistributedTrainer):

    def __init__(self, keras_model, worker_optimizer, loss, metrics=["accuracy"], num_workers=2, batch_size=32,
                 features_col="features", label_col="label", num_epoch=1, communication_window=5,
                 learning_rate=1.0, master_port=5000, loss_weights=None):
        super(Experimental, self).__init__(keras_model, worker_optimizer, loss, metrics, num_workers,
                                           batch_size, features_col, label_col, num_epoch, master_port, loss_weights)
        self.communication_window = communication_window
        self.learning_rate = learning_rate

    def allocate_worker(self):
        worker = ExperimentalWorker(self.master_model, self.worker_optimizer, self.loss, self.loss_weights, self.metrics,
                                    self.features_column, self.label_column, self.batch_size, self.num_epoch,
                                    self.master_host, self.master_port, self.communication_window,
                                    self.num_workers, self.learning_rate)

        return worker

    def allocate_parameter_server(self):
        parameter_server = ExperimentalParameterServer(self.master_model, self.master_port, self.learning_rate)

        return parameter_server
EOF
