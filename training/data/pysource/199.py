
import importlib
import os
import yaml
import gorilla
import tempfile
import shutil

import pandas as pd

from distutils.version import LooseVersion
from kiwi import pyfunc
from kiwi.models import Model
from kiwi.models.model import MLMODEL_FILE_NAME
import kiwi.tracking
from kiwi.exceptions import MlflowException
from kiwi.models.signature import ModelSignature
from kiwi.models.utils import ModelInputExample, _save_example
from kiwi.tracking.artifact_utils import _download_artifact_from_uri
from kiwi.utils.environment import _mlflow_conda_env
from kiwi.utils.model_utils import _get_flavor_configuration
from kiwi.utils.annotations import experimental
from kiwi.utils.autologging_utils import try_mlflow_log, log_fn_args_as_params

FLAVOR_NAME = "keras"

_CUSTOM_OBJECTS_SAVE_PATH = "custom_objects.cloudpickle"
_KERAS_MODULE_SPEC_PATH = "keras_module.txt"

_MODEL_SAVE_PATH = "model.h5"

_CONDA_ENV_SUBPATH = "conda.yaml"

def get_default_conda_env(include_cloudpickle=False, keras_module=None):
    import tensorflow as tf
    conda_deps = []  
    pip_deps = []
    if keras_module is None:
        import keras
        keras_module = keras
    if keras_module.__name__ == "keras":
        if LooseVersion(keras_module.__version__) < LooseVersion('2.3.1'):
            conda_deps.append("keras=={}".format(keras_module.__version__))
        else:
            pip_deps.append("keras=={}".format(keras_module.__version__))
    if include_cloudpickle:
        import cloudpickle
        pip_deps.append("cloudpickle=={}".format(cloudpickle.__version__))
    if LooseVersion(tf.__version__) <= LooseVersion('1.13.2'):
        conda_deps.append("tensorflow=={}".format(tf.__version__))
    else:
        pip_deps.append("tensorflow=={}".format(tf.__version__))

    return _mlflow_conda_env(
        additional_conda_deps=conda_deps,
        additional_pip_deps=pip_deps,
        additional_conda_channels=None)

def save_model(keras_model, path, conda_env=None, mlflow_model=None, custom_objects=None,
               keras_module=None,
               signature: ModelSignature = None, input_example: ModelInputExample = None,
               **kwargs):
    if keras_module is None:
        def _is_plain_keras(model):
            try:
                import keras.engine.network
                return isinstance(model, keras.engine.network.Network)
            except ImportError:
                return False

        def _is_tf_keras(model):
            try:
                import tensorflow.keras.models
                return isinstance(model, tensorflow.keras.models.Model)
            except ImportError:
                return False

        if _is_plain_keras(keras_model):
            keras_module = importlib.import_module("keras")
        elif _is_tf_keras(keras_model):
            keras_module = importlib.import_module("tensorflow.keras")
        else:
            raise MlflowException("Unable to infer keras module from the model, please specify "
                                  "which keras module ('keras' or 'tensorflow.keras') is to be "
                                  "used to save and load the model.")
    elif type(keras_module) == str:
        keras_module = importlib.import_module(keras_module)

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))

    data_subpath = "data"
    data_path = os.path.join(path, data_subpath)
    os.makedirs(data_path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    if custom_objects is not None:
        _save_custom_objects(data_path, custom_objects)

    with open(os.path.join(data_path, _KERAS_MODULE_SPEC_PATH), "w") as f:
        f.write(keras_module.__name__)

    model_subpath = os.path.join(data_subpath, _MODEL_SAVE_PATH)
    model_path = os.path.join(path, model_subpath)
    if path.startswith('/dbfs/'):
        with tempfile.NamedTemporaryFile(suffix='.h5') as f:
            keras_model.save(f.name, **kwargs)
            f.flush()  
            shutil.copyfile(src=f.name, dst=model_path)
    else:
        keras_model.save(model_path, **kwargs)

    mlflow_model.add_flavor(FLAVOR_NAME,
                            keras_module=keras_module.__name__,
                            keras_version=keras_module.__version__,
                            data=data_subpath)

    if conda_env is None:
        conda_env = get_default_conda_env(include_cloudpickle=custom_objects is not None,
                                          keras_module=keras_module)
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, _CONDA_ENV_SUBPATH), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.keras",
                        data=data_subpath, env=_CONDA_ENV_SUBPATH)

    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

def log_model(keras_model, artifact_path, conda_env=None, custom_objects=None, keras_module=None,
              registered_model_name=None, signature: ModelSignature=None,
              input_example: ModelInputExample=None, **kwargs):
    Model.log(artifact_path=artifact_path, flavor=kiwi.keras,
              keras_model=keras_model, conda_env=conda_env, custom_objects=custom_objects,
              keras_module=keras_module, registered_model_name=registered_model_name,
              signature=signature, input_example=input_example,
              **kwargs)

def _save_custom_objects(path, custom_objects):
    import cloudpickle
    custom_objects_path = os.path.join(path, _CUSTOM_OBJECTS_SAVE_PATH)
    with open(custom_objects_path, "wb") as out_f:
        cloudpickle.dump(custom_objects, out_f)

def _load_model(model_path, keras_module, **kwargs):
    keras_models = importlib.import_module(keras_module.__name__ + ".models")
    custom_objects = kwargs.pop("custom_objects", {})
    custom_objects_path = None
    if os.path.isdir(model_path):
        if os.path.isfile(os.path.join(model_path, _CUSTOM_OBJECTS_SAVE_PATH)):
            custom_objects_path = os.path.join(model_path, _CUSTOM_OBJECTS_SAVE_PATH)
        model_path = os.path.join(model_path, _MODEL_SAVE_PATH)
    if custom_objects_path is not None:
        import cloudpickle
        with open(custom_objects_path, "rb") as in_f:
            pickled_custom_objects = cloudpickle.load(in_f)
            pickled_custom_objects.update(custom_objects)
            custom_objects = pickled_custom_objects
    from distutils.version import StrictVersion
    if StrictVersion(keras_module.__version__.split('-')[0]) >= StrictVersion("2.2.3"):
        import h5py
        with h5py.File(os.path.abspath(model_path), "r") as model_path:
            return keras_models.load_model(model_path, custom_objects=custom_objects, **kwargs)
    else:
        return keras_models.load_model(model_path, custom_objects=custom_objects, **kwargs)

class _KerasModelWrapper:
    def __init__(self, keras_model, graph, sess):
        self.keras_model = keras_model
        self._graph = graph
        self._sess = sess

    def predict(self, dataframe):
        if self._graph is not None:
            with self._graph.as_default():
                with self._sess.as_default():
                    predicted = pd.DataFrame(self.keras_model.predict(dataframe.values))
        else:
            predicted = pd.DataFrame(self.keras_model.predict(dataframe.values))
        predicted.index = dataframe.index
        return predicted

def _load_pyfunc(path):
    import tensorflow as tf
    if os.path.isfile(os.path.join(path, _KERAS_MODULE_SPEC_PATH)):
        with open(os.path.join(path, _KERAS_MODULE_SPEC_PATH), "r") as f:
            keras_module = importlib.import_module(f.read())
    else:
        import keras
        keras_module = keras

    K = importlib.import_module(keras_module.__name__ + ".backend")
    if keras_module.__name__ == "tensorflow.keras" or K.backend() == 'tensorflow':
        if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
            graph = tf.Graph()
            sess = tf.Session(graph=graph)
            with graph.as_default():
                with sess.as_default():  
                    K.set_learning_phase(0)
                    m = _load_model(path, keras_module=keras_module, compile=False)
                    return _KerasModelWrapper(m, graph, sess)
        else:
            K.set_learning_phase(0)
            m = _load_model(path, keras_module=keras_module, compile=False)
            return _KerasModelWrapper(m, None, None)

    else:
        raise MlflowException("Unsupported backend '%s'" % K._BACKEND)

def load_model(model_uri, **kwargs):
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    keras_module = importlib.import_module(flavor_conf.get("keras_module", "keras"))
    keras_model_artifacts_path = os.path.join(
        local_model_path,
        flavor_conf.get("data", _MODEL_SAVE_PATH))
    return _load_model(model_path=keras_model_artifacts_path, keras_module=keras_module, **kwargs)

@experimental
def autolog():
    import keras

    class __MLflowKerasCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):  
            try_mlflow_log(kiwi.log_param, 'num_layers', len(self.model.layers))
            try_mlflow_log(kiwi.log_param, 'optimizer_name', type(self.model.optimizer).__name__)
            if hasattr(self.model.optimizer, 'lr'):
                lr = self.model.optimizer.lr if \
                    type(self.model.optimizer.lr) is float \
                    else keras.backend.eval(self.model.optimizer.lr)
                try_mlflow_log(kiwi.log_param, 'learning_rate', lr)
            if hasattr(self.model.optimizer, 'epsilon'):
                epsilon = self.model.optimizer.epsilon if \
                    type(self.model.optimizer.epsilon) is float \
                    else keras.backend.eval(self.model.optimizer.epsilon)
                try_mlflow_log(kiwi.log_param, 'epsilon', epsilon)

            sum_list = []
            self.model.summary(print_fn=sum_list.append)
            summary = '\n'.join(sum_list)
            tempdir = tempfile.mkdtemp()
            try:
                summary_file = os.path.join(tempdir, "model_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write(summary)
                try_mlflow_log(kiwi.log_artifact, local_path=summary_file)
            finally:
                shutil.rmtree(tempdir)

        def on_epoch_end(self, epoch, logs=None):
            if not logs:
                return
            try_mlflow_log(kiwi.log_metrics, logs, step=epoch)

        def on_train_end(self, logs=None):
            try_mlflow_log(log_model, self.model, artifact_path='model')

        def _implements_train_batch_hooks(self): return False

        def _implements_test_batch_hooks(self): return False

        def _implements_predict_batch_hooks(self): return False

    def _early_stop_check(callbacks):
        if LooseVersion(keras.__version__) < LooseVersion('2.3.0'):
            es_callback = keras.callbacks.EarlyStopping
        else:
            es_callback = keras.callbacks.callbacks.EarlyStopping
        for callback in callbacks:
            if isinstance(callback, es_callback):
                return callback
        return None

    def _log_early_stop_callback_params(callback):
        if callback:
            try:
                earlystopping_params = {'monitor': callback.monitor,
                                        'min_delta': callback.min_delta,
                                        'patience': callback.patience,
                                        'baseline': callback.baseline,
                                        'restore_best_weights': callback.restore_best_weights}
                try_mlflow_log(kiwi.log_params, earlystopping_params)
            except Exception:  
                return

    def _get_early_stop_callback_attrs(callback):
        try:
            return callback.stopped_epoch, callback.restore_best_weights, callback.patience
        except Exception:  
            return None

    def _log_early_stop_callback_metrics(callback, history):
        if callback:
            callback_attrs = _get_early_stop_callback_attrs(callback)
            if callback_attrs is None:
                return
            stopped_epoch, restore_best_weights, patience = callback_attrs
            try_mlflow_log(kiwi.log_metric, 'stopped_epoch', stopped_epoch)
            if stopped_epoch != 0 and restore_best_weights:
                restored_epoch = stopped_epoch - max(1, patience)
                try_mlflow_log(kiwi.log_metric, 'restored_epoch', restored_epoch)
                restored_metrics = {key: history.history[key][restored_epoch]
                                    for key in history.history.keys()}
                metric_key = next(iter(history.history), None)
                if metric_key is not None:
                    last_epoch = len(history.history[metric_key])
                    try_mlflow_log(kiwi.log_metrics, restored_metrics, step=last_epoch)

    def _run_and_log_function(self, original, args, kwargs, unlogged_params, callback_arg_index):
        if not kiwi.active_run():
            try_mlflow_log(kiwi.start_run)
            auto_end_run = True
        else:
            auto_end_run = False

        log_fn_args_as_params(original, args, kwargs, unlogged_params)
        early_stop_callback = None

        if len(args) > callback_arg_index:
            tmp_list = list(args)
            early_stop_callback = _early_stop_check(tmp_list[callback_arg_index])
            tmp_list[callback_arg_index] += [__MLflowKerasCallback()]
            args = tuple(tmp_list)
        elif 'callbacks' in kwargs:
            early_stop_callback = _early_stop_check(kwargs['callbacks'])
            kwargs['callbacks'] += [__MLflowKerasCallback()]
        else:
            kwargs['callbacks'] = [__MLflowKerasCallback()]

        _log_early_stop_callback_params(early_stop_callback)

        history = original(self, *args, **kwargs)

        _log_early_stop_callback_metrics(early_stop_callback, history)

        if auto_end_run:
            try_mlflow_log(kiwi.end_run)

        return history

    @gorilla.patch(keras.Model)
    def fit(self, *args, **kwargs):
        original = gorilla.get_original_attribute(keras.Model, 'fit')
        unlogged_params = ['self', 'x', 'y', 'callbacks', 'validation_data', 'verbose']
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 5)

    @gorilla.patch(keras.Model)
    def fit_generator(self, *args, **kwargs):
        original = gorilla.get_original_attribute(keras.Model, 'fit_generator')
        unlogged_params = ['self', 'generator', 'callbacks', 'validation_data', 'verbose']
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 4)

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    gorilla.apply(gorilla.Patch(keras.Model, 'fit', fit, settings=settings))
    gorilla.apply(gorilla.Patch(keras.Model, 'fit_generator', fit_generator, settings=settings))
EOF
