

import os
import json
import logging
import h5py

import six

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim 

from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.keras import _clone_and_build_model as build_keras_model

from tensorflow.python.keras import models

from tensorflow.python.training import monitored_session

from tronn.util.tf_ops import restore_variables_op
from tronn.util.tf_ops import class_weighted_loss_fn
from tronn.util.tf_ops import positives_focused_loss_fn

from tronn.util.tf_utils import print_param_count

from tronn.outlayer import H5Handler

from tronn.learn.estimator import TronnEstimator

from tronn.learn.evaluation import get_global_avg_metrics
from tronn.learn.evaluation import get_regression_metrics

from tronn.learn.learning import RestoreHook
from tronn.learn.learning import KerasRestoreHook
from tronn.learn.learning import DataSetupHook
from tronn.learn.learning import DataCleanupHook

from tronn.nets.filter_nets import produce_confidence_interval_on_outputs
from tronn.nets.nets import net_fns
from tronn.nets.normalization_nets import interpolate_logits_to_labels

from tronn.interpretation.interpret import visualize_region
from tronn.interpretation.dreaming import dream_one_sequence

from tronn.visualization import visualize_debug

from tronn.util.utils import DataKeys
from tronn.util.formats import write_to_json

from tronn.util.tf_utils import setup_tensorflow_session
from tronn.util.tf_utils import close_tensorflow_session

_TRAIN_PHASE = "train"
_EVAL_PHASE = "eval"

class ModelManager(object):
    def __init__(
            self,
            model,
            name="nn_model"):
        self.name = model["name"]
        self.model_fn = net_fns[self.name]
        self.model_params = model.get("params", {})
        self.model_dir = model["model_dir"]
        self.model_checkpoint = model.get("checkpoint")
        self.model_dataset = model.get("dataset", {}) 

    def describe(self):
        model = {
            "name": self.name,
            "params": self.model_params,
            "model_dir": self.model_dir,
            "checkpoint": self.model_checkpoint,
            "dataset": self.model_dataset
        }
        return model
    def build_training_dataflow(
            self,
            inputs,
            optimizer_fn=tf.train.RMSPropOptimizer,
            optimizer_params={
                "learning_rate": 0.002,
                "decay": 0.98,
                "momentum": 0.0},
            features_key=DataKeys.FEATURES,
            labels_key=DataKeys.LABELS,
            logits_key=DataKeys.LOGITS,
            probs_key=DataKeys.PROBABILITIES,
            regression=False,
            logit_indices=[]):
        assert inputs.get(features_key) is not None
        assert inputs.get(labels_key) is not None

        self.model_params.update({"is_training": True})
        outputs, _ = self.model_fn(inputs, self.model_params)

        if len(logit_indices) > 0:
            outputs[logits_key] = tf.gather(outputs[logits_key], logit_indices, axis=1)
        if regression:
            loss = self._add_loss(
                outputs[labels_key],
                outputs[logits_key],
                loss_fn=tf.losses.mean_squared_error)
        else:
            outputs[probs_key] = self._add_final_activation_fn(
                outputs[DataKeys.LOGITS])
            loss = self._add_loss(
                outputs[labels_key],
                outputs[logits_key])

        train_op = self._add_train_op(loss, optimizer_fn, optimizer_params)
        return outputs, loss, train_op

    def build_evaluation_dataflow(
            self,
            inputs,
            features_key=DataKeys.FEATURES,
            labels_key=DataKeys.LABELS,
            logits_key=DataKeys.LOGITS,
            probs_key=DataKeys.PROBABILITIES,
            regression=False,
            logit_indices=[]):
        assert inputs.get(features_key) is not None
        assert inputs.get(labels_key) is not None

        self.model_params.update({"is_training": False})
        outputs, _ = self.model_fn(inputs, self.model_params)
        if len(logit_indices) > 0:
            outputs[logits_key] = tf.gather(outputs[logits_key], logit_indices, axis=1)

        outputs, _ = self._quantile_norm_logits_to_labels(outputs, self.model_params)

        loss = self._add_loss(
            outputs[labels_key],
            outputs[logits_key])

        if regression:
            metrics = self._add_metrics(
                outputs[labels_key],
                outputs[logits_key],
                loss,
                metrics_fn=get_regression_metrics)
        else:
            outputs[probs_key] = self._add_final_activation_fn(
                outputs[logits_key])
            metrics = self._add_metrics(
                outputs[labels_key],
                outputs[probs_key],
                loss)

        return outputs, loss, metrics

    def build_prediction_dataflow(
            self,
            inputs,
            features_key=DataKeys.FEATURES,
            logits_key=DataKeys.LOGITS,
            probs_key=DataKeys.PROBABILITIES,
            regression=False,
            logit_indices=[]):
        assert inputs.get(features_key) is not None

        self.model_params.update({"is_training": False})
        outputs, _ = self.model_fn(inputs, self.model_params)
        if len(logit_indices) > 0:
            outputs[logits_key] = tf.gather(outputs[logits_key], logit_indices, axis=1)

        outputs, _ = self._quantile_norm_logits_to_labels(outputs, self.model_params)

        if not regression:
            outputs[probs_key] = self._add_final_activation_fn(
                outputs[logits_key])
        return outputs

    def build_inference_dataflow(
            self,
            inputs,
            inference_fn,
            inference_params,
            features_key=DataKeys.FEATURES,
            logits_key=DataKeys.LOGITS,
            probs_key=DataKeys.PROBABILITIES,
            regression=False,
            logit_indices=[]):
        assert inputs.get(features_key) is not None
        outputs = self.build_prediction_dataflow(
            inputs,
            features_key=features_key,
            logits_key=logits_key,
            probs_key=probs_key,
            regression=regression,
            logit_indices=logit_indices)

        variables_to_restore = slim.get_model_variables()
        variables_to_restore.append(tf.train.get_or_create_global_step())
        outputs, _ = inference_fn(outputs, inference_params)

        return outputs, variables_to_restore

    def _build_scaffold_with_custom_init_fn(self):

        init_op, init_feed_dict = restore_variables_op(
            self.model_checkpoint,
            skip=["pwm"])

        def init_fn(scaffold, sess):
            sess.run(init_op, init_feed_dict)
        scaffold = monitored_session.Scaffold(
            init_fn=init_fn)
        return scaffold

    def build_estimator(
            self,
            params=None,
            config=None,
            warm_start=None,
            regression=False,
            logit_indices=[],
            out_dir="."):
        if config is None:
            session_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True) 
            session_config.gpu_options.allow_growth = False 
            config = tf.estimator.RunConfig(
                save_summary_steps=30,
                save_checkpoints_secs=None,
                save_checkpoints_steps=10000000000,
                keep_checkpoint_max=None,
                session_config=session_config)

        def estimator_model_fn(
                features,
                labels,
                mode,
                params=None,
                config=config):
            inputs = features
            if mode == tf.estimator.ModeKeys.PREDICT:
                inference_mode = params.get("inference_mode", False)
                if not inference_mode:
                    outputs = self.build_prediction_dataflow(
                        inputs, regression=regression, logit_indices=logit_indices)
                    return tf.estimator.EstimatorSpec(mode, predictions=outputs)
                else:
                    outputs, variables_to_restore = self.build_inference_dataflow(
                        inputs,
                        params["inference_fn"],
                        params,
                        regression=regression,
                        logit_indices=logit_indices)

                    for key in outputs.keys():
                        if key in params["skip_outputs"]:
                            del outputs[key]
                    if self.model_checkpoint is None:
                        logging.info("WARNING: NO CHECKPOINT BEING USED")
                        return tf.estimator.EstimatorSpec(mode, predictions=outputs)
                    else:
                        scaffold = self._build_scaffold_with_custom_init_fn()
                        return tf.estimator.EstimatorSpec(
                            mode, predictions=outputs, scaffold=scaffold)
            elif mode == tf.estimator.ModeKeys.EVAL:
                outputs, loss, metrics = self.build_evaluation_dataflow(
                    inputs, regression=regression, logit_indices=logit_indices)
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
            elif mode == tf.estimator.ModeKeys.TRAIN:
                outputs, loss, train_op = self.build_training_dataflow(
                    inputs, regression=regression, logit_indices=logit_indices)
                print_param_count()
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
            else:
                raise Exception, "mode does not exist!"
            return None
        estimator = TronnEstimator(
            estimator_model_fn,
            model_dir=out_dir,
            params=params,
            config=config)

        return estimator

    def train(
            self,
            input_fn,
            out_dir,
            config=None,
            steps=None,
            hooks=[],
            regression=False,
            logit_indices=[]):
        estimator = self.build_estimator(
            config=config,
            out_dir=out_dir,
            regression=regression,
            logit_indices=logit_indices)
        estimator.train(input_fn=input_fn, max_steps=steps, hooks=hooks)
        return tf.train.latest_checkpoint(out_dir)

    def evaluate(
            self,
            input_fn,
            out_dir,
            config=None,
            steps=None,
            checkpoint=None,
            hooks=[],
            regression=False,
            logit_indices=[]):
        estimator = self.build_estimator(
            config=config,
            out_dir=out_dir,
            regression=regression,
            logit_indices=logit_indices)
        eval_metrics = estimator.evaluate(
            input_fn=input_fn,
            steps=steps,
            checkpoint_path=checkpoint,
            hooks=hooks)
        logging.info("EVAL: {}".format(eval_metrics))

        return eval_metrics

    def predict(
            self,
            input_fn,
            out_dir,
            config=None,
            predict_keys=None,
            checkpoint=None,
            hooks=[],
            yield_single_examples=True,
            logit_indices=[]):
        estimator = self.build_estimator(
            config=config,
            out_dir=out_dir,
            logit_indices=logit_indices)

        return estimator.predict(
            input_fn=input_fn,
            checkpoint_path=checkpoint,
            hooks=hooks)

    def infer(
            self,
            input_fn,
            out_dir,
            inference_params={},
            config=None,
            predict_keys=None,
            checkpoint=None,
            hooks=[],
            yield_single_examples=True,
            logit_indices=[]):
        hooks.append(DataSetupHook())

        inference_params["inference_fn"] = net_fns[
            inference_params["inference_fn_name"]]
        estimator = self.build_estimator(
            params=inference_params,
            config=config,
            out_dir=out_dir,
            logit_indices=logit_indices)

        return estimator.infer(
            input_fn=input_fn,
            checkpoint_path=checkpoint,
            hooks=hooks)

    def dream(
            self,
            dream_dataset,
            input_fn,
            feed_dict,
            out_dir,
            inference_fn,
            inference_params={},
            config=None,
            predict_keys=None,
            checkpoint=None,
            hooks=[],
            yield_single_examples=True):
        params = {
            "checkpoint": checkpoint,
            "inference_mode": True,
            "inference_fn": inference_fn,
            "inference_params": inference_params} 
        estimator = self.build_estimator(
            params=params,
            config=config,
            out_dir=out_dir)

        return estimator.dream_generator(
            dream_dataset,
            input_fn,
            feed_dict,
            predict_keys=predict_keys)

    @staticmethod
    def _setup_train_summary(train_summary):
        train_summary.update({
            "start_epoch": train_summary.get("start_epoch", 0),
            "best_metric_val": train_summary.get("best_metric_val"),
            "consecutive_bad_epochs": int(train_summary.get("consecutive_bad_epochs", 0)),
            "best_epoch": train_summary.get("best_epoch"),
            "best_checkpoint": train_summary.get("best_checkpoint"),
            "last_phase": train_summary.get("last_phase", _EVAL_PHASE)
        })
        if train_summary["best_metric_val"] is not None:
            train_summary["best_metric_val"] == float(train_summary["best_metric_val"])
        return train_summary
    def train_and_evaluate(
            self,
            train_input_fn,
            eval_input_fn,
            out_dir,
            max_epochs=30,
            early_stopping_metric="mean_auprc",
            epoch_patience=3,
            train_steps=None,
            eval_steps=1000,
            warm_start=None,
            warm_start_params={},
            regression=False,
            model_summary_file=None,
            train_summary_file=None,
            early_stopping=True,
            multi_gpu=False,
            logit_indices=[]):
        assert model_summary_file is not None
        assert train_summary_file is not None

        if os.path.isfile(train_summary_file):
            with open(train_summary_file, "r") as fp:
                train_summary = json.load(fp)
            logging.info("Starting from: {}".format(train_summary))
        else:
            train_summary = {}
        train_summary = ModelManager._setup_train_summary(train_summary)

        start_epoch = train_summary["start_epoch"]
        best_metric_val = train_summary["best_metric_val"]
        consecutive_bad_epochs = train_summary["consecutive_bad_epochs"]
        best_checkpoint = train_summary["best_checkpoint"]
        last_phase = train_summary["last_phase"]
        self.model_checkpoint = best_checkpoint
        write_to_json(self.describe(), model_summary_file)
        write_to_json(train_summary, train_summary_file)
        if early_stopping and (consecutive_bad_epochs >= epoch_patience):
            return best_checkpoint
        if multi_gpu:
            distribution = tf.contrib.distribute.MirroredStrategy()
            config = tf.estimator.RunConfig(
                save_summary_steps=30,
                save_checkpoints_secs=None,
                save_checkpoints_steps=10000000000,
                keep_checkpoint_max=None,
                train_distribute=distribution)
        else:
            config = None

        for epoch in xrange(start_epoch, max_epochs):
            logging.info("EPOCH {}".format(epoch))
            train_summary["start_epoch"] = epoch
            write_to_json(train_summary, train_summary_file)
            training_hooks = []
            if (epoch == 0) and warm_start is not None:
                logging.info("Restoring from {}".format(warm_start))
                restore_hook = RestoreHook(
                    warm_start,
                    warm_start_params)
                training_hooks.append(restore_hook)

            if last_phase == _EVAL_PHASE:
                self.train(
                    train_input_fn,
                    "{}/train".format(out_dir),
                    steps=train_steps,
                    config=config,
                    hooks=training_hooks,
                    regression=regression,
                    logit_indices=logit_indices)
                last_phase = _TRAIN_PHASE
                train_summary["last_phase"] = last_phase
                write_to_json(train_summary, train_summary_file)
            if last_phase == _TRAIN_PHASE:
                latest_checkpoint = tf.train.latest_checkpoint(
                    "{}/train".format(out_dir))
                eval_metrics = self.evaluate(
                    eval_input_fn,
                    "{}/eval".format(out_dir),
                    steps=eval_steps,
                    checkpoint=latest_checkpoint,
                    regression=regression,
                    logit_indices=logit_indices)

                last_phase = _EVAL_PHASE
                train_summary["last_phase"] = last_phase
                write_to_json(train_summary, train_summary_file)

            if best_metric_val is None:
                is_good_epoch = True
            elif early_stopping_metric in ["loss", "mse"]:
                is_good_epoch = eval_metrics[early_stopping_metric] < best_metric_val
            else:
                is_good_epoch = eval_metrics[early_stopping_metric] > best_metric_val

            if is_good_epoch:
                best_metric_val = eval_metrics[early_stopping_metric]
                consecutive_bad_epochs = 0
                best_checkpoint = latest_checkpoint
                train_summary["best_metric_val"] = best_metric_val
                train_summary["consecutive_bad_epochs"] = consecutive_bad_epochs
                train_summary["best_epoch"] = epoch
                train_summary["best_checkpoint"] = best_checkpoint
                train_summary["metrics"] = eval_metrics
                self.model_checkpoint = best_checkpoint
            else:
                consecutive_bad_epochs += 1
                train_summary["consecutive_bad_epochs"] = consecutive_bad_epochs
            write_to_json(self.describe(), model_summary_file)
            write_to_json(train_summary, train_summary_file)
            if early_stopping:
                if consecutive_bad_epochs >= epoch_patience:
                    logging.info(
                        "early stopping triggered "
                        "on epoch {} "
                        "with patience {}".format(epoch, epoch_patience))
                    break

        return best_checkpoint

    def _quantile_norm_logits_to_labels(self, inputs, model_params):
        if model_params.get("prediction_sample") is not None:
            if model_params.get("quantile_norm"):
                if "ensemble" in self.name:
                    model_params.update({"is_ensemble": True})
                inputs, _ = interpolate_logits_to_labels(inputs, model_params)
                if "ensemble" in self.name:
                    inputs, _ = produce_confidence_interval_on_outputs(
                        inputs, model_params)

        return inputs, model_params

    def _add_final_activation_fn(
            self,
            logits,
            activation_fn=tf.nn.sigmoid):
        return activation_fn(logits)

    def _add_loss(self, labels, logits, loss_fn=tf.losses.sigmoid_cross_entropy):

        if False:
            pos_weights = get_positive_weights_per_task(self.data_files[data_key])
            if self.finetune:
                pos_weights = [pos_weights[i] for i in self.finetune_tasks]
            self.loss = class_weighted_loss_fn(
                self.loss_fn, labels, logits, pos_weights)
        elif False:
            task_weights, class_weights = get_task_and_class_weights(self.data_files[data_key])
            if self.finetune:
                task_weights = [task_weights[i] for i in self.finetune_tasks]
            if self.finetune:
                class_weights = [class_weights[i] for i in self.finetune_tasks]
            self.loss = positives_focused_loss_fn(
                self.loss_fn, labels, logits, task_weights, class_weights)
        else:
            loss = loss_fn(labels, logits)

        total_loss = tf.losses.get_total_loss()

        return total_loss

    def _add_train_op(
            self,
            loss,
            optimizer_fn,
            optimizer_params):
        optimizer = optimizer_fn(**optimizer_params)
        train_op = slim.learning.create_train_op(
            loss,
            optimizer,
            variables_to_train=None, 
            summarize_gradients=True)

        return train_op
    def _add_metrics(self, labels, probs, loss, metrics_fn=get_global_avg_metrics):
        metric_map = metrics_fn(labels, probs)

        return metric_map

    def _add_summaries(self):
        return None

    @staticmethod
    def infer_and_save_to_h5(generator, h5_file, sample_size, debug=False):
        if debug:
            viz_dir = "{}/viz.debug".format(os.path.dirname(h5_file))
            os.system("mkdir -p {}".format(viz_dir))
        print "starting inference"
        first_example = generator.next()
        with h5py.File(h5_file, "w") as hf:

            h5_handler = H5Handler(
                hf,
                first_example,
                sample_size,
                resizable=True,
                batch_size=min(4096, sample_size),
                is_tensor_input=False)

            h5_handler.store_example(first_example)

            total_examples = 1
            try:
                for i in xrange(1, sample_size):
                    if total_examples % 1000 == 0:
                        logging.info("finished {}".format(total_examples))

                    example = generator.next()
                    h5_handler.store_example(example)
                    total_examples += 1
                    if debug:
                        import ipdb
                        ipdb.set_trace()
                        prefix = "{}/{}".format(viz_dir, os.path.basename(h5_file).split(".h5")[0])
                        visualize_debug(example, prefix)

            except StopIteration:
                print "Done reading data"

            finally:
                h5_handler.flush()
                h5_handler.chomp_datasets()

        return sample_size - total_examples

    @staticmethod
    def dream_and_save_to_h5(generator, h5_handle, group, sample_size=100000):
        logging.info("starting dream")
        first_example = generator.next()
        h5_handler = H5Handler(
            h5_handle,
            first_example,
            sample_size,
            group=group,
            resizable=True,
            batch_size=4096,
            is_tensor_input=False)

        h5_handler.store_example(first_example)

        total_examples = 1
        try:
            for i in xrange(1, sample_size):
                if total_examples % 1000 == 0:
                    print total_examples

                example = generator.next()
                h5_handler.store_example(example)
                total_examples += 1

        except StopIteration:
            print "Done reading data"

        finally:
            h5_handler.flush()
            h5_handler.chomp_datasets()
        return None

    def extract_model_variables(
            self,
            input_fn,
            out_dir,
            prefix,
            skip=["logit", "out"]):
        with tf.Graph().as_default():

            inputs = input_fn()[0]
            self.model_params.update({"is_training": False})
            outputs, _ = self.model_fn(inputs, self.model_params)

            all_trainable_variables = tf.trainable_variables()
            trainable_variables = []
            for v in all_trainable_variables:
                skip_var = False
                for skip_expr in skip:
                    if skip_expr in v.name:
                        skip_var = True
                if not skip_var:
                    trainable_variables.append(v)

            sess, coord, threads = setup_tensorflow_session()

            init_assign_op, init_feed_dict = restore_variables_op(
                self.model_checkpoint, skip=skip)
            sess.run(init_assign_op, init_feed_dict)
            trainable_variables_numpy = sess.run(trainable_variables)
            close_tensorflow_session(coord, threads)

        variables_dict = {}
        variables_order = []
        for v_idx in xrange(len(trainable_variables)):
            variable_name = "{}.{}".format(v_idx, trainable_variables[v_idx].name)
            variables_dict[variable_name] = trainable_variables_numpy[v_idx]
            variables_order.append(variable_name)
        variables_dict["variables_order"] = np.array(variables_order)

        np.savez(
            "{}/{}.model_variables.npz".format(out_dir, prefix),
            **variables_dict)
        return None

class EnsembleModelManager(ModelManager):
    def __init__(
            self,
            models,
            model_params={},
            num_gpus=1,
            name="ensemble_model"):
        assert len(models) != 0
        self.name = name
        self.model_fn = self._build_ensemble_model_fn(models)
        self.model_params = model_params
        self.model_params.update({
            "num_tasks": models[0].model_params["num_tasks"],
            "num_models": len(models),
            "num_gpus": num_gpus})
        self.model_dir = [model.model_dir for model in models]
        self.model_checkpoint = [model.model_checkpoint for model in models]
        self.models = models

    def _build_ensemble_model_fn(
            self,
            models,
            merge_outputs=True):
        def ensemble_model_fn(inputs, params):
            outputs = dict(inputs)
            num_models = params["num_models"]
            num_gpus = params["num_gpus"]
            all_logits = []

            for model_idx in xrange(num_models):
                new_scope = "model_{}".format(model_idx)
                logging.info("calling model {}".format(model_idx))
                model_fn = models[model_idx].model_fn
                model_params = models[model_idx].model_params
                params.update(model_params)

                with tf.variable_scope(new_scope):
                    pseudo_count = num_gpus - (num_models % num_gpus) - 1 
                    device = "/gpu:{}".format((num_models + pseudo_count - model_idx) % num_gpus)
                    print device
                    with tf.device(device):
                        model_outputs, _ = model_fn(inputs, params)
                all_logits.append(model_outputs[DataKeys.LOGITS])
            logits = tf.stack(all_logits, axis=1) 
            if merge_outputs:
                outputs[DataKeys.LOGITS_MULTIMODEL] = logits 
                logits = tf.reduce_mean(logits, axis=1)
            outputs[DataKeys.LOGITS] = logits

            return outputs, params

        return ensemble_model_fn

    def _build_scaffold_with_custom_init_fn(self):
        init_ops = []
        init_feed_dicts = {}
        for model_idx in xrange(len(self.model_checkpoint)):
            new_scope = "model_{}".format(model_idx)
            model_checkpoint = self.model_checkpoint[model_idx]
            init_op, init_feed_dict = restore_variables_op(
                model_checkpoint,
                skip=["pwm"],
                include_scope=new_scope,
                scope_change=["^{}/".format(new_scope), ""])

            init_ops.append(init_op)
            init_feed_dicts.update(init_feed_dict)
        def init_fn(scaffold, sess):
            sess.run(init_ops, init_feed_dicts)
        scaffold = monitored_session.Scaffold(
            init_fn=init_fn)
        return scaffold

class KerasModelManager(ModelManager):

    def __init__(
            self,
            model=None,
            keras_model=None,
            name="keras"):
        self.name = name
        if model is not None:
            keras_model = models.load_model(model["checkpoint"])
            model_params = model.get("params", {})
            self.model_dir = model["model_dir"]
        else:
            assert keras_model is not None, "no model loaded!"
        def keras_estimator_fn(inputs, model_fn_params):
            features = inputs[DataKeys.FEATURES]
            labels = inputs[DataKeys.LABELS]
            outputs = dict(inputs)
            if model_fn_params.get("feature_format", "NHWC") == "NWC":
                features = tf.squeeze(features, axis=1)

            if model_fn_params.get("is_training", False):
                mode = model_fn_lib.ModeKeys.TRAIN
            else:
                mode = model_fn_lib.ModeKeys.PREDICT
            model = build_keras_model(mode, keras_model, model_fn_params, features, labels)
            model.layers.pop()
            model = tf.keras.models.Model(model.input, model.layers[-1].output)
            outputs.update(dict(zip(model.output_names, model.outputs)))
            keras_logits_key = model_fn_params.get("logits_name", DataKeys.LOGITS)
            outputs[DataKeys.LOGITS] = outputs[keras_logits_key]
            for v in tf.global_variables():
                tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)

            if False:
                def init_fn(val):
                    logging.info("called init fn")
                    model.set_weights(keras_weights)
                    return 0
                init_op = tf.py_func(
                    func=init_fn,
                    inp=[0],
                    Tout=[tf.int64],
                    stateful=False,
                    name="init_keras")
                tf.get_collection("KERAS_INIT")
                tf.add_to_collection("KERAS_INIT", init_op)
            return outputs, model_fn_params

        self.model_fn = keras_estimator_fn
        self.model_params = model_params
        self.model_checkpoint = self.save_checkpoint_from_keras_model(keras_model)

    def save_checkpoint_from_keras_model(self, keras_model):
        keras_checkpoint = tf.train.latest_checkpoint(self.model_dir)
        if not keras_checkpoint:
            keras_checkpoint = "{}/keras_model.ckpt".format(self.model_dir)
            keras_weights = keras_model.get_weights()
            mode = model_fn_lib.ModeKeys.TRAIN
            with tf.Graph().as_default() as g:
                tf.train.create_global_step(g)
                model = build_keras_model(
                    mode, keras_model, self.model_params)
                model2 = build_keras_model(
                    mode, keras_model, self.model_params)
                with tf.Session() as sess:
                    model.set_weights(keras_weights)
                    model2.set_weights(keras_weights)

                    if False:
                        print [len(layer.weights) for layer in model.layers]
                        print sum([len(layer.weights) for layer in model.layers])
                        print len([layer.name for layer in model.layers])
                        print len(tf.trainable_variables())
                        print sess.run(tf.trainable_variables()[0])[:,:,0]
                    saver = tf.train.Saver()
                    saver.save(sess, keras_checkpoint)

        if False:
            with tf.Graph().as_default() as g:
                tf.train.create_global_step(g)
                features = tf.placeholder(tf.float32, shape=[64,1,1000,4])
                labels = tf.placeholder(tf.float32, shape=[64,279])
                inputs = {"features": features, "labels": labels}
                self.model_fn(inputs, self.model_params)
                with tf.Session() as sess:
                    saver = tf.train.Saver()
                    saver.restore(sess, keras_checkpoint)
                    for var in tf.trainable_variables():
                        print var.name
                    for var in tf.global_variables():
                        print var.name
                    print tf.trainable_variables()
                    print tf.global_variables()
                    print sess.run(tf.trainable_variables()[0])[:,:,0]
        return keras_checkpoint

    def infer(
            self,
            input_fn,
            out_dir,
            inference_params={},
            config=None,
            predict_keys=None,
            checkpoint=None,
            hooks=[],
            yield_single_examples=True):
        hooks.append(KerasRestoreHook())
        inference_params.update({"model_reuse": False})
        return super(KerasModelManager, self).infer(
            input_fn,
            out_dir,
            inference_params=inference_params,
            config=config,
            predict_keys=predict_keys,
            checkpoint=checkpoint,
            hooks=hooks,
            yield_single_examples=yield_single_examples)
class MetaGraphManager(ModelManager):

    def __init__():
        pass

    def train():
        pass

    def eval():
        pass

    def infer():
        pass

    def _build_metagraph_model_fn(self, metagraph):

        def metagraph_model_fn(inputs, params):
            assert params.get("sess") is not None

            saver = tf.train.import_meta_graph(metagraph)
            saver.restore(params["sess"], metagraph)

            outputs["logits"] = tf.get_collection("logits")
            return outputs, params
        return metagraph_model_fn

class PyTorchModelManager(ModelManager):

    def __init__(
            self,
            model,
            name="pytorch_model"):
        self.name = model["name"]
        self.model_params = model.get("params", {})
        self.model_checkpoint = model.get("checkpoint")
        self.model_dir = model["model_dir"]

        pytorch_model = pytorch_net_fns[model["name"]]()
        def converted_model_fn(inputs, params):
            outputs = dict(inputs)
            seq = inputs[DataKeys.FEATURES]
            seq = tf.squeeze(seq)
            max_batch_size = seq.get_shape().as_list()[0]
            if True:
                rna = np.ones((max_batch_size, 1630))
            else:
                rna = inputs["FEATURES.RNA"]

            pytorch_inputs = [seq, rna, max_batch_size]
            Tout = tf.float32
            outputs[DataKeys.LOGITS] = tf.py_func(
                pytorch_model.output,
                pytorch_inputs,
                Tout,
                name="pytorch_model_logits")
            outputs[DataKeys.LOGITS] = tf.reshape(
                outputs[DataKeys.LOGITS],
                (max_batch_size, 1))
            pytorch_inputs = [seq, rna, False, max_batch_size]
            outputs[DataKeys.IMPORTANCE_GRADIENTS] = tf.py_func(
                pytorch_model.importance_score,
                pytorch_inputs,
                Tout,
                name="pytorch_model_gradients")
            outputs[DataKeys.IMPORTANCE_GRADIENTS] = tf.reshape(
                outputs[DataKeys.IMPORTANCE_GRADIENTS],
                seq.get_shape())
            outputs[DataKeys.IMPORTANCE_GRADIENTS] = tf.expand_dims(
                outputs[DataKeys.IMPORTANCE_GRADIENTS],
                axis=1)
            return outputs, params

        self.model_fn = converted_model_fn

def setup_model_manager(args):
    if args.model_framework == "tensorflow":

        if args.model["name"] != "ensemble":
            model_manager = ModelManager(model=args.model)
        else:
            models = []
            for model_json in args.model["params"]["models"]:

                with open(model_json, "r") as fp:
                    model = json.load(fp)
                sub_model_manager = ModelManager(model=model)
                model_json_dir = os.path.dirname(model_json)
                sub_model_manager.model_checkpoint = "{}/train/{}".format(
                    model_json_dir,
                    os.path.basename(sub_model_manager.model_checkpoint))
                models.append(sub_model_manager)
            model_manager = EnsembleModelManager(
                models,
                model_params=args.model["params"],
                num_gpus=args.num_gpus)

    elif args.model_framework == "keras":
        args.model["model_dir"] = args.out_dir
        model_manager = KerasModelManager(
            model=args.model)

    elif args.model_framework == "pytorch":
        model_manager = PyTorchModelManager(model=args.model)
    else:
        raise ValueError, "unrecognized deep learning framework!"
    return model_manager
EOF
