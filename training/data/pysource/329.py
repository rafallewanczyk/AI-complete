import copy
import importlib
import json
import logging
import pickle
import re
import sys
import warnings
import zlib
from collections import OrderedDict  
from distutils.version import LooseVersion
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import keras
import numpy as np
import pandas as pd
import scipy.sparse

import openml
from openml.exceptions import PyOpenMLError
from openml.extensions import Extension, register_extension
from openml.flows import OpenMLFlow
from openml.runs.trace import OpenMLRunTrace, OpenMLTraceIteration
from openml.tasks import (
    OpenMLTask,
    OpenMLSupervisedTask,
    OpenMLClassificationTask,
    OpenMLRegressionTask,
)

if sys.version_info >= (3, 5):
    from json.decoder import JSONDecodeError
else:
    JSONDecodeError = ValueError

DEPENDENCIES_PATTERN = re.compile(
    r'^(?P<name>[\w\-]+)((?P<operation>==|>=|>)'
    r'(?P<version>(\d+\.)?(\d+\.)?(\d+)?(dev)?[0-9]*))?$'
)

SIMPLE_NUMPY_TYPES = [nptype for type_cat, nptypes in np.sctypes.items()
                      for nptype in nptypes if type_cat != 'others']
SIMPLE_TYPES = tuple([bool, int, float, str] + SIMPLE_NUMPY_TYPES)

LAYER_PATTERN = re.compile(r'layer\d+\_(.*)')

class KerasExtension(Extension):

    @classmethod
    def can_handle_flow(cls, flow: 'OpenMLFlow') -> bool:
        return cls._is_keras_flow(flow)

    @classmethod
    def can_handle_model(cls, model: Any) -> bool:
        return isinstance(model, keras.models.Model)

    def flow_to_model(self, flow: 'OpenMLFlow', initialize_with_defaults: bool = False) -> Any:
        return self._deserialize_keras(flow, initialize_with_defaults=initialize_with_defaults)

    def _deserialize_keras(
            self,
            o: Any,
            components: Optional[Dict] = None,
            initialize_with_defaults: bool = False,
            recursion_depth: int = 0,
    ) -> Any:

        logging.info('-%s flow_to_keras START o=%s, components=%s, '
                     'init_defaults=%s' % ('-' * recursion_depth, o, components,
                                           initialize_with_defaults))
        depth_pp = recursion_depth + 1  

        if isinstance(o, str):
            try:
                o = json.loads(o)
            except JSONDecodeError:
                pass

        rval = None  
        if isinstance(o, dict):
            rval = dict(
                (
                    self._deserialize_keras(
                        o=key,
                        components=components,
                        initialize_with_defaults=initialize_with_defaults,
                        recursion_depth=depth_pp,
                    ),
                    self._deserialize_keras(
                        o=value,
                        components=components,
                        initialize_with_defaults=initialize_with_defaults,
                        recursion_depth=depth_pp,
                    )
                )
                for key, value in sorted(o.items())
            )
        elif isinstance(o, (list, tuple)):
            rval = [
                self._deserialize_keras(
                    o=element,
                    components=components,
                    initialize_with_defaults=initialize_with_defaults,
                    recursion_depth=depth_pp,
                )
                for element in o
            ]
            if isinstance(o, tuple):
                rval = tuple(rval)
        elif isinstance(o, (bool, int, float, str)) or o is None:
            rval = o
        elif isinstance(o, OpenMLFlow):
            if not self._is_keras_flow(o):
                raise ValueError('Only Keras flows can be reinstantiated')
            rval = self._deserialize_model(
                flow=o,
                keep_defaults=initialize_with_defaults,
                recursion_depth=recursion_depth,
            )
        else:
            raise TypeError(o)
        logging.info('-%s flow_to_keras END   o=%s, rval=%s'
                     % ('-' * recursion_depth, o, rval))
        return rval

    def model_to_flow(self, model: Any) -> 'OpenMLFlow':
        return self._serialize_keras(model)

    def _serialize_keras(self, o: Any, parent_model: Optional[Any] = None) -> Any:
        rval = None  

        if self.is_estimator(o):
            rval = self._serialize_model(o)
        elif isinstance(o, (list, tuple)):
            rval = [self._serialize_keras(element, parent_model) for element in o]
            if isinstance(o, tuple):
                rval = tuple(rval)
        elif isinstance(o, SIMPLE_TYPES) or o is None:
            if isinstance(o, tuple(SIMPLE_NUMPY_TYPES)):
                o = o.item()
            rval = o
        elif isinstance(o, dict):
            if not isinstance(o, OrderedDict):
                o = OrderedDict([(key, value) for key, value in sorted(o.items())])

            rval = OrderedDict()
            for key, value in o.items():
                if not isinstance(key, str):
                    raise TypeError('Can only use string as keys, you passed '
                                    'type %s for value %s.' %
                                    (type(key), str(key)))
                key = self._serialize_keras(key, parent_model)
                value = self._serialize_keras(value, parent_model)
                rval[key] = value
            rval = rval
        else:
            raise TypeError(o, type(o))

        return rval

    def get_version_information(self) -> List[str]:

        import keras
        import scipy
        import numpy

        major, minor, micro, _, _ = sys.version_info
        python_version = 'Python_{}.'.format(
            ".".join([str(major), str(minor), str(micro)]))
        keras_version = 'Keras_{}.'.format(keras.__version__)
        numpy_version = 'NumPy_{}.'.format(numpy.__version__)
        scipy_version = 'SciPy_{}.'.format(scipy.__version__)

        return [python_version, keras_version, numpy_version, scipy_version]

    def create_setup_string(self, model: Any) -> str:
        run_environment = " ".join(self.get_version_information())
        return run_environment + " " + str(model)

    @classmethod
    def _is_keras_flow(cls, flow: OpenMLFlow) -> bool:
        return (flow.external_version.startswith('keras==')
                or ',keras==' in flow.external_version)

    def _serialize_model(self, model: Any) -> OpenMLFlow:

        parameters, parameters_meta_info, subcomponents, subcomponents_explicit = \
            self._extract_information_from_model(model)

        class_name = model.__module__ + "." + model.__class__.__name__
        class_name += '.' + format(
            zlib.crc32(json.dumps(parameters, sort_keys=True).encode('utf8')),
            'x'
        )

        external_version = self._get_external_version_string(model, subcomponents)
        name = class_name

        dependencies = '\n'.join([
            self._format_external_version(
                'keras',
                keras.__version__,
            ),
            'numpy>=1.6.1',
            'scipy>=0.9',
        ])

        keras_version = self._format_external_version('keras', keras.__version__)
        keras_version_formatted = keras_version.replace('==', '_')
        flow = OpenMLFlow(name=name,
                          class_name=class_name,
                          description='Automatically created keras flow.',
                          model=model,
                          components=subcomponents,
                          parameters=parameters,
                          parameters_meta_info=parameters_meta_info,
                          external_version=external_version,
                          tags=['openml-python', 'keras',
                                'python', keras_version_formatted,

                                ],
                          language='English',
                          dependencies=dependencies)

        return flow

    def _get_external_version_string(
            self,
            model: Any,
            sub_components: Dict[str, OpenMLFlow],
    ) -> str:
        model_package_name = model.__module__.split('.')[0]
        module = importlib.import_module(model_package_name)
        model_package_version_number = module.__version__  
        external_version = self._format_external_version(
            model_package_name, model_package_version_number,
        )
        openml_version = self._format_external_version('openml', openml.__version__)
        external_versions = set()
        external_versions.add(external_version)
        external_versions.add(openml_version)
        for visitee in sub_components.values():
            for external_version in visitee.external_version.split(','):
                external_versions.add(external_version)

        return ','.join(list(sorted(external_versions)))

    def _from_parameters(self, parameters: 'OrderedDict[str, Any]') -> Any:

        config = {}

        for k, v in parameters.items():
            if not LAYER_PATTERN.match(k):
                config[k] = self._deserialize_keras(v)

        config['config']['layers'] = []
        for k, v in parameters.items():
            if LAYER_PATTERN.match(k):
                v = self._deserialize_keras(v)
                config['config']['layers'].append(v)

        model = keras.layers.deserialize(config)

        if 'optimizer' in parameters:
            training_config = self._deserialize_keras(parameters['optimizer'])
            optimizer_config = training_config['optimizer_config']
            optimizer = keras.optimizers.deserialize(optimizer_config)

            loss = training_config['loss']
            metrics = training_config['metrics']
            sample_weight_mode = training_config['sample_weight_mode']
            loss_weights = training_config['loss_weights']

            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics,
                          loss_weights=loss_weights,
                          sample_weight_mode=sample_weight_mode)
        else:
            warnings.warn('No training configuration found inside the flow: '
                          'the model was *not* compiled. '
                          'Compile it manually.')

        return model

    def _get_parameters(self, model: Any) -> 'OrderedDict[str, Optional[str]]':
        parameters = OrderedDict()  

        model_config = {
            'class_name': model.__class__.__name__,
            'config': model.get_config(),
            'keras_version': keras.__version__,
            'backend': keras.backend.backend()
        }

        layers = model_config['config']['layers']
        del model_config['config']['layers']

        for k, v in model_config.items():
            parameters[k] = self._serialize_keras(v, model)

        max_len = int(np.ceil(np.log10(len(layers))))
        len_format = '{0:0>' + str(max_len) + '}'

        for i, v in enumerate(layers):
            layer = v['config']
            k = 'layer' + len_format.format(i) + "_" + layer['name']
            parameters[k] = self._serialize_keras(v, model)

        if model.optimizer:
            parameters['optimizer'] = self._serialize_keras({
                'optimizer_config': {
                    'class_name': model.optimizer.__class__.__name__,
                    'config': model.optimizer.get_config()
                },
                'loss': model.loss,
                'metrics': model.metrics,
                'weighted_metrics': model.weighted_metrics,
                'sample_weight_mode': model.sample_weight_mode,
                'loss_weights': model.loss_weights,
            }, model)

        return parameters

    def _extract_information_from_model(
            self,
            model: Any,
    ) -> Tuple[
        'OrderedDict[str, Optional[str]]',
        'OrderedDict[str, Optional[Dict]]',
        'OrderedDict[str, OpenMLFlow]',
        Set,
    ]:
        sub_components = OrderedDict()  
        sub_components_explicit = set()  
        parameters = OrderedDict()  
        parameters_meta_info = OrderedDict()  

        model_parameters = self._get_parameters(model)
        for k, v in sorted(model_parameters.items(), key=lambda t: t[0]):
            rval = self._serialize_keras(v, model)
            rval = json.dumps(rval)

            parameters[k] = rval
            parameters_meta_info[k] = OrderedDict((('description', None), ('data_type', None)))

        return parameters, parameters_meta_info, sub_components, sub_components_explicit

    def _deserialize_model(
            self,
            flow: OpenMLFlow,
            keep_defaults: bool,
            recursion_depth: int,
    ) -> Any:
        logging.info('-%s deserialize %s' % ('-' * recursion_depth, flow.name))
        self._check_dependencies(flow.dependencies)

        parameters = flow.parameters
        components = flow.components
        parameter_dict = OrderedDict()  

        components_ = copy.copy(components)

        for name in parameters:
            value = parameters.get(name)
            logging.info('--%s flow_parameter=%s, value=%s' %
                         ('-' * recursion_depth, name, value))
            rval = self._deserialize_keras(
                value,
                components=components_,
                initialize_with_defaults=keep_defaults,
                recursion_depth=recursion_depth + 1,
            )
            parameter_dict[name] = rval

        for name in components:
            if name in parameter_dict:
                continue
            if name not in components_:
                continue
            value = components[name]
            logging.info('--%s flow_component=%s, value=%s'
                         % ('-' * recursion_depth, name, value))
            rval = self._deserialize_keras(
                value,
                recursion_depth=recursion_depth + 1,
            )
            parameter_dict[name] = rval

        return self._from_parameters(parameter_dict)

    def _check_dependencies(self, dependencies: str) -> None:
        if not dependencies:
            return

        dependencies_list = dependencies.split('\n')
        for dependency_string in dependencies_list:
            match = DEPENDENCIES_PATTERN.match(dependency_string)
            if not match:
                raise ValueError('Cannot parse dependency %s' % dependency_string)

            dependency_name = match.group('name')
            operation = match.group('operation')
            version = match.group('version')

            module = importlib.import_module(dependency_name)
            required_version = LooseVersion(version)
            installed_version = LooseVersion(module.__version__)  

            if operation == '==':
                check = required_version == installed_version
            elif operation == '>':
                check = installed_version > required_version
            elif operation == '>=':
                check = (installed_version > required_version
                         or installed_version == required_version)
            else:
                raise NotImplementedError(
                    'operation \'%s\' is not supported' % operation)
            if not check:
                raise ValueError('Trying to deserialize a model with dependency '
                                 '%s not satisfied.' % dependency_string)

    def _format_external_version(
            self,
            model_package_name: str,
            model_package_version_number: str,
    ) -> str:
        return '%s==%s' % (model_package_name, model_package_version_number)

    def is_estimator(self, model: Any) -> bool:
        return isinstance(model, keras.models.Model)

    def seed_model(self, model: Any, seed: Optional[int] = None) -> Any:

        return model

    def _run_model_on_fold(
            self,
            model: Any,
            task: 'OpenMLTask',
            X_train: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame],
            rep_no: int,
            fold_no: int,
            y_train: Optional[np.ndarray] = None,
            X_test: Optional[Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        'OrderedDict[str, float]',
        Optional[OpenMLRunTrace],
        Optional[Any]
    ]:

        def _prediction_to_probabilities(y: np.ndarray, classes: List[Any]) -> np.ndarray:
            if not isinstance(classes, list):
                raise ValueError('please convert model classes to list prior to '
                                 'calling this fn')
            result = np.zeros((len(y), len(classes)), dtype=np.float32)
            for obs, prediction_idx in enumerate(y):
                result[obs][prediction_idx] = 1.0
            return result

        if isinstance(task, OpenMLSupervisedTask):
            if y_train is None:
                raise TypeError('argument y_train must not be of type None')
            if X_test is None:
                raise TypeError('argument X_test must not be of type None')

        model_copy = pickle.loads(pickle.dumps(model))

        user_defined_measures = OrderedDict()  

        try:
            if isinstance(task, OpenMLSupervisedTask):
                model_copy.fit(X_train, y_train)
        except AttributeError as e:
            raise PyOpenMLError(str(e))

        if isinstance(task, OpenMLClassificationTask):
            model_classes = keras.backend.argmax(y_train, axis=-1)

        if isinstance(task, OpenMLSupervisedTask):
            pred_y = model_copy.predict(X_test)
            if isinstance(task, OpenMLClassificationTask):
                pred_y = keras.backend.argmax(pred_y)
            elif isinstance(task, OpenMLRegressionTask):
                pred_y = keras.backend.reshape(pred_y, (-1,))
            pred_y = keras.backend.eval(pred_y)
        else:
            raise ValueError(task)

        if isinstance(task, OpenMLClassificationTask):

            try:
                proba_y = model_copy.predict(X_test)
            except AttributeError:
                if task.class_labels is not None:
                    proba_y = _prediction_to_probabilities(pred_y, list(task.class_labels))
                else:
                    raise ValueError('The task has no class labels')
            if task.class_labels is not None:
                if proba_y.shape[1] != len(task.class_labels):
                    proba_y_new = np.zeros((proba_y.shape[0], len(task.class_labels)))
                    for idx, model_class in enumerate(model_classes):
                        proba_y_new[:, model_class] = proba_y[:, idx]
                    proba_y = proba_y_new

                if proba_y.shape[1] != len(task.class_labels):
                    message = "Estimator only predicted for {}/{} classes!".format(
                        proba_y.shape[1], len(task.class_labels),
                    )
                    warnings.warn(message)
                    openml.config.logger.warn(message)

        elif isinstance(task, OpenMLRegressionTask):
            proba_y = None
        else:
            raise TypeError(type(task))

        return pred_y, proba_y, user_defined_measures, None, None

    def compile_additional_information(
            self,
            task: 'OpenMLTask',
            additional_information: List[Tuple[int, int, Any]]
    ) -> Dict[str, Tuple[str, str]]:
        return dict()

    def obtain_parameter_values(
            self,
            flow: 'OpenMLFlow',
            model: Any = None,
    ) -> List[Dict[str, Any]]:
        openml.flows.functions._check_flow_for_server_id(flow)

        def get_flow_dict(_flow):
            flow_map = {_flow.name: _flow.flow_id}
            for subflow in _flow.components:
                flow_map.update(get_flow_dict(_flow.components[subflow]))
            return flow_map

        def extract_parameters(_flow, _flow_dict, component_model,
                               _main_call=False, main_id=None):
            exp_parameters = set(_flow.parameters)
            exp_components = set(_flow.components)

            _model_parameters = self._get_parameters(component_model)

            model_parameters = set(_model_parameters.keys())
            if len((exp_parameters | exp_components) ^ model_parameters) != 0:
                flow_params = sorted(exp_parameters | exp_components)
                model_params = sorted(model_parameters)
                raise ValueError('Parameters of the model do not match the '
                                 'parameters expected by the '
                                 'flow:\nexpected flow parameters: '
                                 '%s\nmodel parameters: %s' % (flow_params,
                                                               model_params))

            _params = []
            for _param_name in _flow.parameters:
                _current = OrderedDict()
                _current['oml:name'] = _param_name

                current_param_values = self.model_to_flow(_model_parameters[_param_name])

                if isinstance(current_param_values, openml.flows.OpenMLFlow):
                    continue

                parsed_values = json.dumps(current_param_values)

                _current['oml:value'] = parsed_values
                if _main_call:
                    _current['oml:component'] = main_id
                else:
                    _current['oml:component'] = _flow_dict[_flow.name]
                _params.append(_current)

            for _identifier in _flow.components:
                subcomponent_model = self._get_parameters(component_model)[_identifier]
                _params.extend(extract_parameters(_flow.components[_identifier],
                                                  _flow_dict, subcomponent_model))
            return _params

        flow_dict = get_flow_dict(flow)
        model = model if model is not None else flow.model
        parameters = extract_parameters(flow, flow_dict, model, True, flow.flow_id)

        return parameters

    def _openml_param_name_to_keras(
            self,
            openml_parameter: openml.setups.OpenMLParameter,
            flow: OpenMLFlow,
    ) -> str:
        if not isinstance(openml_parameter, openml.setups.OpenMLParameter):
            raise ValueError('openml_parameter should be an instance of OpenMLParameter')
        if not isinstance(flow, OpenMLFlow):
            raise ValueError('flow should be an instance of OpenMLFlow')

        flow_structure = flow.get_structure('name')
        if openml_parameter.flow_name not in flow_structure:
            raise ValueError('Obtained OpenMLParameter and OpenMLFlow do not correspond. ')
        name = openml_parameter.flow_name  
        return '__'.join(flow_structure[name] + [openml_parameter.parameter_name])

    def instantiate_model_from_hpo_class(
            self,
            model: Any,
            trace_iteration: OpenMLTraceIteration,
    ) -> Any:

        return model

register_extension(KerasExtension)
EOF
