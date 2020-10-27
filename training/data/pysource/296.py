

import numbers
import operator
from functools import reduce
import sys
import inspect
import string
import html
import copy
import sys
import re
import os

import numpy as np
import keras
import keras.backend as K
from keras.optimizers import (SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam,
                              TFOptimizer)

from .utils import *

ON_RTD = os.environ.get('READTHEDOCS', None) == 'True'

pypandoc = None
if ON_RTD:  
    try:
        import pypandoc
    except:
        pass 

def make_layer(config):
    import conx.layers
    layer = getattr(conx.layers, config["class"])
    return layer(config["name"], *config["args"], **config["params"])

class _BaseLayer():
    ACTIVATION_FUNCTIONS = ('relu', 'sigmoid', 'linear', 'softmax', 'tanh',
                            'elu', 'selu', 'softplus', 'softsign', 'hard_sigmoid')
    CLASS = None

    def __init__(self, name, *args, **params):
        self.config = {
            "class": self.__class__.__name__,
            "name": name,
            "args": args,
            "params": copy.copy(params),
        }
        if not (isinstance(name, str) and len(name) > 0):
            raise Exception('bad layer name: %s' % (name,))
        self._check_layer_name(name)
        self.name = name
        self.params = params
        self.args = args
        self.handle_merge = False
        self.network = None
        params["name"] = name
        self.shape = None
        self.vshape = None
        self.keep_aspect_ratio = False
        self.image_maxdim = None
        self.image_pixels_per_unit = None
        self.visible = True
        self.colormap = None
        self.minmax = None
        self.model = None
        self.decode_model = None
        self.input_names = []
        self.feature = 0
        self.keras_layer = None
        self.max_draw_units = 20
        self.activation = params.get("activation", None) 
        if not isinstance(self.activation, str):
            self.activation = None
        if 'vshape' in params:
            vs = params['vshape']
            del params["vshape"] 
            if not valid_vshape(vs):
                raise Exception('bad vshape: %s' % (vs,))
            else:
                self.vshape = vs

        if 'keep_aspect_ratio' in params:
            ar = params['keep_aspect_ratio']
            del params["keep_aspect_ratio"] 
            self.keep_aspect_ratio = ar

        if 'image_maxdim' in params:
            imd = params['image_maxdim']
            del params["image_maxdim"] 
            if not isinstance(imd, numbers.Integral):
                raise Exception('bad image_maxdim: %s' % (imd,))
            else:
                self.image_maxdim = imd

        if 'image_pixels_per_unit' in params:
            imd = params['image_pixels_per_unit']
            del params["image_pixels_per_image"] 
            if not isinstance(imd, numbers.Integral):
                raise Exception('bad image_pixels_per_unit: %s' % (imd,))
            else:
                self.image_pixels_per_unit = imd

        if 'visible' in params:
            visible = params['visible']
            del params["visible"] 
            self.visible = visible

        if 'colormap' in params:
            colormap = params["colormap"]
            if isinstance(colormap, (tuple, list)):
                if len(colormap) != 3:
                    raise Exception("Invalid colormap format: requires (colormap_name, vmin, vmax)")
                else:
                    self.colormap = colormap[0]
                    self.minmax = colormap[1:]
            else:
                self.colormap = colormap
            del params["colormap"] 

        if 'minmax' in params:
            self.minmax = params['minmax']
            del params["minmax"] 

        if 'dropout' in params:
            dropout = params['dropout']
            del params["dropout"] 
            if dropout == None:
                dropout = 0
                dropout_dim = 0
            elif isinstance(dropout, numbers.Real):
                dropout_dim = 0
            elif isinstance(dropout, (list, tuple)):
                dropout_dim = dropout[1]
                dropout = dropout[0]
            else:
                raise Exception('bad dropout option: %s' % (dropout,))
            if not (0 <= dropout <= 1):
                raise Exception('bad dropout rate: %s' % (dropout,))
            self.dropout = dropout
            self.dropout_dim = dropout_dim
        else:
            self.dropout = 0
            self.dropout_dim = 0

        if 'bidirectional' in params:
            bidirectional = params['bidirectional']
            del params["bidirectional"] 
            if bidirectional not in ['sum', 'mul', 'concat', 'ave', True, None]:
                raise Exception('bad bidirectional value: %s' % (bidirectional,))
            self.bidirectional = bidirectional
        else:
            self.bidirectional = None

        if 'time_distributed' in params:
            time_distributed = params['time_distributed']
            del params["time_distributed"] 
            self.time_distributed = time_distributed
        else:
            self.time_distributed = False

        if 'activation' in params: 
            self.activation = params["activation"]
            if not isinstance(self.activation, str):
                self.activation = None

        self.incoming_connections = []
        self.outgoing_connections = []

    def _check_layer_name(self, layer_name):
        valid_chars = string.ascii_letters + string.digits + "_-%"
        if len(layer_name) == 0:
            raise Exception("layer name must not be length 0: '%s'" % layer_name)
        if not all(char in valid_chars for char in layer_name):
            raise Exception("layer name must only contain letters, numbers, '-', and '_': '%s'" % layer_name)
        if layer_name.count("%") != layer_name.count("%d"):
            raise Exception("layer name must only contain '%%d'; no other formatting allowed: '%s'" % layer_name)
        if layer_name.count("%d") not in [0, 1]:
            raise Exception("layer name must contain at most one %%d: '%s'" % layer_name)

    def on_connect(self, relation, other_layer):
        pass

    def __repr__(self):
        return "<%s name='%s'>" % (self.CLASS.__name__, self.name)

    def kind(self):
        if len(self.incoming_connections) == 0 and len(self.outgoing_connections) == 0:
            return 'unconnected'
        elif len(self.incoming_connections) > 0 and len(self.outgoing_connections) > 0:
            return 'hidden'
        elif len(self.incoming_connections) > 0:
            return 'output'
        else:
            return 'input'

    def make_input_layer_k(self):
        return keras.layers.Input(self.shape, *self.args, **self.params)

    def make_input_layer_k_text(self):
        return "keras.layers.Input(%s, *%s, **%s)" % (self.shape, self.args, self.params)

    def make_keras_function(self):
        return self.CLASS(*self.args, **self.params)

    def make_keras_function_text(self):
        return "keras.layers.%s(*%s, **%s)" % (self.CLASS.__name__, self.args, self.params)

    def make_keras_functions(self):
        from keras.layers import (TimeDistributed, Bidirectional, Dropout,
                                  SpatialDropout1D, SpatialDropout2D, SpatialDropout3D)

        k = self.make_keras_function() 
        if self.bidirectional:
            if self.bidirectional is True:
                k = Bidirectional(k, name=self.name)
            else:
                k = Bidirectional(k, merge_mode=self.bidirectional, name=self.name)
        if self.time_distributed:
            k = TimeDistributed(k, name=self.name)
        k = [k]
        if self.dropout > 0:
            if self.dropout_dim == 0:
                k += [Dropout(self.dropout)]
            elif self.dropout_dim == 1:
                k += [SpatialDropout1D(self.dropout)]
            elif self.dropout_dim == 2:
                k += [SpatialDropout2D(self.dropout)]
            elif self.dropout_dim == 3:
                k += [SpatialDropout3D(self.dropout)]
        return k

    def make_keras_functions_text(self):
        def bidir_mode(name):
            if name in [True, None]:
                return "concat"
            else:
                return name
        program = self.make_keras_function_text()
        if self.time_distributed:
            program = "keras.layers.TimeDistributed(%s, name='%s')" % (program, self.name)
        if self.bidirectional:
            program = "keras.layers.Bidirectional(%s, name='%s', mode='%s')" % (
                program, self.name, bidir_mode(self.bidirectional))
        retval = [program]
        if self.dropout > 0:
            if self.dropout_dim == 0:
                retval += ["keras.layers.Dropout(self.dropout)"]
            elif self.dropout_dim == 1:
                retval += ["keras.layers.SpatialDropout1D(self.dropout)"]
            elif self.dropout_dim == 2:
                retval += ["keras.layers.SpatialDropout2D(self.dropout)"]
            elif self.dropout_dim == 3:
                retval += ["keras.layers.SpatialDropout3D(self.dropout)"]
        return "[" + (", ".join(retval)) + "]"

    def get_colormap(self):
        if self.__class__.__name__ == "FlattenLayer":
            if self.colormap is None:
                return self.incoming_connections[0].get_colormap()
            else:
                return self.colormap
        elif self.kind() == "input":
            return "gray" if self.colormap is None else self.colormap
        else:
            return get_colormap() if self.colormap is None else self.colormap

    def make_image(self, vector, colormap=None, config={}):
        import keras.backend as K
        from matplotlib import cm
        import PIL
        import PIL.ImageDraw
        if self.vshape and self.vshape != self.shape:
            vector = vector.reshape(self.vshape)
        if len(vector.shape) > 2:
            s = slice(None, None)
            args = []
            if K.image_data_format() == 'channels_last':
                for d in range(len(vector.shape)):
                    if d in [0, 1]:
                        args.append(s) 
                    else:
                        args.append(self.feature) 
            else: 
                count = 0
                for d in range(len(vector.shape)):
                    if d in [0]:
                        args.append(self.feature) 
                    else:
                        if count < 2:
                            args.append(s)
                            count += 1
            vector = vector[args]
        vector = scale_output_for_image(vector, self.get_act_minmax(), truncate=True)
        if len(vector.shape) == 1:
            vector = vector.reshape((1, vector.shape[0]))
        size = config.get("pixels_per_unit",1)
        new_width = vector.shape[0] * size 
        new_height = vector.shape[1] * size 
        if colormap is None:
            colormap = self.get_colormap()
        if colormap is not None:
            try:
                cm_hot = cm.get_cmap(colormap)
            except:
                cm_hot = cm.get_cmap("RdGy")
            vector = cm_hot(vector)
        vector = np.uint8(vector * 255)
        if max(vector.shape) <= self.max_draw_units:
            scale = int(250 / max(vector.shape))
            size = size * scale
            image = PIL.Image.new('RGBA', (new_height * scale, new_width * scale), color="white")
            draw = PIL.ImageDraw.Draw(image)
            for row in range(vector.shape[1]):
                for col in range(vector.shape[0]):
                    draw.rectangle((row * size, col * size,
                                  (row + 1) * size - 1, (col + 1) * size - 1),
                                 fill=tuple(vector[col][row]),
                                 outline='black')
        else:
            image = PIL.Image.fromarray(vector)
            image = image.resize((new_height, new_width))
        if config.get("svg_rotate", False):
            output_shape = self.get_output_shape()
            if ((isinstance(output_shape, tuple) and len(output_shape) >= 3) or
                (self.vshape is not None and len(self.vshape) == 2)):
                image = image.rotate(90, expand=1)
        return image

    def make_dummy_vector(self, default_value=0.0):
        if (self.shape is None or
            (isinstance(self.shape, (list, tuple)) and None in self.shape)):
            v = np.ones(100) * default_value
        else:
            v = np.ones(self.shape) * default_value
        lo, hi = self.get_act_minmax()
        v *= (lo + hi) / 2.0
        return v.tolist()

    def get_act_minmax(self):
        if self.minmax is not None: 
            return self.minmax
        else:
            if self.__class__.__name__ == "FlattenLayer":
                in_layer = self.incoming_connections[0]
                return in_layer.get_act_minmax()
            elif self.kind() == "input":
                if self.network and len(self.network.dataset) > 0:
                    bank_idx = self.network.input_bank_order.index(self.name)
                    return self.network.dataset._inputs_range[bank_idx]
                else:
                    return (-2,+2)
            else: 
                if self.activation in ["tanh", 'softsign']:
                    return (-1,+1)
                elif self.activation in ["sigmoid",
                                         "softmax",
                                         'hard_sigmoid']:
                    return (0,+1)
                elif self.activation in ["relu", 'elu', 'softplus']:
                    return (0,+2)
                elif self.activation in ["selu", "linear"]:
                    return (-2,+2)
                else: 
                    return (-2,+2)

    def get_output_shape(self):
        if self.keras_layer is not None:
            if hasattr(self.keras_layer, "output_shape"):
                return self.keras_layer.output_shape
            elif hasattr(self.keras_layer, "_keras_shape"):
                return self.keras_layer._keras_shape

    def tooltip(self):
        def format_range(minmax):
            minv, maxv = minmax
            if minv <= -2:
                minv = "-Infinity"
            if maxv >= +2:
                maxv = "+Infinity"
            return "(%s, %s)" % (minv, maxv)

        kind = self.kind()
        retval = "Layer: %s (%s)" % (html.escape(self.name), kind)
        retval += "\n output range: %s" % (format_range(self.get_act_minmax(),))
        if self.shape:
            retval += "\n shape = %s" % (self.shape, )
        if self.dropout:
            retval += "\n dropout = %s" % self.dropout
            if self.dropout_dim > 0:
                retval += "\n dropout dimension = %s" % self.dropout_dim
        if self.bidirectional:
            retval += "\n bidirectional = %s" % self.bidirectional
        if kind == "input":
            retval += "\n Keras class = Input"
        else:
            retval += "\n Keras class = %s" % self.CLASS.__name__
        for key in self.params:
            if key in ["name"] or self.params[key] is None:
                continue
            retval += "\n %s = %s" % (key, html.escape(str(self.params[key])))
        return retval

class Layer(_BaseLayer):
    CLASS = keras.layers.Dense
    def __init__(self, name: str, shape, **params):
        super().__init__(name, **params)
        self.config.update({
            "class": self.__class__.__name__,
            "name": name,
            "args": [shape],
            "params": copy.copy(params),
        })
        if not valid_shape(shape):
            raise Exception('bad shape: %s' % (shape,))
        if isinstance(shape, numbers.Integral) or shape is None:
            self.shape = (shape,)
            self.size = shape
        else:
            self.shape = shape
            if all([isinstance(n, numbers.Integral) for n in shape]):
                self.size = reduce(operator.mul, shape)
            else:
                self.size = None 

        if 'activation' in params:
            act = params['activation']
            if act == None:
                act = 'linear'
            if not (callable(act) or act in Layer.ACTIVATION_FUNCTIONS):
                raise Exception('unknown activation function: %s' % (act,))
            self.activation = act
            if not isinstance(self.activation, str):
                self.activation = None

    def __repr__(self):
        return "<Layer name='%s', shape=%s, act='%s'>" % (
            self.name, self.shape, self.activation)

    def print_summary(self, fp=sys.stdout):
        super().print_summary(fp)
        if self.activation:
            print("        * **Activation function**:", self.activation, file=fp)
        if self.dropout:
            print("        * **Dropout percent**    :", self.dropout, file=fp)
            if self.dropout_dim > 0:
                print("        * **Dropout dimension**    :", self.dropout_dim, file=fp)
        if self.bidirectional:
            print("        * **Bidirectional mode** :", self.bidirectional, file=fp)

    def make_keras_function(self):
        return self.CLASS(self.size, **self.params)

    def make_keras_function_text(self):
        return "keras.layers.%s(%s, **%s)" % (self.CLASS.__name__, self.size, self.params)

class ImageLayer(Layer):
    def __init__(self, name, dimensions, depth, **params):
        keep_aspect_ratio = params.get("keep_aspect_ratio", True)
        super().__init__(name, dimensions, **params)
        self.config.update({
            "class": self.__class__.__name__,
            "name": name,
            "args": [dimensions, depth],
            "params": copy.copy(params),
        })
        if self.vshape is None:
            self.vshape = self.shape
        self.keep_aspect_ratio = keep_aspect_ratio
        self.dimensions = dimensions
        self.depth = depth
        if K.image_data_format() == "channels_last":
            self.shape = tuple(list(self.shape) + [depth])
            self.image_indexes = (0, 1)
        else:
            self.shape = tuple([depth] + list(self.shape))
            self.image_indexes = (1, 2)

    def make_image(self, vector, colormap=None, config={}):
        import PIL
        v = (vector * 255).astype("uint8")
        if self.depth == 1:
            v = v.squeeze() 
        else:
            v = v.reshape(self.dimensions[0],
                          self.dimensions[1],
                          self.depth)
        image = PIL.Image.fromarray(v)
        if config.get("svg_rotate", False):
            image = image.rotate(90, expand=1)
        return image

class AddLayer(_BaseLayer):
    CLASS = keras.layers.Add
    def __init__(self, name, **params):
        self.layers = []
        super().__init__(name)
        self.config.update({
            "class": self.__class__.__name__,
            "name": name,
            "args": [],
            "params": copy.copy(params),
        })
        self.handle_merge = True

    def make_keras_functions(self):
        return [lambda k: k]

    def make_keras_function(self):
        from keras.layers import Add
        layers = [(layer.k if layer.k is not None else layer.keras_layer) for layer in self.layers]
        return Add(**self.params)(layers)

    def on_connect(self, relation, other_layer):
        if relation == "to":
            self.layers.append(other_layer)

class SubtractLayer(AddLayer):
    CLASS = keras.layers.Subtract
    def make_keras_function(self):
        from keras.layers import Substract
        layers = [(layer.k if layer.k is not None else layer.keras_layer) for layer in self.layers]
        return Subtract(**self.params)(layers)

class MultiplyLayer(AddLayer):
    CLASS = keras.layers.Multiply
    def make_keras_function(self):
        from keras.layers import Multiply
        layers = [(layer.k if layer.k is not None else layer.keras_layer) for layer in self.layers]
        return Multiply(**self.params)(layers)

class AverageLayer(AddLayer):
    CLASS = keras.layers.Average
    def make_keras_function(self):
        from keras.layers import Average
        layers = [(layer.k if layer.k is not None else layer.keras_layer) for layer in self.layers]
        return Average(**self.params)(layers)

class MaximumLayer(AddLayer):
    CLASS = keras.layers.Maximum
    def make_keras_function(self):
        from keras.layers import Maximum
        layers = [(layer.k if layer.k is not None else layer.keras_layer) for layer in self.layers]
        return Maximum(**self.params)(layers)

class ConcatenateLayer(AddLayer):
    CLASS = keras.layers.Concatenate
    def make_keras_function(self):
        from keras.layers import Concatenate
        layers = [(layer.k if layer.k is not None else layer.keras_layer) for layer in self.layers]
        return Concatenate(**self.params)(layers)

class DotLayer(AddLayer):
    CLASS = keras.layers.Dot
    def make_keras_function(self):
        from keras.layers import Dot
        layers = [(layer.k if layer.k is not None else layer.keras_layer) for layer in self.layers]
        return Dot(**self.params)(layers)

class LambdaLayer(Layer):
    CLASS = keras.layers.Lambda
    def __init__(self, name, size, function, **params):
        super().__init__(name, size, **params)
        self.config.update({
            "class": self.__class__.__name__,
            "name": name,
            "args": [size, function],
            "params": copy.copy(params),
        })

    def make_keras_function(self):
        return self.CLASS(**self.params)

    def make_keras_function_text(self):
        return "keras.layers.%s(**%s)" % (self.CLASS.__name__, self.params)

class EmbeddingLayer(Layer):
    def __init__(self, name, in_size, out_size, **params):
        super().__init__(name, in_size, **params)
        self.config.update({
            "class": self.__class__.__name__,
            "name": name,
            "args": [in_size, out_size],
            "params": copy.copy(params),
        })
        if self.vshape is None:
            self.vshape = self.shape
        self.in_size = in_size
        self.out_size = out_size
        self.sequence_size = None 

    def make_keras_function(self):
        from keras.layers import Embedding as KerasEmbedding
        return KerasEmbedding(self.in_size, self.out_size, input_length=self.sequence_size, **self.params)

    def on_connect(self, relation, other_layer):
        if relation == "to":
            self.sequence_size = other_layer.size 
            self.shape = (self.sequence_size, self.out_size)
            if self.sequence_size:
                self.size = self.sequence_size * self.out_size
            else:
                self.size = None
            self.vshape = (self.sequence_size, self.out_size)
            other_layer.size = (None,)  
            other_layer.shape = (self.sequence_size,)  
            other_layer.params["dtype"] = "int32" 
            other_layer.make_dummy_vector = lambda v=0.0: np.zeros(self.sequence_size) * v
            other_layer.minmax = (0, self.in_size)

def process_class_docstring(docstring):
    docstring = re.sub(r'\n    
                       r'\n    __\1__\n\n',
                       docstring)
    docstring = re.sub(r'    ([^\s\\\(]+):(.*)\n',
                       r'    - __\1__:\2\n',
                       docstring)
    docstring = docstring.replace('    ' * 5, '\t\t')
    docstring = docstring.replace('    ' * 3, '\t')
    docstring = docstring.replace('    ', '')
    return docstring

keras_module = sys.modules["keras.layers"]
for (name, obj) in inspect.getmembers(keras_module):
    if name in ["Embedding", "Input", "Dense", "TimeDistributed",
                "Add", "Subtract", "Multiply", "Average",
                "Maximum", "Concatenate", "Dot", "Lambda"]:
        continue
    if type(obj) == type and issubclass(obj, (keras.engine.Layer, )):
        new_name = "%sLayer" % name
        docstring = obj.__doc__
        if pypandoc:
            try:
                docstring_md  = '    **%s**\n\n' % (new_name,)
                docstring_md += obj.__doc__
                docstring = pypandoc.convert(process_class_docstring(docstring_md), "rst", "markdown_github")
            except:
                pass
        locals()[new_name] = type(new_name, (_BaseLayer,),
                                  {"CLASS": obj,
                                   "__doc__": docstring})

DenseLayer = Layer
InputLayer = Layer
AdditionLayer = AddLayer
SubtractionLayer = SubtractLayer
MultiplicationLayer = MultiplyLayer
EOF
