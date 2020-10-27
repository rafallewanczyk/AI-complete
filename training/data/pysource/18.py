

from datetime import datetime as dt
import matplotlib.pyplot as plt
import sys
import os
import socket
import pickle

from imageio import save as imsave

from io import TextIOWrapper, BytesIO
import numpy as np
import itertools
import json
from time import time as tm

import pandas as pd

from collections import OrderedDict

import textwrap

import imp

import codecs

_HTML_START = "<HEAD><meta http-equiv='refresh' content='5' ></HEAD><BODY><pre>"
_HTML_END = "</pre></BODY>"

class Logger():
  def __init__(self, lib_name = "LOGR", lib_ver = "",
               config_file = "",
               base_folder = "", 
               log_suffix = "",
               DEBUG = True, 
               SHOW_TIME = True, 
               TF_KERAS = True,
               HTML = False,
               max_lines=1000):
    self.devices  = {}
    self.max_lines = max_lines
    self.split_part = 1
    self.DEBUG = DEBUG
    self.HTML = HTML
    self.timers = OrderedDict()
    self.app_log = list()
    self.results = list()
    self.printed = list()
    self.TF_KERAS = TF_KERAS
    self.KERAS_CONVERTER = True
    self.TF = False
    self.MACHINE_NAME = self.GetMachineName()
    self.__version__ = "4.1.0"
    self.SHOW_TIME = SHOW_TIME
    self.last_time = tm()
    self.file_prefix = dt.now().strftime("%Y%m%d_%H%M%S")
    self.log_suffix = log_suffix
    self.log_results_file = self.file_prefix + "_RESULTS.txt"
    self.__lib__= lib_name
    self._base_folder  = base_folder
    self.config_data = None

    self._configure_data_and_dirs(config_file)
    self._generate_log_path()
    self.log_results_file = os.path.join(self._logs_dir, self.log_results_file)
    ver = "v.{}".format(lib_ver) if lib_ver != "" else ""
    self.VerboseLog("Library [{} {}] initialized on machine [{}]".format(
                    self.__lib__, ver, self.MACHINE_NAME))
    self.VerboseLog("Logger version: {}".format(self.__version__))
    if self.TF_KERAS:
      self.CheckTF()
    return

  def _configure_data_and_dirs(self, config_file):
    if config_file != "":
      f = open(config_file)
      self.config_data = json.load(f, object_pairs_hook=OrderedDict)
      new_dict = OrderedDict()
      for key in self.config_data.keys():
        new_dict[key.upper()] = self.config_data[key]
      self.config_data = new_dict
      assert ("BASE_FOLDER" in self.config_data.keys())
      assert ("APP_FOLDER" in self.config_data.keys())
      base_folder = self.config_data["BASE_FOLDER"]
      app_folder = self.config_data["APP_FOLDER"]
      if "GOOGLE" in base_folder.upper():
        base_folder = self.GetGoogleDrive()
      self._base_folder  = os.path.join(base_folder,app_folder)
      print("Loaded config [{}]  BASE: {}".format(
          config_file,self._base_folder), flush = True)

    self._logs_dir = os.path.join(self._base_folder,"_logs")
    self._outp_dir = os.path.join(self._base_folder,"_output")
    self._data_dir = os.path.join(self._base_folder,"_data")
    self._modl_dir = os.path.join(self._base_folder,"_models")

    self._setup_folders([self._outp_dir, self._logs_dir, self._data_dir,
                         self._modl_dir])
  def _generate_log_path(self):
    part = '{:03d}'.format(self.split_part)
    lp = self.file_prefix
    ls = self.log_suffix
    if self.HTML:
      self.log_file = lp + '_' + ls + '_' + part +'_log_web.html'
    else:
      self.log_file = lp + '_' + ls + '_' + part + '_log.txt'
    self.log_file = os.path.join(self._logs_dir, self.log_file)
    path_dict = {}
    path_dict['CURRENT_LOG'] = self.log_file
    file_path = os.path.join(self._logs_dir, self.__lib__+'.txt')
    with open(file_path, 'w') as fp:
        json.dump(path_dict, fp, sort_keys=True, indent=4)         
    self._add_log("{} log changed to {}...".format(file_path, self.log_file))    
    return
  def _check_log_size(self):
    if len(self.app_log) >= self.max_lines:
      self._add_log("Ending log part {}".format(self.split_part))
      self._save_log()
      self.app_log = []
      self.split_part += 1
      self._generate_log_path()
      self._add_log("Starting log part {}".format(self.split_part))
      self._save_log()
    return
  def CheckFolder(self, sub_folder):
    sfolder = os.path.join(self.GetBaseFolder(),sub_folder)
    if sfolder not in self.folder_list:
      self.folder_list.append(sfolder)
    if not os.path.isdir(sfolder):
      self.VerboseLog(" Creating folder [...{}]".format(sfolder[-40:]))
      os.makedirs(sfolder)
    return sfolder
  def LoadDataJSON(self, fname):
    datafile = os.path.join(self._data_dir,fname)
    self.VerboseLog('Loading data json: {}'.format(datafile))
    with open(datafile) as f:
      data_json = json.load(f)      
    return data_json
  def GetBaseFolder(self):
    return self._base_folder

  def GetDataFolder(self):
    return self._data_dir

  def GetOutputFolder(self):
    return self._outp_dir
  def GetModelsFolder(self):
    return self._modl_dir
  def GetFileFromFolder(self, s_folder, s_file):
    s_fn = os.path.join(self.GetBaseFolder(),s_folder,s_file)
    if not os.path.isfile(s_fn):
      s_fn = None
    return s_fn
  def GetDataFile(self, s_file):
    fpath = os.path.join(self._data_dir, s_file)
    if not os.path.isfile(fpath):
      fpath = None
    return fpath

  def ModelExists(self, model_file):
    exists = False
    for ext in ['','.h5', '.pb']:
      fpath = os.path.join(self.GetModelsFolder(), model_file + ext)
      if os.path.isfile(fpath):        
        exists = True
    if exists:
      self.P("Detected model {}.".format(fpath))
    else:
      self.P("Model {} NOT found.".format(model_file))
    return exists

  def GetOutputFile(self, s_file):
    fpath = os.path.join(self._outp_dir, s_file)
    if not os.path.isfile(fpath):
      fpath = None
    return fpath
  def GetGoogleDrive(self):
    home_dir = os.path.expanduser("~")
    valid_paths = [
                   os.path.join(home_dir, "Google Drive"),
                   os.path.join(home_dir, "GoogleDrive"),
                   os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                   os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                   os.path.join("C:/", "GoogleDrive"),
                   os.path.join("C:/", "Google Drive"),
                   os.path.join("D:/", "GoogleDrive"),
                   os.path.join("D:/", "Google Drive"),
                   ]
    drive_path = None
    for path in valid_paths:
      if os.path.isdir(path):
        drive_path = path
        break

    if drive_path is None:
      raise Exception("Couldn't find google drive folder!")    

    return drive_path  
  def _setup_folders(self,folder_list):
    self.folder_list = folder_list
    for folder in folder_list:
      if not os.path.isdir(folder):
        print("Creating folder [{}]".format(folder))
        os.makedirs(folder)
    return

  def ShowNotPrinted(self):
    nr_log = len(self.app_log)
    for i in range(nr_log):
      if not self.printed[i]:
        print(self.app_log[i], flush = True)
        self.printed[i] = True
    return

  def SaveDataframe(self, df, fn = ''):
    file_prefix = self.file_prefix + "_"
    csvfile = os.path.join(self._outp_dir,file_prefix+fn+'.csv')
    df.to_csv(csvfile)
    return
  def SaveDataframeCurrentTime(self, df, fn = ''):
    file_prefix = dt.now().strftime("%Y%m%d_%H%M%S")
    csvfile = os.path.join(self._outp_dir,file_prefix+fn+'.csv')
    df.to_csv(csvfile)
    return

  def ShowResults(self):
    for res in self.results:
      self._logger(res, show = True, noprefix = True)
    return

  def _logger(self, logstr, show = True, results = False, noprefix = False, show_time = False):
    elapsed = tm() - self.last_time
    self._add_log(logstr, show = show, results = results, noprefix = noprefix, show_time = show_time)
    self._save_log()
    self.last_time = tm()
    self._check_log_size()
    return elapsed
  def _add_log(self, logstr, show = True, results = False, noprefix = False, show_time = False):
    elapsed = tm() - self.last_time
    nowtime = dt.now()
    prefix = ""
    strnowtime = nowtime.strftime("[{}][%Y-%m-%d %H:%M:%S] ".format(self.__lib__))
    if self.SHOW_TIME and (not noprefix):
      prefix = strnowtime
    if logstr[0]=="\n":
      logstr = logstr[1:]
      prefix = "\n"+prefix
    logstr = prefix + logstr
    if show_time:
      logstr += " [{:.2f}s]".format(elapsed)
    self.app_log.append(logstr)
    if show:
      print(logstr, flush = True)
      self.printed.append(True)
    else:
      self.printed.append(False)
    if results:
      self.results.append(logstr)
    return
  def _save_log(self):
    nowtime = dt.now()
    strnowtime = nowtime.strftime("[{}][%Y-%m-%d %H:%M:%S] ".format(self.__lib__))
    stage = 0
    try:
      log_output = codecs.open(self.log_file, "w", "utf-8")  
      stage += 1
      if self.HTML:
        log_output.write(_HTML_START)
        stage += 1
        iter_list = reversed(self.app_log)
      else:
        iter_list = self.app_log
      for log_item in iter_list:
          log_output.write("{}\n".format(log_item))
          stage += 1
      if self.HTML:
        log_output.write(_HTML_END)
        stage += 1
      log_output.close()
      stage += 1
    except:
      print(strnowtime+"LogWErr S: {} [{}]".format(stage,
            sys.exc_info()[0]), flush = True)
    return
  def SaveFigure(self, label=''):
    self.OutputPyplotImage(label)
    return
  def SaveImage(self, arr, label =''):
    self.OutputImage(arr, label)
    return

  def OutputImage(self, arr, label=''):
    label = label.replace(">","_")
    file_prefix = dt.now().strftime("%Y%m%d_%H%M%S_")
    file_name = os.path.join(self._outp_dir,file_prefix+label+".png")
    self.VerboseLog("Saving figure [...{}]".format(file_name[-40:]))
    if os.path.isfile(file_name):
      self.VerboseLog("Aborting image saving. File already exists.")
    else:
      imsave(file_name, arr)
    return

  def OutputPyplotImage(self, label=''):
    file_prefix = dt.now().strftime("%Y%m%d_%H%M%S")
    part_file_name = "{}_{}{}".format(file_prefix, label, ".png")
    file_name = os.path.join(self._outp_dir,part_file_name)
    self.VerboseLog("Saving pic [..{}]".format(file_name[-50:]))
    plt.savefig(file_name)
    return file_name

  def VerboseLog(self,str_msg, results=False, show_time=False):    
    return self._logger(str_msg, show = True, results = results, show_time = show_time)
  def P(self,str_msg, results = False, show_time = False):    
    return self._logger(str_msg, show = True, results = results, show_time = show_time)
  def PrintPad(self, str_msg, str_text, n=3):
    str_final = str_msg + "\n" +  textwrap.indent(str_text, n * " ")
    self._logger(str_final, show = True, results = False, show_time = False)
    return

  def Log(self,str_msg, show = False, results = False, show_time = False):    
    return self._logger(str_msg, show = show, results = results, show_time = show_time )

  def GetKerasModelSummary(self, model, full_info = False):
    if not full_info:
      old_stdout = sys.stdout
      sys.stdout = TextIOWrapper(BytesIO(), sys.stdout.encoding)
      model.summary()
      sys.stdout.seek(0)      
      out = sys.stdout.read() 
      sys.stdout.close()
      sys.stdout = old_stdout
    else:
      out = model.to_yaml()

    str_result = "Keras Neural Network Layout\n"+out
    return str_result

  def GetDataFrameInfo(self, df):
    old_stdout = sys.stdout
    sys.stdout = TextIOWrapper(BytesIO(), sys.stdout.encoding)
    df.info()
    sys.stdout.seek(0)      
    out = sys.stdout.read() 
    sys.stdout.close()
    sys.stdout = old_stdout

    str_result = "DataFrame info:\n"+out
    return str_result

  def GetKerasModelDesc(self, model):
    short_name = ""
    nr_l = len(model.layers)
    for i in range(nr_l):
      layer = model.layers[i]
      s_layer = "{}".format(layer.name)
      c_layer = s_layer.upper()[0:4]
      if c_layer == "CONV":
        c_layer = "Conv{}".format(layer.filters)
      if c_layer == "DENS":
        c_layer = "DNS{}".format(layer.units)
      if c_layer == "DROP":
        c_layer = "DRP"
      c_layer += "+"
      short_name += c_layer
    short_name = short_name[:-1]  
    return short_name
  def SaveModelNotes(self, l_notes, model_name, cfg, DEBUG=False):
    if DEBUG:
      self.P("  Saving {} notes for {}".format(len(l_notes), model_name))
    fn = os.path.join(self.GetModelsFolder(), model_name + '.txt')
    with open(fn,"w") as fp:
      fp.write("Model: {} [{}]\n\n".format(
          model_name, dt.now().strftime("%Y-%m-%d %H:%M:%S")))
      for _l in l_notes:
        fp.write("{}\n".format(_l))
    self._save_model_config(model_name, cfg)
    return
  def _save_model_config(self, label, cfg, DEBUG=False):
    if cfg is not None:
      cfg['INFERENCE'] = label
      if DEBUG:
        self.VerboseLog("  Saving cfg [{}] to models...".format(label))
      file_path = os.path.join(self.GetModelsFolder(), label+'.json')
      with open(file_path, 'w') as fp:
          json.dump(cfg, fp, sort_keys=False, indent=4)    
    return
  def CompareKerasModels(self, model1, model2):
    layers1 = model1.layers
    layers2 = model2.layers
    _equal = True
    if len(layers1) != len(layers2):
      self.P("Number of layers differs!")
      _equal = False      
    if _equal:
      for i in range(len(layers1)):
        w1_list = layers1[i].get_weights()
        w2_list = layers2[i].get_weights()
        for j in range(len(w1_list)):
          w1 = w1_list[j]
          w2 = w2_list[j]
          diff = np.sum((w1==w2)==False)
          if diff > 0:
            self.P("Found {} diffs in layers {}/{} on weights:{}".format(diff,
                   layers1[i].name, layers2[i].name, j))
            _equal = False
    if not _equal:
      self.P("Models comparison failed!")
    else:
      self.P("Models are equal...")
    return _equal
  def SaveKerasModelWeights(self, filename, model, layers):
    assert len(layers) > 0, 'Unknown list of selected layers'
    file_name = os.path.join(self.GetModelsFolder(),filename+".pkl")
    self.P("Saving weights for {} layers in {}...".format(len(layers),file_name))
    w_dict = OrderedDict()
    for layer in layers:
      w_dict[layer] = model.get_layer(layer).get_weights()
    with open(file_name, 'wb') as f:
      pickle.dump(w_dict, f)
    self.P("Done saving weights [{}].".format(layers), show_time=True)
    return
  def LoadKerasModelWeights(self, filename, model, layers):
    assert len(layers) > 0, 'Unknown list of selected layers'
    file_name = os.path.join(self.GetModelsFolder(),filename+".pkl")
    if not os.path.isfile(file_name):
      self.P("No weights file found")
      return False
    self.P("Loading weights for {} layers from {}...".format(len(layers),file_name))
    with open(file_name, 'rb') as f:
      w_dict = pickle.load(f)
    self.P("Loaded layers: {}".format(list(w_dict.keys())))
    for layer in layers:
      model.get_layer(layer).set_weights(w_dict[layer])
    self.P("Done loading weights.", show_time=True)
    return True

  def SaveKerasModel(self, model, label, use_prefix=False, cfg=None):
    file_prefix = ""
    if label=="":
      label = self.GetKerasModelDesc(model)
    label = label.replace(">","_")
    if use_prefix:
      file_prefix = dt.now().strftime("%Y%m%d_%H%M%S_")
    file_name = os.path.join(self.GetModelsFolder(),file_prefix+label+".h5")
    self.VerboseLog("Saving [...{}]".format(file_name[-40:]))
    model.save(file_name)
    self.VerboseLog("Done saving [...{}]".format(file_name[-40:]), show_time = True)
    self._save_model_config(label, cfg)    
    return
  def LoadKerasModel(self, model_name, custom_objects = None):
    if model_name[-3:] != '.h5':
      model_name += '.h5'
    self.VerboseLog("Trying to load {}...".format(model_name))
    model_full_path = os.path.join(self.GetModelsFolder(), model_name)
    if os.path.isfile(model_full_path):
      from keras.models import load_model
      self.VerboseLog("Loading [...{}]".format(model_full_path[-40:]))
      model = load_model(model_full_path, custom_objects = custom_objects)
      self.VerboseLog("Done loading [...{}]".format(model_full_path[-40:]), show_time = True)
    else:
      self.VerboseLog("File {} not found.".format(model_name))
      model = None
    return model

  def LogKerasModel(self, model):
    self.VerboseLog(self.GetKerasModelSummary(model))
    return

  def GetMachineName(self):
    self.MACHINE_NAME = socket.gethostname()
    return self.MACHINE_NAME

  def _check_keras_avail(self):
    try:
        imp.find_module('keras')
        found = True
        import keras
        self.KERAS_VER = keras.__version__
        self.KERAS = True
    except ImportError:
        found = False
        self.KERAS = False
    return found

  def _check_tf_avail(self):
    try:
        imp.find_module('tensorflow')
        found = True
        import tensorflow as tf
        self.TF_VER = tf.__version__
        self.TF = True
    except ImportError:
        found = False
        self.TF = False
    return found

  def CheckTF(self):
    ret = 0
    if self._check_tf_avail():
      self.TF = True
      from tensorflow.python.client import device_lib
      local_device_protos = device_lib.list_local_devices()
      self.devices ={x.name: x.physical_device_desc for x in local_device_protos}
      types = [x.device_type for x in local_device_protos]
      if 'GPU' in types:
          ret = 2
          self._logger("Found TF {} running on GPU".format(self.TF_VER))
      else:
          self._logger("Found TF {} running on CPU".format(self.TF_VER))
          ret = 1
      try:
        import tensorflow as tf
        self.TF_KERAS_VER = tf.keras.__version__
        self._logger("Found TF.Keras {}".format(self.TF_KERAS_VER))
      except:
        self.TF_KERAS_VER = None
        self._logger("No TF.Keras found.")
      if self._check_keras_avail():
        self._logger("Found Keras {}".format(self.KERAS_VER))
    else:
      self._logger("TF not found")
      self.TF = False
    return ret
  def GetGPU(self):
    res = []
    if self._check_tf_avail():
      self.TF = True
      from tensorflow.python.client import device_lib
      loc = device_lib.list_local_devices()
      res = [x.physical_device_desc for x in loc if x.device_type=='GPU']
    return res

  def LogConfusionMatrix(self,cm, labels=["0","1"], hide_zeroes=False,
                         hide_diagonal=False, hide_threshold=None,
                         show = False):
    columnwidth = max([len(x) for x in labels] + [8])  
    empty_cell = " " * columnwidth
    full_str = "         " + empty_cell +"Preds\n"
    full_str += "    " + empty_cell+ " "
    for label in labels:
      full_str += "%{0}s".format(columnwidth) % label+" "
    full_str+="\n"
    for i, label1 in enumerate(labels):
      if i==0:
        full_str+="GT  %{0}s".format(columnwidth) % label1 +" "
      else:
        full_str+="    %{0}s".format(columnwidth) % label1 +" "
      for j in range(len(labels)):
        cell = '{num:{fill}{width}}'.format(num=cm[i, j], fill=' ', width=columnwidth)
        if hide_zeroes:
            cell = cell if float(cm[i, j]) != 0 else empty_cell
        if hide_diagonal:
            cell = cell if i != j else empty_cell
        if hide_threshold:
            cell = cell if cm[i, j] > hide_threshold else empty_cell
        full_str += cell + " "
      full_str +="\n"
    self._logger("Confusion Matrix:\n{}".format(full_str), show = show)

  def _keras_epoch_callback(self, epoch, logs):
    str_logs = ""
    for key,val in logs.items():
      str_logs += "{}:{:.6f}  ".format(key,val)
    self.P(" Train/Fit: Epoch: {} Results: {}".format(epoch,str_logs))
    return
  def GetKerasEpochCallback(self, predef_callback=None):
    assert self.TF == True
    import tensorflow as tf
    self.VerboseLog("Creating keras epoch end callback...")
    if predef_callback is None:
      return tf.keras.callbacks.LambdaCallback(on_epoch_end=self._keras_epoch_callback)
    return tf.keras.callbacks.LambdaCallback(on_epoch_end=predef_callback)
  def GetKerasLRCallback(self, monitor='loss', patience=3, factor=0.1, 
                         use_tf_keras = False):
    if use_tf_keras:
      assert self.TF
      import tensorflow as tf
      cb = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, patience=patience, factor=factor)
    else:
      from keras.callbacks import ReduceLROnPlateau
      cb = ReduceLROnPlateau(monitor=monitor, patience=patience, factor=factor)
    return cb

  def GetKerasCheckpointCallback(self, model_name='model', monitor='val_loss',
                                 period=1, use_tf_keras = False):
    if use_tf_keras:
      assert self.TF
      import tensorflow as tf
      file_path = model_name + "_tk_E{epoch:02d}_L{" + monitor + ":.6f}" + ".h5"
      self.VerboseLog("Creating tf.keras chekpoint callback...")
      file_path = os.path.join(self.GetModelsFolder(), file_path)
      cb = tf.keras.callbacks.ModelCheckpoint(filepath = file_path, 
                                              monitor=monitor, 
                                              verbose=0,
                                              save_best_only=False, 
                                              save_weights_only=False, 
                                              mode='auto', 
                                              period=period)
    else:
      from keras.callbacks import ModelCheckpoint
      file_path = model_name + "_k_E{epoch:02d}_L{" + monitor + ":.6f}" + ".h5"
      self.VerboseLog("Creating keras chekpoint callback...")
      file_path = os.path.join(self.GetModelsFolder(), file_path)
      cb = ModelCheckpoint(filepath = file_path, 
                           monitor=monitor,
                           verbose=0, 
                           save_best_only=False, 
                           save_weights_only=False, 
                           mode='auto', 
                           period=period)
    return cb

  def GetStandardKerasCallbacks(self, model_name='model', monitor='loss'):
    cb_list = [self.GetKerasCheckpointCallback(model_name=model_name, monitor=monitor), 
               self.GetKerasEpochCallback()]
    return cb_list
  def GetKerasTensorboardCallback(self, use_tf_keras = False):
    if use_tf_keras:
      assert self.TF == True
      import tensorflow as tf
      self.VerboseLog("Creating tf.keras tensorboard callback...")
      self._tensorboard_dir = os.path.join(self._base_folder,'_tf');
      if not os.path.isdir(self._tensorboard_dir):
        os.makedirs(self._tensorboard_dir)
      cb_tboard = tf.keras.callbacks.TensorBoard(log_dir=self._tensorboard_dir, 
                              histogram_freq=1,  
                              write_graph=True, 
                              write_images=True)
    else:
      from keras.callbacks import TensorBoard
      self.VerboseLog("Creating Keras tensorboard callback...")
      self._tensorboard_dir = os.path.join(self._base_folder,'_tf');
      if not os.path.isdir(self._tensorboard_dir):
        os.makedirs(self._tensorboard_dir)
      cb_tboard = TensorBoard(log_dir=self._tensorboard_dir, 
                              histogram_freq=1,  
                              write_graph=True, 
                              write_images=True)      
    return cb_tboard
  def PlotConfusionMatrix(self,cm, classes=["0","1"],
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        s_title = "[Normalized] " + title
    else:
        s_title = "[Standard] " + title

    plt.title(s_title)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
  def start_timer(self, sname):
    if not self.DEBUG:
      return -1

    count_key = sname+"___COUNT"
    start_key = sname+"___START"
    pass_key  = sname+"___PASS"
    if not (count_key in self.timers.keys()):
      self.timers[count_key] = 0
      self.timers[sname] = 0
      self.timers[pass_key] = True
    ctime = tm()
    self.timers[start_key] = ctime
    return ctime

  def end_timer(self, sname, skip_first_timing = True):
    result = 0
    if self.DEBUG:
      count_key = sname+"___COUNT"
      start_key = sname+"___START"
      end_key   = sname+"___END"
      pass_key  = sname+"___PASS"
      self.timers[end_key] = tm()
      result = self.timers[end_key] - self.timers[start_key]
      _count = self.timers[count_key]
      _prev_avg = self.timers[sname]
      avg =  _count *  _prev_avg
      if self.timers[pass_key] and skip_first_timing:
        self.timers[pass_key] = False
        return result 
      self.timers[count_key] = _count + 1
      avg += result
      avg = avg / self.timers[sname+"___COUNT"]
      self.timers[sname] = avg
    return result
  def show_timers(self):
    if self.DEBUG:
      self.VerboseLog("Timing results:")
      for key,val in self.timers.items():
        if not ("___" in key):
          self.VerboseLog(" {} = {:.3f}s".format(key,val))
    else:
      self.VerboseLog("DEBUG not activated!")
    return
  def get_stats(self):
    self.show_timers()
    return
  def show_timings(self):
    self.show_timers()
    return
  def get_timing(self, skey):
    return self.timers[skey]
  def SaveToHDF(self, df, h5_file, h5_format = 'table'):
    assert "/" not in h5_file
    assert "\\" not in h5_file
    table_name, ext = os.path.splitext(h5_file)
    out_file = os.path.join(self._data_dir, h5_file)
    self.VerboseLog("Saving ...{}".format(out_file[-40:]))
    df.to_hdf(out_file, key = 'table_' + table_name, 
              append = False, format = h5_format)
    self.VerboseLog("Done saving ...{}".format(out_file[-40:]), show_time = True)
    return
  def LoadFromHDF(self, h5_file):
    assert "/" not in h5_file
    assert "\\" not in h5_file
    table_name, ext = os.path.splitext(h5_file)
    out_file = os.path.join(self._data_dir, h5_file)
    self.VerboseLog("Loading ...{}".format(out_file[-40:]))
    df = pd.read_hdf(out_file, key = 'table_' + table_name)
    self.VerboseLog("Done loading ...{}".format(out_file[-40:]), show_time = True)
    return df
  def SaveTFGraph(self, tf_saver, tf_session, file_name, sub_folder='', debug=False):
    if file_name[-5] != '.ckpt':
      file_name += '.ckpt'
    mfolder = self.GetModelsFolder()
    folder = os.path.join(mfolder,sub_folder)
    if not os.path.isdir(folder):
      self.P("Creating folder [{}]".format(folder))
      os.makedirs(folder)                          
    path = os.path.join(folder, file_name)
    try:
      if debug:
        self.P("Saving tf checkpoint '{}'".format(file_name))
      tf_saver.save(tf_session, path)
    except:
      self.P("ERROR Saving session for {}".format(path[-40:]))
    return

  def CleanConvertModel(self, backend, model, model_name, 
                        output_layers, 
                        input_layers):
    import tensorflow as tf
    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.framework import graph_io
    backend.set_learning_phase(0)
    tensor_dict = {}
    out_names = []
    for i, input_layer in enumerate(input_layers):
      tensor_dict["INPUT_{}".format(i)] = input_layer.name
    for i, output_layer in enumerate(output_layers):
      tensor_dict["OUTPUT_{}".format(i)] = output_layer.name
      out_names.append(output_layer.name)
    self.SaveConfigDict(model_name, tensor_dict)

    if model_name[-3:] == '.pb':
      model_name = model_name[:-3]
    pb_graphdef =  model_name+'_gdef_.pb'
    pb_frozen =  model_name+'_frozen_.pb'
    sess = backend.get_session()
    train_graph = sess.graph
    inference_graph = tf.graph_util.remove_training_nodes(train_graph.as_graph_def())
    graph_io.write_graph(inference_graph, self.GetModelsFolder(), pb_graphdef)
    freeze_graph.freeze_graph(
            pb_graphdef, 
            '', 
            False, 
            checkpoint_path, 
            out_names, 
            "save/restore_all", 
            "save/Const:0", 
            pb_frozen, 
            False, 
            ""
        )    
    return
  def SaveModelAsGraphToModels(self, model, model_name, 
                               output_layers, 
                               input_layers):
    if model_name[-3:] != '.pb':
      model_name += '.pb'
    pb_file = os.path.join(self.GetModelsFolder(), model_name)
    self.SaveModelAsGraph(model = model, 
                          pb_file = pb_file,
                          output_layers = output_layers,
                          input_layers = input_layers)
    return
  def SaveGraphToModels(self, session, tensor_list, 
                        graph_name, input_names, 
                        output_names = None,
                        ):
    if graph_name[-3:] != '.pb':
      graph_name += '.pb'
    pb_file = os.path.join(self.GetModelsFolder(), graph_name)
    self.SaveGraph(session = session,
                   tensor_list = tensor_list,
                   pb_file = pb_file,
                   check_input_names = input_names,
                   )
    return
  def SaveModelAsGraph(self, model, pb_file, 
                       output_layers, 
                       input_layers):
    import keras.backend as K
    sess = K.get_session()
    input_names = []

    for inp_layer in input_layers:
      input_names.append(inp_layer.name)
    out_tensors = []
    for layer in output_layers:
      out_tensors.append(layer.output)
    self.SaveGraph(session = sess,
                   tensor_list = out_tensors,
                   pb_file = pb_file,
                   input_names = input_names)
    return

  def save_graph_to_file(self, sess, graph, graph_file_name, output_tensor_list):
    from tensorflow.python.framework import graph_util
    from tensorflow.python.platform import gfile
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), output_tensor_list)

    with gfile.FastGFile(graph_file_name, 'wb') as f:
      f.write(output_graph_def.SerializeToString())
    return 
  def SaveGraph(self, session, tensor_list, 
                pb_file, check_input_names = None):
    assert len(tensor_list) != 0

    g = tensor_list[0].graph
    self.VerboseLog("Saving graph with {} input(s) and {} output(s) ".format(
        len(check_input_names), len(tensor_list), ))

    if check_input_names is not None:
      for i,name in enumerate(check_input_names):
        if name[-2:] != ':0':
          name += ':0'
        else:
          check_input_names[i] = check_input_names[i][:-2]
        self.VerboseLog("  Input: {}".format(g.get_tensor_by_name(name)))

    final_output_names = []
    for i,out_tensor in enumerate(tensor_list):
      out_name = out_tensor.name
      if out_name[-2:-1] != ':':
        out_name += ':0'
      final_output_names.append(out_name[:-2])
      self.VerboseLog("  Output: {}".format(
          g.get_tensor_by_name(final_output_names[-1]+":0"))) 

    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    constant_graph = graph_util.convert_variables_to_constants(session, 
                                                               g.as_graph_def(), 
                                                               final_output_names)
    pb_file_path, pb_file_name = os.path.split(pb_file)
    self.VerboseLog(" Saving {} in ...{}".format(pb_file_name,pb_file_path[-30:]))
    graph_io.write_graph(constant_graph, pb_file_path, pb_file_name, 
                         as_text=False)
    return 
  def LoadTFGraph(self, pb_file):   
    self.VerboseLog("Prep graph from [...{}]...".format(pb_file[-30:]))
    detection_graph = None
    if os.path.isfile(pb_file):
      import tensorflow as tf
      start_time = tm()
      detection_graph = tf.Graph()
      with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_file, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')    
      end_time = tm() 
      self.VerboseLog("Done preparing graph in {:.2f}s.".format(end_time-start_time))        
    else:
      self.VerboseLog(" FILE NOT FOUND [...{}]...".format(pb_file[-30:]))
    return detection_graph
  def LoadGraphFromModels(self, model_name):
    if model_name[-3:] != '.pb':
      model_name += '.pb'
    graph_file = os.path.join(self.GetModelsFolder(), model_name)
    return self.LoadTFGraph(graph_file)
  def SaveConfigDict(self, model_name, tensor_names_dict):
    if model_name[-4:] != '.txt':
      model_name += '.txt'
    self.VerboseLog("Saving cfg [{}] to models...".format(model_name))
    file_path = os.path.join(self.GetModelsFolder(), model_name)
    with open(file_path, 'w') as fp:
        json.dump(tensor_names_dict, fp, sort_keys=True, indent=4)    
    return
  def LoadConfigDict(self, model_name):
    if model_name[-4:] != '.txt':
      model_name += '.txt'
    self.VerboseLog("Loading cfg [{}] from models...".format(model_name))
    file_path = os.path.join(self.GetModelsFolder(), model_name)
    with open(file_path, 'r') as fp:
        data = json.load(fp) 
    return data
  def GetTensorsInTFGraph(self, tf_graph):
    return [n.name for n in tf_graph.as_graph_def().node]
  def PlotKerasHistory(self, keras_history_object):
    styles = ['bo','b']
    keys_lists = [
        ['acc','val_acc'],
        ['loss','val_loss'],
        ['recall','val_recall'],
        ['recall_metric','val_recall_metric'],
        ['precision_metric','val_precision_metric'],
        ['precision','val_precision'],
        ]
    plots = []
    if type(keras_history_object) is dict:
      hist = keras_history_object
    else:
      hist = keras_history_object.history
    for keys in keys_lists:
      vals_dict = {}
      for k in keys:
        if k in hist.keys():
          vals = hist[k]
          vals_dict[k] = vals
      if vals_dict != {}:    
        plots.append(vals_dict)

    for plot in plots:
      s_plot = list(plot.keys())
      plot_name = ""
      for i in range(len(s_plot)-1):
        plot_name += "{} vs. ".format(s_plot[i])
      plot_name += s_plot[-1]
      self.P("Plotting '{}' for {} epochs...".format(plot_name, 
                                                   len(plot[s_plot[0]])))
      plt.figure()
      max_len = max([len(sss) for sss in s_plot])
      fmt = " {:<"+str(max_len+1)+"} {}"
      for i,k in enumerate(plot.keys()):
        vals = plot[k]
        s_vals = ""
        for j in range(min(10, len(vals))):
          s_vals +="{:.3f}  ".format(vals[j])        
        self.P(fmt.format(k+':', s_vals))
        plt.plot(range(len(vals)), vals, styles[i], label=k)
        plt.legend()
      plt.title(plot_name)
      plt.show()
      self.OutputPyplotImage(label=k)
    return

  
EOF
