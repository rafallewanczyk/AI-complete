%%time

import numpy as np
import pandas as pd
import tensorflow as tf  
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep 

np.mat("1,2,3;1,5,6")

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.isna().sum()
dataset = dataset.dropna()
dataset.isna().sum()
dataset.tail(11)
origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

train_dataset = dataset.sample(frac=0.8, random_state=0)
train_dataset.shape
test_dataset = dataset.drop(train_dataset.index)
test_dataset.shape

train_stats = train_dataset.describe()
train_stats.pop("MPG")

train_stats = train_stats.transpose()
train_stats

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
test_labels.tail()

preprocessor = prep.StandardScaler().fit(train_dataset.values)
mycl = train_dataset.columns
np_normed_train_data = preprocessor.transform(train_dataset.values)
np_normed_test_data = preprocessor.transform(test_dataset.values)

normed_train_data = pd.DataFrame(np_normed_train_data, columns=mycl)
normed_test_data = pd.DataFrame(np_normed_test_data, columns=mycl)

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  print()
  return model

model = build_model()

model.summary()
normed_train_data.tail()

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 300

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.shape)

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

model_early_stop = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=11)

history_early_stop = model_early_stop.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

hist_early_stop = pd.DataFrame(history_early_stop.history)
hist_early_stop['epoch'] = history_early_stop.epoch
print(hist_early_stop.shape)

loss, mae, mse = model_early_stop.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
print("Testing set Mean Std Error: {:5.2f} MPG".format(mse))

test_predictions = model_early_stop.predict(normed_test_data).flatten()

print(type(test_labels))
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
if 0:
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")

model_early_stop.save_weights('./mm/my_model')

reload_model = build_model()

reload_model.load_weights('./mm/my_model')
reload_model.summary()

reload_test_predictions = reload_model.predict(normed_test_data).flatten()

print (reload_test_predictions == test_predictions)

model_early_stop.save('./mm/h5.h5')
h5_reload_model = tf.keras.models.load_model('./mm/h5.h5')
h5_reload_model.summary()
h5_reload_test_predictions = h5_reload_model.predict(normed_test_data).flatten()

print (h5_reload_test_predictions == test_predictions)

===

%%time 

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep 
from IPython.display import display

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(8,5))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])

def norm_x_y_data(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return [X_train, X_test]

def show_predict(y_test, y_label, y_predict):
    cnt_err = 0
    for i in range(len(y_test)):
        if np.argmax(y_label[i]) != np.argmax(y_predict[i]):
            print (y_test[i])
            print (y_label[i] , " vs " , y_predict[i])
            e_ =  y_test[i]
            print ("- diff is :" , np.abs(np.sqrt(e_[0] * e_[0] + e_[-1] * e_[-1]) - 0.66))
            cnt_err += 1
            print()
    print("- cnt error is ", cnt_err)
    print()

def get_mm_filesize(param_num):
    R0 = 33.9765625
    each_size = 0.01171875
    return R0 + param_num * each_size 

def gen_model(x_dim, y_dim):

    baseline_model = keras.Sequential([
        keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(x_dim,)),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dense(y_dim, activation=tf.nn.sigmoid) ])

    baseline_model.compile(
        optimizer=tf.keras.optimizers.Adam(),

                loss=tf.keras.losses.binary_crossentropy,
                metrics=['accuracy','binary_crossentropy'])
    return baseline_model    

def gen_train_label(rows, x_dim, y_dim):
    x_train = np.random.random([rows,x_dim])
    x_label = np.random.random([rows,y_dim])
    x_label.fill(0.0)

    for idx in range(rows):
        e_ = x_train[idx]

        e_c = (np.sqrt(e_[0] * e_[0] + e_[-1] * e_[-1]) > 0.66) * 1.0
        x_label[idx][int(e_c)] = 1.0

    return [x_train, x_label]

if __name__ == "__main__":

    import os
    if not os.path.exists("./mm"):
        !mkdir  mm
        !ls mm
    rows = 10000
    x_dim = 10
    y_dim = 2    

    key_acc = "acc"
    if "WINDIR" in os.environ :
        key_acc = "accuracy"  

    [x_train, x_label] = gen_train_label(rows, x_dim, y_dim)
    [y_test, y_label] = gen_train_label( int(rows/2), x_dim, y_dim)

    baseline_model = gen_model(x_dim, y_dim)

    early_stop = keras.callbacks.EarlyStopping(monitor="val_" + key_acc,  patience=7)
    !perl -e "print time"
    print()
    baseline_history = baseline_model.fit(x_train,
                                      x_label,
                                      epochs=300,
                                      batch_size=1000,
                                      validation_data=(y_test, y_label),
                                      callbacks=[early_stop],
                                      verbose=0)    
    !perl -e "print time"
    print()

    plot_history( [ ('baseline', baseline_history) ], key=key_acc) 
    plot_history( [ ('baseline', baseline_history) ], key="binary_crossentropy")
    print(baseline_history.history.keys(), "\n")

    baseline_history_ = pd.DataFrame(baseline_history.history)
    baseline_history_['epoch'] = baseline_history.epoch
    print(baseline_history_.shape)
    print()
    display(baseline_history_.tail())
    batch_size = 1000
    s_i = np.random.choice(range(len(y_test)), batch_size)
    y_predict = baseline_model.predict(y_test[s_i])

    print (baseline_model.evaluate(y_test, y_label))

    if 1:
        baseline_model.save('./mm/h5.h5')
        baseline_model_new_h5 = tf.keras.models.load_model('./mm/h5.h5')
        print (baseline_model_new_h5.evaluate(y_test, y_label))

    if 0:
        baseline_model.save_weights('./mm/ckpt')
        baseline_model_new_ckp = gen_model(x_dim, y_dim)
        baseline_model_new_ckp.load_weights('./mm/ckpt')
        print (baseline_model_new_ckp.evaluate(y_test, y_label))
        baseline_model_new_ckp.summary()

    print ("- h5 networks filesize is : " , get_mm_filesize(baseline_model.count_params()) , " kbytes")

===

%%time
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep 
from IPython.display import display

def gen_train_label(rows, x_dim, y_dim):
    x_train = np.random.random([rows,x_dim])
    x_label = np.random.random([rows,y_dim])
    x_label.fill(0.0)

    for idx in range(rows):
        e_ = x_train[idx]

        e_c = ( (e_[0] * 1 + e_[-1] * 2) - 0.66 ) * 1.0

        x_label[idx][0] = e_c
        x_label[idx][-1] = -e_c + 0

    return [x_train, x_label]

def norm_x_y_data(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return [X_train, X_test, preprocessor]

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(8,5))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])

def show_predict(y_test, y_label, y_predict):
    cnt_err = 0
    for i in range(len(y_test)):
        if np.argmax(y_label[i]) != np.argmax(y_predict[i]):
            print (y_test[i])
            print (y_label[i] , " vs " , y_predict[i])
            e_ =  y_test[i]
            print ("- diff is :" , np.abs(np.sqrt(e_[0] * e_[0] + e_[-1] * e_[-1]) - 0.66 ))
            cnt_err += 1
            print()
    print("- cnt error is ", cnt_err)
    print()

def get_mm_filesize(param_num):
    R0 = 33.9765625
    each_size = 0.01171875
    return R0 + param_num * each_size 

def gen_model(x_dim, y_dim):

    baseline_model = keras.Sequential([

        keras.layers.Dense(6, input_shape=(x_dim,)),
         keras.layers.Dense(4, activation="linear"),

         keras.layers.Dense(y_dim, activation="linear")

    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001/2)

    baseline_model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'binary_crossentropy'])    
    return baseline_model    

if __name__ == "__main__":

    preprocessor = None
    import os
    if not os.path.exists("./mm"):
        !mkdir  mm
        !ls mm
    rows = 10000
    x_dim = 3
    y_dim = 2    
    flag_train = 1

    key_acc = "acc"
    if "WINDIR" in os.environ :
        key_acc = "accuracy"  
    if "HOME" in os.environ and os.environ["HOME"] == "/home/bgi902":
        key_acc = "accuracy"

    [x_train, x_label] = gen_train_label(rows, x_dim, y_dim)
    [y_test, y_label] = gen_train_label( int(rows/2), x_dim, y_dim)
    [x_train, y_test, preprocessor] = norm_x_y_data(x_train, y_test)
    if flag_train:
        baseline_model = gen_model(x_dim, y_dim)
        baseline_model.summary()

        !perl -e "print time"
        print()
        baseline_history = baseline_model.fit(x_train,
                                          x_label,
                                          epochs=10,
                                          batch_size=200,
                                          validation_data=(y_test, y_label),
                                          verbose=0)    

        !perl -e "print time"
        print()

        plot_history( [ ('baseline', baseline_history) ], key="loss") 

        print(baseline_history.history.keys(), "\n")

        baseline_history_ = pd.DataFrame(baseline_history.history)
        baseline_history_['epoch'] = baseline_history.epoch
        print(baseline_history_.shape)
        print()
        display(baseline_history_.tail())

        print (baseline_model.evaluate(y_test, y_label))

    if 1:
        if flag_train:
            baseline_model.save('./mm/h5.h5')
        baseline_model_new_h5 = tf.keras.models.load_model('./mm/h5.h5')
        print (baseline_model_new_h5.evaluate(y_test, y_label))

        batch_size = 10
        s_i = np.random.choice(range(len(y_test)), batch_size)

        y_predict = baseline_model_new_h5.predict(y_test[s_i])
        print("______________________")

        for i in range(10):
            print(y_test[s_i][i], " => ", y_predict[i] , " vs ", y_label[s_i][i])
        x = y_predict[:,0] + y_predict[:, -1]
        print (np.sum(np.abs(x)) / len(x))
        print("______________________")

    if 0:
        baseline_model.save_weights('./mm/ckpt')
        baseline_model_new_ckp = gen_model(x_dim, y_dim)
        baseline_model_new_ckp.load_weights('./mm/ckpt')
        print (baseline_model_new_ckp.evaluate(y_test, y_label))
        baseline_model_new_ckp.summary()

    print ("- h5 networks filesize is : " , get_mm_filesize(baseline_model.count_params()) , " kbytes")

===

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep 
from IPython.display import display

def gen_train_label(rows, x_dim, y_dim):
    x_train = np.random.random([rows,x_dim])
    x_label = np.random.random([rows,y_dim])
    x_label.fill(0.0)

    for idx in range(rows):
        e_ = x_train[idx]

        x_label[idx][0] = np.argmax(e_)
        x_label[idx][-1] = np.argmin(e_)

    return [x_train, x_label]

def norm_x_y_data(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return [X_train, X_test, preprocessor]

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(8,5))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])

def show_predict(y_test, y_label, y_predict):
    cnt_err = 0
    for i in range(len(y_test)):
        if np.argmax(y_label[i]) != np.argmax(y_predict[i]):
            print (y_test[i])
            print (y_label[i] , " vs " , y_predict[i])
            e_ =  y_test[i]
            print ("- diff is :" , np.abs(np.sqrt(e_[0] * e_[0] + e_[-1] * e_[-1]) - 0.66 ))
            cnt_err += 1
            print()
    print("- cnt error is ", cnt_err)
    print()

def get_mm_filesize(param_num):
    R0 = 33.9765625
    each_size = 0.01171875
    return R0 + param_num * each_size 

def gen_model(x_dim, y_dim):

    baseline_model = keras.Sequential([

        keras.layers.Dense(128, input_shape=(x_dim,)),
         keras.layers.Dense(128, activation="linear"),
        keras.layers.Dense(64, activation="relu"),

         keras.layers.Dense(y_dim, activation="relu")

    ])
    baseline_model.summary()

    optimizer = tf.keras.optimizers.RMSprop(0.001/2)

    baseline_model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'binary_crossentropy'])    
    return baseline_model    

if __name__ == "__main__":

    preprocessor = None
    import os
    if not os.path.exists("./mm"):
        !mkdir  mm
        !ls mm
    rows = 10000
    x_dim = 3
    y_dim = 2    
    flag_train = 1

    key_acc = "acc"
    if "WINDIR" in os.environ :
        key_acc = "accuracy"  
    if "HOME" in os.environ and os.environ["HOME"] == "/home/bgi902":
        key_acc = "accuracy"

    [x_train, x_label] = gen_train_label(rows, x_dim, y_dim)
    [y_test, y_label] = gen_train_label( int(rows/2), x_dim, y_dim)

    if flag_train:
        baseline_model = gen_model(x_dim, y_dim)
        baseline_model.summary()

        !perl -e "print time"
        print()
        baseline_history = baseline_model.fit(x_train,
                                          x_label,
                                          epochs=40,
                                          batch_size=200,
                                          validation_data=(y_test, y_label),
                                          verbose=0)    

        !perl -e "print time"
        print()

        plot_history( [ ('baseline', baseline_history) ], key="loss") 

        print(baseline_history.history.keys(), "\n")

        baseline_history_ = pd.DataFrame(baseline_history.history)
        baseline_history_['epoch'] = baseline_history.epoch
        print(baseline_history_.shape)
        print()
        display(baseline_history_.tail())

        print (baseline_model.evaluate(y_test, y_label))

    if 1:
        if flag_train:
            baseline_model.save('./mm/h5.h5')
        baseline_model_new_h5 = tf.keras.models.load_model('./mm/h5.h5')
        print (baseline_model_new_h5.evaluate(y_test, y_label))

        batch_size = 10
        s_i = np.random.choice(range(len(y_test)), batch_size)

        y_predict = baseline_model_new_h5.predict(y_test[s_i])
        print("______________________")
        display(y_test[s_i])
        display(y_label[s_i])
        for i in range(10):
            print(y_test[s_i][i], " => ", np.round(y_predict[i]) , " vs ", y_label[s_i][i])
        x = y_predict[:,0] + y_predict[:, -1]
        print (np.sum(np.abs(x)) / len(x))
        print("______________________")

    if 0:
        baseline_model.save_weights('./mm/ckpt')
        baseline_model_new_ckp = gen_model(x_dim, y_dim)
        baseline_model_new_ckp.load_weights('./mm/ckpt')
        print (baseline_model_new_ckp.evaluate(y_test, y_label))
        baseline_model_new_ckp.summary()

    print ("- h5 networks filesize is : " , get_mm_filesize(baseline_model.count_params()) , " kbytes")

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep 
from IPython.display import display

def gen_train_label(rows, img_rows, img_cols, y_dim):
    (x_train, x_label), (y_test, y_label) = keras.datasets.mnist.load_data()  
    x_train = x_train.reshape(-1, img_rows, img_cols, 1)
    y_test = y_test.reshape(-1, img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    y_test = y_test.astype('float32')
    x_train /= 255.0
    y_test /= 255.0
    x_label = keras.utils.to_categorical(x_label, y_dim)
    y_label = keras.utils.to_categorical(y_label, y_dim)

    return [x_train, x_label, y_test, y_label]

def norm_x_y_data(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return [X_train, X_test, preprocessor]

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(8,5))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])

def show_predict(y_test, y_label, y_predict):
    cnt_err = 0
    for i in range(len(y_test)):
        if np.argmax(y_label[i]) != np.argmax(y_predict[i]):
            print (y_test[i])
            print (y_label[i] , " vs " , y_predict[i])
            e_ =  y_test[i]
            print ("- diff is :" , np.abs(np.sqrt(e_[0] * e_[0] + e_[-1] * e_[-1]) - 0.66 ))
            cnt_err += 1
            print()
    print("- cnt error is ", cnt_err)
    print()

def get_mm_filesize(param_num):
    R0 = 33.9765625
    each_size = 0.01171875
    return R0 + param_num * each_size 

def gen_model(img_rows, img_cols, y_dim):

    input_shape= (img_rows, img_cols, 1)
    baseline_model = keras.Sequential([
        keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape), 

        keras.layers.Conv2D(16, kernel_size=(3, 3),  activation='relu'), 
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.22),
        keras.layers.Flatten(),

        keras.layers.Dense(16, activation='relu'),

        keras.layers.Dropout(0.22),
        keras.layers.Dense(y_dim, activation='softmax')

    ])

    baseline_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return baseline_model    

if __name__ == "__main__":

    preprocessor = None
    import os
    if not os.path.exists("./mm"):
        !mkdir  mm
        !ls mm
    rows = 60000
    img_rows = 28
    img_cols = 28
    y_dim = 10
    flag_train = 1

    key_acc = "acc"
    if "WINDIR" in os.environ :
        key_acc = "accuracy"  
    if "HOME" in os.environ and os.environ["HOME"] == "/home/bgi902":
        key_acc = "accuracy"

    [x_train, x_label, y_test, y_label] = gen_train_label(rows, img_rows, img_cols, y_dim)

    if flag_train:
        baseline_model = gen_model(img_rows, img_cols, y_dim)
        baseline_model.summary()

        !perl -e "print time"
        print()
        baseline_history = baseline_model.fit(x_train, x_label,
              batch_size=512,
              epochs=10,
              verbose=0,
              validation_data=(y_test, y_label)
             )

        !perl -e "print time"
        print()

        plot_history( [ ('baseline', baseline_history) ], key="loss") 

        print(baseline_history.history.keys(), "\n")

        baseline_history_ = pd.DataFrame(baseline_history.history)
        baseline_history_['epoch'] = baseline_history.epoch
        print(baseline_history_.shape)
        print()
        display(baseline_history_.tail())

        print (baseline_model.evaluate(y_test, y_label))

    if 1:
        if flag_train:
            baseline_model.save('./mm/h5.h5')
        baseline_model_new_h5 = tf.keras.models.load_model('./mm/h5.h5')
        print (baseline_model_new_h5.evaluate(y_test, y_label))

        batch_size = 10
        s_i = np.random.choice(range(len(y_test)), batch_size)

        y_predict = baseline_model_new_h5.predict(y_test[s_i])
        print("______________________")
        print(y_test[s_i][0].shape)

        display(y_label[s_i])
        for i in range(10):
            print(i , " => "  , np.argmax(y_predict[i]) , " vs ", np.argmax(y_label[s_i][i]))

    if 0:
        baseline_model.save_weights('./mm/ckpt')
        baseline_model_new_ckp = gen_model(img_rows, img_cols, y_dim)
        baseline_model_new_ckp.load_weights('./mm/ckpt')
        print (baseline_model_new_ckp.evaluate(y_test, y_label))
        baseline_model_new_ckp.summary()

    print ("- h5 networks filesize is : " , get_mm_filesize(baseline_model.count_params()) , " kbytes")

===

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow import keras as keras

from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(1337)

(x_train, _), (x_test, _) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
latent_dim = 16

layer_filters = [32, 64]

inputs = Input(shape=input_shape, name='encoder_input')
x = inputs

for filters in layer_filters:
    shape = K.int_shape(x)
    print(shape)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

shape = K.int_shape(x)

print(shape)

x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

encoder = Model(inputs, latent, name='encoder')
encoder.summary()

latent_inputs = Input(shape=(latent_dim,), name='decoder_input')

print(K.int_shape(latent_inputs))

x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
print(K.int_shape(x))
x = Reshape((shape[1], shape[2], shape[3]))(x)
print(K.int_shape(x))

for filters in layer_filters[::-1]:
    shape = K.int_shape(x)
    print(shape)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

shape = K.int_shape(x)
print(shape)   

x = Conv2DTranspose(filters=1,
                    kernel_size=kernel_size,
                    padding='same')(x)
print(K.int_shape(x))

outputs = Activation('sigmoid', name='decoder_output')(x)
print(K.int_shape(outputs))

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')

autoencoder.fit(x_train_noisy,
                x_train,
                validation_data=(x_test_noisy, x_test),
                epochs=30,
                batch_size=batch_size)

x_decoded = autoencoder.predict(x_test_noisy)

rows, cols = 10, 30
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()

===

!pip install tf-nightly
from __future__ import print_function

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.datasets import imdb

max_features = 177
maxlen = 222
batch_size = 32
embedding_dims = 33
filters = 77
kernel_size = 3
hidden_dims = 250
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

word_to_id = imdb.get_word_index()

print(word_to_id)

word_to_id = {k:(v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[id] for id in x_train[0] ))

assert(0==1)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')

print(x_train.shape)
print(x_train[0:3])

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
print(x_train.shape)

print(x_train[0:3])

print()
print(y_train[0:3])

print(x_test.shape)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print(x_test.shape)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))

model.summary()

model.add(Dropout(0.2))

model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))

model.add(GlobalMaxPooling1D())

model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary() 

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

===

from __future__ import print_function

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout
from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model

from functools import reduce
import tarfile
import numpy as np
import re

def tokenize(sent):
    return [x.strip() for x in re.split(r'(\W+)?', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            if only_supporting:
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def get_stories(f, only_supporting=False, max_length=None):
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length]
    return data

def vectorize_stories(data):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return (pad_sequences(inputs, maxlen=story_maxlen),
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(answers))

try:
    path = get_file('babi-tasks-v1-2.tar.gz',
                    origin='https://s3.amazonaws.com/text-datasets/'
                           'babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2'
          '.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise

challenges = {
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_'
                                  'single-supporting-fact_{}.txt',
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_'
                                'two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
with tarfile.open(path) as tar:
    train_stories = get_stories(tar.extractfile(challenge.format('train')))
    print(len(train_stories))
    print(train_stories[-2])
    print("___")
    print(train_stories[-1])
    test_stories = get_stories(tar.extractfile(challenge.format('test')))

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories)

print(inputs_train[-1])
print(queries_train[-1])
print(answers_train[-1])

inputs_test, queries_test, answers_test = vectorize_stories(test_stories)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Compiling...')

input_sequence = Input((story_maxlen,))
question = Input((query_maxlen,))

input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=64))
input_encoder_m.add(Dropout(0.3))

input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=query_maxlen))
input_encoder_c.add(Dropout(0.3))

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=query_maxlen))
question_encoder.add(Dropout(0.3))

input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

print(input_encoded_m.shape)

match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

response = add([match, input_encoded_c])  
response = Permute((2, 1))(response)  

answer = concatenate([response, question_encoded])

answer = LSTM(32)(answer)  

answer = Dropout(0.3)(answer)
answer = Dense(vocab_size)(answer)  

answer = Activation('softmax')(answer)

model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

plot_model(model,  to_file='model.png',show_shapes=True)
assert(0==1)
e

model.fit([inputs_train, queries_train], answers_train,
          batch_size=512,
          epochs=120,
          validation_data=([inputs_test, queries_test], answers_test))

===

import tensorflow as tf 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout
from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model

from functools import reduce
import tarfile
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split

from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt

def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(8,5))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])

def get_model(i_0, i_1):
    m_0 = Sequential([
    Dense(id_label_all.shape[-1]*2, activation='linear', input_shape= i_0.shape[1:])

        ])

    m_1 = Sequential([
        Dense(id_label_all.shape[-1]*3,  activation='linear', input_shape= i_1.shape[1:]),
        Dense(id_label_all.shape[-1]*2, activation='linear'), 
        ])

    id_ans = response = add([m_0(i_0),m_1(i_1)])  

    id_ans = Dense(id_label_all.shape[-1])(id_ans)

    model =  Model([i_0, i_1], id_ans)

    return model

flag_train = 1

x_items = 80000
if not flag_train:
    x_items = int(x_items/100.0)

x_dim_train_0 = 6

id_train_all_0 = np.random.random([x_items, 3,x_dim_train_0] )    

id_train_all_1 = np.random.random([x_items, 3,int(x_dim_train_0/2.0)] )    

id_train_all_1_ = np.concatenate((id_train_all_1, id_train_all_1*2 + 1), axis=2)

id_label_all = id_train_all_0 + id_train_all_1_ 

id_label_all = np.concatenate((id_label_all, id_label_all), axis=2)

x_train_0, y_test_0, x_train_1, y_test_1, x_label, y_label =  train_test_split(id_train_all_0, id_train_all_1, id_label_all, test_size=0.2)

if not os.path.exists("./mm"):
    !mkdir  mm
    !ls mm

model = None

if flag_train:
    i_0 = Input(id_train_all_0.shape[1:])
    i_1 = Input(id_train_all_1.shape[1:])
    model = get_model(i_0, i_1)
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(
        [x_train_0,x_train_1],
        x_label,

        validation_data=([y_test_0,y_test_1], y_label),
        epochs=40,
        batch_size=512,
        verbose=0
    )

    plot_history( [ ('baseline', history) ], key="loss") 
    model.save('./mm/h5.h5')
else:
    model = tf.keras.models.load_model('./mm/h5.h5')

model.summary()   

batch_size = 3
s_i = np.random.choice(range(len(y_test_0)), batch_size)
y_predict =  model.predict([y_test_0[s_i], y_test_1[s_i]])

cnt = 0
print(s_i)

for i in s_i:
    print(y_label[i])
    print("___")
    print(y_predict[cnt])
    cnt = cnt + 1
    print()
    print()

===

import tensorflow as tf 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout
from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model

from functools import reduce
import tarfile
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split

from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt

id_a_np = np.random.random([1,2,2])
id_b_np = np.random.random([1,1,2])

id_a_np[0,0,0] = 1.0
id_a_np[0,0,1] = 0.1
id_a_np[0,1,0] = 2.0
id_a_np[0,1,1] = 0.2

id_b_np[0,0,0] = -3
id_b_np[0,0,1] = -7

id_a = Input(id_a_np.shape[1:])
id_b = Input(id_b_np.shape[1:])

id_ans = dot([id_a, id_b], axes=[2,2])

model = Model([id_a, id_b], id_ans)

model.summary()
model.compile(loss='mse', optimizer='adam')

print(id_a_np)
print()
print(id_b_np)

id_predict =  model.predict([id_a_np, id_b_np])

id_predict.shape

print()
print(id_predict)

===

import tensorflow as tf 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout, Conv2D, Flatten, Reshape
from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model

from functools import reduce
import tarfile
import numpy as np
import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

from IPython.display import display
from PIL import Image

from tensorflow.keras import backend as K

id_shape = [10000, 28,28,1]
x_train_all =  np.random.random(id_shape)
y_label_all =  x_train_all * 3.3 + 4.4

x_train, y_test, x_label, y_label = train_test_split(x_train_all, y_label_all, test_size=0.2)

model = Sequential(
[
    Conv2D(4, (3,3), padding="same", activation="relu", input_shape=x_train.shape[1:]), 
    Dense(1)
]
)

model.summary()

model.compile(loss='mse', optimizer='adam')

history = model.fit(
    x_train, x_label, 
    validation_data=(y_test, y_label),
    epochs=512,
    batch_size=1024,
    verbose=1
)

id_pd = pd.DataFrame(history.history)

id_pd["idx"] = np.arange(len(id_pd))

=== 

import tensorflow as tf 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout, Conv2D, Flatten, Reshape
from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist

from functools import reduce
import tarfile
import numpy as np
import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K

import pandas as pd
import scipy.stats

import math
from collections import Counter

from scipy import stats
import pandas as pd

from pprint import pprint

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

learning_rate = 0.001
training_iters = 20
batch_size = 1024
display_step = 10
n_input = 28
n_step = 28
n_hidden = 128
n_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, n_step, n_input)

pprint(x_train.shape)

x_test = x_test.reshape(-1, n_step, n_input)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)
model = Sequential()
model.add(LSTM(n_hidden,

               input_shape=x_train.shape[1:],
               unroll=True))
model.add(Dense(n_classes))
model.add(Activation('softmax'))
adam = Adam(lr=learning_rate)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary() 

plot_model(model,  to_file='model.png',show_shapes=True)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=training_iters,
          verbose=1,
          validation_data=(x_test, y_test))
scores = model.evaluate(x_test, y_test, verbose=0)
print('LSTM test score:', scores[0])
print('LSTM test accuracy:', scores[1])

import tensorflow as tf 

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout, Conv2D, Flatten, Reshape
from tensorflow.keras.layers import Conv1D

from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM,RNN
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist

from functools import reduce
import tarfile
import numpy as np
import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K

import pandas as pd
import scipy.stats

import math
from collections import Counter

from scipy import stats
import pandas as pd

from pprint import pprint

from tensorflow.keras.optimizers import Adam

x_item = 10000
x_dim = 3
y_dim = 2

x_train_all = np.random.random([x_item, x_dim])
x_label_all = np.random.random([x_item, y_dim])

for idx, e in enumerate(x_train_all):
    e_sum = e[0] + e[1]
    if (e_sum > 1.7):
        x_label_all[idx][0] = 0
        x_label_all[idx][-1] = 0
    elif (e_sum > 1.1):
        x_label_all[idx][0] = 0
        x_label_all[idx][-1] = 1
    elif (e_sum > 0.6):
        x_label_all[idx][0] = 1
        x_label_all[idx][-1] = 0
    else:
        x_label_all[idx][0] = 1
        x_label_all[idx][-1] = 1        

x_train, y_test, x_label, y_label = train_test_split(x_train_all, x_label_all, test_size=0.1)

model = Sequential(
[
   Dense(128, input_shape=x_train.shape[1:]),
    Dropout(0.224),
    Dense(32 ),
    Dense(y_dim, activation='softmax')
]
)
model.compile(loss='mse',optimizer=Adam(), metrics=['acc'])

model.summary()

model.fit(
x_train, x_label,
    validation_data=(y_test, y_label),
    batch_size=1024,
    epochs=15
)

y_p = model.predict(y_test[0:11])

print(y_p)
print()
print(y_label[:11])

import tensorflow as tf
tf.__version__

import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

%matplotlib inline

!ls

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets( "MNIST_data/", one_hot=1)

img = mnist.train.images[50]
plt.imshow(img.reshape((28, 28)), cmap='Greys_r')

def get_inputs(real_size, noise_size):
    real_img = tf.placeholder(tf.float32, [None, real_size], name='real_img')
    noise_img = tf.placeholder(tf.float32, [None, noise_size], name='noise_img')
    return real_img, noise_img

def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    with tf.variable_scope("generator", reuse=reuse):
        hidden1 = tf.layers.dense(noise_img, n_units)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        hidden1 = tf.layers.dropout(hidden1, rate=0.2)

        logits = tf.layers.dense(hidden1, out_dim)
        outputs = tf.tanh(logits)
        return logits, outputs
def get_discriminator(img, n_units, reuse=False, alpha=0.01):
    with tf.variable_scope("discriminator", reuse=reuse):
        hidden1 = tf.layers.dense(img, n_units)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        logits = tf.layers.dense(hidden1, 1)
        outputs = tf.sigmoid(logits)
        return logits, outputs    

img_size = mnist.train.images[0].shape[0]

noise_size = 100

g_units = 128

d_units = 128

alpha = 0.01

learning_rate = 0.001

smooth = 0.1

tf.reset_default_graph()

real_img, noise_img = get_inputs(img_size, noise_size)

g_logits, g_outputs = get_generator(noise_img, g_units, img_size)

d_logits_real, d_outputs_real = get_discriminator(real_img, d_units)
d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, d_units, reuse=True)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, 
                                                                     labels=tf.ones_like(d_logits_real)) * (1 - smooth))

d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                                     labels=tf.zeros_like(d_logits_fake)))

d_loss = tf.add(d_loss_real, d_loss_fake)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                labels=tf.ones_like(d_logits_fake)) * (1 - smooth))

train_vars = tf.trainable_variables()

g_vars = [var for var in train_vars if var.name.startswith("generator")]

d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

batch_size = 64

epochs = 300

n_sample = 25

samples = []

losses = []

saver = tf.train.Saver(var_list = g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for batch_i in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images*2 - 1
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
            _ = sess.run(d_train_opt, feed_dict={real_img: batch_images, noise_img: batch_noise})
            _ = sess.run(g_train_opt, feed_dict={noise_img: batch_noise})
        train_loss_d = sess.run(d_loss, 
                                feed_dict = {real_img: batch_images, 
                                             noise_img: batch_noise})
        train_loss_d_real = sess.run(d_loss_real, 
                                     feed_dict = {real_img: batch_images, 
                                                 noise_img: batch_noise})
        train_loss_d_fake = sess.run(d_loss_fake, 
                                    feed_dict = {real_img: batch_images, 
                                                 noise_img: batch_noise})
        train_loss_g = sess.run(g_loss, 
                                feed_dict = {noise_img: batch_noise})
        print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
              "Generator Loss: {:.4f}".format(train_loss_g))    
        losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))
        sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
        gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),
                               feed_dict={noise_img: sample_noise})
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')

with open('train_samples.pkl', 'wb') as f:
    pickle.dump(samples, f)

fig, ax = plt.subplots(figsize=(20,7))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator Total Loss')
plt.plot(losses.T[1], label='Discriminator Real Loss')
plt.plot(losses.T[2], label='Discriminator Fake Loss')
plt.plot(losses.T[3], label='Generator')
plt.title("Training Losses")
plt.legend()

with open('train_samples.pkl', 'rb') as f:
    samples = pickle.load(f)

def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch][1]): 
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    return fig, axes

_ = view_samples(-1, samples) 

epoch_idx = [0, 5, 10, 20, 40, 60, 80, 100, 150, 250] 
show_imgs = []
for i in epoch_idx:
    show_imgs.append(samples[i][1])

rows, cols = 10, 25
fig, axes = plt.subplots(figsize=(30,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

idx = range(0, epochs, int(epochs/rows))

for sample, ax_row in zip(show_imgs, axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    sample_noise = np.random.uniform(-1, 1, size=(25, noise_size))
    gen_samples = sess.run(get_generator(noise_img, g_units, img_size, reuse=True),
                           feed_dict={noise_img: sample_noise})

_ = view_samples(0, [gen_samples])    
EOF
