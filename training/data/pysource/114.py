

from numpy import loadtxt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils

from keras.wrappers.scikit_learn import KerasClassifier 
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from keras.optimizers        import SGD

from sklearn.preprocessing   import LabelEncoder
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from symbols import DATAPATH, MODELPATH
from util import get_starttime, calc_runtime

def first_nn():
    fnm = f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = loadtxt(fnm, delimiter=',')

    X = dataset[:,0:8]
    y = dataset[:,8]

    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

    model.fit(X, y, epochs=150, batch_size=16)

    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))

def keras_auto_cv():
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",") 

    X = dataset[:,0:8]
    Y = dataset[:,8]
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)

def keras_manual_cv():
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",") 

    X = dataset[:,0:8]
    Y = dataset[:,8]

    kfold = StratifiedKFold(n_splits=10, shuffle=True) 
    cvscores = []
    for train, test in kfold.split(X, Y):

        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) 
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

def scikit_auto_cv():

    def create_model():
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu')) 
        model.add(Dense(8, activation='relu')) 
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
        return model

    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",") 
    X = dataset[:,0:8]
    Y = dataset[:,8]

    model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True) 
    results = cross_val_score(model, X, Y, cv=kfold) 
    print(results.mean())

def scikit_grid_search():
    def create_model(optimizer='rmsprop', init='glorot_uniform'):
        model = Sequential()
        model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu')) 
        model.add(Dense(8, kernel_initializer=init, activation='relu'))
        model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) 
        return model

    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",")  
    X = dataset[:,0:8]
    Y = dataset[:,8]
    model = KerasClassifier(build_fn=create_model, verbose=0)
    optimizers = ['rmsprop', 'adam']
    inits = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 150]
    batches = [5, 10, 20]
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits) 
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X, Y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def iris_multiclass():
    fnm=f'{DATAPATH}iris.csv'
    dataframe = pd.read_csv(fnm)
    cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm" ] 
    target = "Species"
    X = dataframe[cols].astype(float)
    Y = dataframe[target] 
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)

    def baseline_model():
        model = Sequential()
        model.add(Dense(8, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
        return model
    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0) 
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def sonar_classification():
    fnm=f'{DATAPATH}/sonar.csv'
    dataframe = pd.read_csv(fnm, header=None)
    dataset = dataframe.values
    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    def create_baseline():
        model = Sequential()
        model.add(Dense(60, input_dim=60, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
        return model
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100,
                        batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

    print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def small_sonar_classification():
    fnm=f'{DATAPATH}/sonar.csv'
    dataframe = pd.read_csv(fnm, header=None)
    dataset = dataframe.values
    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    def create_baseline():
        model = Sequential()
        model.add(Dense(30, input_dim=60, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
        return model
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100,
                        batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

    print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def large_sonar_classification():
    fnm=f'{DATAPATH}/sonar.csv'
    dataframe = pd.read_csv(fnm, header=None)
    dataset = dataframe.values
    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    def create_baseline():
        model = Sequential()
        model.add(Dense(60, input_dim=60, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
        return model
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100,
                        batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

    print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def grid_sonar_classification():
    fnm=f'{DATAPATH}/sonar.csv'
    dataframe = pd.read_csv(fnm, header=None)
    dataset = dataframe.values
    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    def create_baseline(optimizer='adam', init='glorot_uniform'):
        model = Sequential()
        model.add(Dense(30, input_dim=60, kernel_initializer=init, activation='relu'))
        model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) 
        return model
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)

    optimizers = ['rmsprop', 'adam']
    inits = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 150]
    batches = [1, 5, 10, 20]
    param_grid = dict(mlp__optimizer=optimizers, 
                      mlp__epochs=epochs, 
                      mlp__batch_size=batches, 
                      mlp__init=inits) 
    grid = GridSearchCV(estimator=pipeline, n_jobs=-1, param_grid=param_grid, cv=kfold)
    print(f'pipeline.get_params().keys()={pipeline.get_params().keys()}')
    grid_result = grid.fit(X, Y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))  

def boston_base_regression():

    fnm=f'{DATAPATH}boston.csv'
    dataframe = pd.read_csv(fnm, delim_whitespace=True)
    dataset = dataframe.values

    X = dataset[:,0:13]
    Y = dataset[:,13]

    def baseline_model():
        model = Sequential()
        model.add(Dense(13, input_dim=13, activation='relu')) 
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam') 
        return model
    estimator = KerasRegressor(build_fn=baseline_model, 
                               epochs=100, batch_size=5, verbose=0) 
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, X, Y, cv=kfold, n_jobs=-1)
    print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def boston_standardize_regression():
    fnm=f'{DATAPATH}boston.csv'
    dataframe = pd.read_csv(fnm, delim_whitespace=True)
    dataset = dataframe.values
    X = dataset[:,0:13]
    Y = dataset[:,13]
    def baseline_model():
        model = Sequential()
        model.add(Dense(13, input_dim=13, activation='relu')) 
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam') 
        return model
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, 
                        epochs=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=-1)
    print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def boston_standardize_regression_deep():
    fnm=f'{DATAPATH}boston.csv'
    dataframe = pd.read_csv(fnm, delim_whitespace=True)
    dataset = dataframe.values
    X = dataset[:,0:13]
    Y = dataset[:,13]

    def larger_model():
        model = Sequential()
        model.add(Dense(13, input_dim=13, activation='relu')) 
        model.add(Dense(6, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam') 
        return model

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=larger_model, 
                       epochs=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=-1)
    print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def boston_standardize_regression_wide():
    fnm=f'{DATAPATH}boston.csv'
    dataframe = pd.read_csv(fnm, delim_whitespace=True)
    dataset = dataframe.values
    X = dataset[:,0:13]
    Y = dataset[:,13]

    def wider_model():
        model = Sequential()
        model.add(Dense(20, input_dim=13, activation='relu')) 
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam') 
        return model

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=wider_model, 
                       epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=-1)
    print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def boston_standardize_regression_mixed():
    fnm=f'{DATAPATH}boston.csv'
    dataframe = pd.read_csv(fnm, delim_whitespace=True)
    dataset = dataframe.values
    X = dataset[:,0:13]
    Y = dataset[:,13]

    def wider_model():
        model = Sequential()
        model.add(Dense(20, input_dim=13, activation='relu'))
        model.add(Dense(6, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam') 
        return model

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=wider_model, 
                       epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold, n_jobs=-1)
    print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def save_model_file():
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",") 

    X = dataset[:,0:8]
    Y = dataset[:,8]
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) 
    fnm = f'{MODELPATH}model.h5'
    model.save(fnm)
    print("Saved model to disk")

def load_model_file():
    fnm = f'{MODELPATH}model.h5'
    model = load_model(fnm) 
    model.summary()
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",")
    X = dataset[:,0:8]
    Y = dataset[:,8]
    score = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

def checkpoint_model_improvements():
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",") 

    X = dataset[:,0:8]
    Y = dataset[:,8]

    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

    filepath = f'{MODELPATH}'+'weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, 
            callbacks=callbacks_list, verbose=0)    

def checkpoint_best_model_only():
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",") 

    X = dataset[:,0:8]
    Y = dataset[:,8]
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    filepath=f'{MODELPATH}'+'weights.best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max') 
    callbacks_list = [checkpoint]
    model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, 
              callbacks=callbacks_list, verbose=0)

def plot_history():
    fnm=f'{DATAPATH}pima-indians-diabetes.csv'
    dataset = np.loadtxt(fnm, delimiter=",")
    X = dataset[:,0:8]
    Y = dataset[:,8]

    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)

    print(history.history.keys())

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'][10:]) 
    plt.plot(history.history['val_loss'][10:]) 
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left') 
    plt.show()

def no_dropout():
    fnm=f'{DATAPATH}/sonar.csv'
    dataframe = pd.read_csv(fnm, header=None)
    dataset = dataframe.values

    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    def create_baseline():
        model = Sequential()
        model.add(Dense(60, input_dim=60, activation='relu')) 
        model.add(Dense(30, activation='relu')) 
        model.add(Dense(1, activation='sigmoid'))

        sgd = SGD(lr=0.01, momentum=0.8)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 
        return model

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=300,
                    batch_size=16, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, X, encoded_Y, cv=kfold, n_jobs=-1)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def dropout_input_only():
    fnm=f'{DATAPATH}/sonar.csv'
    dataframe = pd.read_csv(fnm, header=None)
    dataset = dataframe.values

    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    def create_model():
        model = Sequential()
        model.add(Dropout(0.2, input_shape=(60,)))
        model.add(Dense(60, activation='relu', kernel_constraint=maxnorm(3))) 
        model.add(Dense(30, activation='relu', kernel_constraint=maxnorm(3))) 
        model.add(Dense(1, activation='sigmoid'))

        sgd = SGD(lr=0.1, momentum=0.9)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 
        return model

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=300, 
                       batch_size=16, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, X, encoded_Y, cv=kfold, n_jobs=-1)
    print("Visible: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def dropout_all_layers():
    fnm=f'{DATAPATH}/sonar.csv'
    dataframe = pd.read_csv(fnm, header=None) 
    dataset = dataframe.values

    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]

    encoder = LabelEncoder() 
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    def create_model():
        model = Sequential()
        model.add(Dense(60, input_dim=60, activation='relu', kernel_constraint=maxnorm(3))) 
        model.add(Dropout(0.2))
        model.add(Dense(30, activation='relu', kernel_constraint=maxnorm(3))) 
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        sgd = SGD(lr=0.1, momentum=0.9)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 
        return model

    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=300, 
                       batch_size=16, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, X, encoded_Y, cv=kfold, n_jobs=-1)
    print("Hidden: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def lr_schedule_time_based():
    fnm=f'{DATAPATH}/ionosphere.csv'
    dataframe = pd.read_csv(fnm, header=None)
    dataset = dataframe.values

    X = dataset[:,0:34].astype(float)
    Y = dataset[:,34]

    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)

    model = Sequential()
    model.add(Dense(34, input_dim=34, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    epochs = 50
    learning_rate = 0.1
    decay_rate = learning_rate / epochs
    momentum = 0.8
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False) 
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=28, verbose=2)

def lr_schedule_drop_based():
    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop)) 
        return lrate

    fnm=f'{DATAPATH}/ionosphere.csv'
    dataframe = pd.read_csv(fnm, header=None) 
    dataset = dataframe.values

    X = dataset[:,0:34].astype(float)
    Y = dataset[:,34]

    encoder = LabelEncoder() 
    encoder.fit(Y)
    Y = encoder.transform(Y) 
    model = Sequential()
    model.add(Dense(34, input_dim=34, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    sgd = SGD(lr=0.0, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]
    model.fit(X, Y, validation_split=0.33, epochs=50, batch_size=28, 
            callbacks=callbacks_list, verbose=2)

def print_runtime(func):
    start = get_starttime()
    func()
    calc_runtime(start, True)

if __name__ == "__main__":
    funcs_to_run = [ lr_schedule_time_based, lr_schedule_drop_based
                ]

    for i, f in enumerate(funcs_to_run):
        stars="*"*80
        print(f'{stars}\n{stars}')
        print(f'run={i} function={f}\n')
        print_runtime(f)
        print(f'{stars}\n{stars}\n')
EOF
