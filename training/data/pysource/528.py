
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from scipy import interp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import os
import dill
from .entropy import resample
from sklearn import tree
from fractions import Fraction
from collections import namedtuple

Keras_Input_Dense_Parameters = namedtuple('Keras_Input_Dense_Parameters', ['kernel_initializer', 'activation'])
Keras_InputLayer_Parameters = namedtuple('Keras_InputLayer_Parameters', ['notused'])
Keras_Output_Dense_Parameters = namedtuple('Keras_Output_Dense_Parameters', ['kernel_initializer', 'activation'])
Keras_Dense_Parameters = namedtuple('Keras_Dense_Parameters', ['value', 'kernel_initializer', 'activation'])
Keras_Dropout_Parameters = namedtuple('Keras_Dropout_Parameters', 'rate')
Keras_Flatten_Parameters = namedtuple('Keras_Flatten_Parameters', [])
Keras_Conv1D_Parameters = namedtuple('Keras_Conv1D_Parameters', ['filters', 'kernel_size', 'activation', 'pool_size'])

class ML(object):
    def __init__(self, feature_type='rwe', classifier_type='dt', n_classes=6, rwe_windowsize=None, datapoints=None,
                 nnlayers=None, nnoptimizer="adam", *args, **kwargs):
        super(ML, self).__init__()
        self.classifer = None
        self.classifier_type = classifier_type
        if self.classifier_type:
            self.classifier_type = self.classifier_type.lower()
        self.n_classes = n_classes
        self.classifiers = None
        self.X_sc = None
        self.y_labelencoder = None
        self.rwe_windowsize = rwe_windowsize
        self.datapoints = datapoints
        self.nnlayers = nnlayers
        self.nnoptimizer = nnoptimizer
        self.feature_type = feature_type

    def train(self, *args, **kwargs):
        if self.classifier_type == 'ann' or self.classifier_type == 'cnn':
            return self.train_nn(*args, **kwargs)
        else:
            return self.train_scikitlearn(*args, **kwargs)

    def predict(self, *args, **kwargs):
        if self.classifier_type == 'ann' or self.classifier_type == 'cnn':
            return self.predict_nn(*args, **kwargs)
        else:
            return self.predict_scikitlearn(*args, **kwargs)

    def predict_sample(self, sample, *args, **kwargs):
        if self.feature_type == 'rwe':
            ds1 = sample.running_window_entropy(self.rwe_windowsize)
            ds2 = pd.Series(resample(ds1, self.datapoints))
            ds2.name = ds1.name
            rwe = pd.DataFrame([ds2])
            rwe, _ = self.scale_features(rwe.values)
            y = self.decode_classifications(self.predict(rwe))
            return y[0]
        elif self.feature_type == 'gist':
            ds1 = sample.gist_data
            gist = pd.DataFrame([ds1])
            gist, _ = self.scale_features(gist.values)
            y = self.decode_classifications(self.predict(gist))
            return y[0]

    def save_classifier(self, directory):
        if self.classifier_type == 'gridsearch':
            raise TypeError("Gridsearch model saving is not supported!")
        if self.classifier_type == 'dt':
            tree.export_graphviz(self.classifier, out_file=os.path.join(directory, 'tree.dot'))
        if self.classifier_type in ['ann', 'cnn']:
            with open(os.path.join(directory, "nn.json"), 'w') as file:
                file.write(self.classifier.to_json())
            self.classifier.save_weights(os.path.join(directory, 'nn.h5'))
            self.classifier = None
            self.classifiers = None
            self.nnlayers = None
        with open(os.path.join(directory, "ml.dill"), 'wb') as file:
            dill.dump(self, file)

    @staticmethod
    def load_classifier(directory):
        from keras.models import model_from_json
        with open(os.path.join(directory, "ml.dill"), 'rb') as file:
            myself = dill.load(file)
        if myself.classifier_type in ['ann', 'cnn']:
            with open(os.path.join(directory, "nn.json"), 'r') as file:
                myself.classifier = model_from_json(file.read())
            myself.classifier.load_weights(os.path.join(directory, 'nn.h5'))
        return myself

    def preprocess_data(self, X, y):
        y_out, y_encoder = self.encode_classifications(y)
        X_out, X_scaler = self.scale_features(X)
        return X_out, y_out

    @staticmethod
    def build_gridsearch_static(*args, **kwargs):
        classifier = GridSearchCV(*args, **kwargs)
        return classifier

    def build_gridsearch(self, gridsearch_type, *args, **kwargs):
        self.classifier_type = 'gridsearch'
        self.base_classifier_type = gridsearch_type.lower()
        self.classifier = ML.build_gridsearch_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_ovr_static(*args, **kwargs):
        classifier = OneVsRestClassifier(*args, **kwargs)
        return classifier

    def build_ovr(self, ovr_type, *args, **kwargs):
        self.classifier_type = 'ovr'
        self.base_classifier_type = ovr_type.lower()
        self.classifier = ML.build_ovr_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_adaboost_static(*args, **kwargs):
        classifier = AdaBoostClassifier(*args, **kwargs)
        return classifier

    def build_adaboost(self, adaboost_type, *args, **kwargs):
        self.classifier_type = 'adaboost'
        self.base_classifier_type = adaboost_type.lower()
        self.classifier = ML.build_adaboost_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_nc_static(*args, **kwargs):
        classifier = NearestCentroid(*args, **kwargs)
        return classifier

    def build_nc(self, *args, **kwargs):
        self.classifier_type = 'nc'
        self.classifier = ML.build_nc_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_nb_static(*args, **kwargs):
        classifier = GaussianNB(*args, **kwargs)
        return classifier

    def build_nb(self, *args, **kwargs):
        self.classifier_type = 'nb'
        self.classifier = ML.build_nb_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_knn_static(*args, **kwargs):
        classifier = KNeighborsClassifier(*args, **kwargs)
        return classifier

    def build_knn(self, *args, **kwargs):
        self.classifier_type = 'knn'
        self.classifier = ML.build_knn_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_rf_static(*args, **kwargs):
        classifier = RandomForestClassifier(*args, **kwargs)
        return classifier

    def build_rf(self, *args, **kwargs):
        self.classifier_type = 'rf'
        self.classifier = ML.build_rf_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_dt_static(*args, **kwargs):
        classifier = DecisionTreeClassifier(*args, **kwargs)
        return classifier

    def build_dt(self, *args, **kwargs):
        self.classifier_type = 'dt'
        self.classifier = ML.build_dt_static(*args, **kwargs)
        return self.classifier

    @staticmethod
    def build_svm_static(*args, **kwargs):
        classifier = SVC(*args, **kwargs)
        return classifier

    def build_svm(self, *args, **kwargs):
        self.classifier_type = 'svm'
        self.classifier = ML.build_svm_static(*args, **kwargs)
        return self.classifier

    def train_scikitlearn(self, X, y):
        if self.classifier_type in ['svm', 'nb', 'nc', 'adaboost']:
            Y = y.argmax(1)
        elif self.classifier_type in ['gridsearch']:
            if self.base_classifier_type in ['svm', 'nb', 'nc', 'adaboost']:
                Y = y.argmax(1)
            else:
                Y = y
        else:
            Y = y
        self.classifier.fit(X, Y)
        return self.classifier

    def predict_scikitlearn(self, X):
        return self.classifier.predict(X)

    @staticmethod
    def build_ann_static(X, y, layers=[
        Keras_Input_Dense_Parameters(kernel_initializer='uniform', activation='relu'),
        Keras_Dense_Parameters(value="1/2", kernel_initializer='uniform', activation='relu'),
        Keras_Dense_Parameters(value=100, kernel_initializer='uniform', activation='relu'),
        Keras_Output_Dense_Parameters(kernel_initializer='uniform', activation='softmax')
    ], optimizer='adam'):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from keras import optimizers
        datapoints = X.shape[1]
        output_shape = y.shape[1]
        classifier = Sequential()

        for layer in layers:
            if isinstance(layer, Keras_Dense_Parameters):
                value, kernel_initializer, activation = layer.value, layer.kernel_initializer, layer.activation
                if isinstance(value, int):
                    units = value
                else:
                    units = int(Fraction(value) * datapoints)
                classifier.add(Dense(units=units,
                                     kernel_initializer=kernel_initializer,
                                     activation=activation))
            elif isinstance(layer, Keras_Dropout_Parameters):
                rate = layer.rate
                classifier.add(Dropout(rate))
            elif isinstance(layer, Keras_Input_Dense_Parameters):
                kernel_initializer, activation = layer.kernel_initializer, layer.activation
                classifier.add(Dense(units=datapoints, input_dim=datapoints,
                                     kernel_initializer=kernel_initializer,
                                     activation=activation))
            elif isinstance(layer, Keras_Output_Dense_Parameters):
                kernel_initializer, activation = layer.kernel_initializer, layer.activation
                classifier.add(Dense(units=output_shape,
                                     kernel_initializer=kernel_initializer,
                                     activation=activation))
            else:
                raise ValueError('Invalid layer type!')

        classifier.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy', 'accuracy'])
        return classifier

    def build_ann(self, X, y):
        self.classifier_type = 'ann'
        self.classifier = ML.build_ann_static(X, y, layers=self.nnlayers, optimizer=self.nnoptimizer)
        self.classifier.summary()
        return self.classifier

    @staticmethod
    def build_cnn_static(X, y, layers=[
        Keras_InputLayer_Parameters(notused=None),
        Keras_Conv1D_Parameters(filters=10, kernel_size="1/4", activation="relu", pool_size=10),
        Keras_Conv1D_Parameters(filters=10, kernel_size="1/30", activation="relu", pool_size=2),
        Keras_Conv1D_Parameters(filters=10, kernel_size=2, activation='relu', pool_size=2),
        Keras_Flatten_Parameters(),
        Keras_Dense_Parameters(value="1/4", kernel_initializer='uniform', activation='relu'),
        Keras_Dense_Parameters(value="1/8", kernel_initializer='uniform', activation='relu'),
        Keras_Dense_Parameters(value="1/16", kernel_initializer='uniform', activation='relu'),
        Keras_Output_Dense_Parameters(kernel_initializer='uniform', activation='softmax')
    ], optimizer='adam'):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D, InputLayer
        from keras import optimizers
        datapoints = X.shape[1:]
        input_dim = datapoints[0]
        output_shape = y.shape[1]
        classifier = Sequential()

        for layer in layers:
            if isinstance(layer, Keras_Flatten_Parameters):
                classifier.add(Flatten())
            elif isinstance(layer, Keras_Conv1D_Parameters):
                filters, kernel_size, activation, pool_size = layer.filters, layer.kernel_size, \
                                                              layer.activation, layer.pool_size
                if not isinstance(filters, int):
                    filters = int(Fraction(filters) * input_dim)
                if not isinstance(kernel_size, int):
                    kernel_size = int(Fraction(kernel_size) * input_dim)
                if not isinstance(pool_size, int):
                    pool_size = int(Fraction(pool_size) * input_dim)

                classifier.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
                classifier.add(MaxPooling1D(pool_size=pool_size))
            elif isinstance(layer, Keras_Dense_Parameters):
                value, kernel_initializer, activation = layer.value, layer.kernel_initializer, layer.activation
                if isinstance(value, int):
                    units = value
                else:
                    units = int(Fraction(value) * input_dim)

                classifier.add(Dense(units=units,
                                     kernel_initializer=kernel_initializer,
                                     activation=activation))
            elif isinstance(layer, Keras_Dropout_Parameters):
                rate = layer.rate
                classifier.add(Dropout(rate))
            elif isinstance(layer, Keras_InputLayer_Parameters):
                classifier.add(InputLayer(input_shape=datapoints))
            elif isinstance(layer, Keras_Output_Dense_Parameters):
                kernel_initializer, activation = layer.kernel_initializer, layer.activation
                classifier.add(Dense(units=output_shape,
                                     kernel_initializer=kernel_initializer,
                                     activation=activation))
            else:
                raise ValueError('Invalid layer type!')

        classifier.compile(optimizer=optimizer, loss='categorical_crossentropy',
                           metrics=['categorical_accuracy', 'accuracy'])
        return classifier

    def build_cnn(self, X, y):
        if len(X.shape) != 3:
            X = np.expand_dims(X, axis=2)
        else:
            X = X
        self.classifier_type = 'cnn'
        self.classifier = ML.build_cnn_static(X, y, layers=self.nnlayers, optimizer=self.nnoptimizer)
        self.classifier.summary()
        return self.classifier

    def train_nn(self, X_train, y_train,
                 batch_size=50, epochs=100,
                 tensorboard=False, verbose=2):
        import keras.callbacks
        if len(X_train.shape) != 3 and self.classifier_type == 'cnn':
            X_in = np.expand_dims(X_train, axis=2)
        else:
            X_in = X_train
        if tensorboard is True:
            tb = keras.callbacks.TensorBoard(log_dir='Graph',
                                             histogram_freq=0,
                                             write_grads=True,
                                             write_graph=True,
                                             write_images=True)
            self.classifier.fit(X_in, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                callbacks=[tb],
                                verbose=verbose)
        else:
            self.classifier.fit(X_in, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=verbose)
        return self.classifier

    def predict_nn(self, X_test):
        if len(X_test.shape) != 3 and self.classifier_type == 'cnn':
            X_in = np.expand_dims(X_test, axis=2)
        else:
            X_in = X_test
        y_pred = self.classifier.predict(X_in)
        for i in range(0, len(y_pred)):
            row = y_pred[i]
            row[row == row.max()] = 1
            row[row < row.max()] = 0
            y_pred[i] = row
        return y_pred

    @staticmethod
    def confusion_matrix(y_test, y_pred):
        if isinstance(y_test, list):
            y_test = np.array(y_test)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            yp = column_or_1d(y_pred.argmax(1)).tolist()
        else:
            yp = column_or_1d(y_pred).tolist()
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            yt = column_or_1d(y_test.argmax(1)).tolist()
        else:
            yt = column_or_1d(y_test).tolist()
        cm = confusion_matrix(yt, yp)
        return ML._calculate_confusion_matrix(cm)

    @staticmethod
    def _calculate_confusion_matrix(cm):
        accuracy = 0.
        for i in range(0, len(cm)):
            accuracy += cm[i, i]
        accuracy = accuracy/cm.sum()
        return accuracy, cm

    def scale_features(self, X):
        if self.X_sc is None:
            self.X_sc = StandardScaler()
            X_scaled = self.X_sc.fit_transform(X)
        else:
            X_scaled = self.X_sc.transform(X)
        return X_scaled, self.X_sc

    def encode_classifications(self, y):
        if self.y_labelencoder is None:
            self.y_labelencoder = LabelEncoder()
            Y = self.y_labelencoder.fit_transform(y[:, 0])
        else:
            Y = self.y_labelencoder.transform(y)
        Y = label_binarize(Y, classes=range(self.n_classes))
        return Y, self.y_labelencoder

    def decode_classifications(self, y):
        if self.y_labelencoder is not None:
            y = np.argmax(y, axis=1)
            y_out = self.y_labelencoder.inverse_transform(y)
            return y_out
        else:
            return None

    @staticmethod
    def train_test_split(X, y, test_percent=0.2, random_state=0):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_percent,
                                                            random_state=random_state,
                                                            stratify=y)
        return X_train, X_test, y_train, y_test

    def cross_fold_validation(self, X, y, cv=10, n_jobs=10, batch_size=None, epochs=None):
        cvkfold = StratifiedKFold(n_splits=cv)

        Y = y.argmax(1)

        fold = 0
        saved_futures = {}
        classifiers = {}
        print("Start Cross Fold Validation...")

        if n_jobs < 2:
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=n_jobs)

        for train, test in cvkfold.split(X, Y):
            X_train = X[train]
            X_test = X[test]
            y_train = y[train]
            y_test = y[test]
            fold += 1
            print("\tCalculating fold: {0}".format(fold))
            if executor:
                future = executor.submit(self._cfv_runner,
                                         X_train, y_train,
                                         X_test, y_test,
                                         batch_size=batch_size, epochs=epochs, verbose=2)
                saved_futures[future] = fold
                if len(saved_futures) >= n_jobs:
                    keystodel = []
                    futures = wait(saved_futures, return_when=FIRST_COMPLETED)
                    for future in futures.done:
                        print("\tFinished calculating fold: {0}".format(saved_futures[future]))
                        result_dict = future.result()
                        print("\tAccuracy {0} for fold {1}".format(result_dict['accuracy'], saved_futures[future]))
                        classifiers[saved_futures[future]] = result_dict
                        keystodel.append(future)
                    for key in keystodel:
                        del saved_futures[key]
            else:
                result_dict = self._cfv_runner(X_train, y_train, X_test, y_test, batch_size=batch_size,
                                               epochs=epochs, verbose=2)
                print("\tFinished calculating fold: {0}".format(fold))
                print("\tAccuracy {0} for fold {1}".format(result_dict['accuracy'], fold))
                classifiers[fold] = result_dict

        if executor:
            for future in as_completed(saved_futures):
                print("\tFinished calculating fold: {0}".format(saved_futures[future]))
                result_dict = future.result()
                print("\tAccuracy {0} for fold {1}".format(result_dict['accuracy'], saved_futures[future]))
                classifiers[saved_futures[future]] = result_dict
            executor.shutdown(wait=True)

        self.classifiers = classifiers
        accuracies = np.array([classifiers[f]['accuracy'] for f in classifiers])
        mean = accuracies.mean()
        variance = accuracies.std()
        return mean, variance, classifiers, accuracies

    def _cfv_runner(self, X_train, y_train, X_test, y_test, batch_size=None, epochs=None, verbose=0, **kwargs):
        if self.classifier_type in ['ann', 'cnn']:
            from keras.wrappers.scikit_learn import KerasClassifier
            if self.classifier_type == 'cnn':
                X_train_in = np.expand_dims(X_train, axis=2)
            else:
                X_train_in = X_train

            def create_model():
                if self.classifier_type == 'cnn':
                    if self.nnlayers:
                        return ML.build_cnn_static(X_train_in, label_binarize(y_train, classes=range(self.n_classes)),
                                                   layers=self.nnlayers, optimizer=self.nnoptimizer)
                    else:
                        return ML.build_cnn_static(X_train_in, label_binarize(y_train, classes=range(self.n_classes)),
                                                   optimizer=self.nnoptimizer)
                else:
                    if self.nnlayers:
                        return ML.build_ann_static(X_train, label_binarize(y_train, classes=range(self.n_classes)),
                                                   layers=self.nnlayers, optimizer=self.nnoptimizer)
                    else:
                        return ML.build_ann_static(X_train, label_binarize(y_train, classes=range(self.n_classes)),
                                                   optimizer=self.nnoptimizer)
            classifier = KerasClassifier(build_fn=create_model,
                                         batch_size=batch_size,
                                         epochs=epochs)
            classifier.fit(X_train_in, label_binarize(y_train, classes=range(self.n_classes)),
                           batch_size=batch_size, epochs=epochs, verbose=verbose, **kwargs)
        else:
            classifier = self.classifier
            if self.classifier_type in ['svm', 'nb', 'nc', 'adaboost']:
                Y = y_train.argmax(1)
            else:
                Y = y_train
            classifier.fit(X_train, Y, **kwargs)
        if self.classifier_type == 'cnn':
            X_test_in = np.expand_dims(X_test, axis=2)
            y_pred = classifier.predict(X_test_in, **kwargs)
        else:
            y_pred = classifier.predict(X_test, **kwargs)
        accuracy, cm = ML.confusion_matrix(y_test, y_pred)
        return_dict = {'classifier': classifier, 'cm': cm, 'accuracy': accuracy,
                       'y_test': np.array(y_test),
                       'y_pred': np.array(y_pred),
                       'type': self.classifier_type}
        if self.classifier_type in ['ann', 'cnn']:
            classifier_dict = {}
            classifier_dict['json'] = classifier.model.to_json()
            classifier_dict['weights'] = classifier.model.get_weights()
            return_dict['classifier'] = classifier_dict
        return return_dict

    def set_classifier_by_fold(self, fold):
        from keras.models import model_from_json
        if self.classifiers:
            if self.classifiers[fold]['type'] == 'keras':
                self.classifier = model_from_json(self.classifiers[fold]['classifier']['json'])
                self.classifier.set_weights(self.classifiers[fold]['classifier']['weights'])
            else:
                self.classifier = self.classifiers[fold]['classifier']
        else:
            raise AttributeError("Must use CFV before there are classifiers to set.")

    def plot_roc_curves(self, y_test, y_pred, fold=None, filename=None):
        if isinstance(y_test, list):
            y_test = np.array(y_test)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)

        yt = y_test
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            yp = y_pred
        else:
            yp = label_binarize(y_pred.tolist(), classes=range(self.n_classes))
        fpr = dict()
        tpr = dict()
        thresholds = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], thresholds[i] = roc_curve(yt[:, i], yp[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(yt.ravel(), yp.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= self.n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.figure()
        lw = 2
        for i, color in zip(range(self.n_classes),
                            ['aqua', 'darkorange', 'cornflowerblue', 'red',
                             'green', 'yellow']):
            cn = label_binarize([i], classes=range(self.n_classes))
            class_name = self.decode_classifications(cn)[0]
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve for class {0} (area = {1:0.2f})'.format(class_name, roc_auc[i]))
        plt.plot(fpr["micro"], tpr["micro"], color='darkmagenta',
                 lw=lw, label='Micro ROC curve (area = {0:2f})'.format(roc_auc["micro"]))
        plt.plot(fpr["macro"], tpr["macro"], color='darkorange',
                 lw=lw, label='Macro ROC curve (area = {0:2f})'.format(roc_auc["macro"]))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if fold:
            plt.title('Receiver operating characteristic Fold={0}'.format(fold))
        else:
            plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        if filename:
            print("Saving the figure as {0}...".format(filename))
            plt.savefig(filename)
        else:
            try:
                print("Displaying the plot, close it to continue...")
                plt.show()
            except Exception as exc:
                print("UNHANDLED EXCEPTION - Trying to show Matplotlib plot: {0}".format(exc))
                raise
        return plt
EOF
