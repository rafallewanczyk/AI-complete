

from sklearn import preprocessing
import numpy as np
features = np.array([[-100.1, 3240.1],
                    [-200.2, -234.1],
                    [5000.5, 150.1],
                    [6000.6, -125.1],
                    [9000.9, -673.1]])       
scaler = preprocessing.StandardScaler()     
features_standardized = scaler.fit_transform(features)  
features_standardized

print("Mean:", round(features_standardized[:, 0].mean()))
print("Standard deviation", features_standardized[:, 0].std())

from keras import models
from keras import layers
network = models.Sequential()   

network.add(layers.Dense(units=16, activation="relu", input_shape=(10,)))   
network.add(layers.Dense(units=16, activation="relu"))                      
network.add(layers.Dense(units=16, activation="sigmoid"))                   

network.compile(loss="binary_crossentropy",         
                optimizer="rmsprop",                 
                metrics=["accuracy"])               

import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
np.random.seed(0)       
number_of_features = 1000   
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)  

tokenizer = Tokenizer(num_words=number_of_features)         
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()       

network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features,)))   
network.add(layers.Dense(units=16, activation="relu"))                      
network.add(layers.Dense(units=1, activation="sigmoid"))                   

network.compile(loss="binary_crossentropy",         
                optimizer="rmsprop",                 
                metrics=["accuracy"])               

history = network.fit(features_train,               
                      target_train,                 
                      epochs=3,                     
                      verbose=1,                   
                      batch_size=100,               
                      validation_data=(features_test, target_test))     

features_train.shape         

import numpy as np
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
np.random.seed(0)
number_of_features = 5000       

data = reuters.load_data(num_words=number_of_features)
(data_train, target_vector_train), (data_test, target_vector_test) = data       

tokenizer = Tokenizer(num_words=number_of_features)         
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

target_train = to_categorical(target_vector_train)          
target_test = to_categorical(target_vector_test)

network = models.Sequential()                               

network.add(layers.Dense(units=100, activation="relu", input_shape=(number_of_features,)))   
network.add(layers.Dense(units=46, activation="softmax"))                      

network.compile(loss="categorical_crossentropy",         
                optimizer="rmsprop",                 
                metrics=["accuracy"])               

history = network.fit(features_train,               
                      target_train,                 
                      epochs=3,                     
                      verbose=0,                   
                      batch_size=100,               
                      validation_data=(features_test, target_test))     

target_train        

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
np.random.seed(0)
features, target = make_regression(n_samples=10000,
                                   n_features=3,
                                   n_informative=3,
                                   n_targets=1,
                                   noise=0.0,
                                   random_state=0)          

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.33, random_state=0)   

network = models.Sequential()       
network.add(layers.Dense(units=32,
                         activation='relu',
                         input_shape=(features_train.shape[1],)))
network.add(layers.Dense(units=32, activation='relu'))
network.add(layers.Dense(units=1))      

network.compile(loss="mse",             
                optimizer="RMSprop",    
                metrics=["mse"])        

history = network.fit(features_train,
                      target_train,
                      epochs=10,
                      verbose=0,
                      batch_size=100,                                       
                      validation_data=(features_test, target_test))         

import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
np.random.seed(0)
number_of_features = 10000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)             
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()   

network.add(layers.Dense(units=16,
                         activation='relu',
                         input_shape=(number_of_features,)))
network.add(layers.Dense(units=16, activation='relu'))
network.add(layers.Dense(units=1,activation="sigmoid"))

network.compile(loss="binary_crossentropy",         
                optimizer="rmsprop",                 
                metrics=["accuracy"])               

history = network.fit(features_train,               
                      target_train,                 
                      epochs=3,                     
                      verbose=0,                   
                      batch_size=100,               
                      validation_data=(features_test, target_test))     

predicted_target = network.predict(features_test)       

predicted_target[0]         

import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
import matplotlib.pyplot as plt
np.random.seed(0)
number_of_features = 10000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)             
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()   

network.add(layers.Dense(units=16,
                         activation='relu',
                         input_shape=(number_of_features,)))
network.add(layers.Dense(units=16, activation='relu'))
network.add(layers.Dense(units=1, activation="sigmoid"))

network.compile(loss="binary_crossentropy",         
                optimizer="rmsprop",                 
                metrics=["accuracy"])               

history = network.fit(features_train,               
                      target_train,                 
                      epochs=15,                     
                      verbose=0,                   
                      batch_size=1000,               
                      validation_data=(features_test, target_test))     

training_loss = history.history["loss"]         
test_loss = history.history["val_loss"]

epoch_count = range(1, len(training_loss)+1)    

plt.plot(epoch_count, training_loss, "r--")     
plt.plot(epoch_count, test_loss, "b-")
plt.legend(['Training Loss', "Test Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

training_accuracy = history.history["accuracy"]      
test_accuracy = history.history["val_accuracy"]
plt.plot(epoch_count, training_accuracy, "r--")
plt.plot(epoch_count, test_accuracy, "b-")

plt.legend(["Training Accuracy", "Test Accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy Score")
plt.show()

import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras import regularizers
np.random.seed(0)
number_of_features = 1000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)             
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()   

network.add(layers.Dense(units=16,
                         activation='relu',
                         kernel_regularizer=regularizers.l2(0.01),
                         input_shape=(number_of_features,)))
network.add(layers.Dense(units=16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
network.add(layers.Dense(units=1, activation="sigmoid"))

network.compile(loss="binary_crossentropy",         
                optimizer="rmsprop",                 
                metrics=["accuracy"])               

history = network.fit(features_train,               
                      target_train,                 
                      epochs=3,                     
                      verbose=0,                   
                      batch_size=100,               
                      validation_data=(features_test, target_test))     

import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
np.random.seed(0)
number_of_features = 1000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)             
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()   

network.add(layers.Dense(units=16,
                         activation='relu',
                         input_shape=(number_of_features,)))
network.add(layers.Dense(units=16, activation='relu'))
network.add(layers.Dense(units=1, activation="sigmoid"))

network.compile(loss="binary_crossentropy",         
                optimizer="rmsprop",                 
                metrics=["accuracy"])               

callbacks = [EarlyStopping(monitor="val_loss", patience=2),
             ModelCheckpoint(filepath="best_model.h5",
                             monitor="val_loss",
                             save_best_only=True)]      

history = network.fit(features_train,               
                      target_train,                 
                      epochs=20,                     
                      callbacks=callbacks,          
                      verbose=1,                   
                      batch_size=100,               
                      validation_data=(features_test, target_test))     

import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
np.random.seed(0)
number_of_features = 1000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)             
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()   
network.add(layers.Dropout(0.2, input_shape=(number_of_features,)))       
network.add(layers.Dense(units=16, activation="relu"))                  
network.add(layers.Dropout(0.5))        
network.add(layers.Dense(units=16,activation="relu"))       
network.add(layers.Dropout(0.5))        
network.add(layers.Dense(units=1, activation="sigmoid"))    

network.compile(loss="binary_crossentropy",         
                optimizer="rmsprop",                 
                metrics=["accuracy"])               

history = network.fit(features_train,               
                      target_train,                 
                      epochs=3,                     
                      verbose=0,                   
                      batch_size=100,               
                      validation_data=(features_test, target_test))     

import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.callbacks import ModelCheckpoint
np.random.seed(0)
number_of_features = 1000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

tokenizer = Tokenizer(num_words=number_of_features)             
features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")

network = models.Sequential()   
network.add(layers.Dense(units=16,activation="relu",input_shape=(number_of_features,)))       
network.add(layers.Dense(units=16,activation="relu"))       
network.add(layers.Dense(units=1, activation="sigmoid"))    

network.compile(loss="binary_crossentropy",         
                optimizer="rmsprop",                 
                metrics=["accuracy"])               

checkpoint = [ModelCheckpoint(filepath="model.hdf5")]

history = network.fit(features_train,               
                      target_train,                 
                      epochs=3,                     
                      callbacks=checkpoint,         
                      verbose=0,                   
                      batch_size=100,               
                      validation_data=(features_test, target_test))     

import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
np.random.seed(0)
number_of_features = 100
features, target = make_classification(n_samples=10000,
                                       n_features=number_of_features,
                                       n_informative=3,
                                       n_redundant=0,
                                       n_classes=2,
                                       weights=[.5, .5],
                                       random_state=0)

def create_network():   
    network = models.Sequential()
    network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features,)))
    network.add(layers.Dense(units=16, activation="relu"))
    network.add(layers.Dense(units=1, activation="sigmoid"))
    network.compile(loss="binary_crossentropy",
                    optimizer="rmsprop",
                    metrics=["accuracy"])
    return network

neural_network = KerasClassifier(build_fn=create_network,           
                                 epochs=10,
                                 batch_size=100,
                                 verbose=0)

cross_val_score(neural_network, features, target, cv=3)     

import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
np.random.seed(0)
number_of_features = 100
features, target = make_classification(n_samples=10000,
                                       n_features=number_of_features,
                                       n_informative=3,
                                       n_redundant=0,
                                       n_classes=2,
                                       weights=[.5, .5],
                                       random_state=0)

def create_network(optimizer="rmsprop"):   
    network = models.Sequential()
    network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features,)))
    network.add(layers.Dense(units=16, activation="relu"))
    network.add(layers.Dense(units=1, activation="sigmoid"))
    network.compile(loss="binary_crossentropy",
                    optimizer=optimizer,
                    metrics=["accuracy"])
    return network

neural_network = KerasClassifier(build_fn=create_network, verbose=0)           

epochs = [5, 10]
batches = [5, 10, 100]
optimizers = ["rmsprop", "adam"]

hyperparameters = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)     

grid = GridSearchCV(estimator=neural_network, param_grid=hyperparameters)           

grid_result = grid.fit(features, target)

grid_result.best_params_        

from keras import models
from keras import layers
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

network = models.Sequential()
network.add(layers.Dense(units=16, activation="relu", input_shape=(10, 1)))
network.add(layers.Dense(units=16, activation="relu"))
network.add(layers.Dense(units=1, activation="sigmoid"))

SVG(model_to_dot(network, show_shapes=True).create(prog="dot", format="svg"))    

plot_model(network, show_shapes=True,to_file="network.png")                      

SVG(model_to_dot(network, show_shapes=False).create(prog="dot", format="svg"))

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_data_format("channels_first")   
np.random.seed(0)

channels = 1
height = 28
width = 28

(data_train, target_train), (data_test, target_test) = mnist.load_data()     
data_train = data_train.reshape(data_train.shape[0], channels, height, width)   
data_test = data_test.reshape(data_test.shape[0], channels, height, width)    

features_train = data_train / 255   
features_test = data_test / 255

target_train = np_utils.to_categorical(target_train)        
target_test = np_utils.to_categorical(target_test)
number_of_classes = target_test.shape[1]

network = Sequential()

network.add(Conv2D(filters=64,
                   kernel_size=(5, 5),
                   input_shape=(channels, width, height),
                   activation="relu"))      
network.add(MaxPooling2D(pool_size=(2, 2)))     
network.add(Dropout(0.5))       
network.add(Flatten())      
network.add(Dense(128, activation="relu"))      
network.add(Dropout(0.5))       
network.add(Dense(number_of_classes, activation="softmax"))      

network.compile(loss="categorical_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])

network.fit(features_train,
            target_train,
            epochs=2,
            verbose=0,
            batch_size=1000,
            validation_data=(features_test, target_test))

from keras.preprocessing.image import ImageDataGenerator

augmentation = ImageDataGenerator(featurewise_center=True,      
                                  zoom_range=0.3,   
                                  width_shift_range=0.2,    
                                  horizontal_flip=True,     
                                  rotation_range=90)        

augment_images = augmentation.flow_from_directory("/Users/caleb/Documents/Python Scripts/python Ji Qi Xue Xi Shou Ce /simulated_datasets-master/images/raw/images",      
                                                  batch_size=32,    
                                                  class_mode="binary",      
                                                  save_to_dir="/Users/caleb/Documents/Python Scripts/python Ji Qi Xue Xi Shou Ce /simulated_datasets-master/images/processed/images")

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import models
from keras import layers
np.random.seed(0)
number_of_features = 1000
(data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)

features_train = sequence.pad_sequences(data_train, maxlen=400)     
features_test = sequence.pad_sequences(data_test, maxlen=400)

network = models.Sequential()
network.add(layers.Embedding(input_dim=number_of_features, output_dim=128))     
network.add(layers.LSTM(units=128))     
network.add(layers.Dense(units=1, activation="sigmoid"))        

network.compile(loss="binary_crossentropy",
                optimizer="Adam",
                metrics=["accuracy"])

history = network.fit(features_train,
                      target_train,
                      epochs=3,
                      verbose=0,
                      batch_size=1000,
                      validation_data=(features_test,target_test))      

print(data_train[0])

print(features_test[0])


EOF
