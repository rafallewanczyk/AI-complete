

import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader as DataLoader

import time
from system.misc import *
from system.initnet import *

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras import callbacks

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras import callbacks

import statistics as stat
from scipy import stats

WHICH_PLOT = ["ACT"] 

dataset = SimulateData()
training_set, validation_set, test_set = SplitData(dataset)

saveData([training_set, validation_set, test_set])

LEN_TRAINING_SET    = len(training_set)
LEN_VALIDATION_SET  = len(validation_set)
LEN_TEST_SET        = len(test_set)

training_data_list, training_labels_list = ModDataKeras(training_set)
validation_data_list, validation_labels_list = ModDataKeras(validation_set)
test_data_list, test_labels_list = ModDataKeras(test_set)

training_set = ModDataTorch(training_set)
validation_set = ModDataTorch(validation_set)
test_set = ModDataTorch(test_set)

tr_labels = torch.zeros(len(training_set[:,0]), DIM_OUT)
for i in range(len(training_set[:,0])): tr_labels[i] = training_set[i,2]
val_labels = torch.zeros(len(validation_set[:,0]), DIM_OUT)
for i in range(len(validation_set[:,0])): val_labels[i] = validation_set[i,2]
test_labels = torch.zeros(len(test_set[:,0]), DIM_OUT)
for i in range(len(test_set[:,0])): test_labels[i] = test_set[i,2]

if CUDA:
    training_set_var = Variable(training_set[:,:2].cuda())
    tr_labels_var = Variable(tr_labels.cuda())
    test_set_var = Variable(test_set[:,:2].cuda())
    test_labels_var = Variable(test_labels.cuda())
    validation_set_var = Variable(validation_set[:,:2].cuda())
    validation_labels_var = Variable(val_labels.cuda())
else:
    training_set_var = Variable(training_set[:,:2])
    tr_labels_var = Variable(tr_labels)
    test_set_var = Variable(test_set[:,:2])
    test_labels_var = Variable(test_labels)
    validation_set_var = Variable(validation_set[:,:2])
    validation_labels_var = Variable(val_labels)

print ("DATA SIZE:")
print (training_set.size())
print (validation_set.size())
print (test_set.size())

if "SS" in WHICH_PLOT:
    model = Sequential()
    model.add(Dense(16, activation="relu", input_shape=(2,)))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="linear" ))
    model.summary()
    model.compile ( loss="mean_squared_error", optimizer="adam", metrics=["mse"] )
    history=model.fit ( np.array(training_data_list), np.array(training_labels_list), \
                       validation_data = ( np.array(test_data_list), np.array(test_labels_list) ), 
                        batch_size=MINI_BATCH_SIZE, epochs=EPOCH_NUM)
    print ("Keras trained.")
    netdata = CreateNet(
            4, 
            16, 
            "rel", 
            "lin", 
            "MSE", 
            "Adam", 
            MINI_BATCH_SIZE, 
            1e-3 
            )  
    loss_fn = initloss(netdata["lossf"])
    optimizer = initopt(netdata["optim"], netdata["model"], netdata["lrate"])
    trainloader = DataLoader(training_set, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=4)
    for epoch in range(EPOCH_NUM):
        for i, data in enumerate(trainloader):
            inputs, labels = modelinputs(data)  
            labels_pred = netdata["model"](inputs)
            loss_minibatch = loss_fn(labels_pred, labels)
            optimizer.zero_grad()
            loss_minibatch.backward()
            optimizer.step()
        preds = netdata["model"](test_set_var)
        loss = loss_fn(preds, test_labels_var)
        print ("Epoch: ", epoch, " Loss Pytorch: ", loss)
    print ("Pytorch trained.")
    list_batch_len = []
    list_predtime_keras = []
    list_predstd_keras = []
    list_predtime_torch = []
    list_predstd_torch = []
    for batchlen in range(1,LEN_TRAINING_SET,int(LEN_TRAINING_SET/50)):
        batchkeras = training_data_list[0:batchlen]
        batchtorch = training_set_var[0:batchlen]
        timekeras = []
        for i in range(int(ANALYSIS_SAMPLE_SIZE)):
            timedummy = time.time()
            predskeras = model.predict ( np.array(batchkeras) )
            timekeras.append(time.time() - timedummy)
        timetorch = []
        for i in range(ANALYSIS_SAMPLE_SIZE):
            timedummy = time.time()
            predstorch = netdata["model"](batchtorch)
            timetorch.append(time.time() - timedummy)
        list_batch_len.append(batchlen)
        list_predtime_keras.append(stat.mean(timekeras))
        list_predstd_keras.append(stat.stdev(timekeras))
        list_predtime_torch.append(stat.mean(timetorch))
        list_predstd_torch.append(stat.stdev(timetorch))        
    print("Times saved.")

    slope_keras, intercept_keras, r_value_keras, p_value_keras, std_err_keras = stats.linregress(
            list_batch_len, list_predtime_keras)
    slope_torch, intercept_torch, r_value_torch, p_value_torch, std_err_torch = stats.linregress(
            list_batch_len, list_predtime_torch)    
    fig1, ax1 = plt.subplots()
    ax1.errorbar(np.array(list_batch_len), 
             np.array(list_predtime_keras), 
             yerr=list_predstd_keras, 
             fmt = 'o', color='b', label = 'Keras Measurements')
    ax1.plot(list_batch_len, 
             intercept_keras + slope_keras*np.array(list_batch_len), 
             'g', label='Linear Regression')
    ax1.set(xlabel='Batch Length (Samples)', ylabel='Prediction Time Keras (s)')
    ax1.grid()
    ax1.legend()
    fig1.savefig(PATH_PLOTS + "KerasPyTorch/BatchLengthKeras.eps", format = 'eps')

    fig2, ax2 = plt.subplots()    
    ax2.errorbar(np.array(list_batch_len), 
             np.array(list_predtime_torch), 
             yerr=list_predstd_torch, 
             fmt = 'o', color='r', label = 'PyTorch')
    ax2.plot(list_batch_len, 
             intercept_torch + slope_torch*np.array(list_batch_len), 
             'orange', label='Linear Regression')
    ax2.set(xlabel='Batch Length (Samples)', ylabel='Prediction Time PyTorch (s)')
    ax2.grid()
    ax2.legend()
    fig2.savefig(PATH_PLOTS + "KerasPyTorch/BatchLengthPyTorch.eps", format = 'eps')
    print("Plots done.")
    with open(PATH_PLOTS + "KerasPyTorch/BatchLength_README.txt", 'w') as f:
        f.write("ANALYSIS_ID  = {} \n".format(ANALYSIS_ID))
        f.write("TXNAME = {} \n".format(TXNAME))
        f.write("MOTHER_LOW = {} \n".format(MOTHER_LOW))
        f.write("MOTHER_UP = {} \n".format(MOTHER_UP))
        f.write("MOTHER_STEP = {} \n".format(MOTHER_STEP))
        f.write("LSP_LOW = {} \n".format(LSP_LOW))
        f.write("LSP_STEP = {} \n".format(LSP_STEP))
        f.write("LEN_TEST_SET = {} \n".format(LEN_TEST_SET))
        f.write("LEN_TRAINING_SET = {} \n".format(LEN_TRAINING_SET))
        f.write("LEN_VALIDATION_SET = {} \n".format(LEN_VALIDATION_SET))
        f.write("HIDDEN LAYERS = {} \n".format(netdata["layer"]))
        f.write("NODES = {} \n".format(netdata["nodes"]))
        f.write("ACTIVATION FUNCTION = {} \n".format(netdata["activ"]))
        f.write("SHAPE = {} \n".format(netdata["shape"]))
        f.write("MSE = {} \n".format(netdata["lossf"]))
        f.write("OPTIMIZER = {} \n".format(netdata["optim"]))
        f.write("MINI_BATCH_SIZE = {} \n".format(MINI_BATCH_SIZE))
        f.write("LEARNING RATE = {} \n".format(netdata["lrate"]))
        f.write("ANALYSIS_SAMPLE_SIZE = {} \n".format(ANALYSIS_SAMPLE_SIZE))
        f.write("EPOCH_NUM = {} \n".format(EPOCH_NUM))
        f.write("Run on Intel Core i7-3517U CPU @ 1.90GHz x 4, \
                with an Asus s400c Notebook.")
        f.write("Data for the KERAS Linear Regression:")
        f.write("Slope = {} \n".format(slope_keras)) 
        f.write("Intercept: = {} \n".format(intercept_keras))
        f.write("R_Value = {} \n".format(r_value_keras))
        f.write("P_Value = {} \n".format(p_value_keras))
        f.write("Std_Error = {} \n".format(std_err_keras))
        f.write("Data for the PYTORCH Linear Regression:")
        f.write("Slope = {} \n".format(slope_torch)) 
        f.write("Intercept: = {} \n".format(intercept_torch))
        f.write("R_Value = {} \n".format(r_value_torch))
        f.write("P_Value = {} \n".format(p_value_torch))
        f.write("Std_Error = {} \n".format(std_err_torch))
    with open(PATH_PLOTS + "KerasPyTorch/BatchLength_Length.txt", 'w') as filehandle:  
        for listitem in list_batch_len:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/BatchLength_KerasTime.txt", 'w') as filehandle:  
        for listitem in list_predtime_keras:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/BatchLength_PyTorchTime.txt", 'w') as filehandle:  
        for listitem in list_predtime_torch:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/BatchLength_KerasStd.txt", 'w') as filehandle:  
        for listitem in list_predstd_keras:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/BatchLength_PyTorchStd.txt", 'w') as filehandle:  
        for listitem in list_predstd_torch:
            filehandle.write('%s\n' % listitem)            

if "LAY" in WHICH_PLOT:  

    list_layers = []
    list_predtime_keras = []
    list_predstd_keras = []    
    list_predtime_torch = []
    list_predstd_torch = []

    for layers in HID_LAY:
        model = Sequential()
        model.add(Dense(16, activation="relu", input_shape=(2,)))
        for i in range(layers-1):
            model.add(Dense(16, activation="relu"))
        model.add(Dense(1, activation="linear" ))
        model.summary()
        model.compile ( loss="mean_squared_error", optimizer="adam", metrics=["mse"] )
        history=model.fit ( np.array(training_data_list), np.array(training_labels_list), \
                       validation_data = ( np.array(test_data_list), np.array(test_labels_list) ), 
                        batch_size=MINI_BATCH_SIZE, epochs=EPOCH_NUM)
        print ("Keras trained with ", layers, "layers.")
        netdata = CreateNet(
                layers, 
                16, 
                "rel", 
                "lin", 
                "MSE", 
                "Adam", 
                MINI_BATCH_SIZE, 
                1e-3 
                )  
        loss_fn = initloss(netdata["lossf"])
        optimizer = initopt(netdata["optim"], netdata["model"], netdata["lrate"])
        trainloader = DataLoader(training_set, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=4)
        for epoch in range(EPOCH_NUM):
            for i, data in enumerate(trainloader):
                inputs, labels = modelinputs(data)  
                labels_pred = netdata["model"](inputs)
                loss_minibatch = loss_fn(labels_pred, labels)
                optimizer.zero_grad()
                loss_minibatch.backward()
                optimizer.step()
            preds = netdata["model"](test_set_var)
            loss = loss_fn(preds, test_labels_var)
            print ("Epoch: ", epoch, " Loss Pytorch: ", loss)
        print ("Pytorch trained with ", layers, "layers.")

        timekeras = []
        for i in range(int(ANALYSIS_SAMPLE_SIZE)):
            timedummy = time.time()
            predskeras = model.predict ( np.array(test_data_list) )
            timekeras.append(time.time() - timedummy)
        timetorch = []
        for i in range(ANALYSIS_SAMPLE_SIZE):
            timedummy = time.time()
            predstorch = netdata["model"](test_set_var)
            timetorch.append(time.time() - timedummy)
        list_layers.append(layers)
        list_predtime_keras.append(stat.mean(timekeras))
        list_predstd_keras.append(stat.stdev(timekeras))
        list_predtime_torch.append(stat.mean(timetorch))
        list_predstd_torch.append(stat.stdev(timetorch))        
        print("Times saved for ", layers, " layers.")

    slope_keras, intercept_keras, r_value_keras, p_value_keras, std_err_keras = stats.linregress(
            list_layers, list_predtime_keras)
    slope_torch, intercept_torch, r_value_torch, p_value_torch, std_err_torch = stats.linregress(
            list_layers, list_predtime_torch)    

    fig3, ax3 = plt.subplots()
    ax3.errorbar(np.array(list_layers), 
             np.array(list_predtime_keras), 
             yerr=list_predstd_keras, 
             fmt = 'o', color='b', label = 'Keras Measurements')
    ax3.plot(list_layers, 
             intercept_keras + slope_keras*np.array(list_layers), 
             'g', label='Linear Regression')
    ax3.set(xlabel='Hidden Layers', ylabel='Prediction Time Keras (s)')
    ax3.grid()
    ax3.legend()
    fig3.savefig(PATH_PLOTS + "KerasPyTorch/LayersKeras.eps", format = 'eps')

    fig4, ax4 = plt.subplots()
    ax4.errorbar(np.array(list_layers), 
             np.array(list_predtime_torch), 
             yerr=list_predstd_torch, 
             fmt = 'o', color='r', label = 'PyTorch')
    ax4.plot(list_layers, 
             intercept_torch + slope_torch*np.array(list_layers), 
             'orange', label='Linear Regression')
    ax4.set(xlabel='Hidden Layers', ylabel='Prediction Time PyTorch (s)')
    ax4.grid()
    ax4.legend()
    fig4.savefig(PATH_PLOTS + "KerasPyTorch/LayersPyTorch.eps", format = 'eps')
    print("Plots done.")

    with open(PATH_PLOTS + "KerasPyTorch/Layers_README.txt", 'w') as f:
        f.write("ANALYSIS_ID  = {} \n".format(ANALYSIS_ID))
        f.write("TXNAME = {} \n".format(TXNAME))
        f.write("MOTHER_LOW = {} \n".format(MOTHER_LOW))
        f.write("MOTHER_UP = {} \n".format(MOTHER_UP))
        f.write("MOTHER_STEP = {} \n".format(MOTHER_STEP))
        f.write("LSP_LOW = {} \n".format(LSP_LOW))
        f.write("LSP_STEP = {} \n".format(LSP_STEP))
        f.write("LEN_TEST_SET = {} \n".format(LEN_TEST_SET))
        f.write("LEN_TRAINING_SET = {} \n".format(LEN_TRAINING_SET))
        f.write("LEN_VALIDATION_SET = {} \n".format(LEN_VALIDATION_SET))
        f.write("NODES = {} \n".format(netdata["nodes"]))
        f.write("ACTIVATION FUNCTION = {} \n".format(netdata["activ"]))
        f.write("SHAPE = {} \n".format(netdata["shape"]))
        f.write("MSE = {} \n".format(netdata["lossf"]))
        f.write("OPTIMIZER = {} \n".format(netdata["optim"]))
        f.write("MINI_BATCH_SIZE = {} \n".format(MINI_BATCH_SIZE))
        f.write("LEARNING RATE = {} \n".format(netdata["lrate"]))
        f.write("ANALYSIS_SAMPLE_SIZE = {} \n".format(ANALYSIS_SAMPLE_SIZE))
        f.write("EPOCH_NUM = {} \n".format(EPOCH_NUM))
        f.write("The prediction is done for the test set, so for a single prediction it \
                has to be divided by the length of the test set. \
                Run on Intel Core i7-3517U CPU @ 1.90GHz x 4, \
                with an Asus s400c Notebook.")
        f.write("Data for the KERAS Linear Regression:")
        f.write("Slope = {} \n".format(slope_keras)) 
        f.write("Intercept: = {} \n".format(intercept_keras))
        f.write("R_Value = {} \n".format(r_value_keras))
        f.write("P_Value = {} \n".format(p_value_keras))
        f.write("Std_Error = {} \n".format(std_err_keras))
        f.write("Data for the PYTORCH Linear Regression:")
        f.write("Slope = {} \n".format(slope_torch)) 
        f.write("Intercept: = {} \n".format(intercept_torch))
        f.write("R_Value = {} \n".format(r_value_torch))
        f.write("P_Value = {} \n".format(p_value_torch))
        f.write("Std_Error = {} \n".format(std_err_torch)) 
    with open(PATH_PLOTS + "KerasPyTorch/Layers_Layers.txt", 'w') as filehandle:  
        for listitem in list_layers:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/Layers_KerasTime.txt", 'w') as filehandle:  
        for listitem in list_predtime_keras:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/Layers_PyTorchTime.txt", 'w') as filehandle:  
        for listitem in list_predtime_torch:
            filehandle.write('%s\n' % listitem)

    with open(PATH_PLOTS + "KerasPyTorch/Layers_KerasStd.txt", 'w') as filehandle:  
        for listitem in list_predstd_keras:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/Layers_PyTorchStd.txt", 'w') as filehandle:  
        for listitem in list_predstd_torch:
            filehandle.write('%s\n' % listitem)

if "NOD" in WHICH_PLOT:  

    list_nodes = []
    list_predtime_keras = []
    list_predstd_keras = []
    list_predtime_torch = []
    list_predstd_torch = []

    for nodes in NOD:
        model = Sequential()
        model.add(Dense(nodes, activation="relu", input_shape=(2,)))
        model.add(Dense(nodes, activation="relu"))
        model.add(Dense(nodes, activation="relu"))
        model.add(Dense(nodes, activation="relu"))
        model.add(Dense(1, activation="linear" ))
        model.summary()
        model.compile ( loss="mean_squared_error", optimizer="adam", metrics=["mse"] )
        history=model.fit ( np.array(training_data_list), np.array(training_labels_list), \
                       validation_data = ( np.array(test_data_list), np.array(test_labels_list) ), 
                        batch_size=MINI_BATCH_SIZE, epochs=EPOCH_NUM)
        print ("Keras trained with ", nodes, "nodes.")
        netdata = CreateNet(
                4, 
                nodes, 
                "rel", 
                "lin", 
                "MSE", 
                "Adam", 
                MINI_BATCH_SIZE, 
                1e-3 
                )  
        loss_fn = initloss(netdata["lossf"])
        optimizer = initopt(netdata["optim"], netdata["model"], netdata["lrate"])
        trainloader = DataLoader(training_set, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=4)
        for epoch in range(EPOCH_NUM):
            for i, data in enumerate(trainloader):
                inputs, labels = modelinputs(data)  
                labels_pred = netdata["model"](inputs)
                loss_minibatch = loss_fn(labels_pred, labels)
                optimizer.zero_grad()
                loss_minibatch.backward()
                optimizer.step()
            preds = netdata["model"](test_set_var)
            loss = loss_fn(preds, test_labels_var)
            print ("Epoch: ", epoch, " Loss Pytorch: ", loss)
        print ("Pytorch trained with ", nodes, "nodes.")

        timekeras = []
        for i in range(int(ANALYSIS_SAMPLE_SIZE)):
            timedummy = time.time()
            predskeras = model.predict ( np.array(test_data_list) )
            timekeras.append(time.time() - timedummy)
        timetorch = []
        for i in range(ANALYSIS_SAMPLE_SIZE):
            timedummy = time.time()
            predstorch = netdata["model"](test_set_var)
            timetorch.append(time.time() - timedummy)
        list_nodes.append(nodes)
        list_predtime_keras.append(stat.mean(timekeras))
        list_predstd_keras.append(stat.stdev(timekeras))
        list_predtime_torch.append(stat.mean(timetorch))
        list_predstd_torch.append(stat.stdev(timetorch))        
        print("Times saved for ", nodes, " nodes.")

    slope_keras, intercept_keras, r_value_keras, p_value_keras, std_err_keras = stats.linregress(
            list_nodes, list_predtime_keras)
    slope_torch, intercept_torch, r_value_torch, p_value_torch, std_err_torch = stats.linregress(
            list_nodes, list_predtime_torch)    

    fig5, ax5 = plt.subplots()
    ax5.errorbar(np.array(list_nodes), 
             np.array(list_predtime_keras), 
             yerr=list_predstd_keras, 
             fmt = 'o', color='b', label = 'Keras Measurements')
    ax5.plot(list_nodes, 
             intercept_keras + slope_keras*np.array(list_nodes), 
             'g', label='Linear Regression')
    ax5.set(xlabel='Nodes', ylabel='Prediction Time Keras (s)')
    ax5.grid()
    ax5.legend()
    fig5.savefig(PATH_PLOTS + "KerasPyTorch/NodesKeras.eps", format = 'eps')

    fig6, ax6 = plt.subplots()
    ax6.errorbar(np.array(list_nodes), 
             np.array(list_predtime_torch), 
             yerr=list_predstd_torch, 
             fmt = 'o', color='r', label = 'PyTorch')
    ax6.plot(list_nodes, 
             intercept_torch + slope_torch*np.array(list_nodes), 
             'orange', label='Linear Regression')
    ax6.set(xlabel='Nodes', ylabel='Prediction Time PyTorch (s)')
    ax6.grid()
    ax6.legend()
    fig6.savefig(PATH_PLOTS + "KerasPyTorch/NodesPyTorch.eps", format = 'eps')
    print("Plots done.")

    with open(PATH_PLOTS + "KerasPyTorch/Nodes_README.txt", 'w') as f:
        f.write("ANALYSIS_ID  = {} \n".format(ANALYSIS_ID))
        f.write("TXNAME = {} \n".format(TXNAME))
        f.write("MOTHER_LOW = {} \n".format(MOTHER_LOW))
        f.write("MOTHER_UP = {} \n".format(MOTHER_UP))
        f.write("MOTHER_STEP = {} \n".format(MOTHER_STEP))
        f.write("LSP_LOW = {} \n".format(LSP_LOW))
        f.write("LSP_STEP = {} \n".format(LSP_STEP))
        f.write("LEN_TEST_SET = {} \n".format(LEN_TEST_SET))
        f.write("LEN_TRAINING_SET = {} \n".format(LEN_TRAINING_SET))
        f.write("LEN_VALIDATION_SET = {} \n".format(LEN_VALIDATION_SET))
        f.write("HIDDEN LAYERS = {} \n".format(netdata["layer"]))
        f.write("ACTIVATION FUNCTION = {} \n".format(netdata["activ"]))
        f.write("SHAPE = {} \n".format(netdata["shape"]))
        f.write("MSE = {} \n".format(netdata["lossf"]))
        f.write("OPTIMIZER = {} \n".format(netdata["optim"]))
        f.write("MINI_BATCH_SIZE = {} \n".format(MINI_BATCH_SIZE))
        f.write("LEARNING RATE = {} \n".format(netdata["lrate"]))
        f.write("ANALYSIS_SAMPLE_SIZE = {} \n".format(ANALYSIS_SAMPLE_SIZE))
        f.write("EPOCH_NUM = {} \n".format(EPOCH_NUM))
        f.write("The prediction is done for the test set, so for a single prediction it \
                has to be divided by the length of the test set. \
                Run on Intel Core i7-3517U CPU @ 1.90GHz x 4, \
                with an Asus s400c Notebook.")
        f.write("Data for the KERAS Linear Regression:")
        f.write("Slope = {} \n".format(slope_keras)) 
        f.write("Intercept: = {} \n".format(intercept_keras))
        f.write("R_Value = {} \n".format(r_value_keras))
        f.write("P_Value = {} \n".format(p_value_keras))
        f.write("Std_Error = {} \n".format(std_err_keras))
        f.write("Data for the PYTORCH Linear Regression:")
        f.write("Slope = {} \n".format(slope_torch)) 
        f.write("Intercept: = {} \n".format(intercept_torch))
        f.write("R_Value = {} \n".format(r_value_torch))
        f.write("P_Value = {} \n".format(p_value_torch))
        f.write("Std_Error = {} \n".format(std_err_torch))
    with open(PATH_PLOTS + "KerasPyTorch/Nodes_Nodes.txt", 'w') as filehandle:  
        for listitem in list_nodes:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/Nodes_KerasTime.txt", 'w') as filehandle:  
        for listitem in list_predtime_keras:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/Nodes_PyTorchTime.txt", 'w') as filehandle:  
        for listitem in list_predtime_torch:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/Nodes_KerasStd.txt", 'w') as filehandle:  
        for listitem in list_predstd_keras:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/Nodes_PyTorchStd.txt", 'w') as filehandle:  
        for listitem in list_predstd_torch:
            filehandle.write('%s\n' % listitem)            

if "ACT" in WHICH_PLOT:  

    list_activ = []
    list_predtime_keras = []
    list_predstd_keras = []
    list_predtime_torch = []
    list_predstd_torch = []

    for activ in ACTIVATION_FUNCTIONS:
        model = Sequential()
        if activ == "sig":
            model.add(Dense(16, activation="sigmoid", input_shape=(2,)))
            model.add(Dense(16, activation="sigmoid"))
            model.add(Dense(16, activation="sigmoid"))
            model.add(Dense(16, activation="sigmoid"))
            model.add(Dense(1, activation="linear" ))
        elif activ == "tah":
            model.add(Dense(16, activation="tanh", input_shape=(2,)))
            model.add(Dense(16, activation="tanh"))
            model.add(Dense(16, activation="tanh"))
            model.add(Dense(16, activation="tanh"))
            model.add(Dense(1, activation="linear" ))
        else:
            if activ != "rel":
                print("Activation function not implemented, using ReLU instead.")
            model.add(Dense(16, activation="relu", input_shape=(2,)))
            model.add(Dense(16, activation="relu"))
            model.add(Dense(16, activation="relu"))
            model.add(Dense(16, activation="relu"))
            model.add(Dense(1, activation="linear" ))            
        model.summary()
        model.compile ( loss="mean_squared_error", optimizer="adam", metrics=["mse"] )
        history=model.fit ( np.array(training_data_list), np.array(training_labels_list), \
                       validation_data = ( np.array(test_data_list), np.array(test_labels_list) ), 
                        batch_size=MINI_BATCH_SIZE, epochs=EPOCH_NUM)
        print ("Keras trained with ", activ, " activation function.")
        netdata = CreateNet(
                4, 
                16, 
                activ, 
                "lin", 
                "MSE", 
                "Adam", 
                MINI_BATCH_SIZE, 
                1e-3 
                )  
        loss_fn = initloss(netdata["lossf"])
        optimizer = initopt(netdata["optim"], netdata["model"], netdata["lrate"])
        trainloader = DataLoader(training_set, batch_size=MINI_BATCH_SIZE, shuffle=True, num_workers=4)
        for epoch in range(EPOCH_NUM):
            for i, data in enumerate(trainloader):
                inputs, labels = modelinputs(data)  
                labels_pred = netdata["model"](inputs)
                loss_minibatch = loss_fn(labels_pred, labels)
                optimizer.zero_grad()
                loss_minibatch.backward()
                optimizer.step()
            preds = netdata["model"](test_set_var)
            loss = loss_fn(preds, test_labels_var)
            print ("Epoch: ", epoch, " Loss Pytorch: ", loss)
        print ("Pytorch trained with ", activ, " activation function.")

        timekeras = []
        for i in range(int(ANALYSIS_SAMPLE_SIZE)):
            timedummy = time.time()
            predskeras = model.predict ( np.array(test_data_list) )
            timekeras.append(time.time() - timedummy)
        timetorch = []
        for i in range(ANALYSIS_SAMPLE_SIZE):
            timedummy = time.time()
            predstorch = netdata["model"](test_set_var)
            timetorch.append(time.time() - timedummy)
        list_activ.append(activ)
        list_predtime_keras.append(stat.mean(timekeras))
        list_predstd_keras.append(stat.stdev(timekeras))
        list_predtime_torch.append(stat.mean(timetorch))
        list_predstd_torch.append(stat.stdev(timetorch))        
        print("Times saved for ", activ, " activation function.")

    fig7, ax7 = plt.subplots()
    index = np.arange(len(list_activ))
    bar_width = 0.7
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects = plt.bar(x = index, height = list_predtime_keras, 
                    width = bar_width,
                     alpha=opacity,
                     color='blue',
                     yerr=list_predstd_keras,
                     error_kw=error_config,
                     label = 'Keras')
    plt.xlabel('Activation Function')
    plt.ylabel('Prediction Time (s)')
    plt.xticks(index, list_activ)
    plt.legend()
    plt.tight_layout()
    fig7.savefig(PATH_PLOTS + "KerasPyTorch/ActivKeras.eps", format = 'eps')

    fig9, ax9 = plt.subplots()
    index = np.arange(len(list_activ))
    bar_width = 0.7
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects = plt.bar(x = index, height = list_predtime_keras, 
                    width = bar_width,
                     alpha=opacity,
                     color='blue',
                     error_kw=error_config,
                     label = 'Keras')
    plt.xlabel('Activation Function')
    plt.ylabel('Prediction Time (s)')
    plt.xticks(index, list_activ)
    plt.legend()
    plt.tight_layout()
    fig9.savefig(PATH_PLOTS + "KerasPyTorch/ActivKeras_noerr.eps", format = 'eps')

    fig8, ax8 = plt.subplots()
    index = np.arange(len(list_activ))
    rects = plt.bar(x = index, height = list_predtime_torch, 
                    width = bar_width,
                     alpha=opacity,
                     color='red',
                     yerr=list_predstd_torch,
                     error_kw=error_config,
                     label = 'PyTorch')
    plt.xlabel('Activation Function')
    plt.ylabel('Prediction Time (s)')
    plt.xticks(index, list_activ)
    plt.legend()
    plt.tight_layout()
    fig8.savefig(PATH_PLOTS + "KerasPyTorch/ActivPyTorch.eps", format = 'eps')
    print("Plots done.")

    with open(PATH_PLOTS + "KerasPyTorch/Activ_README.txt", 'w') as f:
        f.write("ANALYSIS_ID  = {} \n".format(ANALYSIS_ID))
        f.write("TXNAME = {} \n".format(TXNAME))
        f.write("MOTHER_LOW = {} \n".format(MOTHER_LOW))
        f.write("MOTHER_UP = {} \n".format(MOTHER_UP))
        f.write("MOTHER_STEP = {} \n".format(MOTHER_STEP))
        f.write("LSP_LOW = {} \n".format(LSP_LOW))
        f.write("LSP_STEP = {} \n".format(LSP_STEP))
        f.write("LEN_TEST_SET = {} \n".format(LEN_TEST_SET))
        f.write("LEN_TRAINING_SET = {} \n".format(LEN_TRAINING_SET))
        f.write("LEN_VALIDATION_SET = {} \n".format(LEN_VALIDATION_SET))
        f.write("HIDDEN LAYERS = {} \n".format(netdata["layer"]))
        f.write("NODES = {} \n".format(netdata["nodes"]))
        f.write("SHAPE = {} \n".format(netdata["shape"]))
        f.write("MSE = {} \n".format(netdata["lossf"]))
        f.write("OPTIMIZER = {} \n".format(netdata["optim"]))
        f.write("MINI_BATCH_SIZE = {} \n".format(MINI_BATCH_SIZE))
        f.write("LEARNING RATE = {} \n".format(netdata["lrate"]))
        f.write("ANALYSIS_SAMPLE_SIZE = {} \n".format(ANALYSIS_SAMPLE_SIZE))
        f.write("EPOCH_NUM = {} \n".format(EPOCH_NUM))
        f.write("The prediction is done for the test set, so for a single prediction it \
                has to be divided by the length of the test set. \
                Run on Intel Core i7-3517U CPU @ 1.90GHz x 4, \
                with an Asus s400c Notebook.")
    with open(PATH_PLOTS + "KerasPyTorch/Activ_Activ.txt", 'w') as filehandle:  
        for listitem in list_activ:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/Activ_KerasTime.txt", 'w') as filehandle:  
        for listitem in list_predtime_keras:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/Activ_PyTorchTime.txt", 'w') as filehandle:  
        for listitem in list_predtime_torch:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/Activ_KerasStd.txt", 'w') as filehandle:  
        for listitem in list_predstd_keras:
            filehandle.write('%s\n' % listitem)
    with open(PATH_PLOTS + "KerasPyTorch/Activ_PyTorchStd.txt", 'w') as filehandle:  
        for listitem in list_predstd_torch:
            filehandle.write('%s\n' % listitem)
            
EOF
