

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import multiprocessing
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
logging.info(tf.__version__)
from scipy.optimize import minimize
from itertools import combinations
from ase.visualize import view
from ase.db import connect
import random
from ase.build import sort
from ase import Atom
from ase.io import read, write

seed = 1234
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

units = 256
ptnc = 20
bat_sz = 128

logging.getLogger().setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] -\
    %(levelname)s: %(message)s',
                    level=logging.DEBUG)

gpu_name = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_name    

schnetpack_preinstall = False 
if(schnetpack_preinstall):  
    from schnetpack.datasets import QM9
    qm9data = QM9('./qm9.db', download=True)
    logging.debug('Number of reference calculations: %d', len(qm9data))
else:  
    db = connect('./qm9.db')

class Data_Generater:
    def get_atoms_by_idx(self, index):
        if(schnetpack_preinstall):
            at = qm9data.get_atoms(idx=index)
        else:  
            at = db.get_atoms(id=index+1)
        return at

    def get_data_counts(self):
        if(schnetpack_preinstall):
            return len(qm9data)
        else:  
            return db.count()

    def data_handler(self, idx_lst, a, b, gen_db): 
        bar = Bar('split qm9.db to train.db vali.db test.db', max=b-a)
        gen_times = 1
        for i in range(a, b):
            for j in range(gen_times):
                orig_atoms = self.get_atoms_by_idx(idx_lst[i])
                del_atom_idx = random.randint(0,\
                    orig_atoms.get_positions().shape[0]-1)
                del_position = orig_atoms.get_positions()[del_atom_idx]
                del_element = orig_atoms.get_chemical_symbols()[del_atom_idx]
                del orig_atoms[del_atom_idx]
                gen_db.write(orig_atoms, data={'del_pos': del_position,
                                               'del_element': del_element,
                                               'del_atom_idx': del_atom_idx,
                                               'id_in_orig_db': idx_lst[i]+1})
            bar.next()
        bar.finish()

    def gen_train_validation_test_dataset(self, train_cnt, vali_cnt):
        os.system('rm -rf train.db vali.db test.db')
        db_sz = self.get_data_counts()
        if(train_cnt + vali_cnt > db_sz or \
            train_cnt <=0 or vali_cnt < 0):
            raise AssertionError
        idx_lst = list(range(db_sz))
        random.shuffle(idx_lst)
        with connect('train.db') as gen_db:
            self.data_handler(idx_lst, 0, train_cnt, gen_db)
        with connect('vali.db') as gen_db:
            self.data_handler(idx_lst, train_cnt, train_cnt+vali_cnt, gen_db)
        with connect('test.db') as gen_db:
            self.data_handler(idx_lst, train_cnt+vali_cnt, db_sz, gen_db)

mat_sz = 32
class Agent:
    model = None
    train_images = None
    test_images = None
    train_labels = None
    test_labels = None
    ucfc_input_sz = mat_sz

    atom_names = ['H', 'C', 'O', 'F', 'N']
    atom_dict = {'H': 0, 'C':1, 'O':2, 'F':3, 'N':4}
    electro_negtive = {1: 2.3, 6: 2.544, 8: 3.61, 9:4.193, 7:3.066}

    def sort_atoms(self, row):
        return sort(row.toatoms())

    def atoms2matrices(self):
        logging.debug('Father class empty func')
    def atoms2matrices_xyz(self):
        logging.debug('Father class empty func')
    def atoms2matrices_ucfc(self):
        logging.debug('Father class empty func')

    def train_UC_FC(self): 
        self.atoms2matrices_ucfc()
        self.model = keras.Sequential([
            keras.layers.Dense(units=units, 
                               activation=tf.nn.relu, input_shape=(self.ucfc_input_sz,)),
            keras.layers.Dense(units=units, 
                               activation=tf.nn.relu),
            keras.layers.Dense(units=units, 
                               activation=tf.nn.relu),
            keras.layers.Dense(units=units, 
                               activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
        self.model.compile(optimizer='adam',
                        loss = 'binary_crossentropy',
                        metrics=['accuracy'])

        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=ptnc)
        chkpt_callback = tf.keras.callbacks.ModelCheckpoint(
            './models/best_model', monitor='val_loss', save_best_only=True, save_weights_only=True)
        os.system('rm -rf ./models/best_model')
        self.model.fit(self.train_images, self.train_labels, epochs=500,
                       batch_size=bat_sz, validation_split=0.1, callbacks=[earlystop_callback, chkpt_callback])
        self.model.load_weights('./models/best_model')

        pass

    def train_xyz(self):
        self.atoms2matrices_xyz()
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(mat_sz, mat_sz)),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(3)
        ])
        self.model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        logging.debug('train')
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=ptnc)
        chkpt_callback = tf.keras.callbacks.ModelCheckpoint(
            './models/best_model', monitor='val_loss', save_best_only=True, save_weights_only=True)
        os.system('rm -rf ./models/best_model')
        self.model.fit(self.train_images, self.train_labels, epochs=500,
                       batch_size=bat_sz, validation_split=0.1, callbacks=[earlystop_callback, chkpt_callback])
        self.model.load_weights('./models/best_model')

    def train(self):
        self.atoms2matrices()
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(mat_sz, mat_sz)),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(5, activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        logging.debug('train')

        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=ptnc)
        chkpt_callback = tf.keras.callbacks.ModelCheckpoint(
            './models/best_model', monitor='val_loss', save_best_only=True, save_weights_only=True)
        os.system('rm -rf ./models/best_model')
        self.model.fit(self.train_images, self.train_labels, epochs=500,
                       batch_size=bat_sz, validation_split=0.1, callbacks=[earlystop_callback, chkpt_callback])
        self.model.load_weights('./models/best_model')
    def test(self):
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        logging.info('Test accuracy:' + str(test_acc)\
            + ', test loss: ' + str(test_loss))
        test_predictions = self.model.predict(self.test_images)
        classification_metrics = np.zeros([5, 5])
        for i in range(len(self.test_images)):
            max_it = 0
            idx_pred = -1
            for idx, it in enumerate(test_predictions[i]):
                if it > max_it:
                    max_it = it
                    idx_pred = idx
            classification_metrics[self.test_labels[i]][idx_pred] += 1
        print(classification_metrics)
        return test_acc, test_loss

    def test_xyz(self):
        loss, mae, mse = self.model.evaluate(self.test_images, self.test_labels)
        logging.info(
            'LOSS: ' + str(loss)\
            + ', MAE: ' + str(mae)\
            + ', MSE: ' + str(mse))
        return [mae], loss 
    def test_UC_FC(self):
        d = dict()
        for idx in range(len(self.test_images)):
            key = (self.test_labels[idx], self.test_images[idx][0], self.test_images[idx][1])
            if key in d.keys():
                d[key] += 1
            else:
                d[key] = 1
        sum = 0
        for it in d:
            print(it, d[it])
            sum += d[it]
        print(sum)

        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels)
        logging.info('Test accuracy:' + str(test_acc)\
            + ', test loss: ' + str(test_loss))

        return test_acc, test_loss

class Agent_2Body_Distance_Mat(Agent): 
    sature_or_not_distance = 1.7
    ucfc_cutoff = {'H': 1.65, 'C':1.86, 'O':1.86, 'F':1.85, 'N':1.83}
    def fill_descriptor(self, row):
        atoms = self.sort_atoms(row)
        img = np.zeros([mat_sz, mat_sz])
        for j in range(len(atoms)):
            for k in range(len(atoms)):
                img[j][k] = atoms.get_distance(j, k)
        return img

    def fill_descriptor_atom_saturation(self, row):
        atoms = self.sort_atoms(row)
        img = np.zeros([mat_sz, mat_sz])
        ret = []
        for j in range(len(atoms)):
            for k in range(len(atoms)):
                if(j == k):
                    img[j][0] = atoms.get_distance(j, k)
                elif(j > k):
                    img[j][k+1] = atoms.get_distance(j, k)
                elif(j < k):
                    img[j][k] = atoms.get_distance(j, k)
                else:
                    pass

        for idx in range(len(img)):
            if(img[idx][1] != 0):
                ret.append(img[idx])
        return ret
    def fill_labels(self, row):
        return self.atom_dict[row.data.del_element]

    def fill_labels_xyz(self, row):
        return np.array(row.data.del_pos)

    def fill_labels_atom_saturation(self, row):
        atoms = self.sort_atoms(row)
        atoms.append(Atom(row.data.del_element, row.data.del_pos))
        ret = []
        for i in range(0, len(atoms)-1):
            if(atoms.get_distance(i, len(atoms)-1) < self.ucfc_cutoff[atoms[i].symbol]):
                ret.append(0) 
            else:
                ret.append(1)
        return ret

    def atoms2matrices(self, xyz=False, normalization=True, saturation_judgement=False):
        train_db = connect('./train.db')
        test_db = connect('./test.db')

        rows = list(train_db.select(sort='id'))
        logging.info('converting train data ' + str(len(rows)) + ' cases, 3 mins...')
        logging.debug(len(rows))
        pool = multiprocessing.Pool()
        if(xyz):
            tmp = pool.map(self.fill_descriptor, rows)
            tmp1 = pool.map(self.fill_labels_xyz, rows)
        elif(saturation_judgement):
            ret = pool.map(self.fill_descriptor_atom_saturation, rows)
            tmp = []
            for t in ret:
                tmp += t
            ret1 = pool.map(self.fill_labels_atom_saturation, rows)
            tmp1 = []
            for t in ret1:
                tmp1 += t
        else:
            tmp = pool.map(self.fill_descriptor, rows)
            tmp1 = pool.map(self.fill_labels, rows)

        self.train_images = np.array(tmp)
        self.train_labels = np.array(tmp1)
        pool.close()
        pool.join()

        rows = list(test_db.select(sort='id'))
        logging.info('converting test data ' +
                     str(len(rows))+' cases, 1 mins...')

        pool = multiprocessing.Pool()
        if(xyz):
            tmp = pool.map(self.fill_descriptor, rows)
            tmp1 = pool.map(self.fill_labels_xyz, rows)
        elif(saturation_judgement):
            ret = pool.map(self.fill_descriptor_atom_saturation, rows)
            tmp = []
            for t in ret:
                tmp += t
            ret1 = pool.map(self.fill_labels_atom_saturation, rows)
            tmp1 = []
            for t in ret1:
                tmp1 += t
        else:
            tmp = pool.map(self.fill_descriptor, rows)
            tmp1 = pool.map(self.fill_labels, rows)

        self.test_images = np.array(tmp)
        self.test_labels = np.array(tmp1)

        pool.close()
        pool.join()
        max_element = max(np.max(self.train_images), np.max(self.test_images))
        logging.info('max element in train + test dataset is: ' + str(max_element))

        if(normalization==True):
            self.train_images = self.train_images / max_element
            self.test_images = self.test_images / max_element

    def atoms2matrices_xyz(self):
        self.atoms2matrices(xyz=True)
    def atoms2matrices_ucfc(self):
        self.atoms2matrices(xyz=False, saturation_judgement=True)

class Agent_BOC(Agent_2Body_Distance_Mat):
    clst_tuple_list = []
    total_clst_cnt = 0
    ele_clst_idx_dict = {'H':0, 'C':1, 'O':2, 'F':3, 'N':4} 
    def generate_cluster_list(self):
        import cluster_dict as cd
        self.clst_tuple_list, self.total_clst_cnt = cd.get_distance_lst()
        self.total_clst_cnt += 5 
        self.clst_tuple_list.sort()
        logging.info('cluster is:'+ str(self.clst_tuple_list))

    def cal_clst_lst_single_atom(self, atoms, index):
        ret = np.zeros([self.total_clst_cnt,])
        ret[self.ele_clst_idx_dict[atoms[index].symbol]] += 1
        for i in range(len(atoms)):
            if(i != index):
                key1 = atoms[i].symbol + atoms[index].symbol
                key2 = atoms[index].symbol + atoms[i].symbol
                min_dis = 9999
                min_idx = -1
                for idx, k in enumerate(self.clst_tuple_list):
                    if(k[0] == key1 or k[0] == key2):
                        if(atoms.get_distance(i, index) < 5 and 
                                abs(k[1]-atoms.get_distance(i, index)*1000) < min_dis):
                            min_dis = abs(
                                k[1]-atoms.get_distance(i, index)*1000)
                            min_idx = idx
                if(min_idx != -1):
                    ret[min_idx + 5] += 1
        return ret

    def cal_clst_lst(self, atoms):
        ret = np.zeros([self.total_clst_cnt,])
        for i in range(len(atoms)):
            ret += self.cal_clst_lst_single_atom(atoms, i)
        ret[5:] /= 2
        return ret

    def fill_descriptor(self, row):
        atoms = self.sort_atoms(row)
        ret = self.cal_clst_lst(atoms)
        return ret

    def fill_descriptor_atom_saturation(self, row):
        atoms = self.sort_atoms(row)
        ret = []
        for i in range(len(atoms)):
            ret.append(self.cal_clst_lst_single_atom(atoms, i))
        return ret

    def train(self):
        self.generate_cluster_list()
        self.atoms2matrices(normalization=False, saturation_judgement=False)
        self.model = keras.Sequential([
            keras.layers.Dense(units=units,
                               activation=tf.nn.relu, input_shape=(self.total_clst_cnt,)),
            keras.layers.Dense(units=units,
                               activation=tf.nn.relu),
            keras.layers.Dense(units=units,
                               activation=tf.nn.relu),
            keras.layers.Dense(units=units,
                               activation=tf.nn.relu),
            keras.layers.Dense(5, activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        logging.debug('train')
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=ptnc)
        chkpt_callback = tf.keras.callbacks.ModelCheckpoint(
            './models/best_model', monitor='val_loss', save_best_only=True, save_weights_only=True)
        os.system('rm -rf ./models/best_model')
        self.model.fit(self.train_images, self.train_labels, epochs=500,
                       batch_size=bat_sz, validation_split=0.1, callbacks=[earlystop_callback, chkpt_callback])
        self.model.load_weights('./models/best_model')

    def train_xyz(self):
        self.generate_cluster_list()
        self.atoms2matrices(normalization=False, saturation_judgement=True)
        self.model = keras.Sequential([
            keras.layers.Dense(units=units, 
                               activation=tf.nn.relu, input_shape=(self.total_clst_cnt,)),
            keras.layers.Dense(units=units, 
                               activation=tf.nn.relu),
            keras.layers.Dense(units=units, 
                               activation=tf.nn.relu),
            keras.layers.Dense(units=units, 
                               activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])
        self.model.compile(optimizer='adam',
                        loss = 'binary_crossentropy',
                        metrics=['accuracy'])

        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=ptnc)
        chkpt_callback = tf.keras.callbacks.ModelCheckpoint(
            './models/best_model', monitor='val_loss', save_best_only=True, save_weights_only=True)
        os.system('rm -rf ./models/best_model')
        self.model.fit(self.train_images, self.train_labels, epochs=500,
                       batch_size=bat_sz, validation_split=0.1, callbacks=[earlystop_callback, chkpt_callback])
        self.model.load_weights('./models/best_model')

    def train_UC_FC(self): 
        self.train_xyz()
    def cluster_punish_func(self, dis, sym):
        min_dis_with_clst = 9999
        if(dis > 5):
            return 0
        for clst in self.clst_tuple_list:
            if(sym in clst[0]): 
                min_dis_with_clst = min(abs(clst[1]/1000 - dis), min_dis_with_clst)
        if(min_dis_with_clst == 9999):
            min_dis_with_clst = 0
        return min_dis_with_clst

    def loss(self, x, atoms, unsatured_idx):
        d = self.sature_or_not_distance
        too_close_val = 1
        scale = 1000
        power = 2
        at = atoms.copy()
        at.append(Atom('H', (x[0], x[1], x[2]))) 
        unsatured_total_dis = 0
        satured_total_dis = 0
        too_close_punish = 0
        cluster_punish = 0
        for i in range(len(at)-1):
            dis = at.get_distance(i, len(at)-1)

            if(dis < 0.001):
                dis = 0
            if(abs(dis-d) < 0.001):
                dis = d
            if(abs(dis-too_close_val) < 0.001):
                dis = too_close_val

            if(i in unsatured_idx):
                unsatured_total_dis += ((scale*abs(max(dis, d)-d))**power)
            else:
                satured_total_dis += ((scale*abs(min(dis, d)-d))**power)
            too_close_punish += ((scale*abs(min(dis, too_close_val) - too_close_val))**power)
        return unsatured_total_dis + satured_total_dis + too_close_punish
    def test_xyz(self):
        loss, acc = self.model.evaluate(self.test_images, self.test_labels)
        logging.info('LOSS: ' + str(loss) + ', ACC: ' + str(acc))
        test_predictions = self.model.predict(self.test_images)
        success = 0
        failure_1 = 0
        failure_0 = 0
        impossible = 0
        for idx, t in enumerate(test_predictions):
            if(abs(self.test_labels[idx] - t[0]) < 0.5):
                success += 1
            elif(t[0] > 0.5):
                failure_1 += 1
            elif(t[0] <= 0.5):
                failure_0 += 1
            else:
                impossible += 1
        print(success, failure_0, failure_1, impossible, success/len(list(test_predictions)))
        gen_db = connect('./test.db')
        rows = list(gen_db.select(sort='id'))
        bar01 = 0.5
        total_mean_cnt = 0
        total_mean_sum = 0
        total_opt_cnt = 0
        total_opt_sum = 0
        total_opt_fail = 0
        total_no_unsatured_cnts = 0
        for row in rows:
            answer_xyz = row.data.del_pos
            answer_sym = row.data.del_element
            atoms = self.sort_atoms(row)
            write('./traj/' + str(row.id)+'.traj', atoms)
            des = np.array(self.fill_descriptor_atom_saturation(row))
            answer_label = np.array(self.fill_labels_atom_saturation(row))
            predict_label = self.model.predict(des)
            unsatured_xyz = []
            unsatured_idx = []
            min_l = 999999
            min_l_idx = -1
            for idx, l in enumerate(predict_label):
                if(l < min_l):
                    min_l = l
                    min_l_idx = idx
                if(l < bar01):
                    unsatured_xyz.append(atoms.get_positions()[idx])
                    unsatured_idx.append(idx)
            if(len(unsatured_xyz)==0 and min_l_idx!=-1):
                unsatured_xyz.append(atoms.get_positions()[min_l_idx])
                unsatured_idx.append(min_l_idx)
                print('no under bar01, so choose the min as the unsatured candidate')
            mean_xyz = np.mean(np.array(unsatured_xyz), axis=0)
            if(len(unsatured_xyz) > 0):
                total_mean_cnt += 3
                total_mean_sum = total_mean_sum + abs(mean_xyz[0]-answer_xyz[0])\
                    + abs(mean_xyz[1]-answer_xyz[1])\
                    + abs(mean_xyz[2]-answer_xyz[2])
                x0 = np.asarray(tuple(mean_xyz))  
                res = minimize(self.loss, x0, args=(
                    atoms, unsatured_idx), method='CG') 
                if(res.success):
                    total_opt_cnt += 3
                    total_opt_sum = total_opt_sum + abs(res.x[0]-answer_xyz[0])\
                        + abs(res.x[1]-answer_xyz[1])\
                        + abs(res.x[2]-answer_xyz[2])
                else:
                    total_opt_fail += 1
                    total_opt_cnt += 3
                    total_opt_sum = total_opt_sum + abs(mean_xyz[0]-answer_xyz[0])\
                        + abs(mean_xyz[1]-answer_xyz[1])\
                        + abs(mean_xyz[2]-answer_xyz[2])
                    atoms.append(Atom('Au', (res.x[0], res.x[1], res.x[2]))) 
                    write('./traj/' + str(row.id)+'.traj', atoms)
                    print('-------------------------------------------------------------')
                    print('answer is: ', answer_xyz)
                    print('mean is: ', mean_xyz)
                    print('mean err is: ', total_mean_sum/total_mean_cnt)
                    print('row.id: ', row.id)
                    print('opt err is: ', total_opt_sum/total_opt_cnt)
                    print('loss: ', res.fun)
                    print('opt xyz: ', res.x)
                    print('opt msg: ', res.message)
                    print('opt Number of iterations: ', res.nit)
            else:
                total_no_unsatured_cnts += 1
        print('opt with mean if opt fail err: ', total_opt_sum/total_opt_cnt)
        print('there are total x atoms don\'t have unsatured atom, x=', total_no_unsatured_cnts)
        print('total opt fail: ', total_opt_fail)
        logging.info('LOSS: ' + str(loss) + ', ACC: ' + str(acc))
        return [total_opt_sum/total_opt_cnt, total_mean_sum/total_mean_cnt], loss

class Agent_BOC_regression(Agent_BOC):
    def train_xyz(self):
        self.generate_cluster_list()
        self.atoms2matrices(xyz=True, normalization=False, saturation_judgement=False)
        self.model = keras.Sequential([
            keras.layers.Dense(units, activation=tf.nn.relu, input_shape=(self.total_clst_cnt,)),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(3)
        ])
        logging.debug('train_xyz')
        self.model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error', 'mean_squared_error'])

        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=ptnc)
        chkpt_callback = tf.keras.callbacks.ModelCheckpoint(
            './models/best_model', monitor='val_loss', save_best_only=True, save_weights_only=True)
        os.system('rm -rf ./models/best_model')
        self.model.fit(self.train_images, self.train_labels, epochs=500,
                       batch_size=bat_sz, validation_split=0.1, callbacks=[earlystop_callback, chkpt_callback])
        self.model.load_weights('./models/best_model')
    def test_xyz(self):
        loss, mae, mse = self.model.evaluate(self.test_images, self.test_labels)
        logging.info(
            'LOSS: ' + str(loss)\
            + ', MAE: ' + str(mae)\
            + ', MSE: ' + str(mse))
        return [mae], loss 

class Agent_Sorted_Coulomb_Mat(Agent_2Body_Distance_Mat):
    def fill_descriptor(self, row):
        atoms = self.sort_atoms(row)
        img = np.zeros([mat_sz, mat_sz])
        for j in range(len(atoms)):
            for k in range(len(atoms)):
                if(atoms.get_distance(j, k) < 0.00000000000001):
                    continue
                else:
                    img[j][k] = atoms.numbers[j]*atoms.numbers[k]/atoms.get_distance(j, k)*0.529
            img[j][j] = 0.5*(atoms.numbers[j]**2.4)
        return img
    def fill_descriptor_atom_saturation(self, row):
        atoms = self.sort_atoms(row)
        img = np.zeros([mat_sz, mat_sz])
        ret = []
        for j in range(len(atoms)):
            for k in range(len(atoms)):
                if(j == k):
                    img[j][0] = 0.5*(atoms.numbers[j]**2.4)
                elif(j > k):
                    img[j][k+1] = atoms.numbers[j]*atoms.numbers[k]/atoms.get_distance(j, k)*0.529
                elif(j < k):
                    img[j][k] = atoms.numbers[j]*atoms.numbers[k]/atoms.get_distance(j, k)*0.529
                else:
                    pass

        for idx in range(len(img)):
            if(img[idx][1] != 0):
                ret.append(img[idx])
        return ret

class Agent_Electron_Counting(Agent_2Body_Distance_Mat):
    ucfc_input_sz = 2
    electron_dict = {'H': 1, 'C':6, 'O':8, 'F':9, 'N':7}
    def fill_descriptor_atom_saturation(self, row):
        atoms = self.sort_atoms(row)
        ret = []
        for j in range(len(atoms)):
            img = np.zeros(2)
            img[0] = self.atom_dict[atoms[j].symbol]
            for k in range(len(atoms)):
                if(j == k):
                    continue
                else:
                    if(atoms.get_distance(j, k) < self.ucfc_cutoff[atoms[j].symbol]):
                        img[1] += self.electron_dict[atoms[k].symbol]
            ret.append(img)
        return ret
    def atoms2matrices_ucfc(self):
        self.atoms2matrices(xyz=False, normalization=False, saturation_judgement=True)

class Agent_Sorted_Coulomb_2Body_Combine(Agent_2Body_Distance_Mat):
    def fill_descriptor(self, row):
        atoms = self.sort_atoms(row)
        img = np.zeros([mat_sz, mat_sz])
        for j in range(len(atoms)):
            for k in range(len(atoms)):
                if(atoms.get_distance(j, k) < 0.00000000000001):
                    continue
                else:
                    if(j>k):
                        img[j][k] = atoms.numbers[j]*atoms.numbers[k]/atoms.get_distance(j, k)*0.529
                    else:
                        img[j][k] = atoms.get_distance(j, k)
            img[j][j] = 0.5*(atoms.numbers[j]**2.4)
        return img

class Agent_Sorted_Coulomb_2Body_Combine_reciprocal(Agent_2Body_Distance_Mat):
    def fill_descriptor(self, row):
        atoms = self.sort_atoms(row)
        img = np.zeros([mat_sz, mat_sz])
        for j in range(len(atoms)):
            for k in range(len(atoms)):
                if(atoms.get_distance(j, k) < 0.00000000000001):
                    continue
                else:
                    if(j>k):
                        img[j][k] = atoms.numbers[j]*atoms.numbers[k]/atoms.get_distance(j, k)*0.529
                    else:
                        img[j][k] = 1.0/atoms.get_distance(j, k)
            img[j][j] = 0.5*(atoms.numbers[j]**2.4)
        return img

class Agent_Electro_Neg(Agent_2Body_Distance_Mat):
    def fill_descriptor(self, row):
        atoms = self.sort_atoms(row)
        img = np.zeros([mat_sz, mat_sz])
        total_electron = 0
        total_modified_electron = 0
        for i in range(len(atoms)):
            total_electron += atoms.numbers[i]
            total_modified_electron += \
                (atoms.numbers[i] * self.electro_negtive[atoms.numbers[i]])
        for i in range(len(atoms)):
            img[i][i] = atoms.numbers[i] * self.electro_negtive[atoms.numbers[i]]/\
                total_modified_electron*total_electron

        for j in range(len(atoms)):
            for k in range(len(atoms)):
                if(atoms.get_distance(j, k) < 0.00000000000001):
                    continue
                else:
                    if(j>k):
                        img[j][k] =\
                        img[j][j]*img[k][k]/\
                            atoms.get_distance(j, k)*0.529
                    else:
                        img[j][k] = atoms.get_distance(j, k)
        for i in range(len(atoms)):
            img[i][i] =  0.5*(img[i][i]**2.4)

        return img

class Agent_1D_xyz_pc(Agent_2Body_Distance_Mat): 
    ucfc_input_sz = mat_sz * 4
    def fill_descriptor(self, row):
        atoms = self.sort_atoms(row)
        img = np.zeros([mat_sz * 4,])
        for j in range(len(atoms)):
            img[j*4]   = atoms.positions[j][0]
            img[j*4+1] = atoms.positions[j][1]
            img[j*4+2] = atoms.positions[j][2]
            img[j*4+3] = atoms.numbers[j]
        return img

    def fill_descriptor_atom_saturation(self, row):
        atoms = self.sort_atoms(row)
        img = np.zeros([mat_sz*4, mat_sz*4])
        ret = []
        for j in range(len(atoms)):
            img[j][j*4] = atoms.positions[j][0]
            img[j][j*4+1] = atoms.positions[j][1]
            img[j][j*4+2] = atoms.positions[j][2]
            img[j][j*4+3] = atoms.numbers[j]
        for idx in range(len(atoms)):
            ret.append(img[idx])
        return ret

    def train(self):
        self.atoms2matrices()
        self.model = keras.Sequential([
            keras.layers.Dense(units, activation=tf.nn.relu, input_shape=(mat_sz*4,)),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(5, activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        logging.debug('train')
        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=ptnc)
        chkpt_callback = tf.keras.callbacks.ModelCheckpoint(
            './models/best_model', monitor='val_loss', save_best_only=True, save_weights_only=True)
        os.system('rm -rf ./models/best_model')
        self.model.fit(self.train_images, self.train_labels, epochs=500,
                       batch_size=bat_sz, validation_split=0.1, callbacks=[earlystop_callback, chkpt_callback])
        self.model.load_weights('./models/best_model')

    def train_xyz(self):
        self.atoms2matrices_xyz()
        self.model = keras.Sequential([
            keras.layers.Dense(units, activation=tf.nn.relu, input_shape=(mat_sz*4,)),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(units, activation=tf.nn.relu),
            keras.layers.Dense(3)
        ])
        logging.debug('train_xyz')
        self.model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error', 'mean_squared_error'])

        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=ptnc)
        chkpt_callback = tf.keras.callbacks.ModelCheckpoint(
            './models/best_model', monitor='val_loss', save_best_only=True, save_weights_only=True)
        os.system('rm -rf ./models/best_model')
        self.model.fit(self.train_images, self.train_labels, epochs=500,
                       batch_size=bat_sz, validation_split=0.1, callbacks=[earlystop_callback, chkpt_callback])
        self.model.load_weights('./models/best_model')

class Agent_2Body_Diagonal_proton_cnt(Agent_2Body_Distance_Mat):
    def fill_descriptor(self, row):
        atoms = self.sort_atoms(row)
        img = np.zeros([mat_sz, mat_sz])
        for j in range(len(atoms)):
            for k in range(len(atoms)):
                img[j][k] = atoms.get_distance(j, k)
            img[j][j] = atoms.numbers[j]
        return img

    def fill_descriptor_atom_saturation(self, row):
        atoms = self.sort_atoms(row)
        img = np.zeros([mat_sz, mat_sz])
        ret = []
        for j in range(len(atoms)):
            for k in range(len(atoms)):
                if(j == k):
                    img[j][0] = self.atom_dict[atoms[j].symbol]
                elif(j > k):
                    img[j][k+1] = atoms.get_distance(j, k)
                elif(j < k):
                    img[j][k] = atoms.get_distance(j, k)
                else:
                    pass

        for idx in range(len(img)):
            if(img[idx][1] != 0):
                ret.append(img[idx])
        return ret

import argparse
if __name__ == '__main__':
    os.system('mkdir traj')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required = True,
            choices=['type', 'xyz', 'ucfc'],
            help = '\'xyz\' for positon, \
            \'type\' for missing element classification.\
            \'ucfc\' for test atom UC or FC (unsatured or satured)')
    args = parser.parse_args()
    run_times = 10
    class_types = 9
    acc_lst = np.zeros([class_types, run_times])
    loss_lst = np.zeros([class_types, run_times])
    for i in range(run_times):
        data_maker = Data_Generater()
        data_maker.gen_train_validation_test_dataset(110000, 0)
        agent_boc = Agent_BOC()
        agent_1dxyz = Agent_1D_xyz_pc()
        agent_2body = Agent_2Body_Distance_Mat()
        agent_2body_pc = Agent_2Body_Diagonal_proton_cnt()
        agent_scm = Agent_Sorted_Coulomb_Mat()
        agent_scm_2body = Agent_Sorted_Coulomb_2Body_Combine()
        agent_scm_2body_re = Agent_Sorted_Coulomb_2Body_Combine_reciprocal()
        agent_electro_neg = Agent_Electro_Neg()
        agent_electro_counting = Agent_Electron_Counting()

        max_acc = 0
        max_idx = 0
        cur_idx = 0
        for agent in [agent_boc, agent_1dxyz, agent_2body, agent_2body_pc,
                agent_scm, agent_scm_2body, agent_scm_2body_re]: 

            if(args.mode == 'type'):
                agent.train()
                acc, loss = agent.test()
                acc_lst[cur_idx][i] = acc
                loss_lst[cur_idx][i] = loss
            elif(args.mode == 'xyz'):
                agent.train_xyz()
                acc, loss = agent.test_xyz()
                acc_lst[cur_idx][i] = acc[0]
                if(len(acc) > 1):
                    cur_idx += 1
                    acc_lst[cur_idx][i] = acc[1]
                loss_lst[cur_idx][i] = loss
            elif(args.mode == 'ucfc'):
                agent.train_UC_FC()
                acc, loss = agent.test_UC_FC()
                acc_lst[cur_idx][i] = acc
                loss_lst[cur_idx][i] = loss
            else:
                pass
            print(acc_lst)
            print(loss_lst)
            cur_idx += 1
        os.system('mkdir ' + str(i))
        if(args.mode == 'type'):
            np.savetxt('acc_lst_'+str(i), acc_lst)
            os.system('mv acc_lst_'+str(i) + ' '+str(i))
        elif(args.mode == 'xyz'):
            np.savetxt('dis_err_lst_'+str(i), acc_lst)
            os.system('mv dis_err_lst_'+str(i) + ' '+str(i))
        elif(args.mode == 'ucfc'):
            np.savetxt('ucfc_acc_lst_'+str(i), acc_lst)
            os.system('mv ucfc_acc_lst_'+str(i) + ' '+str(i))
        else:
            pass
        np.savetxt('loss_lst_'+str(i), loss_lst)
        os.system('mv test.db ' + str(i))
        os.system('mv vali.db ' + str(i))
        os.system('mv train.db ' + str(i))
        os.system('mv loss_lst_'+str(i) + ' '+str(i))
EOF
