

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from pylightgbm.models import GBMRegressor
from sklearn.grid_search import GridSearchCV
from bayes_opt import BayesianOptimization
from IPython.display import display
import json
import pickle
import sys
import subprocess
import time
from scipy.optimize import minimize
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler

ID = 'id'
TARGET = 'loss'
NFOLDS = 4
SEED = 0
NROWS = None
DATA_DIR = "../../input"

TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
TEST_FILE = "{0}/test.csv".format(DATA_DIR)
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)
USE_PICKLED = False

if not USE_PICKLED:
    print("Loading training data from {}".format(TRAIN_FILE))
    train = pd.read_csv(TRAIN_FILE, nrows=NROWS)
    print("Loading test data from {}".format(TEST_FILE))
    test = pd.read_csv(TEST_FILE, nrows=NROWS)
    y_train = train[TARGET].ravel()
    train.drop([ID, TARGET], axis=1, inplace=True)
    test.drop([ID], axis=1, inplace=True)
    print("Data shapes: Train = {}, Test = {}".format(train.shape, test.shape))
    ntrain = train.shape[0]
    ntest = test.shape[0]
    train_test = pd.concat((train, test)).reset_index(drop=True)
    features = train.columns
    cats = [feat for feat in features if 'cat' in feat]
    for feat in cats:
        train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]
    numeric_feats = [ feat for feat in features if 'cont' in feat ]
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    print("\nSkew in numeric features:")
    print(skewed_feats)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])

    print ("Head ( train_test ) : ")
    print (train_test.head())

    x_train = np.array(train_test.iloc[:ntrain,:])
    x_test = np.array(train_test.iloc[ntrain:,:])
    with open('data.pkl', 'wb') as pkl_file:
        pickle.dump( (x_train, x_test, y_train), pkl_file)
else:
    with open('data.pkl', 'rb') as pkl_file:
        (x_train, x_test, y_train) = pickle.load(pkl_file)
        ntrain = x_train.shape[0]
        ntest  = x_test.shape[0]

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=False, random_state=SEED)

def get_timestamp():
    return time.strftime("%Y%m%dT%H%M%S")

def load_submission(DATA_DIR="../../input"):
    SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)
    submission = pd.read_csv(SUBMISSION_FILE)
    return submission

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train, x_valid, y_valid):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 10000)
        self.shift = params.pop('shift', 200)
        self.num_folds_trained = 1

    def evalerror(self, preds, dtrain):
        return 'mae', mean_absolute_error(np.exp(preds) - self.shift,
                                          np.exp(dtrain.get_label()) - self.shift)

    def train(self, x_train, y_train, x_valid, y_valid):
        dtrain = xgb.DMatrix(x_train, label=np.log(y_train + self.shift))
        eval_error_func = lambda p,d : self.evalerror(p,d)

        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return np.exp(self.gbdt.predict(xgb.DMatrix(x))) - self.shift

class LightgbmWrapper(object):
    def __init__(self, seed=0, params=None):
        self.params = params
        self.clf = GBMRegressor(**params)
        self.params['seed'] = seed

    def train(self, x_train, y_train, x_valid, y_valid):
        if self.params['application'] == "regression":
            self.clf.fit(x_train, np.log1p(y_train), [(x_valid, np.log1p(y_valid))])
        else:
            self.clf.fit(x_train, y_train, [(x_valid, y_valid)])

    def predict(self, x):
        if self.params['application'] == "regression":
            return np.expm1(self.clf.predict(x))
        else:
            return self.clf.predict(x)

def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        y_te = y_train[test_index]

        clf.train(x_tr, y_tr, x_te, y_te)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def save_results_to_json(name, params, oof_test, oof_train):
    with open(name + '.json', 'w') as outfile:
        json.dump({ 'name': name,
                    'params': params,
                    'oof_train': oof_train.tolist(),
                    'oof_test': oof_test.tolist() },
                  outfile,
                  indent = 4,
                  sort_keys = False)

def load_results_from_json(filename):
    with open(filename, 'r') as infile:
        res = json.load(infile)
        res['oof_train'] = np.array(res['oof_train'])
        res['oof_test']  = np.array(res['oof_test'])
        print('Loaded {}'.format(res['name']))
        return res

def load_results_from_csv(oof_train_filename, oof_test_filename):
    res = dict({'name': oof_train_filename})
    res['oof_train'] = np.array(pd.read_csv(oof_train_filename)['loss']).reshape(-1,1)
    res['oof_test']  = np.array(pd.read_csv(oof_test_filename)['loss']).reshape(-1,1)
    print('Loaded {}'.format(res['name']))
    return res

LOAD_L1_FROM_CSV = True
if LOAD_L1_FROM_CSV:
    print('Loading level-0 models from csv\'s')
    res_lg_fair = load_results_from_json('model_fair_c_2_w_100_lr_0.002_trees_20K.json')
    res_et = load_results_from_json('model_et.json')
    res_rf_depth_12 = load_results_from_json('skl_rf-20161108T012715.json')
    res_keras = load_results_from_csv(
            'keras_1/oof_train_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv',
            'keras_1/submission_keras_400_0.4_200_0.2_nbags_4_nepochs_55_nfolds_5.csv')
    res_keras_2 = load_results_from_csv(
            'keras_2/oof_train.csv',
            'keras_2/submission.csv')
    res_keras_3 = load_results_from_csv(
            'keras_3/oof_train.csv',
            'keras_3/submission.csv')
    res_keras_A4 = load_results_from_csv(
            'keras_A4/oof_train.csv',
            'keras_A4/submission.csv')
    res_lg_l2_1 = load_results_from_json('model_l2_lr_0.01_trees_7K.json')
    res_lg_l2_2 = load_results_from_json('model_l2_bopt_run1_index75.json')
    res_lg_l2_3 = load_results_from_json('model_l2_bopt_run2_index92.json')
    res_xgb = load_results_from_json('model_xgb_3.json')
    res_xgb_vlad_bs1_fast = load_results_from_json('../stacked_ensembles/test_xgb_A1-20161103T205502.json')
    res_xgb_vlad_bs1 = load_results_from_json('../stacked_ensembles/xgb_Vladimir_base_score_1-20161103T230621.json')
    res_xgb_1117_A1 = load_results_from_json('./xgb-1117-attempt1-20161107T234751.json')
    def elem_wise_mean(x, y):
        shp = x.shape
        return np.mean(np.concatenate((x, y), axis=1), axis=1).reshape(shp)
    res_AM_1 = {
            'name': 'mean_keras_2_xgb_Vladimir_base_score_1-20161103T230621',
            'oof_train': elem_wise_mean(res_keras_2['oof_train'], res_xgb_vlad_bs1['oof_train']),
            'oof_test': elem_wise_mean(res_keras_2['oof_test'], res_xgb_vlad_bs1['oof_test']) }
    res_keras_A5_1 = load_results_from_csv(
            './keras_A5_1/oof_train.csv',
            './keras_A5_1/oof_test.csv' )
    res_keras_A5_2 = load_results_from_csv(
            './keras_A5_2/oof_train.csv',
            './keras_A5_2/oof_test.csv' )
    res_lgbm_l2_ll_extremin_1112_1 = load_results_from_json('lgbm_l2_loglossshift_extremin_1112-20161110T023325.json')
    res_lgbm_l2_ll_extremin_1112_2 = load_results_from_json('lgbm_l2_loglossshift_extremin_1112-lr_0.003-20161110T031110.json')
    res_lgbm_l2_ll_extremin_1112_3 = load_results_from_json('lgbm_l2_loglossshift500_extremin_1112-lr_0.01-20161113T203540.json')
    res_keras_A5_11 = load_results_from_csv(
            './keras_A5_11/oof_train.csv',
            './keras_A5_11/oof_test.csv' )
    res_keras_A6_1 = load_results_from_csv(
            './keras_A6_1/oof_train.csv',
            './keras_A6_1/oof_test.csv' )
    res_keras_A6_2 = load_results_from_csv(
            './keras_A6_2/oof_train.csv',
            './keras_A6_2/oof_test.csv' )
    res_keras_A7_1 = load_results_from_csv(
            './keras_A7_1/oof_train.csv',
            './keras_A7_1/oof_test.csv' )
    res_keras_A8_1 = load_results_from_csv(
            './keras_A8_1/oof_train.csv',
            './keras_A8_1/oof_test.csv' )
    res_xgb_1109_A1 = load_results_from_json('xgb-1109-attempt_1-20161114T223803.json')
    res_xgb_1109_A2 = load_results_from_json('xgb-1109-attempt_2-20161115T231535.json')
    res_xgb_1109_A3 = load_results_from_json('xgb-1109-attempt_3-20161116T091620.json')
    res_ridge_quad_1000_1  = load_results_from_json('skl_Ridge-quad-100020161117T103454.json')
    res_ridge_quad_1000_2  = load_results_from_json('skl_Ridge-quad-100020161117T104911.json')
    res_ridge_quad_1000_3  = load_results_from_json('skl_Ridge-quad-100020161117T110254.json')
    res_ridge_quad_1000_4  = load_results_from_json('skl_Ridge-quad-100020161117T111651.json')
    res_ridge_quad_1000_5  = load_results_from_json('skl_Ridge-quad-100020161117T113232.json')
    res_ridge_quad_1000_6  = load_results_from_json('skl_Ridge-quad-100020161117T114551.json')
    res_ridge_quad_1000_7  = load_results_from_json('skl_Ridge-quad-100020161117T120118.json')
    res_ridge_quad_1000_8  = load_results_from_json('skl_Ridge-quad-100020161117T121525.json')
    res_ridge_quad_1000_9  = load_results_from_json('skl_Ridge-quad-100020161117T123053.json')
    res_ridge_quad_1000_10 = load_results_from_json('skl_Ridge-quad-100020161117T124430.json')
    res_xgb_1109_A4 = load_results_from_json('xgb-1109-attempt_4-base_score_7.7-20161119T202505.json')
    res_xgb_1109_A5 = load_results_from_json('xgb-1109-attempt_5-preprocessor_pow_0.25_shift_1-20161119T225103.json')
    res_xgb_1109_A5b = load_results_from_json('xgb-1109-attempt_5-preprocessor_pow_0.2_shift_10-20161120T055959.json')
    res_xgb_1109_A8 = load_results_from_json('xgb-1109-attempt_8-obj_reglinear-preprocessor_pow_0.2_shift_10-20161120T225122.json')
    res_xgb_1109_A9 = load_results_from_json('xgb-1109-attempt_9-obj_reglinear-preprocessor_log_shift_200-20161121T092757.json')
    res_xgb_mrooijer_1 = load_results_from_json('xgb-mrooijer_1130-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T215220.json')
    res_xgb_mrooijer_2 = load_results_from_json('xgb-mrooijer_1130-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161121T232803.json')
    res_xgb_mrooijer_5 = load_results_from_json('xgb-mrooijer_1130-attempt_5-obj_fair0.7-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161122T214600.json')
    res_xgb_mrooijer_lin_A1 = load_results_from_json('xgb-mrooijer-lin-attempt_1-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T001744.json')
    res_xgb_mrooijer_lin_A2 = load_results_from_json('xgb-mrooijer-lin-attempt_2-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161123T013341.json')
    w = 65
    p = 7
    AM_2_train = (w/100*res_xgb_mrooijer_lin_A2['oof_train'] + res_keras_2['oof_train']*(1-w/100))**(1+p/10000)
    AM_2_test  = (w/100*res_xgb_mrooijer_lin_A2['oof_test']  + res_keras_2['oof_test']*(1-w/100))**(1+p/10000)
    res_AM_2 = {
            'name': 'weighted_mean_keras_2_xgb_moo_lin_A2_w_65_p_7',
            'oof_train': AM_2_train,
            'oof_test': AM_2_test }
    res_keras_A9_1 = load_results_from_csv(
            './keras_A9_1/oof_train.csv',
            './keras_A9_1/oof_test.csv' )
    res_keras_A9_2 = load_results_from_csv(
            './keras_A9_2/oof_train.csv',
            './keras_A9_2/oof_test.csv' )
    res_keras_A9_3 = load_results_from_csv(
            './keras_A9_3/oof_train.csv',
            './keras_A9_3/oof_test.csv' )
    res_keras_A9_4 = load_results_from_csv(
            './keras_A9_4/oof_train.csv',
            './keras_A9_4/oof_test.csv' )
    res_keras_A9_5 = load_results_from_csv(
            './keras_A9_5/oof_train.csv',
            './keras_A9_5/oof_test.csv' )
    res_keras_A9_5_adadelta = load_results_from_csv(
            './keras_A9_5_adadelta/oof_train.csv',
            './keras_A9_5_adadelta/oof_test.csv' )
    res_keras_A9_6 = load_results_from_csv(
            './keras_A9_6_nfold10_nbags3_batch128/oof_train.csv',
            './keras_A9_6_nfold10_nbags3_batch128/oof_test.csv' )
    res_keras_10_A1 = load_results_from_csv(
            './keras_10_A1/oof_train.csv',
            './keras_10_A1/oof_test.csv' )
    res_xgb_moo_lin_A3 = load_results_from_json('xgb-mrooijer-lin-attempt_3-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161203T211419.json')
    res_xgb_moo_lin_A4 = load_results_from_json('xgb-mrooijer-lin-attempt_4-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161204T123341.json')
    res_xgb_moo_lin_A5 = load_results_from_json('xgb-mrooijer-lin-attempt_5-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161205T101649.json')
    res_xgb_moo_lin_A6 = load_results_from_json('xgb-mrooijer-lin-attempt_6-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161207T230841.json')
    res_keras_classify_10_A1 = load_results_from_csv(
            './keras_classify_10_A1/oof_train.csv',
            './keras_classify_10_A1/oof_test.csv' )
    res_keras_11_BN_deep_arch_A1 = load_results_from_csv(
            './keras_11_BN_deep_arch_A1/oof_train.csv',
            './keras_11_BN_deep_arch_A1/oof_test.csv' )
    res_lgbm_fair15_A1 = load_results_from_json('lgbm_fair_c_1.5_w_100-leaf_2000-lr_0.01-max_3500_trees-20161209T182238.json')
    res_xgb_moo_lin_A8 = load_results_from_json('xgb-mrooijer-lin-attempt_8-obj_fair1.5-preprocessor_pow_0.32_shift_1-base_score-13.60279000422951-20161210T172004.json')
    res_xgb_moo_lin_A9 = load_results_from_json('xgb-mrooijer-lin-attempt_9-obj_fair0.7-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161210T194415.json')
    res_xgb_moo_lin_A11 = load_results_from_json('xgb-mrooijer-lin-attempt_11-obj_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161211T011318.json')

    res_keras_12_mae_on_pow_032_shift_1_Stratified = load_results_from_csv(
            './keras_12_mae_on_pow_0.32_shift_1_Stratified/oof_train.csv',
            './keras_12_mae_on_pow_0.32_shift_1_Stratified/oof_test.csv' )

    res_xgb_moo_lin_A12 = load_results_from_json('xgb-mrooijer-lin-attempt_12-obj_logy_fair1.5-preprocessor_pow_0.25_shift_1-base_score-8.4738505029380491-20161211T233814.json')
    res_xgb_moo_lin_A13 = load_results_from_json('xgb-mrooijer-lin-attempt_13-obj_fair0.7-preprocessor_pow_0.32_shift_1-base_score-13.60279000422951-20161212T001354.json')

    res_array = [
            res_lg_fair,
            res_et,
            res_rf_depth_12,
            res_keras,
            res_keras_2,
            res_keras_3,
            res_keras_A4,
            res_lg_l2_1,
            res_lg_l2_2,
            res_lg_l2_3,
            res_xgb,
            res_xgb_vlad_bs1_fast,
            res_xgb_vlad_bs1,
            res_xgb_1117_A1,
            res_keras_A5_1,
            res_keras_A5_2,
            res_keras_A5_11,
            res_lgbm_l2_ll_extremin_1112_1,
            res_lgbm_l2_ll_extremin_1112_2,
    	    res_lgbm_l2_ll_extremin_1112_3,
            res_keras_A6_1,
            res_keras_A6_2,
            res_keras_A7_1,
            res_keras_A8_1,
            res_xgb_1109_A1,
            res_xgb_1109_A2,
            res_xgb_1109_A3,
            res_ridge_quad_1000_1,
            res_ridge_quad_1000_2,
            res_ridge_quad_1000_3,
            res_ridge_quad_1000_4,
            res_ridge_quad_1000_5,
            res_ridge_quad_1000_6,
            res_ridge_quad_1000_7,
            res_ridge_quad_1000_8,
            res_ridge_quad_1000_9,
            res_ridge_quad_1000_10,
            res_xgb_1109_A4,
            res_xgb_1109_A5,
            res_xgb_1109_A5b,
            res_xgb_1109_A8,
            res_xgb_1109_A9,
            res_xgb_mrooijer_1,
            res_xgb_mrooijer_2,
            res_xgb_mrooijer_5,
            res_xgb_mrooijer_lin_A1,
            res_xgb_mrooijer_lin_A2,
            res_keras_A9_1,
            res_keras_A9_2,
            res_keras_A9_4,
            res_keras_A9_5,
            res_keras_A9_5_adadelta,
            res_keras_A9_6,
            res_keras_10_A1,
            res_AM_2,
            res_xgb_moo_lin_A3,
            res_xgb_moo_lin_A4,
            res_xgb_moo_lin_A5,
            res_xgb_moo_lin_A6,
            res_keras_classify_10_A1,
            res_keras_11_BN_deep_arch_A1,
            res_lgbm_fair15_A1,
            res_xgb_moo_lin_A8,
            res_xgb_moo_lin_A9,
            res_xgb_moo_lin_A11,
            res_keras_12_mae_on_pow_032_shift_1_Stratified,
            res_xgb_moo_lin_A12,
            res_xgb_moo_lin_A13,
            ]
    l1_x_train = np.concatenate([r['oof_train'] for r in res_array], axis=1)
    l1_x_test  = np.concatenate([r['oof_test' ] for r in res_array], axis=1)
    with open('l1_data.pkl', 'wb') as pkl_file:
        pickle.dump( (l1_x_train, l1_x_test, y_train, res_array), pkl_file)
else:
    print('Loading level-0 models from l1_data.pkl')
    with open('l1_data.pkl', 'rb') as pkl_file:
        (l1_x_train, l1_x_test, y_train, res_array) = pickle.load(pkl_file)

for i, r in enumerate(res_array):
    cv_err  = np.abs(y_train - r['oof_train'].flatten())
    cv_mean = np.mean(cv_err)
    cv_std  = np.std(cv_err)
    print ("Model {0}: \tCV = {2:.3f}+{3:.1f}, \tName = {1} ".format(
        i, r['name'], cv_mean, cv_std))

l1_x_train[l1_x_train < 0] = 1
l1_x_test[l1_x_test < 0] = 1

if False:
    l1_x_train = np.concatenate([x_train, l1_x_train], axis=1)
    l1_x_test  = np.concatenate([x_test,  l1_x_test],  axis=1)
if True:
    lin_train = pd.read_csv('../../input/all-the-allstate-dates-eda/lin_train.csv')
    lin_test  = pd.read_csv('../../input/all-the-allstate-dates-eda/lin_test.csv')
    lin_feats = [feat for feat in lin_train.columns if 'lin_cont' in feat]
    print(lin_feats)
    lin_train_array = lin_train[lin_feats].as_matrix()
    lin_test_array  = lin_test[lin_feats].as_matrix()
    l1_x_train = np.concatenate([lin_train_array, l1_x_train], axis=1)
    l1_x_test  = np.concatenate([lin_test_array,  l1_x_test],  axis=1)
if True:
    COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79'.split(',')
    for c in COMB_FEATURE:
        dummy = pd.get_dummies(train_test[c].astype('category'))
        l1_x_train = np.concatenate([l1_x_train, dummy[:ntrain]], axis=1)
        l1_x_test  = np.concatenate([l1_x_test,  dummy[ntrain:]], axis=1)

print("{},{}".format(l1_x_train.shape, l1_x_test.shape))

if False:
    et_params = {
        'n_jobs': 4,
        'n_estimators': 169,
        'max_features': 0.999,
        'max_depth': 12,
        'min_samples_leaf': 12,
        'verbose': 1,
    }
    et = SklearnWrapper(clf=ExtraTreesRegressor, seed=SEED, params=et_params)
    et_oof_train, et_oof_test = get_oof(et)
    save_results_to_json('model_et', et_params, et_oof_test, et_oof_train)

if False:
    rf_params = {
        'n_jobs': 4,
        'n_estimators': 10,
        'max_features': 0.2,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 1,
    }
    rf = SklearnWrapper(clf=RandomForestRegressor, seed=SEED, params=rf_params)
    rf_oof_train, rf_oof_test = get_oof(rf)
    save_results_to_json('model_rf', rf_params, rf_oof_test, rf_oof_train)

if False:
    xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.5,
        'silent': 1,
        'subsample': 0.8,
        'learning_rate': 0.01,
        'max_depth': 12,
        'num_parallel_tree': 1,
        'min_child_weight': 1,
        'nrounds': 1800,
        'alpha': 1,
        'gamma': 1,
        'verbose_eval': 10,
        'shift': 200
    }
    xg = XgbWrapper(seed=SEED, params=xgb_params)
    xgb_oof_train, xgb_oof_test = get_oof(xg)

if False:
    lightgbm_params_fair = {
        'exec_path': '../../../LightGBM/lightgbm',
        'config': '',
        'application': 'regression-fair',
        'fair_constant': 15.16,
        'fair_scaling': 194.388,
        'num_iterations': 50000,
        'learning_rate': 0.00588,
        'num_leaves': 107,
        'tree_learner': 'serial',
        'num_threads': 4,
        'min_data_in_leaf': 2,
        'metric': 'l1',
        'feature_fraction': 0.6665121,
        'feature_fraction_seed': SEED,
        'bagging_fraction': 0.96029,
        'bagging_freq': 3,
        'bagging_seed': SEED,
        'metric_freq': 100,
        'early_stopping_round': 100,
    }
    lg_fair = LightgbmWrapper(seed=SEED, params=lightgbm_params_fair)
    lg_oof_train_fair, lg_oof_test_fair = get_oof(lg_fair)
    save_results_to_json('model_fair_c_15.16_w_194.388_lr_0.00588_trees_50K', lightgbm_params_fair, lg_oof_test_fair, lg_oof_train_fair)

if False:
    lightgbm_params_l2 = {
        'exec_path': '../../../LightGBM/lightgbm',
        'config': '',
        'application': 'regression',
        'num_iterations': 2000,
        'learning_rate': 0.0251188,
        'num_leaves': 107,
        'tree_learner': 'serial',
        'num_threads': 4,
        'min_data_in_leaf': 215,
        'metric': 'l1exp',
        'feature_fraction': 0.6665121,
        'bagging_fraction': 0.9602939,
        'bagging_freq': 3,
        'metric_freq': 10,
        'early_stopping_round': 100,
        'verbose': True
    }
    lg_l2 = LightgbmWrapper(seed=SEED, params=lightgbm_params_l2)
    lg_oof_train_l2, lg_oof_test_l2 = get_oof(lg_l2)

if False:
    dtrain = xgb.DMatrix(l1_x_train, label=y_train)
    dtest = xgb.DMatrix(l1_x_test)
    max_depth = 6
    min_child_weight = 1
    gamma = 1
    xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.8,
        'silent': 1,
        'subsample': 0.6,
        'learning_rate': 0.01,
        'objective': 'reg:linear',
        'max_depth': int(max_depth), 
        'num_parallel_tree': 1,
        'min_child_weight': int(min_child_weight), 
        'eval_metric': 'mae',
        'gamma': gamma,
    }
    res = xgb.cv(xgb_params, dtrain, num_boost_round=5000, nfold=NFOLDS, seed=SEED,
                 stratified=False, early_stopping_rounds=25, verbose_eval=25, show_stdv=True)
    best_nrounds = res.shape[0] - 1
    cv_mean = res.iloc[-1, 0]
    cv_std = res.iloc[-1, 1]
    print('Ensemble-CV: {0:.3f}+{1:.1f}'.format(cv_mean, cv_std))
    print('Best Rounds: {}'.format(best_nrounds))
    gbdt = xgb.train(xgb_params, dtrain, int(best_nrounds * (1)))
    submission = pd.read_csv(SUBMISSION_FILE)
    submission.iloc[:, 1] = gbdt.predict(dtest)
    submission.to_csv('xgstacker.sub.csv', index=None)

if True:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import KFold, StratifiedKFold
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Dropout, Activation, Lambda
    from keras.layers.normalization import BatchNormalization
    from keras.layers.advanced_activations import PReLU
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint
    from keras.regularizers import l2
    def batch_generator(X, y, batch_size, shuffle):
        number_of_batches = np.ceil(X.shape[0]/batch_size)
        counter = 0
        sample_index = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(sample_index)
        while True:
            batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
            X_batch = X[batch_index,:]
            y_batch = y[batch_index]
            counter += 1
            yield X_batch, y_batch
            if (counter == number_of_batches):
                if shuffle:
                    np.random.shuffle(sample_index)
                counter = 0
    def batch_generatorp(X, batch_size, shuffle):
        number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
        counter = 0
        sample_index = np.arange(X.shape[0])
        while True:
            batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
            X_batch = X[batch_index, :]
            counter += 1
            yield X_batch
            if (counter == number_of_batches):
                counter = 0
    def nn_model():
        model = Sequential()
        model.add(Dense(100, input_dim = l1_x_train.shape[1], init = 'he_normal'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.3))
        model.add(Dense(50, init = 'he_normal'))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.3))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(loss = 'mae', optimizer = 'adam')
        return(model)
    nfolds = 5
    nbags = 3
    nepochs = 60
    es_patience = 10
    nbags_per_fold = 1
    train_batch_size = 128
    USE_STRATIFIED = True
    pred_oob = np.zeros(l1_x_train.shape[0])
    pred_test = np.zeros(l1_x_test.shape[0])
    for J in range(nbags):
        i = 0

        if USE_STRATIFIED:
            perc_values = np.percentile(y_train, range(10,101,10))
            y_perc = np.zeros_like(y_train)
            for v in perc_values[:-1]:
                y_perc += (y_train > v)
            folds = StratifiedKFold(y_perc, n_folds=nfolds, shuffle=True, random_state=None)
        else:
            folds = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=None)

        for (inTr, inTe) in folds:
            xtr = l1_x_train[inTr]
            ytr = y_train[inTr]
            xte = l1_x_train[inTe]
            yte = y_train[inTe]
            pred = np.zeros(xte.shape[0])
            for j in range(nbags_per_fold):
                model = nn_model()
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=es_patience,
                        verbose=0, mode='auto'),
                    ModelCheckpoint('keras.best.hdf5', monitor='val_loss',
                        save_best_only=True, save_weights_only=False, verbose=1),
                ]
                fit = model.fit_generator(generator = batch_generator(xtr, ytr, train_batch_size, True),
                                          nb_epoch = nepochs,
                                          samples_per_epoch = xtr.shape[0],
                                          verbose = 2,
                                          callbacks = callbacks,
                                          validation_data = batch_generator(xte, yte, 800, False),
                                          nb_val_samples = 37600,
                                         )
                model = load_model('keras.best.hdf5')
                pred += model.predict_generator(generator = batch_generatorp(xte, 800, False),
                                                val_samples = xte.shape[0])[:,0]
                pred_test += model.predict_generator(generator = batch_generatorp(l1_x_test,800,False),
                                                     val_samples = l1_x_test.shape[0])[:,0]
            pred /= nbags_per_fold
            pred_oob[inTe] += pred
            score = mean_absolute_error(yte, pred)
            i += 1
            print('Fold ', J, '-', i, '- MAE:', score)
        print('Total (', (J+1), ')- MAE:', mean_absolute_error(y_train, pred_oob/(J+1)))
    pred_oob /= nbags
    print('Total MAE:', mean_absolute_error(y_train, pred_oob))
    pred_test /= (nfolds*nbags*nbags_per_fold)
    oof_test = pred_test
    oof_train = pred_oob
    params = {'model': model.to_yaml(),
              'nfolds': nfolds,
              'USE_STRATIFIED': USE_STRATIFIED,
              'nbags': nbags,
              'nepochs': nepochs,
              'es_patience': es_patience,
              'nbags_per_fold': nbags_per_fold, 
              'train_batch_size': train_batch_size,
             }
    timestamp = get_timestamp()
    model_key = 'Keras.stacker-' + timestamp
    print('Saving model as {}'.format(model_key))
    save_results_to_json(model_key, params, oof_test, oof_train)

    sub = load_submission()
    sub['loss'] = oof_test
    sub.to_csv(model_key + '.sub.csv', index=False)

if False:
    l1_x_train[l1_x_train < 0] = 1
    l1_x_test[l1_x_test < 0] = 1

    def mae_loss_func(weights):
        final_prediction = np.sum(weights * l1_x_train, axis=1)
        err = mean_absolute_error(y_train, final_prediction)
        return err
    starting_values = np.random.uniform(size=l1_x_train.shape[1])
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    bounds = [(0,1)] * l1_x_train.shape[1]
    print('Starting minimize')
    res = minimize(mae_loss_func, 
               starting_values, 
               method = 'SLSQP', 
               bounds = bounds, 
               options={'maxiter': 100, 'disp': True})
    best_score = res['fun']
    weights = res['x']
    print('Ensamble Score: {}'.format(best_score))
    print('Best Weights: {}'.format(weights))
    oof_train = np.sum(weights * l1_x_train, axis=1)
    oof_test = np.sum(weights * l1_x_test, axis=1)
    params = {'weights': weights.tolist(),
              'starting_values' : starting_values.tolist(),
              'bounds': bounds,
             }
    timestamp = get_timestamp()
    model_key = 'WeightedSum.stacker-' + timestamp
    save_results_to_json(model_key, params, oof_test, oof_train)
    sub = load_submission()
    sub['loss'] = oof_test
    sub.to_csv(model_key + '.sub.csv', index=False)
EOF
