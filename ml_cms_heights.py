import cPickle
import os
import pandas as pd
import pdb
import numpy as np
import logging
import sys
import calendar
import matplotlib.pyplot as plt

from math import ceil
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from ConfigParser import SafeConfigParser
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split

import constants
import compute_stats
import rgeo
import utils

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Logging
cur_flname = os.path.splitext(os.path.basename(__file__))[0]
LOG_FILENAME = constants.log_dir + os.sep + 'Log_' + cur_flname + '.txt'
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, filemode='w',
                    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%m-%d %H:%M")  # Logging levels are DEBUG, INFO, WARNING, ERROR, and CRITICAL
# Output to screen
logger = logging.getLogger(cur_flname)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

# loop_countries
# ; Loop over countries x crops
#     train_ml_model
#         create_train_df
#             compute_ml_vars
#         ; create ml model
#     loop_forecasting
#         create_forecast_df
#         do_forecasting


class MLCms:
    """

    """
    def __init__(self, config_file=''):
        # Parse config file
        self.parser = SafeConfigParser()
        self.parser.read(config_file)

        # machine learning specific variables
        self.classify = constants.DO_CLASSIFICATION  # Regress or classify?
        self.vars_features = constants.fixed_vars
        self.vars_target = constants.ML_TARGETS

        if self.classify:
            self.var_target = constants.ML_TARGETS
            self.task = 'classification'
            self.model = RandomForestClassifier(n_estimators=2500, n_jobs=constants.ncpu, random_state=0)
        else:
            self.var_target = constants.ML_TARGETS
            self.task = 'regression'
            self.model = RandomForestRegressor(n_estimators=2500, n_jobs=constants.ncpu, random_state=0)  # SVR()

        # Get path to input
        self.path_inp = constants.base_dir + os.sep + constants.name_inp_fl

        # Output directory is <dir>_<classification>_<2014>
        self.path_out_dir = constants.out_dir
        utils.make_dir_if_missing(self.path_out_dir)

        # Model pickle
        self.path_pickle_model = self.path_out_dir + os.sep + constants.model_pickle
        self.path_pickle_features = self.path_out_dir + os.sep + 'pickled_features'

    def output_model_importance(self, gs, name_gs, num_cols):
        """

        :param gs:
        :param name_gs:
        :param num_cols:
        :return:
        """
        rows_list = []
        name_vars = []

        feature_importance = gs.best_estimator_.named_steps[name_gs].feature_importances_
        importances = 100.0 * (feature_importance / feature_importance.max())

        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        # Store feature ranking in a dataframe
        for f in range(num_cols):
            dict_results = {'Variable': self.vars_features[indices[f]], 'Importance': importances[indices[f]]}
            name_vars.append(self.vars_features[indices[f]])
            rows_list.append(dict_results)

        df_results = pd.DataFrame(rows_list)
        num_cols = 10 if len(indices) > 10 else len(indices)  # Plot upto a maximum of 10 features
        plot.plot_model_importance(num_bars=num_cols, xvals=importances[indices][:num_cols],
                                   std=std[indices][:num_cols], fname=self.task + '_importance_' + self.crop,
                                   title='Importance of variable (' + self.country + ' ' + self.crop_lname + ')',
                                   xlabel=name_vars[:num_cols], out_path=self.path_out_dir)

        df_results.to_csv(self.path_out_dir + os.sep + self.task + '_importance_' + self.crop + '.csv')

    def get_data(self):
        """

        :return:
        """
        df = pd.read_csv(self.path_inp)
        cols = [col for col in df.columns if col not in self.vars_features]
        # cols.extend(['DI', 'PI'])

        # Add information on PI and DI of soils
        # iterate over each row, get lat and lon
        # Find corresponding DI and PI

        lat_lons = zip(df['Long_round'], df['Lat_round'])
        vals_di = []
        vals_pi = []
        # for idx, (lon, lat) in enumerate(lat_lons):
        #     print idx, len(lat_lons)
        #     vals_pi.append(rgeo.get_value_at_point('C:\\Users\\ritvik\\Documents\\PhD\\Projects\\CMS\\Input\\Soils\\PI.tif',
        #                                            lon, lat, replace_ras=False))
        #     vals_di.append(rgeo.get_value_at_point('C:\\Users\\ritvik\\Documents\\PhD\\Projects\\CMS\\Input\\Soils\\DI.tif',
        #                                      lon, lat, replace_ras=False))
        #
        # df['DI'] = vals_di
        # df['PI'] = vals_pi
        df = df[cols]

        data = df.as_matrix(columns=cols[1:])
        target = df.as_matrix(columns=[self.var_target]).ravel()
        # Get training and testing splits
        splits = train_test_split(data, target, test_size=0.2)

        return cols, splits

    def train_ml_model(self):
        """

        :return:
        """
        logger.info('#########################################################################')
        logger.info('train_ml_model')
        logger.info('#########################################################################')

        ######################################################
        # Load dataset
        ######################################################
        cols, splits = self.get_data()
        data_train, data_test, target_train, target_test = splits

        # clf =  ExtraTreesRegressor(500, n_jobs=constants.ncpu)
        # #clf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        # #clf = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
        # data = df_train.as_matrix(columns=cols[1:])  # convert dataframe column to matrix
        # #data = preprocessing.scale(data)
        # target = df_train.as_matrix(columns=[self.var_target]).ravel()  # convert dataframe column to matrix
        # clf.fit(data, target)
        #
        # predict_val = clf.predict(after.as_matrix(columns=cols[1:]))
        # results = compute_stats.ols(predict_val.tolist(), after_target.tolist())
        # print results.rsquared
        # import matplotlib.pyplot as plt
        # plt.scatter(after_target, predict_val)
        # plt.show()
        # pdb.set_trace()
        if not os.path.isfile(self.path_pickle_model):
            # For details in scikit workflow: See http://stackoverflow.com/questions/
            # 35256876/ensuring-right-order-of-operations-in-random-forest-classification-in-scikit-lea
            # TODO Separate out a dataset so that even the grid search cv can be tested
            ############################
            # Select features from model
            ############################
            logger.info('Selecting important features from model')
            if self.classify:
                rf_feature_imp = ExtraTreesRegressor(150, n_jobs=constants.ncpu)
            else:
                rf_feature_imp = ExtraTreesRegressor(150, n_jobs=constants.ncpu)
            feat_selection = SelectFromModel(rf_feature_imp)

            pipeline = Pipeline([
                      ('fs', feat_selection),
                      ('clf', self.model),
                    ])

            #################################
            # Grid search for best parameters
            #################################
            C_range = np.logspace(-2, 10, 13)
            gamma_range = np.logspace(-9, 3, 13)
            logger.info('Tuning hyperparameters')
            param_grid = {
                'fs__threshold': ['mean', 'median'],
                'fs__estimator__max_features': ['auto', 'log2'],
                'clf__max_features': ['auto', 'log2'],
                'clf__n_estimators': [1000, 2000]
                #'clf__gamma': np.logspace(-9, 3, 13),
                #'clf__C': np.logspace(-2, 10, 13)
            }

            gs = GridSearchCV(pipeline, param_grid=param_grid, verbose=2, n_jobs=constants.ncpu, error_score=np.nan)
            # Fir the data before getting the best parameter combination. Different data sets will have
            # different optimized parameter combinations, i.e. without data, there is no optimal parameter combination.
            gs.fit(data_train, target_train)
            logger.info(gs.best_params_)

            data_test = pd.DataFrame(data_test, columns=cols[1:])

            # Update features that should be used in model
            selected_features = gs.best_estimator_.named_steps['fs'].transform([cols[1:]])
            cols = selected_features[0]
            data_test = data_test[cols]

            # Update model with the best parameters learnt in the previous step
            self.model = gs.best_estimator_.named_steps['clf']

            predict_val = self.model.predict(data_test)
            results = compute_stats.ols(predict_val.tolist(), target_test.tolist())
            print results.rsquared
            print cols
            plt.scatter(target_test, predict_val)
            plt.show()
            pdb.set_trace()
            ###################################################################
            # Output and plot importance of model features, and learning curves
            ###################################################################
            self.output_model_importance(gs, 'clf', num_cols=len(cols[1:]))

            if constants.plot_model_importance:
                train_sizes, train_scores, test_scores = learning_curve(self.model, data, target, cv=k_fold,
                                                                        n_jobs=constants.ncpu)
                plot.plot_learning_curve(train_scores, test_scores, train_sizes=train_sizes, fname='learning_curve',
                                         ylim=(0.0, 1.01), title='Learning curves', out_path=self.path_out_dir)

            # Save the model to disk
            logger.info('Saving model and features as pickle on disk')
            with open(self.path_pickle_model, 'wb') as f:
                cPickle.dump(self.model, f)
            with open(self.path_pickle_features, 'wb') as f:
                cPickle.dump(self.vars_features, f)
        else:
            # Read model from pickle on disk
            with open(self.path_pickle_model, 'rb') as f:
                logger.info('Reading model from pickle on disk')
                self.model = cPickle.load(f)

            logger.info('Reading features from pickle on disk')
            self.vars_features = pd.read_pickle(self.path_pickle_features)

        return df_cc

    def do_forecasting(self, df_forecast, mon_names, available_target=False, name_target='yield'):
        """
        1. Does classification/regression based on already built model.
        2. Plots confusion matrix for classification tasks, scatter plot for regression
        3. Plots accuracy statistics for classification/regression
        :param df_forecast:
        :param mon_names:
        :param available_target: Is target array available?
        :param name_target: Name of target array (defaults to yield)
        :return:
        """
        data = df_forecast.as_matrix(columns=self.vars_features)  # convert dataframe column to matrix
        predicted = self.model.predict(data)

        if available_target:
            expected = df_forecast.as_matrix(columns=[name_target]).ravel()
            if not self.classify:  # REGRESSION
                # Compute stats
                results = compute_stats.ols(predicted.tolist(), expected.tolist())
                bias = compute_stats.bias(predicted, expected)
                rmse = compute_stats.rmse(predicted, expected)
                mae = compute_stats.mae(predicted, expected)

                # Plot!
                plot.plot_regression_scatter(expected, np.asarray(predicted),
                                             annotate=r'$r^{2}$ ' + '{:0.2f}'.format(results.rsquared) + '\n' +
                                             'peak NDVI date: ' + self.time_peak_ndvi.strftime('%b %d'),
                                             xlabel='Expected yield',
                                             ylabel='Predicted yield',
                                             title=mon_names + ' ' + str(int(df_forecast[self.season].unique()[0])),
                                             fname=self.task + '_' + '_'.join([mon_names]) + '_' + self.crop,
                                             out_path=self.path_out_dir)

                # global expected vs predicted
                if self.debug:
                    # any non-existing index will add row
                    self.df_global.loc[len(self.df_global)] = [np.nanmean(expected), np.nanmean(predicted), mon_names,
                                                               self.forecast_yr]

                return predicted, {'RMSE': rmse, 'MAE': mae, r'$r^{2}$': results.rsquared, 'Bias': bias}
            else:  # CLASSIFICATION
                # Convert from crop condition class (e.g. 4) to string (e.g. exceptional)
                expected, predicted = compute_stats.remove_nans(expected, predicted)
                cm = confusion_matrix(expected, predicted, labels=self.dict_cc.keys()).T

                # Compute and plot class probabilities
                proba_cc = self.model.predict_proba(data)
                df_proba = pd.DataFrame(proba_cc, columns=self.dict_cc.values())
                plot.plot_class_probabilities(df_proba, fname='proba_' + '_'.join([mon_names]) + '_' + self.crop,
                                              out_path=self.path_out_dir)

                # Plot confusion matrix
                plot.plot_confusion_matrix(cm, normalized=False, fname='cm_' + '_'.join([mon_names]) + '_' + self.crop,
                                           xlabel='True class', ylabel='Predicted class', ticks=self.dict_cc.values(),
                                           out_path=self.path_out_dir)

                # Normalize and plot confusion matrix
                cm_normalized = normalize(cm.astype(float), axis=1, norm='l1')
                plot.plot_confusion_matrix(cm_normalized, fname='norm_cm_' + '_'.join([mon_names]) + '_' + self.crop,
                                           xlabel='True class', ylabel='Predicted class', normalized=True,
                                           ticks=self.dict_cc.values(), out_path=self.path_out_dir)

                score_accuracy = accuracy_score(expected, predicted) * 100.0
                score_precision = precision_score(expected, predicted, average='weighted') * 100.0
                return predicted, {'Accuracy': score_accuracy, 'Precision': score_precision}
        else:
            return predicted, {'RMSE': np.nan, 'MAE': np.nan, r'$r^{2}$': np.nan, 'Bias': np.nan,
                               'Nash-Sutcliff': np.nan}


def do_ml_model():
    obj = MLCms(config_file='config_CMS.txt')
    obj.train_ml_model()

if __name__ == '__main__':
    do_ml_model()

