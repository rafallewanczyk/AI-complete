

def main():
    from utilities.Python.machine_learning import keras_init
    keras_init(gpu_mem_frac=0.5)

    import sys
    sys.path.insert(0, '../../globals/Python')
    from globals import utilities_dir, num_lgc_prcs

    import os
    import warnings
    import time

    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    warnings.simplefilter('ignore')
    import seaborn as sns
    from functools import partial
    from utilities.Python.plotting import KerasPlotLosses

    from ETL import load_data

    import pickle
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, r2_score
    import keras
    import keras.models as ks_models
    from keras import backend as K
    from machine_learning import create_keras_regressor, keras_reg_grid_search, keras_convert_optimizer_obj_to_name, \
                                 reshape_data_from_2D_to_keras

    from analysis import find_Markowitz_optimal_portfolio_weights, calc_CAPM_betas, CAPM_RoR, run_monte_carlo_financial_simulation
    from type_conversions import pandas_dt_to_str
    from utilities.Python.plotting import add_value_text_to_seaborn_barplot, monte_carlo_plot_confidence_band
    from utilities.Python.conversions import returns_to_yearly

    from keras.optimizers import Adam

    load_source = 'csv'
    analysis_data_resolution = 'daily'
    analysis_data_date_range = ('2013-05-01', '2017-10-31') 
    analysis_time_unit = 'days' if analysis_data_resolution == 'daily' else 'hours'
    annualize_returns = partial(returns_to_yearly, resolution=analysis_data_resolution)
    model_data_resolution = 'hourly'
    model_data_date_range = ('2017-07-01', '2018-03-31')
    model_last_extrapolation_time = pd.to_datetime('2018-08-31')
    model_time_unit_singular = 'day' if model_data_resolution == 'daily' else 'hour'
    model_time_unit_plural = 'days' if model_data_resolution == 'daily' else 'hours'

    load_models = True 
    num_lgc_prcs_grd_srch = min(3, num_lgc_prcs)
    num_cv_splits = 3 
    keras_use_cv_grd_srch = False
    keras_grd_srch_verbose = 1

    make_analysis_plots = True
    make_prediction_plots = True
    make_keras_loss_plots = False
    make_keras_param_set_occurrence_plots = False
    make_asset_keras_figures = make_keras_loss_plots
    make_all_assets_figures = make_analysis_plots
    make_selected_assets_figures = make_analysis_plots or make_prediction_plots or make_asset_keras_figures
    make_keras_figures = make_keras_loss_plots or make_keras_param_set_occurrence_plots
    make_any_figures = (make_analysis_plots or make_prediction_plots or
                        make_keras_loss_plots or make_keras_param_set_occurrence_plots)

    figure_size = (12, 6) 
    figure_dpi = 300
    figures_dir = 'figures'
    if make_any_figures and not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    param_sets_dir = 'param_sets'
    if not os.path.exists(param_sets_dir):
        os.makedirs(param_sets_dir)

    param_set_occurrences_dir = 'param_set_occurrences_keras'
    if not os.path.exists(param_set_occurrences_dir):
        os.makedirs(param_set_occurrences_dir)

    num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices = \
        load_data(resolution=analysis_data_resolution, date_range=analysis_data_date_range, source=load_source)

    if make_all_assets_figures:
        currency_figures_subdirs = ['{}/{}'.format(figures_dir, currency_label.replace(' ', '_'))
                                    for currency_label in currencies_labels]
        for currency_figures_subdir in currency_figures_subdirs:
            if not os.path.exists(currency_figures_subdir):
                os.makedirs(currency_figures_subdir)
    keras_figures_dir = None 
    if make_keras_figures:
        keras_figures_dir = '{}/{}'.format(figures_dir, 'keras')
        if not os.path.exists(keras_figures_dir):
            os.makedirs(keras_figures_dir)

    if make_analysis_plots:
        num_cols = 2
        num_rows = int(np.ceil(num_currencies / num_cols))
        for currency_index, column in enumerate(prices.columns):
            currency_label = currencies_labels[currency_index]
            currency_label_no_spaces = currency_label.replace(' ', '_')
            currency_ticker = currencies_tickers[currency_index]
            plt.figure(figsize=figure_size, dpi=figure_dpi)
            plt.plot(prices[column])
            plt.xlabel('Date')
            plt.ylabel('Close')
            plt.title("{} ({}) Closing Value"
                      .format(currency_label, currency_ticker))
            currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
            plt.savefig('{}/{}_price_trend'.format(currency_figures_subdir, currency_label_no_spaces))

    returns = prices.pct_change()

    old_prices_index = prices.index
    prices.index = prices.index.map(lambda date: pandas_dt_to_str(date))
    is_null_prices = pd.isnull(prices)
    if make_analysis_plots:
        ax = sns.heatmap(is_null_prices, yticklabels=120)
        absent_values_fig = ax.get_figure()
        plt.tight_layout()
        absent_values_fig.savefig('{}/absent_values.png'.format(figures_dir))
        prices.index = old_prices_index

    if analysis_data_resolution == 'daily':
        currencies_labels_tickers_to_remove = np.array(
            [['Bitcoin Cash', 'BCH'], ['BitConnect', 'BCC'], ['Ethereum Classic', 'ETC'],
             ['Iota', 'MIOTA'], ['Neo', 'NEO'], ['Numeraire', 'NMR'], ['Omisego', 'OMG'],
             ['Qtum', 'QTUM'], ['Stratis', 'STRAT'], ['Waves', 'WAVES']])
    else:
        currencies_labels_tickers_to_remove = np.array([['Omisego', 'OMG'], ['Ethereum Classic', 'ETC'],
                                                        ['Neo', 'NEO']])
    currencies_labels_to_remove = currencies_labels_tickers_to_remove[:, 0]
    currencies_tickers_to_remove = currencies_labels_tickers_to_remove[:, 1]
    currencies_labels_and_tickers_to_remove = ["{} ({})".format(currencies_label, currencies_ticker)
                                               for currencies_label, currencies_ticker in
                                               zip(currencies_labels_to_remove, currencies_tickers_to_remove)]
    print("Removing currencies from analysis data: {}".format(currencies_labels_and_tickers_to_remove))
    subset_prices = prices.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)
    subset_num_currencies = len(subset_prices.columns)
    subset_prices_nonan = subset_prices.dropna()
    print("Beginning and ending {} with data for remaining currencies in analysis data: {}, {}".
          format(analysis_time_unit, subset_prices_nonan.index[0], subset_prices_nonan.index[-1]))

    num_non_nan_times = len(subset_prices_nonan)
    print("Considering {} {} of price information for analysis (as many as without NaN values)."
          .format(num_non_nan_times, analysis_time_unit))

    selected_currency_figures_subdirs = ['{}/{}'.format(figures_dir, currency_label.replace(' ', '_'))
                                         for currency_label in currencies_labels
                                         if currency_label not in currencies_labels_to_remove]
    if make_selected_assets_figures:
        for currency_figures_subdir in selected_currency_figures_subdirs:
            if not os.path.exists(currency_figures_subdir):
                os.makedirs(currency_figures_subdir)
    keras_figures_subdirs = ['{}/{}'.format(currency_figures_subdir, 'keras')
                             for currency_figures_subdir in selected_currency_figures_subdirs]
    if make_asset_keras_figures:
        for keras_figures_subdir in keras_figures_subdirs:
            if not os.path.exists(keras_figures_subdir):
                os.makedirs(keras_figures_subdir)

    subset_returns = subset_prices_nonan.pct_change()
    subset_returns_yearly = annualize_returns(subset_returns.mean())

    if make_analysis_plots:
        clustermap_params = dict(annot=True, annot_kws=dict(fontsize=12), fmt='.2f', vmin=-1, vmax=1)
        plt.figure(figsize=figure_size, dpi=figure_dpi)
        price_correlations = subset_prices.corr()
        price_correlations_fig = sns.clustermap(price_correlations, **clustermap_params)
        price_correlations_fig.savefig('{}/price_correlations.png'.format(figures_dir))
        plt.clf()
        returns_correlations = subset_returns.corr()
        returns_correlations_fig = sns.clustermap(returns_correlations, **clustermap_params)
        returns_correlations_fig.savefig('{}/return_correlations.png'.format(figures_dir))
        plt.clf()

    if make_analysis_plots:
        returns_std_by_year = subset_returns.groupby(subset_returns.index.year).std()
        returns_std_2017 = returns_std_by_year.loc[2017]
        returns_std_2017.sort_values(ascending=False, inplace=True)
        plt.subplots(figsize=figure_size)
        plotting_data = returns_std_2017.to_frame(name='Volatility').reset_index()
        volatility_plot = sns.barplot(x='Name', y='Volatility', data=plotting_data, palette='viridis')
        add_value_text_to_seaborn_barplot(volatility_plot, plotting_data['Volatility'])
        plt.title('Volatility (2017)')
        plt.savefig('{}/volatility.png'.format(figures_dir), dpi=figure_dpi)

    market_index = 'Bitcoin (BTC)'
    subset_betas = calc_CAPM_betas(subset_returns, market_index)
    subset_betas.sort_values(ascending=False, inplace=True)
    if make_analysis_plots:
        fig, ax = plt.subplots(figsize=figure_size)
        plotting_data = subset_betas.to_frame(name='Beta').reset_index()
        beta_plot = sns.barplot(ax=ax, x='Name', y='Beta', data=plotting_data, palette='viridis')
        add_value_text_to_seaborn_barplot(beta_plot, plotting_data['Beta'])
        plt.title('Betas (Bitcoin (BTC) as Market Index)')
        plt.savefig('{}/CAPM_betas.png'.format(figures_dir), dpi=figure_dpi)

    return_risk_free = 0.025
    CAPM_expected_rates_of_return = CAPM_RoR(subset_betas, subset_returns_yearly, market_index, return_risk_free)
    CAPM_expected_rates_of_return.sort_values(ascending=False, inplace=True)
    if make_analysis_plots:
        fig, ax = plt.subplots(figsize=figure_size)
        plotting_data = CAPM_expected_rates_of_return.to_frame(name='Expected Rate of Return').reset_index()
        CAPM_plot = sns.barplot(ax=ax, x='Name', y='Expected Rate of Return', data=plotting_data, palette='viridis')
        add_value_text_to_seaborn_barplot(CAPM_plot, plotting_data['Expected Rate of Return'], percent=True)
        plt.title('CAPM Expected Rates of Return')
        plt.savefig('{}/CAPM_expected_rates_of_return.png'.format(figures_dir), dpi=figure_dpi)

    if make_analysis_plots:
        plotting_data = subset_returns_yearly.reindex(CAPM_expected_rates_of_return.index, copy=False)
        plotting_data = plotting_data.to_frame(name='Rate of Return').reset_index()
        fig, ax = plt.subplots(figsize=figure_size)
        returns_plot = sns.barplot(ax=ax, x='Name', y='Rate of Return', data=plotting_data, palette='viridis')
        add_value_text_to_seaborn_barplot(returns_plot, plotting_data['Rate of Return'], percent=True)
        plt.title('Rates of Return')
        plt.savefig('{}/rates_of_return.png'.format(figures_dir), dpi=figure_dpi)

    if make_analysis_plots:
        optimal_weights = find_Markowitz_optimal_portfolio_weights(subset_returns, return_risk_free)
        optimal_weights.sort_values(ascending=False, inplace=True)
        fig, ax = plt.subplots(figsize=figure_size)
        plotting_data = optimal_weights.to_frame(name='Weight').reset_index()
        portfolio_weights_plot = sns.barplot(ax=ax, x='Name', y='Weight', data=plotting_data, palette='viridis')
        add_value_text_to_seaborn_barplot(portfolio_weights_plot, plotting_data['Weight'])
        plt.title('Markowitz Optimal Portfolio Weights')
        plt.savefig('{}/optimal_portfolio_weights.png'.format(figures_dir), dpi=figure_dpi)

    num_currencies, currencies_labels, currencies_tickers, currencies_labels_and_tickers, prices = \
        load_data(resolution=model_data_resolution, date_range=model_data_date_range, source=load_source)
    currencies_labels_and_tickers_to_remove = \
        [label_and_ticker_to_remove for label_and_ticker_to_remove in currencies_labels_and_tickers_to_remove
         if label_and_ticker_to_remove in prices.columns]
    subset_prices = prices.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)
    subset_currencies_labels = [currency_label for currency_label in currencies_labels
                                if currency_label not in currencies_labels_to_remove]
    subset_currencies_tickers = [currency_ticker for currency_ticker in currencies_tickers
                                 if currency_ticker not in currencies_tickers_to_remove]
    subset_currencies_labels_and_tickers = subset_prices.columns.values
    subset_num_currencies = len(subset_prices.columns)
    subset_prices_nonan = subset_prices.dropna()
    num_non_nan_times = len(subset_prices_nonan)
    date_times = subset_prices_nonan.index.values
    print("Considering {} {} of price information for model training (as many as without NaN values)."
          .format(num_non_nan_times, model_time_unit_plural))

    model_last_data_time = prices.index[-1]
    model_offset_time = {'days': 1} if model_data_resolution == 'daily' else {'hours': 1}
    model_first_extrapolation_time = model_last_data_time + pd.DateOffset(**model_offset_time)
    model_extrapolation_times = np.array(
        pd.date_range(model_first_extrapolation_time, model_last_extrapolation_time,
                      freq='D' if model_data_resolution == 'daily' else 'H'))
    model_num_extrapolation_times = len(model_extrapolation_times)

    MC_predicted_values_ranges, MC_predicted_values = run_monte_carlo_financial_simulation(prices, model_extrapolation_times)
    subset_monte_carlo_predicted_values_ranges = \
        MC_predicted_values_ranges.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)
    subset_monte_carlo_predicted_values = \
        MC_predicted_values.drop(labels=currencies_labels_and_tickers_to_remove, axis=1)

    currency_models_subdirs = ['{}/{}'.format(models_dir, currency_label.replace(' ', '_'))
                               for currency_label in subset_currencies_labels]
    for currency_models_subdir in currency_models_subdirs:
        if not os.path.exists(currency_models_subdir):
            os.makedirs(currency_models_subdir)

    param_sets_subdirs = ['{}/{}'.format(param_sets_dir, currency_label.replace(' ', '_'))
                          for currency_label in subset_currencies_labels]
    for param_sets_subdir in param_sets_subdirs:
        if not os.path.exists(param_sets_subdir):
            os.makedirs(param_sets_subdir)

    if model_data_resolution == 'daily': 
        window_sizes = np.array([7, 14] + list(range(30, 361, 30)))
    else: 
        window_sizes = 24*7*np.array([1, 2, 4, 8, 12])
    window_sizes = window_sizes[::-1] 

    if make_asset_keras_figures:
        for keras_figures_subdir in keras_figures_subdirs:
            for window_size in window_sizes:
                keras_figures_window_subdir = os.path.join(keras_figures_subdir, 'w_{}'.format(window_size))
                if not os.path.exists(keras_figures_window_subdir):
                    os.makedirs(keras_figures_window_subdir)

    keras_param_names = None 
    param_set_occurrences_pickle_paths = ['{}/param_set_occurrences_{}.pkl'.format(param_set_occurrences_dir, window_size)
                                          for window_size in window_sizes]
    keras_window_size_scores_pickle_path = '{}/window_size_scores.pkl'.format(param_set_occurrences_dir)

    asset_col_str = 'asset'
    window_size_col_str = 'window_size'
    keras_window_size_scores = {} 

    for window_size_ind, window_size in enumerate(window_sizes):
        num_windows = len(subset_prices_nonan) - 2*window_size + 1
        num_features = subset_num_currencies * window_size
        X = np.empty((num_windows, num_features), dtype=np.float64)
        y = np.empty((num_windows, num_features), dtype=np.float64)
        for window_index in range(num_windows):
            X[window_index,:] = \
                subset_prices_nonan[window_index:window_index + window_size].values.flatten()
            y[window_index,:] = \
                subset_prices_nonan[window_index + window_size:window_index + 2*window_size].values.flatten()

        X_scaler = StandardScaler()
        X_scaled = X_scaler.fit_transform(X)
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y)

        single_hidden_layer_size = 96
        neural_net_hidden_layer_sizes = [(single_hidden_layer_size, single_hidden_layer_size // 8)]
        from sklearn.pipeline import Pipeline
        model_neural_net = Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor())
        ])
        params_neural_net = {'model__hidden_layer_sizes': neural_net_hidden_layer_sizes,
                             'model__max_iter': [5000],
                             'model__beta_1': [0.7, 0.8, 0.9],
                             'model__beta_2': [0.9, 0.95, 0.999],
                             'model__alpha': [1e-10, 1e-8, 1e-4]}

        from keras.wrappers.scikit_learn import KerasRegressor
        keras_model_nn = "Keras_NN"
        frst_hdn_lyr_sz_nn_single_asset = 512
        frst_hdn_lyr_sz_nn_collective = 1024

        hdn_lyr_szs_nn_single_asset = [(frst_hdn_lyr_sz_nn_single_asset, frst_hdn_lyr_sz_nn_single_asset)]
        hdn_lyr_szs_nn_collective = [(frst_hdn_lyr_sz_nn_collective, frst_hdn_lyr_sz_nn_collective)]

        keras_params_nn = {
             'batch_size': [48],  
             'hidden_layer_sizes': None, 
             'hidden_layer_type': ['LSTM'],
             'optimizer': [Adam],
             'lr': [1e-3],
             'beta_1': [0.9],
             'beta_2': [0.999]}
        keras_param_names = list(keras_params_nn.keys())
        keras_nn_epochs_grd_srch = 20
        keras_nn_epochs_grd_srch = max(2, keras_nn_epochs_grd_srch) \
            if not keras_use_cv_grd_srch & make_keras_loss_plots else keras_nn_epochs_grd_srch
        keras_nn_epochs = 20

        model_extra_trees = Pipeline([
            ('scaler', StandardScaler()),
            ('model', ExtraTreesRegressor())
        ])
        params_extra_trees = {'model__n_estimators': [500],
                              'model__min_samples_split': [2, 5, 10],
                              'model__max_features': ['auto', 'sqrt', 'log2']}

        models_to_test = [keras_model_nn]  
        param_grids = [keras_params_nn]  

        cv = TimeSeriesSplit(n_splits=num_cv_splits)
        keras_cv = cv if keras_use_cv_grd_srch else None

        keras_param_sets_occurrences_window = {}
        keras_param_set_occurrences_pickle_path = param_set_occurrences_pickle_paths[window_size_ind]

        single_asset_models_paths, single_asset_models_types, single_asset_models_param_sets = [], [], []
        single_asset_models = [None]*subset_num_currencies

        for currency_index in range(subset_num_currencies):
            def get_X_subset(X, asset_index, num_assets):
                return X[:, asset_index::num_assets]

            def get_y_subset(y, asset_index, num_assets):
                return y[:, asset_index::num_assets]

            currency_label = subset_currencies_labels[currency_index]
            currency_label_no_spaces = currency_label.replace(' ', '_')
            currency_models_subdir = '{}/{}'.format(models_dir, currency_label_no_spaces)
            currency_param_sets_subdir = "{}/{}".format(param_sets_dir, currency_label_no_spaces)
            currency_figures_subdir = \
                '{}/{}'.format(figures_dir, currency_label_no_spaces) if make_selected_assets_figures else None
            keras_figures_subdir = \
                '{}/{}'.format(currency_figures_subdir, 'keras') if make_asset_keras_figures else None
            keras_figures_window_subdir = \
                '{}/{}'.format(keras_figures_subdir, 'w_{}'.format(window_size)) if make_asset_keras_figures else None
            model_base_path = "{}/{}_model_{}".format(currency_models_subdir, currency_label_no_spaces, window_size)
            model_pickle_path = "{}.pkl".format(model_base_path)
            param_set_base_path = "{}/{}_param_set_{}".format(currency_param_sets_subdir, currency_label_no_spaces, window_size)
            param_set_pickle_path = "{}.pkl".format(param_set_base_path)
            model_path, model_type = (None, None)

            X_subset_scaled = get_X_subset(X_scaled, currency_index, subset_num_currencies)
            y_subset_scaled = get_y_subset(y_scaled, currency_index, subset_num_currencies)

            if not load_models:
                currency_label_and_ticker = subset_currencies_labels_and_tickers[currency_index]
                print("Currently training a model for {} with a window size of {} {}."
                    .format(currency_label_and_ticker, window_size, model_time_unit_plural))

                best_score, best_model, best_param_set = (-np.inf, None, None)

                for i in range(len(models_to_test)):
                    model_to_test = models_to_test[i]
                    param_grid = param_grids[i]
                    param_grid['hidden_layer_sizes'] = hdn_lyr_szs_nn_single_asset
                    if model_to_test == keras_model_nn:
                        X_subset_scaled_dict = reshape_data_from_2D_to_keras(param_grid, 1, X_subset_scaled)
                        model, score, param_set = \
                            keras_reg_grid_search(X_subset_scaled_dict, y_subset_scaled, build_fn=create_keras_regressor,
                                                  param_grid=param_grid, epochs=keras_nn_epochs,
                                                  cv_epochs=keras_nn_epochs_grd_srch, cv=keras_cv,
                                                  scoring=r2_score, verbose=keras_grd_srch_verbose,
                                                  plot_losses=make_keras_loss_plots, plotting_dir=keras_figures_window_subdir,
                                                  figure_title_prefix='{} (window size {})'.format(currency_label, window_size),
                                                  figure_kwargs={'figsize':figure_size, 'dpi':figure_dpi})
                        param_set['optimizer'] = keras_convert_optimizer_obj_to_name(param_set['optimizer'])
                        param_vals_tup = tuple(param_set.values()) + ('single', window_size)
                        num_occurrences = keras_param_sets_occurrences_window.setdefault(param_vals_tup, 0)
                        keras_param_sets_occurrences_window[param_vals_tup] = num_occurrences + 1
                    else:
                        grid_search = GridSearchCV(model_to_test, param_grid, scoring=make_scorer(r2_score),
                                                   cv=cv, n_jobs=num_lgc_prcs_grd_srch)
                        grid_search.fit(X_subset_scaled, y_subset_scaled.ravel())
                        model = grid_search.best_estimator_
                        score = grid_search.best_score_
                        param_set = grid_search.best_params_
                    best_model = model if score > best_score else best_model
                    best_param_set = param_set if score > best_score else best_param_set
                    best_score = score if score > best_score else best_score
                time.sleep(1)  
                print("Best model and score for asset {} and window size {}: {}"
                      .format(currency_label_and_ticker, window_size, (best_model, best_score)))
                if type(best_model) is keras.models.Sequential:
                    model_path = model_base_path + '.h5'
                    ks_models.save_model(best_model, model_path)
                else:
                    model_path = model_pickle_path
                    with open(model_pickle_path, "wb") as model_outfile:
                        pickle.dump(best_model, model_outfile)
                with open(param_set_pickle_path, "wb") as param_set_outfile:
                    pickle.dump(best_param_set, param_set_outfile)
            else:
                try: 
                    best_model = ks_models.load_model(model_base_path + '.h5')
                    model_path = model_base_path + '.h5'
                except OSError:
                    model_path = model_pickle_path
                    with open(model_pickle_path, "rb") as model_infile:
                        best_model = pickle.load(model_infile)
                with open(param_set_pickle_path, "rb") as param_set_infile:
                    best_param_set = pickle.load(param_set_infile)
            single_asset_models_paths.append(model_path)
            single_asset_models_types.append(type(best_model))
            single_asset_models_param_sets.append(best_param_set)
        best_model = None
        K.clear_session()  

        collective_assets_model_path, collective_assets_model_type, collective_assets_model_param_set = (None, None, None)
        collective_assets_model = None
        model_base_path = "{}/model_{}".format(models_dir, window_size)
        model_pickle_path = "{}.pkl".format(model_base_path)

        param_set_base_path = "{}/param_set_{}".format(param_sets_dir,window_size)
        param_set_pickle_path = "{}.pkl".format(param_set_base_path)

        if not load_models:
            print("Currently training a model for all assets collectively "
                  "with a window size of {} {}.".format(window_size, model_time_unit_plural))

            best_score, best_model, best_param_set = (-np.inf, None, None)

            for i in range(len(models_to_test)):
                model_to_test = models_to_test[i]
                param_grid = param_grids[i]
                param_grid['hidden_layer_sizes'] = hdn_lyr_szs_nn_collective
                if model_to_test == keras_model_nn:
                    X_scaled_dict = reshape_data_from_2D_to_keras(param_grid, subset_num_currencies, X_scaled)
                    model, score, param_set = \
                        keras_reg_grid_search(X_scaled_dict, y_scaled, build_fn=create_keras_regressor,
                                              param_grid=param_grid,
                                              epochs=keras_nn_epochs, cv_epochs=keras_nn_epochs_grd_srch,
                                              cv=keras_cv, scoring=r2_score, verbose=keras_grd_srch_verbose,
                                              plot_losses=make_keras_loss_plots, plotting_dir=keras_figures_dir,
                                              figure_title_prefix='Collective (window size {})'.format(window_size),
                                              figure_kwargs={'figsize': figure_size, 'dpi': figure_dpi})
                    param_set['optimizer'] = keras_convert_optimizer_obj_to_name(param_set['optimizer'])
                    param_vals_tup = tuple(param_set.values()) + ('all', window_size)
                    num_occurrences = keras_param_sets_occurrences_window.setdefault(param_vals_tup, 0)
                    keras_param_sets_occurrences_window[param_vals_tup] = num_occurrences + 1
                else:
                    grid_search = GridSearchCV(model_to_test, param_grid, scoring=make_scorer(r2_score),
                                               cv=cv, n_jobs=num_lgc_prcs_grd_srch)
                    grid_search.fit(X_scaled, y_scaled.ravel())
                    model = grid_search.best_estimator_
                    score = grid_search.best_score_
                    param_set = grid_search.best_params_
                best_model = model if score > best_score else best_model
                best_param_set = param_set if score > best_score else best_param_set
                best_score = score if score > best_score else best_score
            time.sleep(1)  
            print("Best collective model and score for window size {}: {}"
                  .format(window_size, (best_model, best_score)))
            collective_assets_model_type = type(best_model)
            if type(best_model) is keras.models.Sequential:
                collective_assets_model_path = model_base_path + '.h5'
                ks_models.save_model(best_model, model_base_path + '.h5')
            else:
                collective_assets_model_path = model_pickle_path
                with open(model_pickle_path, "wb") as model_outfile:
                    pickle.dump(best_model, model_outfile)
            with open(param_set_pickle_path, "wb") as param_set_outfile:
                pickle.dump(best_param_set, param_set_outfile)
            keras_window_size_scores[window_size] = best_score
            with open(keras_window_size_scores_pickle_path, 'wb') as keras_window_size_scores_outfile:
                pickle.dump(keras_window_size_scores, keras_window_size_scores_outfile)
        else:
            try: 
                best_model = ks_models.load_model(model_base_path + '.h5')
                collective_assets_model_path = model_base_path + '.h5'
            except OSError:
                collective_assets_model_path = model_pickle_path
                with open(model_pickle_path, "rb") as model_infile:
                    best_model = pickle.load(model_infile)
            with open(param_set_pickle_path, "rb") as param_set_infile:
                best_param_set = pickle.load(param_set_infile)
        collective_assets_model_type = type(best_model)
        collective_assets_model_param_set = best_param_set
        best_model = None
        K.clear_session()

        if not load_models:
            with open(keras_param_set_occurrences_pickle_path, "wb") as param_set_occurrences_outfile:
                pickle.dump(keras_param_sets_occurrences_window, param_set_occurrences_outfile)

        if make_prediction_plots:
            print("Predicting prices with a window of {} {} of preceding currency values.".format(window_size, model_time_unit_plural))

            def single_asset_models_predict(X):
                single_asset_models_pred = np.empty_like(X)
                for currency_index in range(subset_num_currencies):
                    param_set = single_asset_models_param_sets[currency_index]
                    X_subset = get_X_subset(X, currency_index, subset_num_currencies)
                    single_asset_model = single_asset_models[currency_index]
                    if type(single_asset_model) is keras.models.Sequential:
                        X_subset = reshape_data_from_2D_to_keras(param_set, 1, X_subset)
                    single_asset_models_pred[:, currency_index::subset_num_currencies] = single_asset_model.predict(X_subset)
                return single_asset_models_pred

            def collective_assets_model_predict(X):
                param_set = collective_assets_model_param_set
                if type(collective_assets_model) is keras.models.Sequential:
                    X = reshape_data_from_2D_to_keras(param_set, subset_num_currencies, X)
                collective_assets_model_pred = collective_assets_model.predict(X)
                return collective_assets_model_pred

            def load_model(model_type, model_path):
                if model_type is keras.models.Sequential:
                    model = ks_models.load_model(model_path)
                else:
                    with open(model_path, "rb") as model_infile:
                        model = pickle.load(model_infile)
                return model

            def load_single_asset_models():
                for currency_index in range(subset_num_currencies):
                    single_asset_models[currency_index] = \
                        load_model(single_asset_models_types[currency_index], single_asset_models_paths[currency_index])

            def load_collective_assets_model():
                nonlocal collective_assets_model
                collective_assets_model = load_model(collective_assets_model_type, collective_assets_model_path)

            def fmt_pred_for_vis(predictions, num_times_excess, num_times_expected, num_features):
                return np.concatenate(
                    [*predictions[:-1], predictions[-1, num_times_excess*num_features:]])\
                    .reshape((num_times_expected, num_features))

            X_scaled_fmt_for_pred = X_scaled[::window_size]
            y_fmt_for_vis = y[::window_size]
            if X_scaled.shape[0] % window_size != 0:
                X_scaled_fmt_for_pred = np.concatenate([X_scaled_fmt_for_pred, X_scaled[[-1]]])
                y_fmt_for_vis = np.concatenate([y_fmt_for_vis, y[[-1]]])
            pred_num_times = int(np.prod(np.array(X_scaled_fmt_for_pred.shape)) / subset_num_currencies)
            pred_num_times_expected = num_non_nan_times - window_size
            pred_num_times_excess = pred_num_times - pred_num_times_expected
            fmt_pred_for_vis_val = \
                partial(fmt_pred_for_vis, num_times_excess=pred_num_times_excess,
                        num_times_expected=pred_num_times_expected, num_features=subset_num_currencies)

            num_orig_extrp_rows = int(np.ceil(model_num_extrapolation_times / window_size))
            extrp_y_shape = (num_orig_extrp_rows, subset_num_currencies * window_size)
            extrp_pred_num_times = int(num_orig_extrp_rows * window_size)
            extrp_pred_num_times_expected = model_num_extrapolation_times
            extrp_pred_num_times_excess = extrp_pred_num_times - extrp_pred_num_times_expected
            fmt_pred_for_vis_extrp = \
                partial(fmt_pred_for_vis, num_times_excess=extrp_pred_num_times_excess,
                        num_times_expected=extrp_pred_num_times_expected, num_features=subset_num_currencies)

            y_fmt_for_vis = fmt_pred_for_vis_val(y_fmt_for_vis) 

            load_single_asset_models()
            single_asset_models_pred = \
                fmt_pred_for_vis_val(y_scaler.inverse_transform(single_asset_models_predict(X_scaled_fmt_for_pred)))
            single_asset_models = [None]*len(single_asset_models) 
            K.clear_session()

            load_collective_assets_model()
            collective_assets_model_pred = \
                fmt_pred_for_vis_val(y_scaler.inverse_transform(collective_assets_model_predict(X_scaled_fmt_for_pred)))
            collective_assets_model = None
            K.clear_session()

            actual_values_color = 'blue'
            actual_values_label = 'True'
            ml_model_single_asset_color = 'red'
            ml_model_single_asset_label = 'ML Model (Single Asset Predictors)'
            ml_model_collective_color = 'orange'
            ml_model_collective_label = 'ML Model (Collective Predictor)'
            monte_carlo_color = 'green'
            monte_carlo_label = 'Monte Carlo AVG'

            subset_num_cols = 2
            subset_num_rows = int(np.ceil(subset_num_currencies / subset_num_cols))
            collective_fig = plt.figure(figsize=(12 * subset_num_cols, 6 * subset_num_rows))
            for currency_index in range(subset_num_currencies):
                currency_label = subset_currencies_labels[currency_index]
                currency_label_no_spaces = currency_label.replace(' ', '_')
                currency_ticker = subset_currencies_tickers[currency_index]
                collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, currency_index + 1)
                collective_ax_current.plot(date_times[window_size:], y_fmt_for_vis[:, currency_index],
                                           color=actual_values_color, alpha=0.5, label=actual_values_label)
                collective_ax_current.plot(date_times[window_size:], single_asset_models_pred[:, currency_index],
                                           color=ml_model_single_asset_color, alpha=0.5, label=ml_model_single_asset_label)
                collective_ax_current.plot(date_times[window_size:], collective_assets_model_pred[:, currency_index],
                                           color=ml_model_collective_color, alpha=0.5, label=ml_model_collective_label)
                collective_ax_current.set_xlabel('Date')
                collective_ax_current.set_ylabel('Close')
                collective_ax_current.set_title("{} ({}) Closing Value ({} {} window)"
                                                .format(currency_label, currency_ticker, window_size,
                                                        model_time_unit_singular))
                collective_ax_current.legend()
                indiv_fig = plt.figure(figsize=(12, 6))
                indiv_fig_ax = indiv_fig.add_subplot(111)
                indiv_fig_ax.plot(date_times[window_size:], y_fmt_for_vis[:, currency_index],
                                  color=actual_values_color, alpha=0.5, label=actual_values_label)
                indiv_fig_ax.plot(date_times[window_size:], single_asset_models_pred[:, currency_index],
                                  color=ml_model_single_asset_color, alpha=0.5, label=ml_model_single_asset_label)
                indiv_fig_ax.plot(date_times[window_size:], collective_assets_model_pred[:, currency_index],
                                  color=ml_model_collective_color, alpha=0.5, label=ml_model_collective_label)
                indiv_fig_ax.set_xlabel('Date')
                indiv_fig_ax.set_ylabel('Close')
                indiv_fig_ax.set_title("{} ({}) Closing Value ({} {} window)"
                                       .format(currency_label, currency_ticker, window_size,
                                               model_time_unit_singular))
                indiv_fig_ax.legend()
                currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
                indiv_fig.savefig('{}/{}_validation_{}.png'
                                  .format(currency_figures_subdir, currency_label_no_spaces, window_size),
                                  dpi=figure_dpi)
            collective_fig.savefig('{}/validation_{}.png'
                                   .format(figures_dir, window_size), dpi=figure_dpi)
            collective_fig.clf()

            load_single_asset_models()
            single_asset_models_extrapolation_y = \
                np.zeros((extrp_y_shape), dtype=np.float64)
            single_asset_models_extrapolation_y[0] = \
                y_scaler.inverse_transform(single_asset_models_predict(
                    X_scaled[-1].reshape(1,-1)))
            for window_index in range(1, len(single_asset_models_extrapolation_y)):
                single_asset_models_extrapolation_y[window_index] = \
                    y_scaler.inverse_transform(single_asset_models_predict(
                        X_scaler.transform(single_asset_models_extrapolation_y[window_index-1].reshape(1,-1))))
            single_asset_models = [None] * len(single_asset_models)
            K.clear_session()
            single_asset_models_extrapolation_y = fmt_pred_for_vis_extrp(single_asset_models_extrapolation_y)

            load_collective_assets_model()
            collective_assets_model_extrapolation_y = \
                np.zeros((int(np.ceil(model_num_extrapolation_times / window_size)),
                          subset_num_currencies * window_size), dtype=np.float64)
            collective_assets_model_extrapolation_y[0] = \
                y_scaler.inverse_transform(collective_assets_model_predict(
                    X_scaled[-1].reshape(1, -1)))
            for window_index in range(1, len(collective_assets_model_extrapolation_y)):
                collective_assets_model_extrapolation_y[window_index] = \
                    y_scaler.inverse_transform(collective_assets_model_predict(
                        X_scaler.transform(collective_assets_model_extrapolation_y[window_index - 1].reshape(1, -1))))
            collective_assets_model = None
            K.clear_session()
            collective_assets_model_extrapolation_y = fmt_pred_for_vis_extrp(collective_assets_model_extrapolation_y)

            for currency_index in range(subset_num_currencies):
                currency_label = subset_currencies_labels[currency_index]
                currency_label_no_spaces = currency_label.replace(' ', '_')
                currency_ticker = subset_currencies_tickers[currency_index]
                label_and_ticker = "{} ({})".format(currency_label, currency_ticker)
                collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, currency_index + 1)
                collective_ax_current.plot(model_extrapolation_times, single_asset_models_extrapolation_y[:, currency_index],
                                           color=ml_model_single_asset_color, label=ml_model_single_asset_label)
                collective_ax_current.plot(model_extrapolation_times, collective_assets_model_extrapolation_y[:, currency_index],
                                           color=ml_model_collective_color, label=ml_model_collective_label)
                collective_ax_current.plot(model_extrapolation_times, subset_monte_carlo_predicted_values[label_and_ticker],
                                           color=monte_carlo_color, label=monte_carlo_label)
                monte_carlo_plot_confidence_band(collective_ax_current, model_extrapolation_times,
                                                 subset_monte_carlo_predicted_values_ranges, label_and_ticker, color='cyan')
                collective_ax_current.set_xlabel('Date')
                collective_ax_current.set_ylabel('Close')
                collective_ax_current.set_title("{} ({}) Predicted Closing Value ({} {} window)"
                                                .format(currency_label, currency_ticker, window_size,
                                                        model_time_unit_singular))
                collective_ax_current.legend()
                indiv_fig = plt.figure(figsize=(12, 6))
                indiv_fig_ax = indiv_fig.add_subplot(111)
                indiv_fig_ax.plot(model_extrapolation_times, single_asset_models_extrapolation_y[:, currency_index],
                                  color=ml_model_single_asset_color, label=ml_model_single_asset_label)
                indiv_fig_ax.plot(model_extrapolation_times, collective_assets_model_extrapolation_y[:, currency_index],
                                  color=ml_model_collective_color, label=ml_model_collective_label)
                indiv_fig_ax.plot(model_extrapolation_times, subset_monte_carlo_predicted_values[label_and_ticker],
                                  color=monte_carlo_color, label=monte_carlo_label)
                monte_carlo_plot_confidence_band(indiv_fig_ax, model_extrapolation_times,
                                                 subset_monte_carlo_predicted_values_ranges, label_and_ticker, color='cyan')
                indiv_fig_ax.set_xlabel('Date')
                indiv_fig_ax.set_ylabel('Close')
                indiv_fig_ax.set_title("{} ({}) Predicted Closing Value ({} {} window)"
                                       .format(currency_label, currency_ticker, window_size,
                                               model_time_unit_singular))
                indiv_fig_ax.legend()
                currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
                indiv_fig.savefig('{}/{}_predictions_{}.png'
                                  .format(currency_figures_subdir, currency_label_no_spaces, window_size),
                                  dpi=figure_dpi)
            collective_fig.savefig('{}/predictions_{}.png'
                                   .format(figures_dir, window_size), dpi=figure_dpi)
            collective_fig.clf()

            for currency_index in range(subset_num_currencies):
                currency_label = subset_currencies_labels[currency_index]
                currency_label_no_spaces = currency_label.replace(' ', '_')
                currency_ticker = subset_currencies_tickers[currency_index]
                label_and_ticker = "{} ({})".format(currency_label, currency_ticker)
                collective_ax_current = collective_fig.add_subplot(subset_num_rows, subset_num_cols, currency_index + 1)
                collective_ax_current.plot(date_times[window_size:], y_fmt_for_vis[:, currency_index],
                                           color=actual_values_color, label=actual_values_label)
                collective_ax_current.plot(model_extrapolation_times, single_asset_models_extrapolation_y[:, currency_index],
                                           color=ml_model_single_asset_color, label=ml_model_single_asset_label)
                collective_ax_current.plot(model_extrapolation_times, collective_assets_model_extrapolation_y[:, currency_index],
                                           color=ml_model_collective_color, label=ml_model_collective_label)
                collective_ax_current.plot(model_extrapolation_times, subset_monte_carlo_predicted_values[label_and_ticker],
                                           color=monte_carlo_color, label=monte_carlo_label)
                monte_carlo_plot_confidence_band(collective_ax_current, model_extrapolation_times,
                                                 subset_monte_carlo_predicted_values_ranges, label_and_ticker, color='cyan')
                collective_ax_current.set_xlabel('Date')
                collective_ax_current.set_ylabel('Close')
                collective_ax_current.set_title("{} ({}) True + Predicted Closing Value ({} {} window)"
                                                .format(currency_label, currency_ticker, window_size,
                                                        model_time_unit_singular))
                collective_ax_current.legend()
                indiv_fig = plt.figure(figsize=(12, 6))
                indiv_fig_ax = indiv_fig.add_subplot(111)
                indiv_fig_ax.plot(date_times[window_size:], y_fmt_for_vis[:, currency_index],
                                  color=actual_values_color, label=actual_values_label)
                indiv_fig_ax.plot(model_extrapolation_times, single_asset_models_extrapolation_y[:, currency_index],
                                  color=ml_model_single_asset_color, label=ml_model_single_asset_label)
                indiv_fig_ax.plot(model_extrapolation_times, collective_assets_model_extrapolation_y[:, currency_index],
                                  color=ml_model_collective_color, label=ml_model_collective_label)
                indiv_fig_ax.plot(model_extrapolation_times, subset_monte_carlo_predicted_values[label_and_ticker],
                                  color=monte_carlo_color, label=monte_carlo_label)
                monte_carlo_plot_confidence_band(indiv_fig_ax, model_extrapolation_times,
                                                 subset_monte_carlo_predicted_values_ranges, label_and_ticker, color='cyan')
                indiv_fig_ax.set_xlabel('Date')
                indiv_fig_ax.set_ylabel('Close')
                indiv_fig_ax.set_title("{} ({}) True + Predicted Closing Value ({} {} window)"
                                       .format(currency_label, currency_ticker, window_size,
                                               model_time_unit_singular))
                indiv_fig_ax.legend()
                currency_figures_subdir = '{}/{}'.format(figures_dir, currency_label_no_spaces)
                indiv_fig.savefig('{}/{}_actual_plus_predictions_{}.png'
                                  .format(currency_figures_subdir, currency_label_no_spaces, window_size),
                                  dpi=figure_dpi)
            collective_fig.savefig('{}/actual_plus_predictions_{}.png'.format(figures_dir, window_size), dpi=figure_dpi)
        plt.close('all')

    if make_keras_param_set_occurrence_plots:

        with open(keras_window_size_scores_pickle_path, 'rb') as keras_window_size_scores_infile:
            keras_window_size_scores = pickle.load(keras_window_size_scores_infile)
        keras_best_window_size = max(keras_window_size_scores, key=keras_window_size_scores.get)
        keras_param_sets_occurrences = {}
        for keras_param_set_occurrences_pickle_path in param_set_occurrences_pickle_paths:
            with open(keras_param_set_occurrences_pickle_path, "rb") as model_infile:
                param_set_occurrences_window = pickle.load(model_infile)
            for k,v in param_set_occurrences_window.items():
                keras_param_sets_occurrences[k] = keras_param_sets_occurrences.get(k,0) + param_set_occurrences_window[k]

        keras_unique_param_sets_tuples = list(keras_param_sets_occurrences.keys())
        col_names_to_add = [asset_col_str, window_size_col_str]  
        index = pd.MultiIndex.from_tuples(keras_unique_param_sets_tuples, names=keras_param_names + col_names_to_add)
        keras_param_sets_occurrences_series = pd.Series(list(keras_param_sets_occurrences.values()), index=index)
        num_occurrences_str = 'num_occurrences'
        keras_param_sets_occurrences_frame = keras_param_sets_occurrences_series.to_frame(num_occurrences_str)
        keras_param_sets_occurrences_frame.reset_index(inplace=True)
        plotting_data = keras_param_sets_occurrences_frame.loc[
            keras_param_sets_occurrences_frame[window_size_col_str] == keras_best_window_size]
        plotting_data.drop(window_size_col_str, axis=1, inplace=True)
        plotting_index_headers = list(set(plotting_data.columns) - {num_occurrences_str, window_size_col_str, asset_col_str})
        index_col_str = ", ".join(plotting_index_headers)
        plotting_data[index_col_str] = plotting_data[plotting_index_headers].apply(tuple, axis=1)
        plotting_data.drop(plotting_index_headers, axis=1, inplace=True)
        plotting_data.sort_values(by=num_occurrences_str, ascending=False, inplace=True)
        fig, ax = plt.subplots()
        sns.barplot(x=index_col_str, y=num_occurrences_str, hue=asset_col_str, data=plotting_data, ax=ax)
        x_ticks_rotation_amt = 15
        plt.xticks(rotation=x_ticks_rotation_amt, fontsize=6)
        plt.title('Parameter set occurrences for window size {}'.format(keras_best_window_size))
        plt.tight_layout()
        plt.savefig('{}/param_set_occurrences_keras_{}_{}.png'
                    .format(keras_figures_dir, frst_hdn_lyr_sz_nn_single_asset,
                            frst_hdn_lyr_sz_nn_collective), dpi=figure_dpi)

if __name__ == '__main__':
    main()
EOF
