import optuna
import time
import torch.multiprocessing as torch_mp
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

class ModelOptimizer:
    """

    Apply Bayesian Optimization to look for optimized hyperparameters for training model in deep reinforcement learning

    Methods
    -------
    plot_optuna_result()

    optimize_model(optimize_tuning_max_eval)

    """
    def __init__(self,training_data_batch,train_start,config_settings,logger):
        """

        Parameters
        ----------
        training_data_batch : dict[str -> pandas.DataFrame]
            A dictionary of data batch  with each feature as key and value as the dataframe of historical data
        train_start : pandas.Timestamp
            Start date of training data
        config_settings : configSetter
            Configuration reference to config.json
        logger : Logger
            current logger

        """

        config_settings.training_steps = config_settings.opt_training_steps

        self._runs_path = config_settings.runs_path
        self._set_opt_objective(training_data_batch, train_start, config_settings, logger)

        sampler = optuna.samplers.TPESampler(seed=10)

        self._set_optuna_logging_handlers(logger)
        self._study = optuna.create_study(direction="minimize", sampler=sampler)

    def _set_optuna_logging_handlers(self,logger):
        """

        Set the logger for optuna to log the optimization progress (e.g. the current performance result and best hyperparameter so far) and the final best hyperparameter and its result

        Parameters
        ----------
        logger : Logger
            current logger

        """
        optuna.logging.disable_default_handler()
        for h in logger.handlers:
            optuna.logging._get_library_root_logger().addHandler(h)

    def _set_opt_objective(self,training_data_batch, train_start, config_settings, logger):
        """

        Set the objective function for optuna's optimization run

        It will conduct a rolling window cross validation.
        The training window is increasing and starts from number of years before fold to number of years before fold + number of years per fold * number of folds. Each fold increases by number of years per fold.
        The validation window is fixed as the out-of-sample number of years per fold

        For example:
        number of years before fold: 5 years
        number of years per fold: 1 year
        number of folds: 5
        For 1st fold: 5 years of training set and 1 year of validation set
        For 2nd fold: 5+1*1 years of training set and 1 year of validation set
        For 3rd fold: 5+1*2 years of training set and 1 year of validation set
        ...
        For 5th fold: 5+1*4 years of training set and 1 year of validation set

        The following hyperparameters are tuned:

        Categorical search:

        - batch size
        - window size
        - regularization norm
        - conv_layer_outputs
        - eiie_dense_out_channels
        - risk penalty

        Uniform distribution search:

        - learning rate
        - weight decay

        Parameters
        ----------
        training_data_batch : dict[str -> pandas.DataFrame]
            A dictionary of data batch  with each feature as key and value as the dataframe of historical data
        train_start : pandas.Timestamp
            Start date of training data
        config_settings : configSetter
            Configuration reference to config.json
        logger : Logger
            current logger

        """

        from drl_robo_advisor.agent import Agent
        def objective(trial):
            batch_size = trial.suggest_categorical('batch_size',
                                                   config_settings.optimize_hyperparameters['batch_sizes'])
            window_size = trial.suggest_categorical('window_size',
                                                    config_settings.optimize_hyperparameters['window_sizes'])
            learning_rate = trial.suggest_uniform('learning_rate',
                                                  config_settings.optimize_hyperparameters['learning_rates'][0],
                                                  config_settings.optimize_hyperparameters['learning_rates'][1])
            weight_decay = trial.suggest_uniform('weight_decay',
                                                 config_settings.optimize_hyperparameters['weight_decays'][0],
                                                 config_settings.optimize_hyperparameters['weight_decays'][1])
            reg_norm = trial.suggest_categorical('reg_norm', config_settings.optimize_hyperparameters['reg_norms'])
            conv_layer_outputs = trial.suggest_categorical('conv_layer_outputs',
                                                           config_settings.optimize_hyperparameters[
                                                               'conv_layer_outputs_list'])
            eiie_dense_out_channels = trial.suggest_categorical('eiie_dense_out_channels',
                                                                config_settings.optimize_hyperparameters[
                                                                    'eiie_dense_out_channels_list'])
            risk_penalty = trial.suggest_categorical('risk_penalty',
                                                     config_settings.optimize_hyperparameters['risk_penalties'])

            hyperparams_for_tuning = {
                'batch_size': batch_size,
                'window_size': window_size,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'reg_norm': reg_norm,
                'conv_layer_outputs': conv_layer_outputs,
                'eiie_dense_out_channels': eiie_dense_out_channels,
                'risk_penalty': risk_penalty,
                "geometric_bias": config_settings.hyperparameters['geometric_bias'],
                "adversarial_flag": config_settings.hyperparameters['adversarial_flag'],
                "noise_normal_mean": config_settings.hyperparameters['noise_normal_mean'],
                "noise_normal_var": config_settings.hyperparameters['noise_normal_var']
            }

            config_settings.set_hyperparameters(hyperparams_for_tuning)

            for hyperparam, hyperparam_value in hyperparams_for_tuning.items():
                logger.info(f'Current space for tuning hyperparameter {hyperparam}: {hyperparam_value}')

            valid_losses = []

            for fold in tqdm(range(1, config_settings.num_folds + 1),
                             desc=f"Optimization Fold of trial {trial.number}"):
                train_to_valid_end = config_settings.initial_start_date + pd.offsets.DateOffset(
                    years=config_settings.num_years_before_fold) + pd.offsets.DateOffset(
                    years=config_settings.num_years_per_fold * fold)
                valid_split_portion = config_settings.num_years_per_fold / (
                        config_settings.num_years_before_fold + config_settings.num_years_per_fold * fold)
                logger.debug(f'-----Start Fold {fold} from {train_start} to {train_to_valid_end}-----')
                training_data_batch_fold = dict(
                    (k, v.loc[train_start:train_to_valid_end]) for k, v in training_data_batch.items())
                config_settings.valid_portion = valid_split_portion
                advisor = Agent(observed_price_feature_history=training_data_batch_fold, config_settings=config_settings,
                                logger=logger,load_network=False)
                advisor.set_mode(mode='opt-train', config_settings=config_settings)
                _, avg_valid_loss = advisor.train_agent(write_to_summary=False, plot_dist_indexs=False,
                                                        save_model=False,
                                                        do_validation=True, desc_tqdm="Optimization Training Steps")
                valid_losses.append(avg_valid_loss)
                logger.info(f'Average Validating Loss at fold {fold}: {avg_valid_loss}')
                logger.debug(f'-----Complete Fold {fold}-----')

            avg_loss = np.mean(valid_losses)
            logger.info(f'Average Validating Loss for {config_settings.num_folds} folds: {avg_loss}')
            return avg_loss

        self._objective_func = objective

    def optimize_model(self,optimize_tuning_max_eval):
        """

        Create the optuna model object for optimization

        Parameters
        ----------
        optimize_tuning_max_eval : int
            number of maximum iterations of the optimization

        """
        start_time = time.time()
        self._study.optimize(self._objective_func, n_trials=optimize_tuning_max_eval,
                       n_jobs=torch_mp.cpu_count())
        print("---Time for Optuna optimization: %s seconds ---" % (time.time() - start_time))
        best_hyperparams = self._study.best_params

        return best_hyperparams

    def plot_optuna_result(self):
        """

        Plot the optimization results after optuna tunning and output to HTML

        - Optimization history
        - Parameter importance
        - Parameter relationship as contour plot
        - High-dimentional parameter relationships

        References
        ----------
        https://optuna.readthedocs.io/en/v2.1.0/reference/visualization.html

        """
        fig = optuna.visualization.plot_optimization_history(self._study)
        fig.write_html(Path(self._runs_path, 'optimization_history.html'))
        fig = optuna.visualization.plot_param_importances(self._study)
        fig.write_html(Path(self._runs_path, 'optimization_param_importances.html'))
        fig = optuna.visualization.plot_contour(self._study)
        fig.write_html(Path(self._runs_path, 'optimization_contour.html'))
        fig = optuna.visualization.plot_parallel_coordinate(self._study, params=['batch_size',
                                                                           'window_size',
                                                                           'learning_rate',
                                                                           'reg_norm',
                                                                            'weight_decay',
                                                                           'eiie_dense_out_channels',
                                                                           'risk_penalty'])
        fig.write_html(Path(self._runs_path, 'optimization_parallel_coordinate.html'))