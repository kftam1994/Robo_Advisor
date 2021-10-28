from drl_robo_advisor.config_setter import ConfigSetter
from drl_robo_advisor.logging_handler import LoggingHandler
from drl_robo_advisor.prepare_zipline_data import ZiplineDataPreparation
from drl_robo_advisor.agent import Agent
import sys
from drl_robo_advisor.zipline_data_handler import ZiplineDataHandler
from drl_robo_advisor.zipline_back_test_handler import ZiplineBacktestHandler
import pandas as pd
from drl_robo_advisor.model_optimizer import ModelOptimizer
from pathlib import Path
from drl_robo_advisor.utils import plot_dist
from argparse import ArgumentParser
import torch
import numpy as np
import random

config_json_file_path = './config.json'

def set_global_seed():
    seed = 55
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)

def data_preparation(config_settings,logger):
    """

    Download and load data to zipline

    Parameters
    ----------
    config_settings : ConfigSetter
        Configuration reference to config.json
    logger : Logger
        current logger

    """
    zipline_data_preparation = ZiplineDataPreparation(config_settings=config_settings,
                                                      logger=logger)
    zipline_data_preparation.load_data_to_zipline()

def construct_training_dates(config_settings,logger):
    """

    Parameters
    ----------
    config_settings : ConfigSetter
        Configuration reference to config.json
    logger : Logger
        current logger

    Returns
    -------
    train_start : pandas.Timestamp
    train_end : pandas.Timestamp

    """
    train_start = config_settings.initial_start_date
    train_end = config_settings.initial_start_date + pd.offsets.DateOffset(
        years=config_settings.num_years_before_fold) + pd.offsets.DateOffset(
        years=config_settings.num_years_per_fold * config_settings.num_folds)
    logger.info(f'Load history data from {train_start} to {train_end}')
    return train_start,train_end

def construct_testing_dates(config_settings,logger):
    """

    Parameters
    ----------
    config_settings : ConfigSetter
        Configuration reference to config.json
    logger : Logger
        current logger

    Returns
    -------
    test_start : pandas.Timestamp
    test_end : pandas.Timestamp

    """
    test_start = config_settings.initial_start_date + pd.offsets.DateOffset(
        years=config_settings.num_years_before_fold) + pd.offsets.DateOffset(
        years=config_settings.num_years_per_fold * config_settings.num_folds) + pd.offsets.DateOffset(days=1)
    test_end = test_start + pd.offsets.DateOffset(years=6)
    logger.info(f'start testing from {test_start} to {test_end}')
    return test_start,test_end

def run_training(training_data_batch,config_settings,logger):
    """

    Initialize an object of Agent, set mode to training and start to run training

    Parameters
    ----------
    training_data_batch : dict[str -> pandas.DataFrame]
            A dictionary of data batch with each feature as key and value as the dataframe of historical data
    config_settings : ConfigSetter
        Configuration reference to config.json
    logger : Logger
        current logger

    """
    logger.info(f'----------Start Training----------')
    advisor = Agent(observed_price_feature_history=training_data_batch, config_settings=config_settings, logger=logger)
    advisor.set_mode(mode='train', config_settings=config_settings)
    advisor.train_agent()
    logger.info(f'----------End Training----------')

def run_hyperparam_optimization(train_start,training_data_batch,config_settings,logger):
    """

    Initialize an object of modelOptimizer and start to run hyperparameters optimization

    Parameters
    ----------
    train_start : pandas.Timestamp
    training_data_batch : dict[str -> pandas.DataFrame]
            A dictionary of data batch with each feature as key and value as the dataframe of historical data
    config_settings : ConfigSetter
        Configuration reference to config.json
    logger : Logger
        current logger

    """
    logger.info(f'----------Start Training Optimization----------')
    model_optimizer = ModelOptimizer(training_data_batch=training_data_batch, train_start=train_start,
                                     config_settings=config_settings, logger=logger)
    best_hyperparams = model_optimizer.optimize_model(optimize_tuning_max_eval=config_settings.optimize_tuning_max_eval)
    model_optimizer.plot_optuna_result()
    logger.info(f'Best hyperparameters for training: {best_hyperparams}')
    logger.info(f'----------End Training Optimization----------')

def run_back_testing(test_start, test_end,training_data_batch,zipline_data_handler,config_settings,logging_handler,logger):
    """

    Copy the saved trained model and then initialize an object of Agent, set mode to predict
    After initialize the object of ZiplineBacktestHandler, start to run back-testing

    Parameters
    ----------
    test_start : pandas.Timestamp
    test_end : pandas.Timestamp
    training_data_batch : dict[str -> pandas.DataFrame]
            A dictionary of data batch with each feature as key and value as the dataframe of historical data
    zipline_data_handler : ZiplineDataHandler
    config_settings : ConfigSetter
        Configuration reference to config.json
    logging_handler : LoggingHandler
    logger : Logger
        current logger

    """
    logger.info(f'----------Start Testing----------')
    Agent.copy_model_roll_train(from_path=Path(config_settings.model_path, config_settings.model_filename),
                                to_path=Path(config_settings.model_path, config_settings.rolltrain_model_filename),
                                logger=logger)
    config_settings.model_filename = config_settings.rolltrain_model_filename
    config_settings.hyperparameters = config_settings.backtest_rolltrain_hyperparameters
    advisor = Agent(observed_price_feature_history=training_data_batch, config_settings=config_settings, logger=logger)
    advisor.set_mode(mode='predict', config_settings=config_settings)

    zipline_backtest_handler = ZiplineBacktestHandler(agent=advisor,
                                                      custom_data_loaders=zipline_data_handler.custom_data_loaders,
                                                      make_pipeline=zipline_data_handler.make_pipeline,
                                                      config_settings=config_settings, logger=logger)
    selected_sessions = zipline_data_handler.get_selected_trading_sessions(test_start, test_end)
    zipline_backtest_handler.run_back_test(selected_sessions)
    zipline_backtest_handler.plot_back_test_result(suffix=logging_handler.logger_filename_suffix)
    plot_dist(advisor.model_root_path, advisor.portfolio_trainer.predict_save_indexs, advisor.mode + '_rolltrain')

    advisor.portfolio_trainer.plot_integrate_gradient()

    logger.info(f'----------End Testing----------')

def main():
    args_parser = ArgumentParser()
    args_parser.add_argument("--download_data",dest="download_data",help="Download data and load to zipline",
                             action='store_true')
    args_parser.add_argument("--train", dest="train", help="Start training",
                             action='store_true')
    args_parser.add_argument("--opt_train", dest="opt_train", help="Optimization of hyperparameters of training",
                             action='store_true')
    args_parser.add_argument("--back_test", dest="back_test", help="Start back-testing",
                             action='store_true')

    set_global_seed()

    config_settings = ConfigSetter(config_json_file_path)
    logging_handler = LoggingHandler(runs_path=config_settings.runs_path)
    logger = logging_handler.get_logger()
    sys.excepthook = logging_handler.exception_hook
    logger.info(f'base path is in {config_settings.base_path}')

    args = args_parser.parse_args()
    args.train = True if args.opt_train == True else args.train

    if args.download_data == True:
        data_preparation(config_settings,logger)

    if args.train==True or args.opt_train==True or args.back_test==True:
        train_start, train_end = construct_training_dates(config_settings, logger)
        zipline_data_handler = ZiplineDataHandler(train_start,train_end,config_settings=config_settings,
                                                          logger=logger)

        config_settings.asset_symbols = zipline_data_handler.asset_symbols
        training_data_batch = zipline_data_handler.get_data_batch(train_start, train_end)

        if args.train==True:
            if args.opt_train != True:
                run_training(training_data_batch,config_settings,logger)
                logging_handler.shut_down_logging()
            else:
                run_hyperparam_optimization(train_start,training_data_batch,config_settings,logger)
                logging_handler.shut_down_logging()

        if args.back_test==True:
            test_start, test_end = construct_testing_dates(config_settings, logger)
            run_back_testing(test_start, test_end, training_data_batch, zipline_data_handler, config_settings,
                             logging_handler, logger)
            logging_handler.shut_down_logging()

if __name__ == "__main__":
    main()