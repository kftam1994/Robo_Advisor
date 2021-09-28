from pathlib import Path
import json
from datetime import datetime
import os
import pandas as pd
from calendar import monthrange
import pytz
import torch

class ConfigSetter:
    """

    Set configuration and constant from configuration JSON file

    Notes
    -----
    It also sets the environment variable for zipline's data directory, which will be under the base path in config.json

    """
    def __init__(self,config_json_file_path):
        """

        Parameters
        ----------
        config_json_file_path : str

        """
        with open(config_json_file_path,'r') as config_json_file:
            config_json = json.load(config_json_file)

        self.base_path = Path(config_json['BASE_PATH_str']).absolute()
        # Set the environment variable for zipline's data directory
        os.environ["ZIPLINE_ROOT"] = str(self.base_path)

        self.runs_path = Path(self.base_path,config_json['RUNS_FOLDER_NAME'])
        self.runs_path.mkdir(parents=True, exist_ok=True)

        self.model_path = Path(self.base_path,config_json['MODEL_FOLDER_NAME'])
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.model_filename = config_json['MODEL_FILENAME']
        self.rolltrain_model_filename = config_json['ROLLTRAIN_MODEL_FILENAME']

        with open(Path(self.base_path,config_json['TICKER_LIST_FILENAME']), 'r') as tickers_list_file:
            stock_tickers = [line.rstrip() for line in tickers_list_file]
            self.stock_tickers = stock_tickers

        self.benchmark_stock_ticker = config_json['BENCHMARK_STOCK_TICKER']

        self.data_folder_path = Path(self.base_path,config_json['DATA_FOLDER_NAME'])
        self.data_folder_path.mkdir(parents=True, exist_ok=True)

        if 'YEARS_PERIOD' in config_json:
            self.whole_year_periods = config_json['YEARS_PERIOD']
            self.start_date = datetime.strptime(f'{self.whole_year_periods[0]}-01-01', '%Y-%m-%d')
            self.end_date = datetime.strptime(f'{self.whole_year_periods[1]}-12-{monthrange(int(self.whole_year_periods[1]),1)[1]}', '%Y-%m-%d')
        else:
            self.start_date = datetime.strptime(config_json['START_DATE'], '%Y-%m-%d')
            self.end_date = datetime.strptime(config_json['END_DATE'], '%Y-%m-%d')

        self.bundle_name = config_json['BUNDLE_NAME']

        self.frequency = config_json['FREQUENCY']

        self.stock_exchange = config_json['STOCK_EXCHANGE']

        self.basic_features = config_json['BASIC_FEATURES']
        self.macroecon_features = config_json['MACROECON_FEATURES']
        self.stock_specific_feautres = config_json['STOCK_SPECIFIC_FEATURES']
        self.feature_names = self.basic_features+self.macroecon_features+self.stock_specific_feautres

        self.initial_start_date = pd.Timestamp(f"{config_json['INITIAL_START_DATE']} 00:00:00", tz=pytz.utc)
        self.num_years_before_fold = config_json['NUM_YEAR_BEFORE_FOLD']
        self.num_years_per_fold = config_json['NUM_YEAR_PER_FOLD']
        self.num_folds = config_json['NUM_FOLDS']

        self.commission_rate = config_json['COMMISSION_RATE']
        self.total_capital = config_json['TOTAL_CAPITAL']
        self.training_steps = config_json['TRAINING_STEPS']
        self.opt_training_steps = config_json['OPT_TRAINING_STEPS']
        self.predict_rolltrain_steps = config_json['PREDICT_ROLLTRAIN_STEPS']
        self.valid_portion = config_json['VALID_PORTION']

        self.hyperparameters = config_json['HYPERPARAMETERS']
        self.optimize_hyperparameters = config_json['OPTIMIZE_HYPERPARAMETERS']
        self.backtest_rolltrain_hyperparameters = config_json['BACKTEST_ROLLTRAIN_HYPERPARAMETERS']

        self.backtest_capital_base = config_json['BACKTEST_CAPITAL_BASE']

        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        self.optimize_tuning_max_eval = config_json['OPTIMIZE_TUNING_MAX_EVAL']

        self.asset_symbols = None

    def set_hyperparameters(self,hyperparameters):
        """

        Parameters
        ----------
        hyperparameters : dict[str -> ]

        """
        self.hyperparameters = hyperparameters