from drl_robo_advisor.experience_batch import ExperienceBatch
from drl_robo_advisor.portfolio_trainer import PortfolioTrainer
from drl_robo_advisor.network import Network
from pathlib import Path
import numpy as np
import torch

class Agent:
    """

    Agent is the class to manage historical data and conduct training and after observing new data, predict the next portfolio weight to optimize total return
    It includes

    - experience_batch: after reading historical data, store it and allow extraction of batches
    - network: the neural network model
    - portfolio_trainer: to train a model and predict by the model

    Methods
    ------
    read_new_data(asset_name,asset_prices)

    consol_new_data()

    load_network_if_exist()

    copy_model_roll_train(from_path,to_path,logger)

    set_mode(mode,config_settings)

    train_agent(write_to_summary,plot_dist_indexs,save_model,do_validation,desc_tqdm)

    predict(cur_time)

    """
    def __init__(self,observed_price_feature_history,config_settings,logger,load_network=True):
        """

        Parameters
        ----------
        observed_price_feature_history : dict[str -> pandas.DataFrame]
            A dictionary of data batch with each feature as key and value as the dataframe of historical data
        config_settings : ConfigSetter
            Configuration reference to config.json
        logger : Logger
            current logger
        load_network : bool
            Decide whether to load a network if it exists in the model directory

        """
        self._asset_names = config_settings.stock_tickers
        self._asset_symbols = config_settings.asset_symbols
        self._benchmark_stock_ticker = config_settings.benchmark_stock_ticker
        self._num_assets = len(self._asset_names)
        self._price_feature_names = config_settings.feature_names
        self._feature_number = len(self._price_feature_names)
        self._basic_features = config_settings.basic_features
        self._window_size = config_settings.hyperparameters['window_size']
        self._logger = logger

        self._model_path = Path(config_settings.model_path, config_settings.model_filename)
        self._model_root_path = config_settings.model_path
        self._model_filename = config_settings.model_filename


        self._create_experiences(observed_price_feature_history,config_settings,logger)

        self._create_network(config_settings,load_network)
        self._logger.info(f'NN network created: {self._network}')

    @property
    def asset_names(self):
        """

        Returns
        -------
        asset_names : list[str]

        """
        return self._asset_names

    @property
    def asset_symbols(self):
        """

        Returns
        -------
        asset_symbols : list[zipline.assets.Equity]

        """
        return self._asset_symbols

    @property
    def num_assets(self):
        """

        Returns
        -------
        num_assets : int

        """
        return self._num_assets

    @property
    def price_feature_names(self):
        """

        Returns
        -------
        price_feature_names : list[str]

        """
        return self._price_feature_names

    @property
    def basic_features(self):
        """

        Returns
        -------
        basic_features : list[str]

        """
        return self._basic_features

    @property
    def feature_number(self):
        """

        Returns
        -------
        feature_number : int

        """
        return self._feature_number

    @property
    def portfolio_vector_memory(self):
        """

        Returns
        -------
        portfolio_vector_memory : numpy.array

        """
        return self._portfolio_vector_memory

    @portfolio_vector_memory.setter
    def portfolio_vector_memory(self,portfolio_vector_memory):
        self._portfolio_vector_memory = portfolio_vector_memory

    @property
    def network(self):
        """

        Returns
        -------
        network : Network

        """
        return self._network

    @property
    def model_path(self):
        """

        Returns
        -------
        model_path : pathlib.Path

        """
        return self._model_path

    @property
    def model_root_path(self):
        """

        Returns
        -------
        model_root_path : pathlib.Path

        """
        return self._model_root_path

    @property
    def mode(self):
        """

        Returns
        -------
        mode : str

        """
        return self._mode

    @property
    def portfolio_trainer(self):
        """

        Returns
        -------
        portfolio_trainer : PortfolioTrainer

        """
        return self._portfolio_trainer

    @property
    def experience_batch(self):
        """

        Returns
        -------
        experience_batch : ExperienceBatch

        """
        return self._experience_batch

    @property
    def last_trade_w(self):
        """

        Returns
        -------
        last_trade_w : numpy.array

        """
        return  self._last_trade_w

    @last_trade_w.setter
    def last_trade_w(self,last_trade_w):
        self._last_trade_w=last_trade_w

    def _create_experiences(self,price_feature_history,config_settings,logger):
        """

        Create an object of ExperienceBatch

        Parameters
        ----------
        price_feature_history : dict[str -> pandas.DataFrame]
            A dictionary of data batch  with each feature as key and value as the dataframe of historical data
        config_settings : ConfigSetter
            Configuration reference to config.json
        logger : Logger
            current logger

        """
        self._experience_batch = ExperienceBatch(price_feature_history=price_feature_history,valid_portion=config_settings.valid_portion,agent=self,
                                                 config_settings=config_settings,logger=logger)

    def read_new_data(self,asset_symbol,asset_prices):
        """

        Read a new window of data for the stock

        Parameters
        ----------
        asset_symbol : zipline.assets.Equity
        asset_prices : pandas.DataFrame

        """
        self._experience_batch.observe_price(asset_symbol,asset_prices)

    def consol_new_data(self):
        """

        Add one day to memory and experiences after observing one day of data and add an initialized set of memory of size 1 day with all zero

        """
        #
        self._experience_batch.combine_obervation()
        self._portfolio_vector_memory = np.concatenate((self._portfolio_trainer.portfolio_vector_memory,np.full([1, self._num_assets], 0.0,dtype=np.float32)))

    def _create_network(self,config_settings,load_network):
        """

        Create an object of Network and then load parameters from trained model if load_network is True

        Parameters
        ----------
        config_settings : ConfigSetter
            Configuration reference to config.json
        load_network : bool

        """
        self._network = Network(feature_number=self._feature_number, rows=self._num_assets,
                                    columns=config_settings.hyperparameters['window_size'],
                                    conv_layer_outputs=config_settings.hyperparameters['conv_layer_outputs'],
                                    eiie_dense_out_channels=config_settings.hyperparameters['eiie_dense_out_channels']) \
                .to(config_settings.device)
        self._logger.info(f'NN model created at seed {torch.seed()}')
        if load_network==True:
            self.load_network_if_exist()

    def _create_portfolio_trainer(self,config_settings):
        """

        Create an object of PortfolioTrainer

        Parameters
        ----------
        config_settings : ConfigSetter
            Configuration reference to config.json

        """
        self._portfolio_trainer = PortfolioTrainer(config_settings=config_settings,agent=self,logger=self._logger)

    def load_network_if_exist(self):
        """

        Check if there is saved model in the model directory and load its parameters

        """
        if self._model_path.exists():
            loaded_network = torch.load(self._model_path, map_location=torch.device('cpu'))
            self._network.load_state_dict(loaded_network)
            self._logger.info(f'NN model loaded from {self._model_path}')

    @staticmethod
    def copy_model_roll_train(from_path,to_path,logger):
        """

        Copy the saved model to prevent overwriting of original trained model during back-test roll training

        Parameters
        ----------
        from_path : pathlib.Path
        to_path : pathlib.Path
        logger : Logger
            current logger

        """
        to_path.write_bytes(from_path.read_bytes())
        logger.info(f'NN model copied from {from_path} to {to_path}')

    def _initial_trade_w(self):
        """

        Initialize the last predicted portfolio weights as all zero

        """
        self._last_trade_w = np.full([1, self._num_assets], 0)
        self._last_trade_w = np.concatenate(([1], self._last_trade_w ), axis=None)

    def _initial_baseline_w(self, initial_total_capital=1):
        """

        Initialize the last predicted portfolio weights as all zero and total capital for equiweight, all cash and benchmark portfolio

        Parameters
        ----------
        initial_total_capital : float, default=1

        """
        self.equiweight_w = np.full([1, self._num_assets], 1 / self._num_assets)
        self.equiweight_w = np.concatenate(([0], self.equiweight_w), axis=None)
        self.total_capital_equiweight_w = initial_total_capital

        self.allcash_w = np.full([1, self._num_assets], 0)
        self.allcash_w = np.concatenate(([1], self.allcash_w), axis=None)
        self.total_capital_allcash_w = initial_total_capital

        self.benchmark_w = np.full([1, self._num_assets], 0)
        self.benchmark_w[0][self._asset_names.index(self._benchmark_stock_ticker)] = 1
        self.benchmark_w = np.concatenate(([0], self.benchmark_w), axis=None)
        self.total_capital_benchmark_w = initial_total_capital

    def set_mode(self,mode,config_settings):
        """

        According to the model, initialize the past weights memory, portfolio trainer and others

        Parameters
        ----------
        mode : str
            The modes include {train,opt-train,predict}
        config_settings : ConfigSetter
            Configuration reference to config.json

        """
        self._mode = mode

        if mode == 'train':
            assert self._experience_batch.price_feature_history is not None
            self._portfolio_vector_memory = np.full([self._experience_batch.experiences.shape[1], self._num_assets],0.0,dtype=np.float32)
            self._create_portfolio_trainer(config_settings)
            self._portfolio_trainer.set_train_writer()
        elif mode == 'opt-train':
            assert self._experience_batch.price_feature_history is not None
            self._portfolio_vector_memory = np.full([self._experience_batch.experiences.shape[1], self._num_assets],0.0,dtype=np.float32)
            self._create_portfolio_trainer(config_settings)
        elif mode == 'predict':
            if self._experience_batch.price_feature_history is not None:
                self._portfolio_vector_memory = np.full([self._experience_batch.experiences.shape[1], self._num_assets],0.0,dtype=np.float32)
            self._create_portfolio_trainer(config_settings)
            self._initial_trade_w()
            self._initial_baseline_w()
            self._original_history_size = self._experience_batch.price_feature_history[self._price_feature_names[0]].shape[0]
            self._portfolio_trainer.set_predict_writer()
        self._portfolio_trainer.agent = self

    def train_agent(self,write_to_summary=True,plot_dist_indexs=True,save_model=True,do_validation=False,desc_tqdm="Training Steps"):
        """

        If there are enough historical data, at least one batch, and then start training of network of agent

        Parameters
        ----------
        write_to_summary : bool
        plot_dist_indexs : bool
        save_model : bool
        do_validation : bool
        desc_tqdm : str

        Returns
        -------
        save_indexs : numpy.array
        avg_valid_loss : numpy.array

        """
        flag_enough_data = self._experience_batch.check_if_enough_data()
        if flag_enough_data==True:
            save_indexs, avg_valid_loss = self._portfolio_trainer.train(write_to_summary=write_to_summary,plot_dist_indexs=plot_dist_indexs,save_model=save_model,do_validation=do_validation,desc_tqdm=desc_tqdm)
            return save_indexs, avg_valid_loss
        else:
            self._logger.error('Not enough training data')

    def predict(self,cur_time_step,current_datetime):
        """

        Wait until enough observation, at least one batch of data, and then start predicting by the trained model
        Otherwise, return a dummy prediction with all zero

        Parameters
        ----------
        cur_time_step : int
            The current time step of prediction
        current_datetime : datetime
            The current datetime of prediction

        Returns
        -------
        predicted_assets_weight : numpy.array

        """
        if self._experience_batch.check_if_enough_data(self._original_history_size)==True:
            predicted_assets_weights = self._portfolio_trainer.predict(self._experience_batch, self._portfolio_vector_memory, cur_time_step,current_datetime)
            return predicted_assets_weights
        else:
            self._portfolio_trainer.portfolio_vector_memory = self._portfolio_vector_memory
            predicted_assets_weights = np.full([self._num_assets], 0)
            self._logger.debug('Waiting for enough data length of window size')
            return predicted_assets_weights
