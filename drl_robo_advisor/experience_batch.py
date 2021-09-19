import numpy as np
import pandas as pd
from collections import namedtuple
import math

class Batch(namedtuple('Batch',field_names=['start_idx','batch_X','batch_y','last_w'])):
    """

    A batch of data with window size

    - starting index: the index of the first record of the batch in the whole experiences
    - batch X: the normalized stock data features
    - batch y: the relative change of price within the day
    - last w: the last predicted weight

    """

class ExperienceBatch:
    """

    Take the historical stock data features and generate batch(es) for training, validating and predicting

    Methods
    -------
    check_if_enough_data(additional_size=0)

    """
    def __init__(self,price_feature_history,valid_portion,agent,config_settings,logger):
        """

        Parameters
        ----------
        price_feature_history : dict[str -> pandas.DataFrame]
            A dictionary of data batch  with each feature as key and value as the dataframe of historical data
        valid_portion : float
            The proportion specified for validation so it is kept as out-of-sample data and will not be extracted in batch(es)
        agent : Agent
            An agent object which will retrieve batch(es) of data
        config_settings : ConfigSetter
            Configuration reference to config.json
        logger : Logger
            current logger

        """
        self._asset_symbols = agent.asset_symbols
        self._num_assets = agent.num_assets
        self._price_feature_names = agent.price_feature_names
        self._feature_number = agent.feature_number
        self._basic_features_indexs = [idx for idx,i in enumerate(agent.price_feature_names) if i in agent.basic_features]
        self._logger = logger

        self._window_size = config_settings.hyperparameters['window_size']
        self._batch_size = config_settings.hyperparameters['batch_size']

        if price_feature_history is None:
            self._initialize_price_feature_daily_history()
            self._price_feature_history = self._price_feature_daily_history
        else:
            self._initialize_price_feature_daily_history()
            self._price_feature_history = price_feature_history

            experiences = self._convert_price_feature_history_to_experiences()

            if valid_portion is not None:
                assert price_feature_history is not None
                total_size = experiences.shape[1]
                valid_size = int(total_size*valid_portion)
                train_size = total_size-valid_size
                self._experiences = experiences[:,:train_size,:]
                self._valid_experiences = experiences[:, train_size:, :]
                self._valid_experiences_size = self._valid_experiences.shape[1]
                assert self._valid_experiences_size==valid_size
                self._logger.debug(f'Shape of current training buffer: {self._experiences.shape[1]}')
                self._logger.debug(f'Shape of current validating buffer: {self._valid_experiences.shape}')
            else:
                self._experiences = experiences
                self._logger.debug(f'Shape of current buffer: {self._experiences.shape[1]}')

    @property
    def price_feature_history(self):
        """

        Returns
        -------
        price_feature_history : dict[str -> pandas.DataFrame]

        """
        return self._price_feature_history

    @property
    def experiences(self):
        """

        Returns
        -------
        experiences : numpy.array
            A numpy array converted from price_feature_history, excluding validation portion

        """
        return self._experiences

    def _initialize_price_feature_daily_history(self):
        """

        Initialize a empty dataset for a new period of historical data

        """
        self._price_feature_daily_history = {}
        for f in self._price_feature_names:
            self._price_feature_daily_history[f] = pd.DataFrame(columns=self._asset_symbols)

    def check_if_enough_data(self, additional_size=0):
        """

        A utility function to check if enough historical data to form a batch with specified window size

        Parameters
        ----------
        additional_size : int
            A size in additional to window size specified
            It is used for back testing to ensure at least a window of out-of-sample data is available

        Returns
        -------
        flag : bool

        """
        flag = None
        if not all([f_hist.shape[0] >= additional_size + self._window_size for f, f_hist in
                    self.price_feature_history.items()]):
            self._logger.debug('Not enough data length of window size')
            flag = False
            return flag
        else:
            self._logger.debug('Enough data length of window size')
            flag = True
            return flag

    def observe_price(self,asset_symbol,asset_prices):
        """

        After observing a new period of price data for a stock, structure it to the dataset for the new period of historical data

        Parameters
        ----------
        asset_symbol : zipline.assets.Equity
        asset_prices : pandas.DataFrame

        """
        for price_feature in self._price_feature_names:
            self._price_feature_daily_history[price_feature][asset_symbol] = asset_prices[price_feature]

    def combine_obervation(self):
        """

        Merge the new period of historical data to the original historical data, and then convert it to experiences array

        """
        self._combine_to_history()
        self._experiences = self._convert_price_feature_history_to_experiences()
        self._logger.debug(f'experiences updated to size {self._experiences.shape[1]}')


    def _convert_price_feature_history_to_experiences(self):
        """

        Convert dataframe of historical data to experiences array

        """
        experiences = np.array(list(self._price_feature_history.values()))
        return experiences

    def _combine_to_history(self):
        """

        Merge the new period of historical data to the original historical data dataframe

        """
        for price_feature in self._price_feature_names:
            self._price_feature_history[price_feature] = pd.concat([self._price_feature_history[price_feature],self._price_feature_daily_history[price_feature]],axis=0)
        self._initialize_price_feature_daily_history()


    def _sample_a_batch(self,start_idx=0,bias=5e-5):
        """

        Sample randomly the starting index of a batch of data from experiences
        As the batch of data is consecutive for the length of window, only sampling of starting index is needed

        Parameters
        ----------
        start_idx : int
            The minimum of starting index
        bias : float
            The parameter for geometric distribution

        Returns
        -------
        random_start_idx : int

        """
        random_num = np.random.geometric(bias)
        last_start_idx = self._experiences.shape[1] - self._window_size - self._batch_size
        # sample again if random number is larger than the maximum possible starting index
        while random_num > last_start_idx - start_idx:
            random_num = np.random.geometric(bias)
        random_start_idx = last_start_idx - random_num
        print('last_start_idx: ', last_start_idx)
        print('random_start: ',random_start_idx)
        self._logger.debug(f'random_start: {random_start_idx}')
        return random_start_idx

    def _check_batch(self,batch_data):
        """

        Check if all of the data in a batch has enough data of a window size

        Parameters
        ----------
        batch_data : list[Batch]

        Raises
        ------
        AssertionError

        """
        try:
            assert all([len(n) == self._window_size for b in batch_data for n in b.batch_X]), 'some batches have data fewer than window size'
        except AssertionError as error:
            self._logger.exception('some batches have data fewer than window size')
            raise error
        except Exception as exception:
            self._logger.exception(exception, False)
            raise exception

    def _normalize_next_X(self,next_X):
        """

        If it is basic features ("close", "high", "low", "open"), it is divided by the latest closing price. Else it is divided by the latest value of the feature

        Parameters
        ----------
        next_X : numpy.array

        Returns
        -------
        next_X_norm : numpy.array

        """
        next_X_norm = np.array([next_X[i, :-1, :] / next_X[self._price_feature_names.index('close'), -2, :] \
                      if i in self._basic_features_indexs else \
                      np.divide(next_X[i, :-1, :], next_X[i, -2, :], out=np.zeros_like(next_X[i, :-1, :]),
                                where=next_X[i, -2, :] != 0) \
                  for i in range(next_X.shape[0])])
        return next_X_norm

    def _compute_rel_price_chg(self,next_X):
        """

        Compute the relative change of price for the current day

        Parameters
        ----------
        next_X : numpy.array

        Returns
        -------
        y : numpy.array

        """
        y = next_X[self._price_feature_names.index('close'), -1, :] / next_X[self._price_feature_names.index('close'), -2,:]
        return y

    def extract_next_batches(self,geometric_bias=5e-5,adversarial_flag=True,noise_normal_mean=0,noise_normal_var=0.002):
        """

        Extract a number of random batches each with window size for training
        The batches are consecutive through the time. Three features are within a batch.

        - X: Stock's normalized batch historical data, starting index to t
        - y: The relative change of price from the current day from t to t+1 in the batch
        - last_w: None at this moment and will be filled with the last predicted weight for the batch

        Parameters
        ----------
        geometric_bias : float
            The parameter for geometric distribution
        adversarial_flag : bool
            A flag to decide whether to add normal noise to the data of relative change of price, y
        noise_normal_mean : float
            The parameter mean of normal distribution for the noise
        noise_normal_var : float
            The parameter variance of normal distribution for the noise

        Returns
        -------
        next_batches : list[Batch]

        """
        random_start_idx = self._sample_a_batch(bias=geometric_bias)
        next_batches = []
        for i in range(self._batch_size):
            next_X = self._experiences[:, random_start_idx + i:random_start_idx + self._window_size + i + 1, :]
            next_X_norm = self._normalize_next_X(next_X)
            y = self._compute_rel_price_chg(next_X)
            if adversarial_flag==True:
                y = y+np.random.normal(noise_normal_mean,noise_normal_var,(y.shape[0]))
            next_batch = Batch(start_idx=random_start_idx + i,
                               batch_X=next_X_norm,
                               batch_y=y,
                               last_w=None)
            next_batches.append(next_batch)
        self._check_batch(next_batches)
        return next_batches

    def extract_valid_batch(self):
        """

        Extract a number of batches each with window size for validation from all of the out-of-sample portion of data
        The batches are consecutive through the time. Three features are within a batch.

        - X: Stock's batch historical data, starting index to t
        - y: The relative change of price from the current day t to t+1 in the batch
        - last_w: None at this moment and will be filled with the last predicted weight for the batch

        Returns
        -------
        next_batches : list[Batch]

        """
        batch_size = int(math.ceil(self._valid_experiences_size / self._window_size))
        next_batches = []
        start_idx = 0
        for i in range(batch_size):
            next_X = self._valid_experiences[:, start_idx + i:start_idx + self._window_size + i + 1, :]
            next_X_norm = self._normalize_next_X(next_X)
            y = self._compute_rel_price_chg(next_X)
            next_batch = Batch(start_idx=start_idx + i,
                           batch_X=next_X_norm,
                           batch_y=y,
                           last_w=None)
            next_batches.append(next_batch)
        self._check_batch(next_batches)
        return next_batches

    def construct_predict_batch(self):
        """

        Construct the batch of data for prediction by extracting the latest batch of data with window size
        As the current day's change of price is supposed to be unknown, it is until current day t

        - X: Stock's batch historical data, t-window size to t
        - y: The relative change of price from last day t-1 to the current day t
        - last_w: None at this moment and will be filled with the last predicted weight

        Returns
        -------
        next_batches : list[Batch]

        """
        next_X = self._experiences[:, -self._window_size:, :]
        next_X_norm = np.array([next_X[i, :, :] / next_X[self._price_feature_names.index('close'), -1, :] \
                                    if i in self._basic_features_indexs else \
                                    np.divide(next_X[i, :, :], next_X[i, -1, :], out=np.zeros_like(next_X[i, :, :]),where=next_X[i, -1, :] != 0) \
                                for i in range(next_X.shape[0])])  # X_t
        y = self._compute_rel_price_chg(next_X)
        next_batch = Batch(start_idx=None,
                          batch_X=next_X_norm,
                          batch_y=y,
                          last_w=None)
        next_batches = [next_batch]
        self._check_batch(next_batches)
        return next_batches