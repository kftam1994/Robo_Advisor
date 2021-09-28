import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time
from drl_robo_advisor.utils import plot_dist
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from pathlib import Path

torch.autograd.set_detect_anomaly(True)

class PortfolioTrainer:
    """

    This class is used for:

    - Train a network and then validation, especially for hyperparameters tuning
    - Prediction or back-testing

    Methods
    -------
    train(write_to_summary,plot_dist_indexs,save_model,do_validation,desc_tqdm="Training Steps")

    predict(experience_batch, portfolio_vector_memory,cur_time)

    """
    def __init__(self,agent,config_settings,logger,valid_portion=None):
        """

        Parameters
        ----------
        agent : Agent
            An agent object which will conduct training or prediction
        config_settings : ConfigSetter
            Configuration reference to config.json
        logger : Logger
            current logger
        valid_portion : float

        Notes
        -----
        Total capital for training is initialized as 1

        """
        self._window_size = config_settings.hyperparameters['window_size']  # length of a batch
        self._price_feature_names = config_settings.feature_names
        self._feature_number = len(self._price_feature_names)
        self._agent = agent
        self._experience_batch = self._agent.experience_batch
        self._network = self._agent.network
        self._portfolio_vector_memory = self._agent.portfolio_vector_memory

        self._batch_size = config_settings.hyperparameters['batch_size']
        self._learning_rate = config_settings.hyperparameters['learning_rate']
        self._weight_decay = config_settings.hyperparameters['weight_decay']
        self._conv_layer_outputs = config_settings.hyperparameters['conv_layer_outputs']
        self._eiie_dense_out_channels = config_settings.hyperparameters['eiie_dense_out_channels']
        self._reg_norm = config_settings.hyperparameters['reg_norm']
        self._risk_penalty = config_settings.hyperparameters['risk_penalty']
        self._geometric_bias = config_settings.hyperparameters['geometric_bias']
        self._adversarial_flag = config_settings.hyperparameters['adversarial_flag']
        self._noise_normal_mean = config_settings.hyperparameters['noise_normal_mean']
        self._noise_normal_var = config_settings.hyperparameters['noise_normal_var']

        self._commission_rate = config_settings.commission_rate
        self._device = config_settings.device
        self._logger = logger

        self._total_capital = 1
        if valid_portion is not None:
            self._valid_portion = valid_portion
        else:
            self._valid_portion = config_settings.valid_portion
        self._training_steps = config_settings.training_steps
        self._predict_rolltrain_steps = config_settings.predict_rolltrain_steps

        self._logger.info(f'NN network to be trained: {self._network}')

        self._create_optimizer()

        self._y_t_1 = None
        self.predict_save_indexs = np.array([])
        self.IG_output_all_frames = []

    @property
    def agent(self):
        """

        Returns
        -------
        agent : Agent
            An agent object which will conduct training or prediction

        """
        return self._agent

    @agent.setter
    def agent(self,agent):
        self._agent = agent

    @property
    def portfolio_vector_memory(self):
        """

        Returns
        -------
        portfolio_vector_memory : numpy.array
            An array with size the same as experiences to record all past predicted portfolio weights

        """
        return self._portfolio_vector_memory

    @portfolio_vector_memory.setter
    def portfolio_vector_memory(self,portfolio_vector_memory):
        self._portfolio_vector_memory = portfolio_vector_memory

    def _create_optimizer(self):
        """

        Create optimizer for Pytorch model training
        It is Adaptive Moment Estimation (Adam) with L2 penalty given by weight decay

        """
        self._optimizer = optim.Adam(self._network.parameters(), lr=self._learning_rate,weight_decay=self._weight_decay)

    def set_train_writer(self,comment_name="-rl-portfolio-train"):
        """

        Set up the Tensorboard writer for training

        Parameters
        ----------
        comment_name : str

        """
        self._train_writer = SummaryWriter(comment=comment_name)

    def set_predict_writer(self,comment_name="-rl-portfolio-predict"):
        """

        Set up the Tensorboard writer for training

        Parameters
        ----------
        comment_name : str, default="-rl-portfolio-predict"

        """
        self._predict_writer = SummaryWriter(comment=comment_name)

    def _calculate_trans_remainder_train(self):
        """

        Calculate the transaction remainder factor during training
        After the change of price during the day, transactions are conducted to revert the portfolio weights back to target weights but it results in transaction cost (i.e. commission)
        Therefore, this factor determine the final portfolio value after transactions

        Returns
        -------
        mu : float

        References
        ----------
        https://github.com/ZhengyaoJiang/PGPortfolio/blob/master/pgportfolio/learn/nnagent.py

        """
        w_t = self._future_trade_day_w[:self._input_bath_size-1]
        w_t1 = self._network_output[1:self._input_bath_size]
        mu = 1-torch.sum(torch.abs(w_t1[:,1:] - w_t[:,1:]), axis=1)*self._commission_rate
        return mu

    def _calculate_regularization(self,norm,reg_lambda):
        """

        Calculate L1/L2 norm penalty to Pytorch loss

        Parameters
        ----------
        norm : int
            L1 or L2 norm
        reg_lambda : float
            parameter for regularization

        Returns
        -------
        penalty : float

        """
        reg_loss = None
        for W in self._network.parameters():
            if reg_loss is None:
                reg_loss = W.norm(p=norm)
            else:
                reg_loss = reg_loss + W.norm(p=norm)
        penalty = reg_loss * reg_lambda
        return penalty

    def _calculate_loss(self, indexs=None):
        """

        Calculate the loss for the iteration of training, which is the average logarithmic cumulated return of the portfolio and is also the maximization objective

        - predict t th day weight by t-1 th day's closing price and note that t th day's close price are not known to the model
        - portfolio value change calculated by the price movement from t-1 th closing price to t th closing price

        Parameters
        ----------
        indexs : numpy.array, optional
            Indexes of the current batch in the past weight history/portfolio vector memory

        """
        self._network_output = self._network(self._x, self._last_w)
        if indexs is not None:
            self._set_cur_w(self._network_output[-1][1:].cpu().detach().numpy(),indexs)

        self._future_price_movement = torch.cat([torch.ones([self._input_bath_size, 1]).to(self._device), self._y], 1)
        self._future_trade_day_w = (self._future_price_movement * self._network_output) / \
                             torch.sum(self._future_price_movement * self._network_output, axis=1)[:, None]
        self._portfolio_value_vector = torch.sum(self._future_price_movement * self._network_output, axis=1) * (
                                                torch.cat([torch.ones(1).to(self._device), self._calculate_trans_remainder_train()], axis=0))
        self._log_mean_free = torch.mean(torch.log(torch.sum(self._future_price_movement * self._network_output, axis=1)))
        self._portfolio_value = torch.prod(self._portfolio_value_vector)

        self._mean = torch.mean(self._portfolio_value_vector)
        self._log_mean = torch.mean(torch.log(self._portfolio_value_vector))
        self._standard_deviation = torch.sqrt(torch.mean((self._portfolio_value_vector - self._mean) ** 2))
        self._sharp_ratio = (self._mean - 1) / self._standard_deviation

        if self._reg_norm is not None:
            self._loss = -torch.mean(torch.log(self._portfolio_value_vector)) + (-self._calculate_regularization(norm=self._reg_norm,reg_lambda=1e-5))
        else:
            self._loss = -torch.mean(torch.log(self._portfolio_value_vector))

        if self._risk_penalty is not None:
            self._loss += self._standard_deviation*self._risk_penalty


    def _calculate_trans_remainder_rolltrain(self,w1, w0, commission_rate):
        """

        Calculate the transaction remainder factor during rolltrain and solve it through iterative update of mu1

        Parameters
        ----------
        w1 : numpy.array
            target portfolio weight vector
        w0 : numpy.array
            rebalanced last period portfolio weight vector
        commission_rate : float
            rate of commission fee, proportional to the transaction cost

        Returns
        -------
        mu1 : float

        References
        ----------
        https://github.com/ZhengyaoJiang/PGPortfolio/blob/master/pgportfolio/tools/trade.py

        """
        mu0 = 1
        mu1 = 1 - 2 * commission_rate + commission_rate ** 2
        while abs(mu1 - mu0) > 1e-10:
            mu0 = mu1
            mu1 = (1 - commission_rate * w0[0] -
                   (2 * commission_rate - commission_rate ** 2) *
                   np.sum(np.maximum(w0[1:] - mu1 * w1[1:], 0))) / \
                  (1 - commission_rate * w1[0])
        return mu1



    def _check_input_data(self,input_data):
        """

        Check if input batch data contain any NaN/null value

        Parameters
        ----------
        input_data : list[Batch]

        Raises
        ------
        AssertionError

        """
        try:
            for each_input in input_data:
                assert np.isnan(each_input.batch_X).any()==False, 'input_data batch_X contains NaN'
                assert np.isnan(each_input.batch_y).any() == False, 'input_data batch_X contains NaN'
        except AssertionError as error:
            # Output expected AssertionErrors.
            self._logger.exception('input_data contains NaN')
            raise error
        except Exception as exception:
            # Output unexpected Exceptions.
            self._logger.exception(exception, False)
            raise exception

    def _create_input_set(self, batches):
        """

        Convert the batches of data into target input format to network model

        Parameters
        ----------
        batches : list[Batch]

        Returns
        -------
        X : numpy.array
        y : numpy.array
        last_w : numpy.array

        """
        X = []
        y = []
        last_w = []
        for batch in batches:
            x = batch.batch_X
            x_reshape = np.reshape(x,[x.shape[0], x.shape[2], x.shape[1]])
            X.append(x_reshape)
            y.append(batch.batch_y)
            last_w.append(batch.last_w)
        X = np.array(X)
        y = np.array(y)
        last_w = np.array(last_w)
        return X, y, last_w

    def _retry_save_model(self):
        """

        Save Pytorch model trained parameters

        """
        error = None
        for _ in range(50):
            try:
                return torch.save(self._network.state_dict(), self._agent.model_path)
            except Exception as exception:
                error = exception
                time.sleep(5)
                pass
        raise error

    def _write_to_summary(self,writer,step):
        """

        Write the following training step result to Tensorboard

        - portfolio value
        - average logarithmic cumulated return
        - training loss
        - average logarithmic cumulated return free of influence of transaction cost
        - sharp ratio
        - standard deviation
        - histogram of gradients for each parameter

        Parameters
        ----------
        writer : torch.utils.tensorboard.SummaryWriter
        step : int

        """
        writer.add_scalar(f'{self._agent.mode}/portfolio_value', self._portfolio_value.item(), step)
        writer.add_scalar(f'{self._agent.mode}/log_mean', self._log_mean.item(), step)
        writer.add_scalar(f'{self._agent.mode}/loss', self._loss.item(), step)
        writer.add_scalar(f"{self._agent.mode}/log_mean_free", self._log_mean_free.item(), step)
        writer.add_scalar(f"{self._agent.mode}/sharp_ratio", self._sharp_ratio.item(), step)
        writer.add_scalar(f"{self._agent.mode}/standard_deviation", self._standard_deviation.item(),step)

        for name, weight in self._network.named_parameters():
            writer.add_histogram(name, weight, step)
            writer.add_histogram(f'{name}.grad', weight.grad, step)

    def _set_cur_w(self,cur_w,indexs):
        self._portfolio_vector_memory[indexs, :] = cur_w

    def train(self,write_to_summary,plot_dist_indexs,save_model,do_validation,desc_tqdm="Training Steps"):
        """

        The major training procedure to train the model for the number of training steps

        Parameters
        ----------
        write_to_summary : bool
        plot_dist_indexs : bool
        save_model : bool
        do_validation : bool
        desc_tqdm : str, default="Training Steps"

        Returns
        -------
        save_indexs : numpy.array
        mean_valid_losses : numpy.array

        """
        save_indexs = np.array([])
        #load model for each train during rolltrain
        if self._agent.mode=='predict':
            self._agent.load_network_if_exist()
            self._network = self._agent.network

        X = None
        last_w = None

        valid_losses = []
        for i in tqdm(range(1,self._training_steps+1), desc=desc_tqdm):
            self._logger.info(f'training step {i}')
            batches = self._experience_batch.extract_next_batches(geometric_bias=self._geometric_bias,adversarial_flag=self._adversarial_flag,noise_normal_mean=self._noise_normal_mean,noise_normal_var=self._noise_normal_var)

            self._check_input_data(batches)

            indexs = np.array([b.start_idx for b in batches])
            print(f'current train batches indexs: {indexs}')
            self._logger.debug(f'current train batches indexs: {indexs}')
            save_indexs = np.concatenate((save_indexs, indexs)) if save_indexs.size else indexs


            for batch_idx,batch in enumerate(batches):
                last_w = self._portfolio_vector_memory[indexs[batch_idx] - 1, :]
                batches[batch_idx] = batch._replace(last_w = last_w)

            X, y_t_1, last_w = self._create_input_set(batches)
            self._train_network(x=X, y=y_t_1, last_w=last_w, indexs=indexs)

            if do_validation:
                valid_loss = self._validate(self._experience_batch)
                valid_losses.append(valid_loss)

            if write_to_summary:
                self._write_to_summary(self._train_writer,i)
                self._logger.debug(f'tensorboard summary written')

            if save_model:
                self._retry_save_model()
                self._logger.debug(f'model save to {str(self._agent.model_path)}')
                if i == 8000 or i==1:
                    self._logger.debug('at start and end')
                    for name, weight in self._network.named_parameters():
                        self._logger.debug(name)
                        self._logger.debug(weight)

        if plot_dist_indexs:
            plot_dist(self._agent.model_root_path,save_indexs,prefix=self._agent.mode)

        if write_to_summary:
            self._train_writer.flush()

        mean_valid_losses = np.mean(valid_losses)

        return save_indexs,mean_valid_losses

    def _validate(self,experience_batch):
        """

        Validate the trained model with out-of-sample validation batches from experience

        Parameters
        ----------
        experience_batch : ExperienceBatch

        Returns
        -------
        valid_loss : float

        """

        batches = experience_batch.extract_valid_batch()

        #Check if validate set has more than one batch
        try:
            assert len(batches)>=4, f'validate batches size is {len(batches)} and smaller than 4'
        except AssertionError as error:
            self._logger.exception(f'validate batches size is {len(batches)} and smaller than 4')
            raise error
        except Exception as exception:
            self._logger.exception(exception, False)
            raise exception

        self._check_input_data(batches)

        indexs = np.array([b.start_idx for b in batches])
        shapes_batch_X = np.average(np.array([b.batch_X.shape for b in batches]), axis=0)
        self._logger.debug(f'current valid batches indexs of window size {self._window_size}: {indexs}')
        self._logger.debug(f'each batch X of valid batches with shape {shapes_batch_X}')

        for batch_idx, batch in enumerate(batches):
            last_w = self._portfolio_vector_memory[indexs[batch_idx] - 1, :]
            batches[batch_idx] = batch._replace(last_w=last_w)

        X, y_t_1, last_w = self._create_input_set(batches)

        self._test_network(x=X, y=y_t_1, last_w=last_w, indexs=indexs)

        valid_loss = self._loss.item()

        return valid_loss


    def _train_network(self, x, y, last_w, indexs):
        self._network.train()
        self._input_bath_size = x.shape[0]
        self._y = torch.tensor(y, requires_grad=False, dtype=torch.float32).to(self._device)
        self._x = torch.tensor(x, dtype=torch.float32).to(self._device)
        self._last_w = torch.tensor(last_w, requires_grad=False, dtype=torch.float32).to(self._device)
        self._optimizer.zero_grad()
        self._calculate_loss(indexs)

        self._loss.backward()
        self._optimizer.step()

    def _test_network(self, x, y, last_w, indexs):
        self._network.eval()
        self._input_bath_size = x.shape[0]
        self._y = torch.tensor(y, requires_grad=False, dtype=torch.float32).to(self._device)
        self._x = torch.tensor(x, dtype=torch.float32).to(self._device)
        self._last_w = torch.tensor(last_w, requires_grad=False, dtype=torch.float32).to(self._device)

        self._calculate_loss(indexs)

    def _write_predict_performance_to_summary(self,w_assets,cur_time):
        """

        Write the following predict step result to Tensorboard and compare with equiweight, all cash and benchmark portfolio

        - predicted portfolio weights
        - change of portfolio value
        - total capital value of portfolio

        Parameters
        ----------
        w_assets : numpy.array
            The predicted portfolio weights
        cur_time : int
            Current time step the model is predicting

        """
        self._predict_writer.add_scalars(f'predict_compare/asset_weight', w_assets, cur_time)

        self._predict_writer.add_scalar("predict_compare/model_portfolio_change", self._portfolio_change, cur_time)
        self._predict_writer.add_scalar("predict_compare/model_total_assets_value", self._total_capital, cur_time)

        self._predict_writer.add_scalars("predict_compare/compare_portfolio_change",
                                         {'model_w': self._portfolio_change,
                                          'equiweight_w': self._portfolio_change_equiweight_w,
                                          'allcash_w': self._portfolio_change_allcash_w,
                                          'benchmark_w': self._portfolio_change_benchmark_w}, cur_time)

        self._predict_writer.add_scalars("predict_compare/compare_total_assets_value",
                                         {'model_w': self._total_capital,
                                          'equiweight_w': self._agent.total_capital_equiweight_w,
                                          'allcash_w': self._agent.total_capital_allcash_w,
                                          'benchmark_w': self._agent.total_capital_benchmark_w}, cur_time)

    def _evaluate_performance(self, target_w, y_t_1, last_trade_w,total_capital):
        """

        Evaluate the performance of portfolio during back-testing and calculate

        - last traded weights
        - change of portfolio value
        - current total capital value of the portfolio

        As the predicted weight will be orders executed the next day, this function update the executed weight the next day as last traded weight

        Parameters
        ----------
        target_w
        y_t_1
        last_trade_w
        total_capital

        Returns
        -------
        last_traded_w
        portfolio_value_change
        total_capital

        References
        ----------
        https://github.com/ZhengyaoJiang/PGPortfolio/blob/master/pgportfolio/trade/backtest.py

        """
        future_price = np.concatenate((np.ones(1), y_t_1))
        portfolio_value_after_commission = self._calculate_trans_remainder_rolltrain(target_w, last_trade_w, self._commission_rate)
        portfolio_value_change = portfolio_value_after_commission * np.dot(target_w, future_price)
        total_capital *= portfolio_value_change
        last_traded_w = portfolio_value_after_commission * target_w * future_price / portfolio_value_change

        return last_traded_w,portfolio_value_change,total_capital

    def predict(self, experience_batch, portfolio_vector_memory,cur_time):
        """

        Conduct prediction with trained model based on the latest window of data and then a re-training with the whole historical data
        It generates a comparison with equiweight, all cash and benchmark portfolio and writes the comparison to Tensorboard

                y               w         last_w
        t-1  v t-1/v t-2    initilize 0     0
        t    v t  /v t-1    predicted       0
        t+1  v t+1/v t      predicted   last predicted

        Parameters
        ----------
        experience_batch : ExperienceBatch
            The updated experience with the latest window of data
        portfolio_vector_memory :  : numpy.array
            An array with size the same as experiences above to record all past predicted portfolio weights
        cur_time : int
            The current time step of prediction

        Returns
        -------
        predicted_assets_weight : numpy.array
            The predicted portfolio weights from the model

        """
        # To include new observation
        self._experience_batch = experience_batch
        self._portfolio_vector_memory = portfolio_vector_memory
        batches = self._experience_batch.construct_predict_batch()

        self._check_input_data(batches)

        batches[0] = batches[0]._replace(last_w=self._agent.last_trade_w[1:])
        X, y_t, last_w = self._create_input_set(batches)
        IG_output = self._predict_by_network(x=X, last_w=last_w,cur_time=cur_time)
        self.IG_output_all_frames.append(IG_output)
        w = self._network_output.cpu().detach().numpy()

        self._logger.info(f'predict asset weight of time {cur_time}: {w}')

        predicted_assets_weight = w[0][1:]

        w_assets = {self._agent.asset_symbols[idx].symbol: w[0][idx] for idx in range(1, len(self._agent.asset_symbols))}
        w_assets['CASH'] = w[0][0]


        if self._y_t_1 is None:
            self._y_t_1 = np.squeeze(y_t)
        else:
            self._agent.last_trade_w, self._portfolio_change, self._total_capital = self._evaluate_performance(
                np.squeeze(w), self._y_t_1, self._agent.last_trade_w, self._total_capital)
            self._y_t_1 = np.squeeze(y_t)

            self._logger.info(f'predict portfolio change of time {cur_time}: {self._portfolio_change}')
            self._logger.info(f'predict total assets value until time {cur_time}: {self._total_capital}')

            _, self._portfolio_change_equiweight_w, self._agent.total_capital_equiweight_w = self._evaluate_performance(
                np.squeeze(self._agent.equiweight_w), self._y_t_1, self._agent.equiweight_w,
                self._agent.total_capital_equiweight_w)

            _, self._portfolio_change_allcash_w, self._agent.total_capital_allcash_w = self._evaluate_performance(
                np.squeeze(self._agent.allcash_w), self._y_t_1, self._agent.allcash_w,
                self._agent.total_capital_allcash_w)

            _, self._portfolio_change_benchmark_w, self._agent.total_capital_benchmark_w = self._evaluate_performance(
                np.squeeze(self._agent.benchmark_w), self._y_t_1, self._agent.benchmark_w,
                self._agent.total_capital_benchmark_w)

            self._write_predict_performance_to_summary(w_assets,cur_time)

        self._training_steps = self._predict_rolltrain_steps

        self._logger.debug(f'start roll train at time step {cur_time}')
        save_indexs,_ = self.train(write_to_summary=False,plot_dist_indexs=False,save_model=True,do_validation=False,desc_tqdm="Roll Training Steps")
        self.predict_save_indexs = np.concatenate((self.predict_save_indexs, save_indexs)) if self.predict_save_indexs.size else save_indexs
        self._predict_writer.flush()
        return predicted_assets_weight

    def _predict_by_network(self,x,last_w,cur_time):
        self._network.eval()
        self._input_bath_size = x.shape[0]
        self._x = torch.tensor(x, dtype=torch.float32).to(self._device)
        self._last_w = torch.tensor(last_w, requires_grad=False, dtype=torch.float32).to(self._device)
        self._network_output = self._network(self._x, self._last_w)

        ig = IntegratedGradients(self._network)
        attribution = ig.attribute(inputs=(self._x, self._last_w),
                                   target=list(range(self._network_output.shape[0])))

        IG_output = attribution[0].detach().numpy()[0]

        return IG_output
