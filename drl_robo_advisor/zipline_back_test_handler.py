from zipline.api import symbol,symbols, order_percent,order,set_benchmark, record, attach_pipeline, pipeline_output,set_long_only,set_max_leverage,set_slippage,get_open_orders
from zipline.finance import commission, slippage
import pandas as pd
import numpy as np
import zipline
from matplotlib import pyplot as plt
from matplotlib import ticker
from drl_robo_advisor.utils import export_df_to_csv
from pathlib import Path

class ZiplineBacktestHandler:
    """

    Execute back-testing on out-of-sample data with trained deep reinforcement learning model

    Methods
    -------
    run_back_test(start_end_ts_sessions)

    plot_back_test_result(suffix=None,figsize=(20, 10 * 5))

    """
    def __init__(self,agent,custom_data_loaders,make_pipeline,config_settings,logger):
        """

        Parameters
        ----------
        agent : Agent
            An agent object which load a trained model
        custom_data_loaders : dict[zipline.pipeline.data.Column -> zipline.pipeline.loaders.frame.DataFrameLoader]
            A dictionary to map each feature column to corresponding DataFrameLoader
        make_pipeline : function
            A function to create pipeline to feed stock data
        config_settings : ConfigSetter
            Configuration reference to config.json
        logger : Logger
            current logger

        """
        self._bundle_name = config_settings.bundle_name
        self._backtest_capital_base = config_settings.backtest_capital_base
        self._runs_path = config_settings.runs_path
        self._custom_data_loaders = custom_data_loaders
        self._create_initialize_func(agent,make_pipeline,config_settings,logger)
        self._create_handle_data_test_func()

    def _create_initialize_func(self,agent,make_pipeline,config_settings,logger):
        """

        Create initialize function for zipline

        Parameters
        ----------
        agent : Agent
            An agent object which load a trained model
        make_pipeline : function
            A function to create pipeline to feed stock data
        config_settings : ConfigSetter
            Configuration reference to config.json
        logger : Logger
            current logger

        """
        def initialize(context):
            context.set_commission(commission.PerShare(cost=0.0, min_trade_cost=0))
            context.logger = logger
            context.assets = symbols(*config_settings.stock_tickers)
            context.asset_names = config_settings.stock_tickers
            context.time = 1
            context.agent = agent
            context.steps = config_settings.predict_rolltrain_steps
            context.benchmark_asset_name = config_settings.benchmark_stock_ticker
            set_benchmark(symbol(config_settings.benchmark_stock_ticker))
            set_long_only()
            set_slippage(slippage.FixedSlippage(spread=0.0))
            attach_pipeline(
                make_pipeline(),
                'custom_data_pipeline'
            )

        self._initialize = initialize

    def _compute_num_shares_to_order(self,target_weight,asset_price,position_amount,portfolio_value):
        """

        Compare the current asset value and target asset value and compute the number of shares to be ordered
        If the calculated number of shares to be sold is larger the number of shares owned, it only sells the number of shares owned to avoid shorting

        Parameters
        ----------
        target_weight : float
        asset_price : float
        position_amount : float
        portfolio_value : float

        Returns
        -------
        number_of_shares_to_order : float
        target_asset_value : float

        """

        current_asset_value = position_amount * asset_price
        target_asset_value = target_weight * portfolio_value
        diff_asset_value = target_asset_value - current_asset_value
        number_of_shares_to_order = np.floor(diff_asset_value / asset_price)

        # Prevent shorting
        if number_of_shares_to_order < 0 and abs(position_amount) < abs(number_of_shares_to_order):
            number_of_shares_to_order = -1 * abs(position_amount)

        return number_of_shares_to_order, target_asset_value

    def _create_handle_data_test_func(self):
        """

        Handle the back-test trading operation of zipline, from observing new data, predict the target weight of each stock in the portfolio to sumitting orders for trading

        """

        def handle_data_test(context, data):
            order_weights = []
            print('current time: ', context.time)
            context.logger.info(f'current time: {context.time}')
            print('current datetime: ', data.current_dt)
            context.logger.info(f'current datetime: {data.current_dt}')

            one_period_history = pipeline_output('custom_data_pipeline')
            for asset_idx, asset in enumerate(context.assets):
                prices = one_period_history.filter(items=[asset], axis='index')
                prices = prices.set_index([pd.Index([context.blotter.current_dt])])
                context.agent.read_new_data(asset, prices)
            context.agent.consol_new_data()
            ordering_weights = context.agent.predict(context.time)

            for idx, asset in enumerate(context.assets):
                if data.can_trade(asset) :
                    number_of_shares_to_order,target_asset_value = self._compute_num_shares_to_order(ordering_weights[idx],data[asset].price,context.portfolio.positions[asset].amount,context.portfolio.portfolio_value)
                    if abs(number_of_shares_to_order) >= 1.0:
                        context.logger.debug(f"Target weight for {asset.symbol}: {ordering_weights[idx]*100.0}%")
                        context.logger.debug(f"Target asset value for {asset.symbol}: {target_asset_value}")
                        context.logger.debug(f"Order number of shares for {asset.symbol}: {number_of_shares_to_order}")
                        order(asset, number_of_shares_to_order)
            context.time += 1

        self._handle_data_test = handle_data_test

    def run_back_test(self,start_end_ts_sessions):
        """

        Start running back-test within a range of dates

        Parameters
        ----------
        start_end_ts_sessions : list[pandas.Datetime]
            The list of pandas.Datetime for selected sessions in trading calendar

        Notes
        -----
        Shorting is avoided by the private method _compute_num_shares_to_order()

        """
        self.run_test_result = zipline.run_algorithm(start=start_end_ts_sessions[0],
                                         end=start_end_ts_sessions[-1],
                                         initialize=self._initialize,
                                         capital_base=self._backtest_capital_base,
                                         handle_data=self._handle_data_test,
                                         bundle=self._bundle_name,
                                         data_frame_loaders=self._custom_data_loaders)


    def plot_back_test_result(self,suffix=None,figsize=(20, 10 * 5)):
        """

        Plot the result after back-testing

        Plot and Compare against benchmark return

        - Return
        - Portfolio Value
        - Volatility

        And also plot the gross leverage and short exposure

        Parameters
        ----------
        suffix : str
        figsize : tuple(int,int)


        """
        export_df_to_csv(self.run_test_result,Path(self._runs_path, f'run_test_{suffix}.csv'))

        fig, axes = plt.subplots(5, 1, figsize=figsize)
        # https://analyzingalpha.com/a-simple-trading-strategy-in-zipline-and-jupyter
        # algorithm_period_return and benchmark_period_return are cumulative returns
        axes[0].plot(self.run_test_result.index, self.run_test_result.algorithm_period_return, color='blue')
        axes[0].plot(self.run_test_result.index, self.run_test_result.benchmark_period_return, color='red')
        axes[0].legend(['Model', 'Benchmark'])
        axes[0].set_ylabel("Returns", color='black', size=20)
        axes[1].plot(self.run_test_result.index, self.run_test_result.portfolio_value)
        axes[1].set_ylabel("Portfolio Value", color='black', size=20)
        axes[1].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        axes[2].plot(self.run_test_result.index, self.run_test_result.algo_volatility, color='blue')
        axes[2].plot(self.run_test_result.index, self.run_test_result.benchmark_volatility, color='red')
        axes[2].legend(['Model', 'Benchmark'])
        axes[2].set_ylabel("Volatility", color='black', size=20)
        axes[3].plot(self.run_test_result.index, self.run_test_result.gross_leverage)
        axes[3].set_ylabel("Gross Leverage", color='black', size=20)
        axes[4].plot(self.run_test_result.index, self.run_test_result.short_exposure)
        axes[4].set_ylabel("Short Exposure", color='black', size=20)

        plt.savefig(Path(self._runs_path, f'zipline_result_plot_{suffix}.png'))

        self.run_test_result.to_pickle(str(Path(self._runs_path, f'run_test_{suffix}.pkl')))

