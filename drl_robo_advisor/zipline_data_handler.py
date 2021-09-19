import drl_robo_advisor.extension
from drl_robo_advisor.utils import convert_to_daily,create_or_load_minmaxscaler

import pandas as pd
from zipline.data.data_portal import DataPortal
from zipline.data import bundles
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.loaders import USEquityPricingLoader
from trading_calendars import get_calendar
from zipline.pipeline.filters import StaticAssets
from zipline.pipeline.engine import SimplePipelineEngine

from zipline.pipeline.data import DataSet
from zipline.pipeline.data import Column
from zipline.pipeline.domain import US_EQUITIES
import pandas_datareader as pdr
from zipline.pipeline.loaders.frame import DataFrameLoader
from pathlib import Path
import pytz

class CustomData(DataSet):
    volume = Column(dtype=float)
    # return = Column(dtype=float)
    beta_2 = Column(dtype=float)
    alpha_2 = Column(dtype=float)
    beta_5 = Column(dtype=float)
    alpha_5 = Column(dtype=float)
    MA_50 = Column(dtype=float)
    MA_200 = Column(dtype=float)
    EMA_50 = Column(dtype=float)
    EMA_200 = Column(dtype=float)
    RSI_14 = Column(dtype=float)
    ATR_14 = Column(dtype=float)
    MACD = Column(dtype=float)
    ADL = Column(dtype=float)

    DGS1MO = Column(dtype=float)
    DGS3MO = Column(dtype=float)
    DGS1 = Column(dtype=float)
    DGS3 = Column(dtype=float)
    DGS10 = Column(dtype=float)
    CPALTT01USM657N = Column(dtype=float)
    DLTIIT = Column(dtype=float)
    T5YIFR = Column(dtype=float)
    T10YIE = Column(dtype=float)
    GEPUCURRENT = Column(dtype=float)
    UNRATE = Column(dtype=float)
    MRTSSM44X72USS = Column(dtype=float)
    JTSJOL = Column(dtype=float)
    INDPRO = Column(dtype=float)

    MICH = Column(dtype=float)
    VIXCLS = Column(dtype=float)
    UMCSENT = Column(dtype=float)
    STLFSI2 = Column(dtype=float)

    T10Y2Y = Column(dtype=float)
    DCOILBRENTEU = Column(dtype=float)
    GOLDAMGBD228NLBM = Column(dtype=float)

    IC4WSA = Column(dtype=float)
    WM1NS = Column(dtype=float)
    WM2NS = Column(dtype=float)
    BOGMBASE = Column(dtype=float)

    domain = US_EQUITIES

class ZiplineDataHandler:
    """

    Load and retrieve custom data in zipline


    Methods
    -------
    get_all_trading_sessions()

    get_selected_trading_sessions(start_ts,end_ts)

    get_data_batch(start_ts,end_ts)

    read_other_stock_specific_data(asset_names, sids, stock_specific_field, data_folder_path)

    read_FREDData(sids, macro_indicator, data_folder_path)

    """
    def __init__(self,config_settings,logger):
        """

        Parameters
        ----------
        config_settings : ConfigSetter
            Configuration reference to config.json
        logger : Logger
            current logger

        """
        self._bundle_data = bundles.load(config_settings.bundle_name)
        self._feature_names = config_settings.feature_names
        self._asset_names = config_settings.stock_tickers
        self._asset_symbols = self._bundle_data.asset_finder.lookup_symbols(config_settings.stock_tickers, as_of_date=None)
        self._benchmark_asset_symbol = self._bundle_data.asset_finder.lookup_symbol(config_settings.benchmark_stock_ticker, as_of_date=None)
        self._assets_sids = pd.Int64Index([asset.sid for asset in self._asset_symbols])
        self._data_portal = DataPortal(asset_finder=self._bundle_data.asset_finder,
                                 trading_calendar=get_calendar(config_settings.stock_exchange),
                                 first_trading_day=self._bundle_data.equity_daily_bar_reader.first_trading_day,
                                 equity_daily_reader=self._bundle_data.equity_daily_bar_reader)
        self._macroecon_features = config_settings.macroecon_features
        self._stock_specific_feautres = config_settings.stock_specific_feautres
        self._data_folder_path = config_settings.data_folder_path
        self._logger = logger

        self._build_engine()
    
    @property
    def asset_symbols(self):
        """

        Returns
        -------
        asset_symbols : list[zipline.assets.Equity]

        """
        return self._asset_symbols
    
    @property
    def benchmark_asset_symbol(self):
        """

        Returns
        -------
        benchmark_asset_symbol : zipline.assets.Equity

        """
        return self._benchmark_asset_symbol

    @property
    def assets_sids(self):
        """

        Returns
        -------
        assets_sids : pandas.Index

        """
        return self._assets_sids
        
    @property
    def data_portal(self):
        """

        Returns
        -------
        data_portal : zipline.data.data_portal.DataPortal

        """
        return self._data_portal

    @property
    def engine(self):
        """

        Returns
        -------
        engine : zipline.pipeline.engine.SimplePipelineEngine

        """
        return self._engine

    @property
    def make_pipeline(self):
        """

        Returns
        -------
        make_pipeline : function

        """
        return self._make_pipeline

    @property
    def custom_data_loaders(self):
        """

        Returns
        -------
        custom_data_loaders : dict[zipline.pipeline.data.Column -> zipline.pipeline.loaders.frame.DataFrameLoader]

        """
        return self._custom_data_loaders
    
    def get_all_trading_sessions(self):
        """

        Get all trading sessions in the loaded Zipline data bundle

        Returns
        -------
        all_sessions : pandas.DatetimeIndex
            The list of datetime of sessions in trading calendar

        """
        all_sessions = self._data_portal.trading_calendar.all_sessions
        return all_sessions
    
    def get_selected_trading_sessions(self,start_ts,end_ts):
        """

        Get trading sessions within the start and end timestamp in the loaded Zipline data bundle

        Parameters
        ----------
        start_ts : pandas.Timestamp
            desired start time
        end_ts : pandas.Timestamp
            desired end time

        Returns
        -------
        selected_trading_sessions : list[pandas.Datetime]
            The list of pandas.Datetime for sessions in trading calendar

        """
        all_trading_sessions = self.get_all_trading_sessions()
        selected_trading_sessions = all_trading_sessions[all_trading_sessions.to_series().between(start_ts, end_ts)]
        return selected_trading_sessions

    def _create_pipeline(self, custom_data_loaders):
        """

        Create zipline pipeline from custom zipline data loaders

        Parameters
        ----------
        custom_data_loaders : dict[zipline.pipeline.data.Column -> zipline.pipeline.loaders.frame.DataFrameLoader]
            A dictionary to map each feature column to corresponding DataFrameLoader

        """
        pipeline_loader = USEquityPricingLoader.without_fx(
            self._bundle_data.equity_daily_bar_reader,
            self._bundle_data.adjustment_reader
        )

        def choose_loader(column):
            if column in USEquityPricing.columns:
                return pipeline_loader
            return custom_data_loaders[column]

        self._choose_loader = choose_loader

        def make_pipeline():
            pipeline_columns = {
                'close': USEquityPricing.close.latest,
                'high': USEquityPricing.high.latest,
                'low': USEquityPricing.low.latest,
                'open': USEquityPricing.open.latest}

            for field in self._macroecon_features + self._stock_specific_feautres:
                pipeline_columns[field] = eval(f'CustomData.{field}.latest')

            p = Pipeline(
                columns=pipeline_columns,
                screen=StaticAssets(self._asset_symbols)
            )
            return p

        self._make_pipeline = make_pipeline

    def _build_engine(self):
        """

        Create custom zipline data loaders and then zipline pipeline from custom zipline data loaders

        """

        self._start_date = self.get_all_trading_sessions().to_series()[0]
        self._end_date = self.get_all_trading_sessions().to_series()[-1]

        custom_data_loaders = self._get_data_loaders(self._stock_specific_feautres+self._macroecon_features)

        self._create_pipeline(custom_data_loaders)

        self._engine = SimplePipelineEngine(
            get_loader=self._choose_loader,
            asset_finder=self._bundle_data.asset_finder,
            default_domain=US_EQUITIES
        )

        self._custom_data_loaders = custom_data_loaders



    def _get_historical_data(self,start_ts,end_ts):
        """

        Retrieve the historical data within the start and end timestamp

        Parameters
        ----------
        start_ts : pandas.Timestamp
            desired start time
        end_ts : pandas.Timestamp
            desired end time

        Returns
        -------
        history_data : pandas.DataFrame
            A dataframe of historical data within the range of dates

        """
        selected_sessions = self.get_selected_trading_sessions(start_ts,end_ts)
        self._history_data = self._engine.run_pipeline(
                                    self._make_pipeline(),
                                    selected_sessions[0],
                                    selected_sessions[-1]
                                )

    def get_data_batch(self,start_ts,end_ts):
        """

        Get historical data dataframe from zipline engine and then pivot the dataframe

        Parameters
        ----------
        start_ts : pandas.Timestamp
            desired start time
        end_ts : pandas.Timestamp
            desired end time

        Returns
        -------
        data_batch : dict[str -> pandas.DataFrame]
            A dictionary of data batch  with each feature as key and value as the dataframe of historical data

        """

        self._get_historical_data(start_ts,end_ts)
        data_batch = {}
        for f in self._feature_names:
            data_batch[f] = self._history_data[f].unstack(level=1)

        return data_batch

    def _get_data_loaders(self,data_fields):
        """

        Parameters
        ----------
        data_fields : list[str]
            A list of feature name which is one of the attributes in CustomData

        Returns
        -------
        custom_data_loaders : dict[zipline.pipeline.data.Column -> zipline.pipeline.loaders.frame.DataFrameLoader]
            A dictionary to map each feature column to corresponding DataFrameLoader

        """

        custom_data_loaders = {}
        for data_field in data_fields:
            if data_field in self._stock_specific_feautres:
                fundam_data_daily = self.read_other_stock_specific_data(self._asset_names, self._assets_sids, data_field,
                                                                self._data_folder_path)
                custom_data_loaders[eval(f'CustomData.{data_field}')] = DataFrameLoader(eval(f'CustomData.{data_field}'), fundam_data_daily)
                self._logger.debug('fundamental data custom data loader complete')
            elif data_field in self._macroecon_features:
                macroecon_data_daily = self.read_FREDData(self._assets_sids, data_field,
                                                                self._data_folder_path)

                custom_data_loaders[eval(f'CustomData.{data_field}')] = DataFrameLoader(eval(f'CustomData.{data_field}'),macroecon_data_daily)
                self._logger.debug('macroecon data custom data loader complete')

        return custom_data_loaders

    def read_other_stock_specific_data(self,asset_names, sids, stock_specific_field, data_folder_path):
        """

        Read features specific to each stock including:

        - volume: trading volume of the stock, Note: volume is not handled by zipline pipeline as min-max normalization is needed
        - beta_2: Stock beta of 2-year period
        - alpha_2: Stock alpha of 2-year period
        - beta_5: Stock beta of 5-year period
        - alpha_5: Stock alpha of 5-year period
        - MA_50: Moving Average of 50-day windows
        - MA_200: Moving Average of 200-day windows
        - EMA_50: Exponential Moving Average (EMA) for 50-day windows
        - EMA_200: Exponential Moving Average (EMA) for 200-day windows
        - RSI_14: Relative Strength Indicator (RSI) for 14-day period
        - ATR_14: Average True Range (ATR) for 14-day period

        and then normalize them with min max scaler

        Parameters
        ----------
        asset_names : list[str]
            A list of stock tickers of assets
        sids : pandas.Index
            A list of pandas ID for each asset
        stock_specific_field : str
            One of the feature listed above
        data_folder_path : pathlib.Path
            The path to folder storing stock historical data

        Returns
        -------
        other_stock_specific_data_daily : pandas.DataFrame
            A dataframe of normalized data features specific to each stock

        """

        other_stock_specific_data_daily = pd.DataFrame()
        for asset_name, sid in zip(asset_names, sids):
            output_file_path = Path(data_folder_path, f'{asset_name}.csv')
            other_stock_specific_data_daily_asset_all = pd.read_csv(output_file_path, index_col=0, parse_dates=True)[stock_specific_field]
            #self._logger.debug(f'stock data loaded from {output_file_path}')
            minmaxscaler = create_or_load_minmaxscaler(f'{stock_specific_field}_{asset_name}',other_stock_specific_data_daily_asset_all,data_folder_path,self._logger)

            other_stock_specific_data_daily_asset = pd.read_csv(output_file_path, index_col=0, parse_dates=True)

            other_stock_specific_data_daily_asset.index = other_stock_specific_data_daily_asset.index.tz_localize(tz=pytz.utc)

            # Min Max Scaling
            other_stock_specific_data_daily_asset[stock_specific_field] = minmaxscaler.transform(
                other_stock_specific_data_daily_asset[stock_specific_field].values.reshape(-1, 1))

            other_stock_specific_data_daily[sid] = other_stock_specific_data_daily_asset[stock_specific_field]

        return other_stock_specific_data_daily

    def read_FREDData(self,sids, macro_indicator, data_folder_path):
        """

        Download features of macroeconomic data from pandas FredReader including:

        US Treasury

        - DGS1MO: 1-Month Treasury Constant Maturity Rate
        - DGS3MO: 3-Month Treasury Constant Maturity Rate
        - DGS1: 1-Year Treasury Constant Maturity Rate
        - DGS3: 3-Year Treasury Constant Maturity Rate
        - DGS10: 10-Year Treasury Constant Maturity Rate
        - T10Y2Y: 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity

        US Inflation

        - CPALTT01USM657N: Consumer Price Index: Total All Items for the United States
        - DLTIIT: Treasury Inflation-Indexed Long-Term Average Yield
        - T5YIFR: 5-Year Forward Inflation Expectation Rate
        - T10YIE: 10-Year Breakeven Inflation Rate
        - MICH: University of Michigan: Inflation Expectation

        Money Supply

        - WM1NS: M1 Money Stock
        - WM2NS: M2 Money Stock
        - BOGMBASE: Monetary Base

        Employment

        - UNRATE: US Unemployment Rate
        - JTSJOL: Job Openings: Total Nonfarm
        - IC4WSA: 4-Week Moving Average of Initial Claims

        Production and Consumption

        - INDPRO: Industrial Production: Total Index
        - MRTSSM44X72USS: Retail Sales: Retail Trade and Food Services
        - UMCSENT: University of Michigan: Consumer Sentiment

        Market Sentiment

        - VIXCLS: CBOE Volatility Index: VIX
        - STLFSI2: St. Louis Fed Financial Stress Index

        Commodity Price

        - DCOILBRENTEU: Crude Oil Prices: Brent - Europe
        - GOLDAMGBD228NLBM: Gold Fixing Price

        Global Growth

        - GEPUCURRENT: Global Economic Policy Uncertainty Index: Current Price Adjusted GDP (a GDP-weighted average of national EPU indices for 20 countries)

        They are converted to daily by backward and forward filling and then normalized with min max scaler

        Parameters
        ----------
        sids : pandas.Index
            A list of pandas ID for each asset
        macro_indicator : list[str]

        data_folder_path : pathlib.Path
            The path to folder storing stock historical data

        Returns
        -------
        macroecon_data_daily : pandas.DataFrame
            A dataframe of normalized macroeconomic data features

        """

        output_file_path = Path(data_folder_path,f'{macro_indicator}.csv')

        if output_file_path.exists():
            macroecon_data_all = pd.read_csv(output_file_path,index_col=0, parse_dates=True)
            self._logger.debug(f'macroecon data loaded from {output_file_path}')
        else:
            try:
                # !! zipline's equity_pricing_loader.py load_adjusted_array() will shift allquery dates back by a trading session so the start date is offset 1
                macroecon_data_all = pdr.fred.FredReader(macro_indicator, start=pd.Timestamp(self._start_date)- pd.DateOffset(1),end=pd.Timestamp(self._end_date)).read()
                macroecon_data_all.to_csv(output_file_path)
                #self._logger.debug(f'macroecon data saved to {output_file_path}')
            except Exception as e:
                self._logger.debug(f'Failed to read {macro_indicator} data from pandas datareader: ' + str(e))
                raise e

        minmaxscaler = create_or_load_minmaxscaler(macro_indicator,macroecon_data_all,data_folder_path,self._logger)

        if macroecon_data_all.index.tzinfo is None:
            macroecon_data_all.index = macroecon_data_all.index.tz_localize(tz=pytz.utc)

        macroecon_data_daily = convert_to_daily(macroecon_data_all, self._start_date, self._end_date)

        macroecon_data_daily = macroecon_data_daily.fillna(method='bfill')
        macroecon_data_daily = macroecon_data_daily.fillna(method='ffill')

        # Min Max Scaling
        macroecon_data_daily[macro_indicator] = minmaxscaler.transform(macroecon_data_daily[macro_indicator].values.reshape(-1, 1))

        for sid in sids:
            macroecon_data_daily[sid] = macroecon_data_daily.loc[:, macro_indicator]

        return macroecon_data_daily




