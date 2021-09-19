from datetime import datetime
import pandas_market_calendars as mcal
import pandas as pd
import os
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from yahoofinancials import YahooFinancials
import subprocess
import talib
import numpy as np
from scipy import stats
import sqlite3

class ZiplineDataPreparation:
    """

    Downlaod and load historical stock price data to Zipline database


    Methods
    -------
    load_data_to_zipline()

    """
    def __init__(self,config_settings,logger):
        """

        Parameters
        ----------
        config_settings : configSetter
            Configuration reference to config.json
        logger : Logger
            current logger

        """
        self._stock_tickers = config_settings.stock_tickers
        self._start_date = config_settings.start_date
        self._end_date = config_settings.end_date
        self._start_date_str = datetime.strftime(self._start_date, '%Y-%m-%d')
        self._end_date_str = datetime.strftime(self._end_date, '%Y-%m-%d')

        mkt_calendar = mcal.get_calendar(config_settings.stock_exchange)
        self._stock_exchange = config_settings.stock_exchange
        nyse_dates = mkt_calendar.schedule(start_date=self._start_date, end_date=self._end_date)
        self._all_dates_df = pd.DataFrame(index=nyse_dates.index)

        self._base_path = config_settings.base_path

        self._data_root_path = config_settings.data_folder_path
        self._bundle_name = config_settings.bundle_name
        self._frequency = config_settings.frequency

        self._logger = logger

    def _retrieve_data(self,ticker, freq):
        """

        Retrieve historical stock price data from Yahoo Finance and calculate the return from adjusted close price

        The price data includes:

        - open
        - high
        - low
        - adjclose
        - volume
        - dividend
        - split

        Parameters
        ----------
        ticker : str
            price data of the stock ticker to be downloaded
        freq : str
            frequency or interval between records of price data to be downloaded

        Returns
        -------
        df_all : pandas.DataFrame
            A dataframe of historical price data

        """
        yahoo_financials = YahooFinancials(ticker)

        df = yahoo_financials.get_historical_price_data(self._start_date_str, self._end_date_str, freq)
        df_prices = pd.DataFrame(df[ticker]['prices']).drop(['date'], axis=1) \
                        .rename(columns={'formatted_date': 'date'}) \
                        .loc[:, ['date', 'open', 'high', 'low', 'adjclose', 'volume']] \
            .set_index('date') \
            .rename(columns={'adjclose': 'close'})

        if 'dividends' in df[ticker]['eventsData']:
            df_dividends = pd.DataFrame(df[ticker]['eventsData']['dividends']).T.drop(['date'], axis=1) \
                .rename(columns={'formatted_date': 'date', 'amount': 'dividends'}).set_index('date')
            df_all = df_prices.join(df_dividends).fillna(0)
        else:
            df_all = df_prices
            df_all['dividend'] = 0
            df_all['split'] = 1

        df_all.index = pd.to_datetime(df_all.index)

        df_all['return'] = df_all['close'].pct_change()[1:]

        return df_all

    def _calculate_beta_alpha(self,df, num_trading_days_per_year, len_period):
        """

        Calculate the beta and alpha of a stock by linear regression of return against benchmark return

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe with stock closing price and return, and benchmark return
        num_trading_days_per_year : int
            The number of trading days in a year, which should be 252 days
        len_period : int
            The length of observing period to calculate beta and alpha

        Returns
        -------
        df : pandas.DataFrame
            A dataframe with beta and alpha columns added

        """
        num_trading_days_per_period = len_period * num_trading_days_per_year

        if df.shape[0] < num_trading_days_per_period:
            num_trading_days_per_period = df.shape[0]

        betas = [np.nan] * (num_trading_days_per_period - 1)
        alphas = [np.nan] * (num_trading_days_per_period - 1)
        for i in range(num_trading_days_per_period, df.shape[0] + 1):
            (beta, alpha) = stats.linregress(df['return'].values[i - num_trading_days_per_period:i],
                                             df['benchmark_return'].values[i - num_trading_days_per_period:i])[0:2]
            betas.append(beta)
            alphas.append(alpha)
        try:
            df[f'beta_{len_period}'] = np.array(betas)
            df[f'alpha_{len_period}'] = np.array(alphas)
        except:
            print(betas)
            raise
        return df

    def _calculate_technical_indicators(self,df):
        """

        Calculate the technical indicators of a stock including:

        - MA_50: Moving Average of 50-day windows
        - MA_200: Moving Average of 200-day windows
        - EMA_50: Exponential Moving Average (EMA) for 50-day windows
        - EMA_200: Exponential Moving Average (EMA) for 200-day windows
        - RSI_14: Relative Strength Indicator (RSI) for 14-day period
        - MACD: Moving Average Convergence/Divergence
        - ATR_14: Average True Range (ATR) for 14-day period
        - ADL: Accumulation/Distribution Line

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe with stock high, low and closing price, and trading volume

        Returns
        -------
        df : pandas.DataFrame
            A dataframe with technical indicator columns added

        """
        df['MA_50'] = talib.MA(df['close'], timeperiod=50, matype=0)
        df['MA_200'] = talib.MA(df['close'], timeperiod=200, matype=0)
        df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
        df['EMA_200'] = talib.EMA(df['close'], timeperiod=200)
        df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
        macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['ATR_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['ADL'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        return df

    def _plot_data(self,img_path,ticker,df_all_dates):
        """

        Plot the time series of stock closing price

        Parameters
        ----------
        img_path : pathlib.Path
            Output image's path
        ticker : str
            Stock ticker to be plotted
        df_all_dates : pandas.DataFrame
            A dataframe of stock closing price data with all dates instead of only trading days

        """
        plt.figure(figsize=(30, 10))
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.plot(df_all_dates.index, df_all_dates['close'])
        ax.set_title('{} prices --- {}:{}'.format(ticker, self._start_date_str, self._end_date_str))
        plt.xticks(rotation=90)
        plt.savefig(img_path)

    def _download_data_to_csv(self, ticker, output_path, benchmark_ticker='SPY',num_trading_days_per_year=252):
        """

        Retrieve stock data and save to CSV for zipline to load

        Parameters
        ----------
        ticker : str
            Data of the stock ticker to be retrieved
        output_path : pathlib.Path
            Output path of CSV
        benchmark_ticker : str
            Indicate the stock for benchmarking in calculation in beta and alpha
        num_trading_days_per_year : int, default=252
            Number of trading days in a year, by default 252 days

        """
        df_all = self._retrieve_data(ticker, self._frequency)
        benchmark_all = self._retrieve_data(benchmark_ticker, self._frequency)

        benchmarking_cal_df = pd.merge(df_all.iloc[1:, -1].to_frame(),
                                       benchmark_all.iloc[1:, -1].to_frame().rename(
                                           columns={'return': 'benchmark_return'}),
                                       how='inner',
                                       left_index=True, right_index=True)

        for j in [2, 5]:
            benchmarking_cal_df = self._calculate_beta_alpha(benchmarking_cal_df, num_trading_days_per_year, j)

        df_all = self._calculate_technical_indicators(df_all)

        df_all = pd.merge(df_all, benchmarking_cal_df[['beta_2', 'alpha_2', 'beta_5', 'alpha_5']], how='left',
                          left_index=True, right_index=True)

        df_all_dates = pd.merge(df_all, self._all_dates_df, how='right', left_index=True, right_index=True)

        print(f'first non-NaN row index: {df_all_dates.first_valid_index()}')
        print(f'first non-NaN row:')
        print(df_all_dates.loc[df_all_dates.first_valid_index(), :])

        df_all_dates = df_all_dates.fillna(method='bfill')
        df_all_dates = df_all_dates.fillna(method='ffill')

        # save data to csv for later ingestion
        df_all_dates.to_csv(output_path, header=True, index=True)

        self._plot_data(Path(str(output_path).replace('.csv', '.png')),ticker,df_all_dates)

    def _modify_extension_file(self):
        """

        Modify extension file to fill in details of configuration before ingestion of zipline

        """
        script = f"""
import pandas as pd

from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities
start_session = pd.Timestamp('{self._start_date_str}', tz='utc')
end_session = pd.Timestamp('{self._end_date_str}', tz='utc')

# register the bundle
register(
    '{self._bundle_name}',  # name we select for the bundle
    csvdir_equities(
        # name of the directory as specified above (named after data frequency)
        ['{self._frequency}'],
        # path to directory containing the
        r'{str(self._data_root_path.parent)}',
    ),
    calendar_name='{self._stock_exchange}',  # New York Stock Exchange https://github.com/quantopian/trading_calendars
    start_session=start_session,
    end_session=end_session
                )"""
        ext_path = Path(self._base_path,'extension.py')
        with open(ext_path,'w') as ext_file:
            ext_file.write(script)
        self._logger.info(f"{ext_path} updated")

    def _update_zipline_sqlite_db(self):
        """

        To fix the error of ?? country_code

        Notes
        -----
        The error is caused because the country_code in the Database is somehow messed up.
        This can be resolved by changing the SQLite Database (normally under ~/.zipline/data/quandl/TIMESTAMP/assets-n.sqlite)

        - Modify the 'exchanges' Table and change country_code to 'US', which was set to '???' originally

        References
        ----------
        https://github.com/quantopian/zipline/issues/2517

        """
        db_path = Path(list(Path(self._data_root_path.parent,f'{self._bundle_name}').glob('**/*'))[0],'assets-7.sqlite')

        try:
            sqliteConnection = sqlite3.connect(str(db_path))
            cursor = sqliteConnection.cursor()
            self._logger.info(f"Connected to SQLite DB in {db_path}")

            sql_update_query = """Update exchanges set country_code = 'US'"""
            cursor.execute(sql_update_query)
            sqliteConnection.commit()
            self._logger.info("country_code fixed")
            cursor.close()
        except sqlite3.Error as error:
            cursor.close()
            self._logger.error("Failed to update exchanges table", error)
            raise error

    def load_data_to_zipline(self):
        """

        Download data, process calculation and then ingest data to zipline database (in DATA_FOLDER_NAME of config.json)

        1. Check what is current bundles
        2. Download data and calculation of beta, alpha and technical indicators for each ticker
        3. Load the data to Zipline database

        Notes
        -----
        The environment variable about zipline's data directory (ZIPLINE_ROOT) was set in config_setter.py. So config_setter.py should be run before this function.

        """
        # extension.py will be created under the environment variable about zipline's data directory (ZIPLINE_ROOT)
        run_output = subprocess.run(args=['zipline', 'bundles'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._logger.info(f'bundles before ingestion: {run_output.stdout}')

        self._modify_extension_file()

        for ticker in self._stock_tickers:
            output_path = Path(self._data_root_path, f'{ticker}.csv')
            self._logger.info(f"Start to prepare {output_path}")

            self._download_data_to_csv(ticker, output_path)

        run_output = subprocess.run(args=['zipline', 'ingest', "--bundle", f'{self._bundle_name}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._logger.info(f'during ingestion: {run_output.stdout}')
        run_output = subprocess.run(args=['zipline', 'bundles'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._logger.info(f'bundles after ingestion: {run_output.stdout}')

        self._update_zipline_sqlite_db()

        self._logger.info(f'Data preparation in zipline completed')