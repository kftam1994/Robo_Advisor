
import pandas as pd

from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities
start_session = pd.Timestamp('1990-01-01', tz='utc')
end_session = pd.Timestamp('2020-12-31', tz='utc')

# register the bundle
register(
    'ETF_2021_08_30_1990_to_2020_etf2005',  # name we select for the bundle
    csvdir_equities(
        # name of the directory as specified above (named after data frequency)
        ['daily'],
        # path to directory containing the
        r'C:\Users\AndyTam\Downloads\robo advisor\RL\Final\data',
    ),
    calendar_name='NYSE',  # New York Stock Exchange https://github.com/quantopian/trading_calendars
    start_session=start_session,
    end_session=end_session
                )