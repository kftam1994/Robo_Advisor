import pandas as pd
from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler
import pytz
import seaborn as sns
from matplotlib import pyplot as plt

def convert_to_daily(df, start_date, end_date):
    """

    Convert dataframe to daily data with all dates within the start and end dates

    Parameters
    ----------
    df
    start_date
    end_date

    Returns
    -------

    """
    start_date = pd.Timestamp(start_date, tz=pytz.utc)
    end_date = pd.Timestamp(end_date, tz=pytz.utc)
    if not df.empty:
        dates_daily = pd.date_range(start_date, end_date, freq='D')
    else:
        dates_daily = pd.date_range(start_date, end_date, freq='D')
        fill_df = pd.DataFrame(index=dates_daily, columns=df.columns)
        fill_df = fill_df.fillna(0)
        df = fill_df
    dates_daily.name = 'date'
    df = df.reindex(dates_daily, method='ffill')
    return df

def create_or_load_minmaxscaler(filename,all_df,data_folder_path,logger,save_folder_name='minmaxscaler'):
    """

    Create a new or load an existing min-max scaler to conduct min-max normalization

    Parameters
    ----------
    filename : str
    all_df : pandas.DataFrame
    data_folder_path : pathlib.Path
    logger : Logger
    save_folder_name : str

    Returns
    -------
    scaler : sklearn.preprocessing.MinMaxScaler

    """

    Path(data_folder_path, save_folder_name).mkdir(parents=True, exist_ok=True)
    save_path = Path(data_folder_path,save_folder_name,f'{filename}.pkl')
    if save_path.exists():
        scaler = pickle.load(open(save_path, 'rb'))
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(all_df.values.reshape(-1, 1))

        pickle.dump(scaler, open(save_path, 'wb'))
        logger.debug(f'min max scaler saved to {save_path}')
    return scaler

def plot_dist(save_folder_path,data,prefix=None):
    """

    Plot distribution plots

    Parameters
    ----------
    save_folder_path : pathlib.Path
    data : numpy.array
    prefix : str

    """
    plt.clf()
    sns.stripplot(x=data)
    plt.savefig(Path(save_folder_path, f'{prefix}_strip.png'))
    plt.clf()
    sns.displot(data, kde=True, rug=True)
    plt.savefig(Path(save_folder_path, f'{prefix}_dist.png'))
    plt.clf()

def export_df_to_csv(df,output_path):
    """

    Save a pandas DataFrame to csv file

    Parameters
    ----------
    df : pandas.DataFrame
    output_path : pathlib.Path


    """
    df.to_csv(output_path, header=True,index=False, sep='|')

def merge_list_in_dict(original_dict,new_dict):
    """

    To merge a new dictioanry of list to the original dictioanry of list so that the list in the original dictioanry with respect to each key is appended the list of same key from the new dictioanry

    Reference: https://stackoverflow.com/questions/33931259/to-merge-two-dictionaries-of-list-in-python

    Parameters
    ----------
    main_dict : dict
    new_dict : dict

    Returns
    -------
    merged_dict : dict

    """
    merged_dict = {key: original_dict.get(key, []) + new_dict.get(key, []) for key in set(list(original_dict.keys()) + list(new_dict.keys()))}
    return merged_dict
 