
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

def convert_time_series(train_path = None, test_path = None):
    id_time_series_train = defaultdict()
    if train_path is not None:
        df_train = pd.read_parquet(train_path)
        df_train.set_index('id', inplace= True)
        print('Converting train time series')
        for id in tqdm(df_train.index):
            id_time_series_train[id] = {
                'time_serie': pd.DataFrame(data = df_train.loc[id]['values'],
                                            index= df_train.loc[id]['dates'],
                                            columns= ['values']), 
                'target': df_train.loc[id]['label']}
            
    id_time_series_test = defaultdict()
    if test_path is not None:
        df_test = pd.read_parquet(test_path)
        df_test.set_index('id', inplace= True)
        print('Converting test time series')
        for id in tqdm(df_test.index):
            id_time_series_test[id] = {
                'time_serie': pd.DataFrame(data = df_test.loc[id]['values'],
                                            index= df_test.loc[id]['dates'],
                                            columns= ['values'])
                                            }
    return id_time_series_train, id_time_series_test

def rolling_ts(ts):
    rolling_trand = ts.rolling(
        window = 12,
        center = True,
        min_periods = 6).mean()
    return rolling_trand

def convert_rolling_time_series(train_path = None, test_path = None):
    id_time_series_train_rolling = defaultdict()
    if train_path is not None:
        df = pd.read_parquet(train_path)
        df.set_index('id', inplace= True)
        print('Converting train time series')
        for id in tqdm(df.index):
            id_time_series_train_rolling[id] = {
                'time_serie': rolling_ts(pd.DataFrame(data = df.loc[id]['values'],
                                            index= df.loc[id]['dates'],
                                            columns= ['values'])
                ),
                'target': df.loc[id]['label']
            }
    id_time_series_test_rolling = defaultdict()
    if test_path is not None:
        df_ = pd.read_parquet(test_path)
        df_.set_index('id', inplace= True)
        print('Converting test time series')
        for id in tqdm(df_.index):
            id_time_series_test_rolling[id] = {
                'time_serie': rolling_ts(pd.DataFrame(data = df_.loc[id]['values'],
                                            index= df_.loc[id]['dates'],
                                            columns= ['values'])
                ), 
            }
    return id_time_series_train_rolling, id_time_series_test_rolling


def split_year_intervals(k = 8): # Разобью по годам
    time_intervals = []
    start_interval = pd.to_datetime('2016-01-01').date()
    for _ in range(k):
        end_interval = (start_interval + pd.offsets.YearEnd(0)).date()
        time_intervals.append((start_interval, end_interval))
        start_interval = (end_interval + pd.offsets.YearBegin(0)).date()
    time_intervals[-1] = (time_intervals[-1][0], (time_intervals[-1][1] + pd.offsets.YearBegin(0)).date())
    return time_intervals


def create_interval_df(time_series,  agg_func, k = 8):
    intevals_dict = defaultdict(dict)
    time_intervals = split_year_intervals(k = k)
    for i in range(k):
        interval_values = time_series.loc[time_intervals[i][0]: time_intervals[i][1]]['values']
        intevals_dict[f'interval_{i}'] = {name: func(interval_values) for name, func in agg_func}
    df = pd.DataFrame(pd.DataFrame(intevals_dict).unstack()).T

    df = (df
    .set_axis(df.columns.map('_'.join), axis=1)
    )

    return df

def create_dataset(agg_func, id_ts):
    final_df = create_interval_df(id_ts.get(0, id_ts[2])['time_serie'], agg_func).head(0)
    rows = []
    print('Create dataset')
    for id, ts in tqdm(id_ts.items()):
        str_ = create_interval_df(ts['time_serie'], agg_func)
        str_['id'] = id
        if 'target' in ts:
            str_['target'] = ts['target']
        rows.append(str_)
    final_df = pd.concat(rows, ignore_index=True)
    final_df.set_index('id', inplace = True)
    return final_df

