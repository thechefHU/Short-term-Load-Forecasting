import pandas as pd
import numpy as np
import os
import holidays

def load(gran: str, include_lags: bool, include_generation: bool, include_weather: bool, fNG: bool):
    ZONE = "DK1"
    START = "2024-03-01"
    END = "2026-03-01"

    if gran == "h":
       freq = '1h'
    elif gran =="q":
       freq = '15min'
    df_load = pd.read_csv(os.getcwd()+"/data/DK1_load_2y.csv")
    df_load = df_load.set_index('timestamp')
    df_load.index = pd.to_datetime(df_load.index, utc=True)
    df_load = df_load.rename(columns={"value": "load"})
    
    start_date = '2024-03-01 00:00:00'
    end_date = '2026-03-01 23:45:00'

    # continuous spine for timeframe
    spine_index = pd.date_range(
        start=start_date, 
        end=end_date, 
        freq=freq, 
        tz='UTC' 
    )

    df_main = pd.DataFrame(index=spine_index)

    # indexing
    df_main.index.name = 'timestamp'
    # joining with load
    df_main = df_main.join(df_load['load'], how='left')
    df_main = df_main.resample(freq).asfreq()
    df_main['load'] = df_main['load'].interpolate(method='cubic')
    df_main['load'] = df_main['load'].ffill()

    #Daylight Savings & Sinusoidal time
    local_time_index = df_main.index.tz_convert('Europe/Copenhagen')

    local_time_of_day = local_time_index.hour + (local_time_index.minute / 60.0)

    # Time of Day
    df_main['tod_sin'] = np.sin(2 * np.pi * local_time_of_day / 24.0)
    df_main['tod_cos'] = np.cos(2 * np.pi * local_time_of_day / 24.0)

    #Week of the Month??

    #Day of Week & Day of Year
    local_day_of_week = local_time_index.dayofweek
    local_day_of_year = local_time_index.dayofyear

    df_main['dow_sin'] = np.sin(2 * np.pi * local_day_of_week / 7.0)
    df_main['dow_cos'] = np.cos(2 * np.pi * local_day_of_week / 7.0)

    df_main['doy_sin'] = np.sin(2 * np.pi * local_day_of_year / 365.25)
    df_main['doy_cos'] = np.cos(2 * np.pi * local_day_of_year / 365.25)

    zone_mapping = {
        'DK1': 'DK',
        'DK2': 'DK',
        'SE1': 'SE',
        'SE2': 'SE',
        'SE3': 'SE',
        'SE4': 'SE',
        'DE-LU': 'DE',
        'NO1': 'NO'
    }

    country_code = zone_mapping.get(ZONE)
    if not country_code:
        raise ValueError(f"Bidding zone {ZONE} not found in mapping!")

    years_in_data = local_time_index.year.unique().tolist()

    local_holidays = holidays.country_holidays(country_code, years=years_in_data)

    df_main['is_holiday'] = [1 if d in local_holidays else 0 for d in local_time_index.date]
    df_main.index = pd.to_datetime(df_main.index, utc=True)
    
    if gran == 'h':
        periods = 1
    elif gran == 'q':
        periods = 4
    #lags
    if include_lags == True:
        #df_main['lag_1d'] = df_main['load'].shift(96)   # 1 days * 96 periods
        df_main['lag_7d'] = df_main['load'].shift(7*24*periods)   # 7 days * 96 periods
        df_main['lag_14d'] = df_main['load'].shift(14*24*periods) # 14 days * 96 periods
        df_main['lag_21d'] = df_main['load'].shift(21*24*periods) # 21 days * 96 periods
    
    #generation
    if include_generation == True:
        df_gen = pd.read_csv(os.getcwd()+"/data/dk1_renewable_forecast_clean.csv")
        cols = ['timestamp', 'wind onshore', 'wind offshore', 'solar']

        existing_cols = [c for c in cols if c in df_gen.columns]
        df_gen= df_gen[existing_cols]
        df_gen = df_gen.set_index('timestamp')
        df_gen.index = pd.to_datetime(df_gen.index, utc=True)
        df_gen = df_gen.rename(columns={
            'wind onshore': 'Wind Onshore',
            'wind offshore': 'Wind Offshore',
            'solar': 'Solar'
            })
        df_main = df_main.join(df_gen[['Wind Onshore', 'Wind Offshore', 'Solar']], how='left')
        target_cols = ['Wind Onshore', 'Wind Offshore', 'Solar']

        for col in target_cols:
            df_main[col] = df_main[col].interpolate(method='cubic').ffill().bfill()
    if include_weather == True:

        df_weather = pd.read_csv(os.getcwd() + "/dk1_population_weighted_weather.csv")
        df_weather['time'] = pd.to_datetime(df_weather['time'])
        df_weather = df_weather.set_index('time')
        
        # Handle DST
        df_weather.index = df_weather.index.tz_localize(
            'Europe/Copenhagen', 
            ambiguous='NaT', 
            nonexistent='NaT'
        ).tz_convert('UTC')
        
        df_weather = df_weather[df_weather.index.notnull()]

        # Rename Columns
        column_mapping = {
            'temperature_2m (°C)': 't2m',
            'relative_humidity_2m (%)': 'rhum',
            'cloud_cover (%)': 'tcc',
            'surface_pressure (hPa)': 'sp',
            'wind_speed_10m (km/h)': 'ws',
            'shortwave_radiation (W/m²)': 'swr'
        }
        df_weather = df_weather.rename(columns=column_mapping)


        df_weather['ws'] = df_weather['ws'] / 3.6
        df_weather = df_weather.resample('15min').interpolate(method='linear')
        df_main = df_main.join(df_weather, how='left')
        
        cols_to_fill = list(column_mapping.values())
        df_main[cols_to_fill] = df_main[cols_to_fill].ffill().bfill()

    if fNG == True:
        unique_dates = pd.Series(local_time_index.date).unique()
        df_daily = pd.DataFrame(index=unique_dates)
        df_daily.index = pd.to_datetime(df_daily.index)

        df_daily['day_of_week'] = df_daily.index.dayofweek
        df_daily['is_holiday'] = [1 if d in local_holidays else 0 for d in df_daily.index.date]

        # DT1
        # 0 = Workday (Mon-Fri)
        # 1 = Day Off (Saturday)
        # 2 = Non-Working (Sunday OR Holiday)
        df_daily['DT1'] = 0 
        df_daily.loc[df_daily['day_of_week'] == 5, 'DT1'] = 1 
        df_daily.loc[(df_daily['day_of_week'] == 6) | (df_daily['is_holiday'] == 1), 'DT1'] = 2 

        # DT2
        # yesterday's DT1 to see the momentum
        df_daily['yesterday_DT1'] = df_daily['DT1'].shift(1)

        # Default 0
        df_daily['DT2'] = 0 
        # Ramp Down = 1
        df_daily.loc[(df_daily['yesterday_DT1'] == 0) & (df_daily['DT1'] >= 1), 'DT2'] = 1
        # Ramp Up = 2
        df_daily.loc[(df_daily['yesterday_DT1'] >= 1) & (df_daily['DT1'] == 0), 'DT2'] = 2

        # DT3 - Bridging days
        df_daily['DT3'] = 0
        df_daily.loc[(df_daily['is_holiday'] == 0) & (((df_daily['day_of_week'] == 0) & (df_daily['is_holiday'].shift(-1) == 1)) | ((df_daily['day_of_week'] == 4) & (df_daily['is_holiday'].shift(1) == 1))), 'DT3'] = 1


        local_dates = pd.to_datetime(local_time_index.date)

        df_main['DT1'] = local_dates.map(df_daily['DT1']).values
        df_main['DT2'] = local_dates.map(df_daily['DT2']).values
        df_main['DT3'] = local_dates.map(df_daily['DT3']).values
  
        #Code section for calculating Total load 2-week average (on working / non-working days)
        work_lags = pd.DataFrame(index=df_main.index)
        nonwork_lags = pd.DataFrame(index=df_main.index)

        periods_per_day = 24*periods

        
        for i in range(1, 15):
            shift_amount = i * periods_per_day
            
            past_load = df_main['load'].shift(shift_amount)
            past_DT1 = df_main['DT1'].shift(shift_amount)

            work_lags[f'day_{i}'] = past_load.where(past_DT1 == 0)

            nonwork_lags[f'day_{i}'] = past_load.where(past_DT1 >= 1)

        df_main['TL2W_w'] = work_lags.mean(axis=1)

    
        TL2W_nw = nonwork_lags.mean(axis=1)

        df_main['dTL2W_w_nw'] = TL2W_nw - df_main['TL2W_w']

        #Code for calculating HDD (Heating degree days)
        T_BASE = 17.0

        df_main['HDD'] = (T_BASE - df_main['t2m']).clip(lower=0)

        rolling_hdd_24h = df_main['HDD'].rolling(window=96, min_periods=1).mean()

        df_main['HDD_mean_D-2'] = rolling_hdd_24h.shift(2*24*periods)
        df_main['HDD_mean_D-7'] = rolling_hdd_24h.shift(7*24*periods)

    return df_main

def add_relative_humidity(df, temp_col, dew_point_col):
    """
    Calculates Relative Humidity (%) using the Magnus-Tetens approximation.
    Assumes inputs are in degrees Celsius.
    """
    # constants for the Magnus equation
    a = 17.625
    b = 243.04
    
    T = df[temp_col]
    Td = df[dew_point_col]
    
    # saturation vapor pressure
    svp = np.exp((a * T) / (b + T))
    
    # actual vapor pressure
    avp = np.exp((a * Td) / (b + Td))
    
    # rhum
    rh = 100 * (avp / svp)
    
    return np.clip(rh, 0, 100)
