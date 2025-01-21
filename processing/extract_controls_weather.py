import pandas as pd
import os
import numpy as np
from datetime import datetime

def add_headers_to_csv():
    # Define the path to the data files
    data_dir = 'data/bleiswijk'
    hps_file = 'dataHPS.csv'
    led_file = 'dataLED.csv'
    
    # Define column names for the dataset
    column_names = [
        'Time',
        'global radiation',
        'air temperature',
        'RH',
        'wind speed',
        'IndoorTemp',
        'IndoorVPD',
        'ThScrPos',
        'BlScrPos',
        'LeeSideVent',
        'WindSideVent',
        'PipeTemp',
        'GrowPipeTemp',
        'TopLight',
        'Interlight',
        'Unused1',
        'Unused2',
        'Unused3',
        'Unused4',
        'Unused5',
        'Unused6',
        'Unused7',
        'Co2Injection',
        'CO2 concentration',
    ]
    # # Process HPS file
    hps_path = os.path.join(data_dir, hps_file)
    if os.path.exists(hps_path):
        df_hps = pd.read_csv(hps_path, header=None)
        df_hps.columns = column_names
        print(f"Processed {hps_file}")
    
    # Process LED file
    led_path = os.path.join(data_dir, led_file)
    if os.path.exists(led_path):
        df_led = pd.read_csv(led_path, header=None)
        df_led.columns = column_names
        print(f"Processed {led_file}")
    return df_hps, df_led

def split_weather_controls(df):
    """
    Split dataframe into weather and control variables
    
    Args:
        df: Input dataframe with all variables
        
    Returns:
        weather_df: Dataframe with only weather variables
        controls_df: Dataframe with only control variables
    """
    # Create uBoil column based on PipeTemp threshold
    df['uBoil'] = (df['PipeTemp'] > df['air temperature']).astype(float)
    # Weather variables
    weather_vars = [
        'Time',
        'global radiation', 
        'air temperature',
        'RH',
        'wind speed',
    ]
    
    # Control variables 
    control_vars = [
        'uBoil',
        'ThScrPos',
        'BlScrPos', 
        'LeeSideVent',
        'WindSideVent',
        'PipeTemp',
        'GrowPipeTemp',
        'TopLight',
        'Interlight',
        'Co2Injection'
    ]
    
    weather_df = df[weather_vars].copy()
    controls_df = df[control_vars].copy()


    return weather_df, controls_df

def compute_sky_temp(air_temp, cloud):
    """
    Compute sky temperature from air temperature and cloud cover.
    Args
        air_temp: Air temperature in °C
        cloud: Cloud cover (0-1)
    Returns
        sky_temp: Sky temperature in °C
    """

    sigma = 5.67e-8 # Stefan-Boltzmann constant
    C2K = 273.15    # Conversion of °C to K

    ld_clear = 213+5.5*air_temp                      # Equation 5.26
    eps_clear = ld_clear/(sigma*(air_temp+C2K)**4)    # Equation 5.22
    eps_cloud = (1-0.84*cloud)*eps_clear+0.84*cloud   # Equation 5.32
    ld_cloud = eps_cloud*sigma*(air_temp+C2K)**4      # Equation 5.22
    sky_temp = (ld_cloud/sigma)**(0.25)-C2K           # Equation 5.22, but here assuming eps=1
    return sky_temp

def format_weather_df(weather_df, seconds, interpolated_cloud):
    """
    Format weather dataframe to match required columns and units
    
    Args:
        weather_df: Input weather dataframe with raw data
        
    Returns:
        formatted_df: Weather dataframe with standardized columns and units
    """
    # Create new dataframe with required columns
    formatted_df = pd.DataFrame()

    # Convert time from days to seconds
    formatted_df['time'] = np.arange(seconds, seconds + len(weather_df)*300, 300)

    # Copy over existing columns that match
    formatted_df['global radiation'] = weather_df['global radiation']
    formatted_df['wind speed'] = weather_df['wind speed'] 
    formatted_df['air temperature'] = weather_df['air temperature']

    # Add dummy sky temperature (4 degrees below air temp)
    formatted_df['sky temperature'] = compute_sky_temp(weather_df['air temperature'], interpolated_cloud)
    
    # Add empty column for unknown variable
    formatted_df['??'] = 0.0

    # Add constant CO2 concentration
    formatted_df['CO2 concentration'] = 400.0
    
    # Extract day number from time
    formatted_df['day number'] = formatted_df['time']/24/3600 

    # Copy RH
    formatted_df['RH'] = weather_df['RH']

    return formatted_df

def compute_seconds_in_2009():
    start = datetime(2009, 1, 1, 0, 0)  # 2009/01/01 00:00
    end = datetime(2009, 10, 19, 15, 15)  # 2009/10/19 15:15
    
    time_diff = end - start
    seconds = time_diff.total_seconds()
    print(f"Seconds between {start} and {end}: {seconds}")
    return seconds

def extract_cloud_cover(time_column):
    """
    Load cloud cover data from Rotterdam .mat file and extract values 
    corresponding to given timestamps.
    
    Args:
        time_column: Pandas series containing timestamps
        
    Returns:
        cloud_cover: Array of cloud cover values matching the timestamps
    """
    try:
        from scipy.io import loadmat
        import numpy as np
        
        # Load the .mat file containing cloud cover data
        cloud_data = loadmat('data/bleiswijk/cloudRotterdam2009_2012.mat')["cloudRotterdam2009_2012"]

        # Extract time and cloud cover arrays from the .mat file
        cloud_time = cloud_data[:,0]  # Assuming time is stored in 'time' variable
        cloud_cover = cloud_data[:,1] # Assuming cloud cover is stored in 'cloud' variable
        
        # Interpolate cloud cover values to match input timestamps
        from scipy.interpolate import interp1d
        f = interp1d(cloud_time, cloud_cover, bounds_error=False, fill_value='extrapolate')
        interpolated_cloud = f(time_column)

        return interpolated_cloud

    except Exception as e:
        print(f"Error loading cloud cover data: {str(e)}")
        # Return zeros if data loading fails
        return np.zeros(len(time_column))

def format_controls_df(controls_df):
    """
    Format controls dataframe to match required columns and units
    """
    np_controls = np.zeros((len(controls_df), 6))
    np_controls[:, 0] = controls_df['uBoil']
    np_controls[:, 1] = controls_df['Co2Injection']
    np_controls[:, 2] = controls_df['ThScrPos']/100
    np_controls[:, 3] = 0.5*(controls_df['WindSideVent'] + controls_df['LeeSideVent'])/100
    np_controls[:, 4] = controls_df['TopLight']/100
    np_controls[:, 5] = controls_df['BlScrPos']/100
    # Check for NaN values in each control column
    for i, control in enumerate(['Boiler', 'CO2', 'Thermal Screen', 'Ventilation', 'Top Light', 'Blackout Screen']):
        nan_count = np.isnan(np_controls[:, i]).sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values found in {control} control")
            # Replace NaNs with 0 to avoid issues
            np_controls[:, i] = np.nan_to_num(np_controls[:, i], 0)
    df_controls = pd.DataFrame(np_controls, columns=['Boiler', 'CO2', 'Thermal Screen', 'Ventilation', 'Top Light', 'Blackout Screen'])
    return df_controls

def process_time_data(df):
    """
    Process the time data by:
    1. Creating a copy of data from day 365 onwards for 2010
    2. Adjusting the time for 2010 data
    3. Filtering 2009 data to keep only days 292-364
    
    Args:
        df: DataFrame with 'day number' and 'time' columns
    
    Returns:
        tuple: (df_2009, df_2010) processed DataFrames for both years
    """
    # Create copy for 2010 data (days >= 365)
    df_2010 = df[df['day number'] >= 365].copy()
    # Adjust day numbers and time for 2010 (subtract 365 days worth of seconds)
    df_2010['day number'] = df_2010['day number'] - 365
    df_2010['time'] = df_2010['time'] - (365 * 86400)  # 86400 seconds per day
    
    # Filter 2009 data (days 292-364)
    df_2009 = df[(df['day number'] >= 292) & (df['day number'] < 365)].copy()
    
    return df_2009, df_2010


if __name__ == "__main__":
    df_hps, df_led = add_headers_to_csv()
    weather_df, controls_df = split_weather_controls(df_led)
    interpolated_cloud = extract_cloud_cover(weather_df['Time'])
    seconds = compute_seconds_in_2009()
    formatted_weather_df = format_weather_df(weather_df, seconds, interpolated_cloud)
    formatted_controls_df = format_controls_df(controls_df)
    df_2009, df_2010 = process_time_data(formatted_weather_df)


    # Save formatted weather data to specified path
    output_path = 'gl_gym/environments/weather/Bleiswijk/GL2009.csv'
    df_2009.to_csv(output_path, index=False)
    # Save 2010 data to specified path 
    output_path_2010 = 'gl_gym/environments/weather/Bleiswijk/GL2010.csv'
    df_2010.to_csv(output_path_2010, index=False)

    # save controls data to specified path
    output_path_controls = 'data/bleiswijk/controls2009.csv'
    pd.DataFrame(formatted_controls_df).to_csv(output_path_controls, index=False)

