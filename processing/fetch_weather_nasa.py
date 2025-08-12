#!/usr/bin/env python3
"""
fetch_and_process_power.py

Download, process and interpolate to 5-minute resolution hourly weather data
from the NASA POWER API for a given city over a range of years.

Steps:
  1. Geocode the city name to latitude/longitude.
  2. Fetch hourly data for each year.
   3. Identify the CSV header line "YEAR,MO,DY,HR".
  4. Read data from that line onward, parse date/time into a datetime index.
  5. Convert global radiation from MJ/h/m² to W/m².
  6. Compute sky temperature from air temperature and cloud cover.
  7. Compute seconds since the start of the year and day number.
  8. Fill CO₂ concentration constant.
  9. Resample to 5-minute intervals using piecewise cubic Hermite (PCHIP).
 10. Save a processed CSV under weather/<city>/<year>.csv with columns:
      time, global radiation, wind speed, air temperature,
      sky temperature, cloud cover, CO₂ concentration, day number, RH

Dependencies:
    pip install requests geopy pandas scipy

Usage:
    python fetch_and_process_power.py "City Name, Country"
"""
import os
import argparse
import requests
import pandas as pd
import calendar
from geopy.geocoders import Nominatim
from io import StringIO
from time import sleep
from requests.exceptions import HTTPError

# for PCHIP interpolation
from scipy.interpolate import PchipInterpolator
from timezonefinder import TimezoneFinder
import numpy as np
import pytz

def compute_sky_temp(air_temp_c, cloud_frac):
    sigma = 5.67e-8
    C2K = 273.15
    # Clear-sky longwave down
    ld_clear = 213 + 5.5 * air_temp_c
    eps_clear = ld_clear / (sigma * (air_temp_c + C2K)**4)
    eps_cloud = (1 - 0.84 * cloud_frac) * eps_clear + 0.84 * cloud_frac
    ld_cloud = eps_cloud * sigma * (air_temp_c + C2K)**4
    sky_temp_k = (ld_cloud / sigma)**0.25
    return sky_temp_k - C2K

def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="weather_fetcher")
    loc = geolocator.geocode(city_name)
    if loc is None:
        raise ValueError(f"Could not geocode '{city_name}'")
    return loc.latitude, loc.longitude


def _parse_power_csv(text):
    """
    Locate 'YEAR,MO,DY,HR' header, read data into DataFrame,
    and build a datetime index.
    """
    lines = text.splitlines()
    header_idx = next((i for i, L in enumerate(lines) if L.startswith("YEAR,MO,DY,HR")), None)
    if header_idx is None:
        raise ValueError("Could not find 'YEAR,MO,DY,HR' header")
    data = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(data))
    df['time'] = pd.to_datetime({
        'year': df['YEAR'],
        'month': df['MO'],
        'day': df['DY'],
        'hour': df['HR']
    })
    df.set_index('time', inplace=True)
    return df


def fetch_process_year(lat, lon, year, city_dir, parameters, community="AG"):
    """
    Fetch, process, interpolate and save data for a single year.
    """
    # compute local start (00:00) and end (23:00) for the year
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=lat, lng=lon)
    if timezone_str is None:
        raise ValueError("Could not determine timezone for the given coordinates")
    tz = pytz.timezone(timezone_str)
    local_start = tz.localize(pd.Timestamp(f"{year}-01-01 00:00:00"))
    local_end   = tz.localize(pd.Timestamp(f"{year}-12-31 23:00:00"))
    # convert to UTC for the API window
    utc_start   = local_start.astimezone(pytz.UTC)
    utc_end     = local_end.astimezone(pytz.UTC)
    
    local_dt = tz.localize(pd.Timestamp(2001, 1, 1, 0, 0, 0))
    # 3. Get the UTC offset
    offset_td    = local_dt.utcoffset()                     # a timedelta
    offset_hours = offset_td.total_seconds() // 3600.0       # convert to hours

    # format YYYYMMDD for NASA POWER
    if offset_hours > 0 and year == 2001:
        start_date  = (utc_start.date() + pd.Timedelta(days=1)).strftime('%Y%m%d')
        print(start_date)
    else:
        start_date  = utc_start.date().strftime('%Y%m%d')
    end_date    = utc_end.date().strftime('%Y%m%d')


    def _build_url(start, end):
        return (
            "https://power.larc.nasa.gov/api/temporal/hourly/point"
            f"?start={start}&end={end}"
            f"&latitude={lat}&longitude={lon}"
            f"&community={community}"
            f"&parameters={','.join(parameters)}"
            "&format=CSV&header=true&time-standard=utc"
        )

    print(f"Fetching full year {year}")
    # url_year = _build_url(f"{year}0101", f"{year}1231")
    url_year = _build_url(start_date, end_date)
    try:
        resp = requests.get(url_year)
        resp.raise_for_status()
        df = _parse_power_csv(resp.text)
    except HTTPError as e:
        if e.response is not None and 500 <= e.response.status_code < 600:
            print(f"  Full-year fetch failed (status {e.response.status_code}), retrying by month...")
            monthly = []
            for m in range(1, 13):
                last = calendar.monthrange(year, m)[1]
                start = f"{year}{m:02d}01"
                end   = f"{year}{m:02d}{last:02d}"
                try:
                    sleep(1)
                    r = requests.get(_build_url(start, end))
                    r.raise_for_status()
                    monthly.append(_parse_power_csv(r.text))
                except Exception as me:
                    print(f"    Month {m:02d} failed: {me}")
            if not monthly:
                raise RuntimeError(f"No data for {year}")
            df = pd.concat(monthly).sort_index()
        else:
            raise

    # Convert units and compute features
    # MJ/m2/h to W/m2
    df['ALLSKY_SFC_SW_DWN'] *= (1e6 / 3600)
    # Cloud fraction
    df['cloud_frac'] = df['CLOUD_AMT'] / 100.0
    # Sky temperature
    df['sky_temperature'] = df.apply(lambda r: compute_sky_temp(r['T2M'], r['cloud_frac']), axis=1)
    # seconds and day number
    start_ts = pd.Timestamp(f"{year}-01-01 00:00:00")
    # CO2 constant

    n = len(df)
    # 3) Compute start/end positions
    if offset_hours > 0:
        # skip first offset_hours rows, and last (offset_hours) rows
        start_idx = 24-offset_hours
        end_idx   = n - (offset_hours)
    elif offset_hours < 0:
        # skip first -offset_hours rows, and last (24+offset_hours) rows
        start_idx = -offset_hours
        end_idx   = n - (24+offset_hours)
    else:
        start_idx, end_idx = 0, n
    print(start_idx, end_idx, n)
    print(end_idx - start_idx)
    # 4) Slice by position (keep the DatetimeIndex!)
    df = df.iloc[int(start_idx):int(end_idx)]

    # Interpolate to 5-minute using PCHIP
    df5 = df.resample('5min').asfreq()
    cols_interp = ['ALLSKY_SFC_SW_DWN', 'WS2M', 'T2M', 'sky_temperature', 'cloud_frac', 'RH2M']
    for col in cols_interp:
        # drop NaNs for interpolation
        series = df[col]
        pchip = PchipInterpolator(series.index.astype(int), series.values)
        df5[col] = pchip(df5.index.astype(int))
    # CO2 constant
    df5['CO2_ppm'] = 400.0
    # Recompute seconds and day number on 5-min grid
    df5['seconds_since_start'] = np.arange(len(df5)) * 300  # 5 minutes = 300 seconds
    df5['day_number'] = df5.index.dayofyear

    # Prepare output
    out = df5[[
        'seconds_since_start', 'ALLSKY_SFC_SW_DWN', 'WS2M', 'T2M',
        'sky_temperature', 'cloud_frac', 'CO2_ppm', 'day_number', 'RH2M'
    ]].copy()
    out.columns = [
        'time', 'global radiation', 'wind speed', 'air temperature',
        'sky temperature', 'cloud cover', 'CO2 concentration',
        'day number', 'RH'
    ]

    filepath = os.path.join(city_dir, f"{year}.csv")
    out.to_csv(filepath, index=False)
    print(f"Saved processed and interpolated data to {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("city", help="City name, e.g. 'Reykjavik, Iceland'")
    parser.add_argument("--output-dir", default="weather")
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year",   type=int, default=2020)
    args = parser.parse_args()

    key = args.city.split(',')[0].capitalize().replace(' ', '_')
    print(key)
    base = os.path.join(args.output_dir, key)
    os.makedirs(base, exist_ok=True)

    print(f"Geocoding {args.city}...")
    lat, lon = get_coordinates(args.city)
    print(f"Coordinates: {lat:.4f}, {lon:.4f}\n")

    params = ["ALLSKY_SFC_SW_DWN", "T2M", "WS2M", "RH2M", "CLOUD_AMT"]
    for yr in range(args.start_year, args.end_year+1):
        try:
            fetch_process_year(lat, lon, yr, base, params)
            sleep(1)
        except Exception as e:
            print(f"Error processing {yr}: {e}")

if __name__ == "__main__":
    main()
