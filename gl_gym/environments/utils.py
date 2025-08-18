from typing import Tuple, SupportsFloat
from os.path import join

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
from pandas._typing import ArrayLike
import pandas as pd
from scipy.interpolate import PchipInterpolator


def init_state(d0, rhMax=90, time_in_days=0):
    # Initialize the state as a NumPy array of zeros with the appropriate size
    state = np.zeros(28)  # Assuming 28 elements based on the original function

    state[0] = d0[3]        # co2Air
    state[1] = state[0]     # co2Top
    state[2] = 16.5         # tAir
    state[3] = state[2]     # tTop
    state[4] = state[2] + 4 # tCan
    state[5] = state[2]     # tCovIn
    state[6] = state[2]     # tCovE
    state[7] = state[2]     # tThScr
    state[8] = state[2]     # tFlr
    state[9] = state[2]     # tPipe
    state[10] = state[2]    # tSoil1
    state[11] = 0.25 * (3. * state[2] + d0[6])      # tSoil2
    state[12] = 0.25 * (2. * state[2] + 2 * d0[6])  # tSoil3
    state[13] = 0.25 * (state[2] + 3 * d0[6])       # tSoil4
    state[14] = d0[6]                               # tSoil5
    state[15] = rhMax / 100. * satVp(state[2])  # vpAir
    state[16] = state[15]   # vpTop
    state[17] = state[2]    # tLamp
    state[18] = state[2]    # tIntLamp
    state[19] = state[2]    # tGroPipe
    state[20] = state[2]    # tBlScr
    state[21] = state[4]    # tCan24
    state[22] = 0.          # cBuf
    state[23] = 9.5283e4    # cLeaf
    state[24] = 2.5107e5    # cStem
    state[25] = 5.5338e4    # cFruit
    state[26] = 3.0978e3    # tCanSum
    state[27] = time_in_days  # time

    return state

def load_weather_data(
                        weatherDataDir: str,
                        location: str,
                        growthYear: int,
                        startDay: int,
                        nDays: int,
                        predHorizon: int,
                        h: float,
                        nd: int
                    ) -> np.ndarray:
    """
    Loads in rawweather data from matlab file and converts it to values GreenLight model uses in numpy array.
    If the solver requires data on a higher frequency we interpolate between available weather data.
    Time interval of matlab data usually is 5 minutes.
    The rawweather data is a file with 9 columns, which we convert to 7 columns used by the GreenLight.

    Args:
        weatherDataDir  - path to raw weather data
        location        - location of the weather data
        growthYear      - start of the growth year of the simulation
        startDay        - at which day of the year do we start the simulation
        nDays           - how many days do we simulate forward in time
        predHorizon     - prediction horizon [days]
        h               - sample time of the solver in seconds
        nd              - number of weather variables
    
    Returns:
        Matrix with following interpolated weather variables:
        d[0]: iGlob         Global radiation [W m^{-2}]
        d[1]: tOut          Outdoor temperature [deg C]    
        d[2]: vpOut         Outdoor vapor pressure [Pa]
        d[3]: co2Out        Outdoor CO2 concentration [mg m^{-3}]
        d[4]: wind          Outdoor wind speed [m s^{-1}]
        d[5]: tSky          Sky temperature [deg C]
        d[6]: tSoOut        Outdoor soil temperature [deg C]
        d[7]: dli           Daily radiation sum [MJ m^{-2} day^{-1}]
        d[8]: isDay         Whether it is day or night [0,1]
        d[9]: isDaySmooth   Whether it is day or night [0,1] with a smooth transition
    """
    weatherDataPath = join(join(weatherDataDir, location), str(growthYear)) + ".csv"

    c = 86400      # seconds in a day
    CO2_PPM = 400  # assumed constant outdoor co2 concentration [ppm]
    rawWeather = pd.read_csv(weatherDataPath, sep=",")

    time = rawWeather["time"].values    # time since start of the year in [s]
    dt = np.mean(np.diff(time-time[0])) # sample period of data [s]
    N0 = int(np.ceil(startDay*c/dt))    # Start index
    Ns = int(np.ceil(nDays*c/dt))       # Number of samples we need from regular data
    Np = int(np.ceil(predHorizon*c/dt))+1 # Number of samples into the future we need from regular data

    # check whether we exceed data length and we are in the final season
    if N0+Ns+Np > len(time):
        rawWeather = expandWeatherData(weatherDataDir, rawWeather, location, growthYear, time, dt)
    weatherData = np.zeros((Ns+Np, nd))                                         # preallocate weather data matrix
    time = rawWeather["time"].values[N0:N0+Ns+Np]                               # time since start of the year in [s]
    weatherData[:, 0] = rawWeather["global radiation"][N0:N0+Ns+Np]             # iGlob
    weatherData[:, 1] = rawWeather["air temperature"][N0:N0+Ns+Np]              # tOut
    vpDensity = rh2vaporDens(weatherData[:, 1], rawWeather["RH"][N0:N0+Ns+Np])  # vp Density
    weatherData[:,2] = vaporDens2pres(weatherData[:, 1], vpDensity)             # vpOut
    weatherData[:,3] = co2ppm2dens(weatherData[:, 1], CO2_PPM)*1e6              # co2Out (converted from kg/m^3 to mg/m^3)
    weatherData[:,4] = rawWeather["wind speed"][N0:N0+Ns+Np]                    # wind
    weatherData[:,5] = rawWeather["sky temperature"][N0:N0+Ns+Np]               # tSky
    weatherData[:,6] = soilTempNl(rawWeather["time"][N0:N0+Ns+Np])              # tSoOut
    weatherData[:, 7] = dailLightSum(time, weatherData[:,0], c)                 # daily sun radiation sum [MJ m^{-2} day^{-1}]
    weatherData[:, 8], weatherData[:,9] = computeisDay(weatherData[:, 0], dt)   # isDay, isDaySmooth

    # number of samples required for the solver
    ns = int((dt/h) * (Ns+Np))

    # interpolate and resample
    interpolation = PchipInterpolator(time, weatherData)
    timeRes = np.linspace(time[0], time[-1], ns)
    weatherDataResampled = interpolation(timeRes)

    # set small radiation values to zero
    weatherDataResampled[:,0][weatherDataResampled[:, 0] < 1e-10] = 0

    return weatherDataResampled

def expandWeatherData(
                    weatherDataDir: str, 
                    rawWeather: pd.DataFrame,
                    location: str,
                    growthYear: int,
                    time: ArrayLike,
                    dt: SupportsFloat,
                    ) -> pd.DataFrame:
    """
    Function that loads in the weather data for the next year and appends it to the current weather data.
    Required when the simulation exceeds the length of the current weather data.
    Args:
        weatherDataDir  - path to raw weather data
        rawWeather      - current weather data
        location        - location of the greenhouse
        growthYear      - year of the growth season
        time            - time since start of the year in [s]
        dt              - sample period of weather data [s]
    Returns:
        rawWeather      - weather data for the next year appended to the current weather data
    """
    weatherDataPath = join(join(weatherDataDir, location), str(growthYear+1)) + ".csv"
    newRawWeather = pd.read_csv(weatherDataPath, sep=",")
    newRawWeather["time"] += time[-1] + dt
    rawWeather = pd.concat([rawWeather, newRawWeather.iloc[:, :]])
    return rawWeather

def days2date(timeInDays: float, referenceDate: str):
    """
    Function that converts the number of days since a reference date
    to a date in the format DD-MM-YYYY.
    Args:
        timeInDays      - number of days since reference date
        referenceDate   - reference date in format DD-MM-YYYY
    Returns:
        targetDatetime  - current date in format DD-MM-YYYY
    """
    referenceDatetime = datetime.strptime(referenceDate, '%d-%m-%Y')
    int_days = np.floor(timeInDays).astype(int)
    time_component = (timeInDays - int_days) * 24           # Convert decimal part to hours
    hours = time_component.astype(int)
    time_component = (time_component - hours) * 60          # Convert remaining decimal part to minutes
    minutes = time_component.astype(int)
    
    target_datetimes = [referenceDatetime + timedelta(days=int(int_day), hours=int(hour), minutes=int(minute)) for int_day, hour, minute in zip(int_days, hours, minutes)]

    return [target_datetime.strftime('%Y-%m-%d %H:%M:%S') for target_datetime in target_datetimes]

def computeisDay(rad: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function that computes whether it is day or night based on the radiation.
    A day is defined as the period between sunrise and sunset.
    There is a tranisition period between day and night.
    To account for the twilight between day and night.
    Smooth transition is based on a sigmoid function.
    Args:
        rad     - radiation [W m^{-2}]
        dt      - sample period of the weather data [s]
    Returns:
        isDay           - whether it is day or night [0,1]
        isDaySmooth     - whether it is day or night [0,1] with a smooth transition
    """
    
    # add transition betwen day and night
    isDay = (rad > 0)*1.0
    isDaySmooth = deepcopy(isDay)
    transSize = int(3600/dt)    # length of transition period between night and day
                                # should be even. 3600/dt is the number of samples in an hour.

    trans = np.linspace(0, 1, transSize)
    transSmooth = 1/(1+np.exp(-10*(trans-0.5)))
    sunset = False  # indicates if we are during sunset

    for k in range(transSize, len(isDay) - transSize):
        if isDay[k] == 0:
            sunset = False
        if isDay[k] == 0 and isDay[k + 1] == 1:
            isDay[k - transSize // 2 : k + transSize // 2] = trans
            isDaySmooth[k - transSize // 2 : k + transSize // 2] = transSmooth
        elif isDay[k] == 1 and isDay[k + 1] == 0 and not sunset:
            isDay[k - transSize // 2: k + transSize // 2] = 1 - trans
            isDaySmooth[k - transSize // 2: k + transSize // 2] = 1 - transSmooth
            sunset = True
    return isDay, isDaySmooth

def dailLightSum(time: np.ndarray, rad: np.ndarray, c: int):
    """
    Function that computes the DLI (Daily Light Integral) from a given radiation time series.
    Args:
        time    - time since start of the year in [s]
        rad     - radiation [W m^{-2}]
        c       - seconds in a day
    Returns:
        lightSum    - DLI [MJ m^{-2} day^{-1}]
    """
    interval = time[1]-time[0] # time interval between samples [s]
    time = time/c               # convert to days

    # index of the midnight before current point
    mnBefore = 0

    # index of the midnight after current point
    mnAfter = np.where(np.diff(np.floor(time)) == 1)[0] +1
    if mnAfter.size == 0:
        mnAfter = len(time)
    else:
        mnAfter = mnAfter[0]
    lightSum =  np.zeros(len(time))

    for i in range(len(time)):
        lightSum[i] = np.sum(rad[mnBefore:mnAfter+1])

        if i == mnAfter -1:
            mnBefore = mnAfter
            mnAfter = np.where(np.diff(np.floor(time[mnBefore+2:])) == 1)[0] + mnBefore+2
            # mnAfter = len(time)
            if mnAfter.size == 0:
                mnAfter = len(time)
            else:
                mnAfter = mnAfter[0]
    return lightSum*interval*1e-6

# def sat_vp(temp):
#     return .61078*np.exp(17.2694*temp/(temp+238.3))

# def actual_vp(temp, rh):
#     return sat_vp(temp)*(rh/100)

# def hum_deficit(temp, rh):
#     return sat_vp(temp) - actual_vp(temp, rh)

def soilTempNl(time):
    # SOILTEMPNL An estimate of the soil temperature in the Netherlands in a given time of year
    # Based on Figure 3 in 
    # Jacobs, A. F. G., Heusinkveld, B. G. & Holtslag, A. A. M. 
    # Long-term record and analysis of soil temperatures and soil heat fluxes in 
    # a grassland area, The Netherlands. Agric. For. Meteorol. 151, 774�780 (2011).
    #
    # Input:
    #   time - seconds since beginning of the year [s]
    # Output:
    #   soilT - soil temperature at 1 meter depth at given time [�C]

    # Calculated based on a sin function approximating the figure in the reference
    
    # David Katzin, Wageningen University
    # david.katzin@wur.nl

    SECS_IN_YEAR = 3600*24*365
    soilT = 10+5*np.sin((2*np.pi*(time+0.625*SECS_IN_YEAR)/SECS_IN_YEAR))
    return soilT

def vaporDens2pres(temp, vaporDens):
    # VAPORDENS2PRES Convert vapor density [kg{H2O} m^{-3}] to vapor pressure [Pa]
    #
    # Usage:
    #   vaporPres = vaporDens2pres(temp, vaporDens)
    # Inputs:
    #   temp        given temperatures [°C] (numeric vector)
    #   vaporDens   vapor density [kg{H2O} m^{-3}] (numeric vector)
    #   Inputs should have identical dimensions
    # Outputs:
    #   vaporPres   vapor pressure [Pa] (numeric vector)
    #
    # Calculation based on 
    #   http://www.conservationphysics.org/atmcalc/atmoclc2.pdf

    # David Katzin, Wageningen University
    # david.katzin@wur.nl
    # david.katzin1@gmail.com
    
    # parameters used in the conversion
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
        # default value is [610.78 238.3 17.2694 -6140.4 273 28.916]
    
    rh = vaporDens/rh2vaporDens(temp, 100) # relative humidity [0-1]
        
    satP = p[0]*np.exp(p[2]*temp/(temp+p[1]))
    # Saturation vapor pressure of air in given temperature [Pa]
    
    return satP*rh

def satVp(temp):
    # saturated vapor pressure (Pa) at temperature temp (�C)
    # Calculation based on 
    #   http://www.conservationphysics.org/atmcalc/atmoclc2.pdf
    # See also file atmoclc2.pdf

    # parameters used in the conversion
    # p = [610.78 238.3 17.2694 -6140.4 273 28.916];
        # default value is [610.78 238.3 17.2694 -6140.4 273 28.916]

        # Saturation vapor pressure of air in given temperature [Pa]
    return 610.78* np.exp(17.2694*temp/(temp+238.3))


def co2ppm2dens(temp, ppm):
    # CO2PPM2DENS Convert CO2 molar concetration [ppm] to density [kg m^{-3}]

    # Usage:
    #   co2Dens = co2ppm2dens(temp, ppm) 
    # Inputs:
    #   temp        given temperatures [�C] (numeric vector)
    #   ppm         CO2 concetration in air (ppm) (numeric vector)
    #   Inputs should have identical dimensions
    # Outputs:
    #   co2Dens     CO2 concentration in air [kg m^{-3}] (numeric vector)

    # Calculation based on ideal gas law pV=nRT, with pressure at 1 atm

    # David Katzin, Wageningen University
    # david.katzin@wur.nl
    # david.katzin1@gmail.com

    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    M_CO2 = 44.01e-3 # molar mass of CO2 [kg mol^-{1}]
    P = 101325 # pressure (assumed to be 1 atm) [Pa]
    
    # number of moles n=m/M_CO2 where m is the mass [kg] and M_CO2 is the
    # molar mass [kg mol^{-1}]. So m=p*V*M_CO2*P/RT where V is 10^-6*ppm    
    return P*10**-6*ppm*M_CO2/(R*(temp+C2K))

def co2dens2ppm(temp, dens):
    '''
    Convert CO2 density [kg m^{-3}] to molar concentration [ppm]
    '''
    R = 8.3144598        # Molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15         # Conversion from Celsius to Kelvin [K]
    M_CO2 = 44.01e-3     # Molar mass of CO2 [kg mol^{-1}]
    P = 101325           # Pressure (assumed to be 1 atm) [Pa]
    
    return 1e6 * R * (temp + C2K) * dens / (P * M_CO2)

def vaporPres2rh(temp, vaporPres):
    return np.clip(100*vaporPres/satVp(temp), a_min=0., a_max=100.)

def vaporDens2rh(temp, vaporDens):
    """
    vaporDens2rh Convert vapor density [kg{H2O} m^{-3}] to relative humidity [%]

    Usage:
    rh = vaporDens2rh(temp, vaporDens)
    Inputs:
    temp        given temperatures [°C] (numeric vector)
    vaporDens   absolute humidity [kg{H20} m^{-3}] (numeric vector)
    Inputs should have identical dimensions
    Outputs:
    rh          relative humidity [%] between 0 and 100 (numeric vector)

    Calculation based on 
    http://www.conservationphysics.org/atmcalc/atmoclc2.pdf

    David Katzin, Wageningen University
    david.katzin@wur.nl
    """
    # constants
    # molar gas constant [J mol^{-1} K^{-1}]
    R = 8.3144598 
    # conversion from Celsius to Kelvin [K]
    C2K = 273.15  
    # molar mass of water [kg mol^-{1}]
    Mw = 18.01528e-3  
    
    # parameters used in the conversion
    # default value is [610.78 238.3 17.2694 -6140.4 273 28.916]
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
    
    # Saturation vapor pressure of air in given temperature [Pa]
    satP = p[0]*np.exp(p[2]*temp/(temp+p[1])) 
    # convert to relative humidity using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.
    relhumid = 100*R*(temp+C2K)/(Mw*satP)*vaporDens
    # if np.isinf(relhumid).any():
    #     print(temp, vaporDens)
    return np.clip(relhumid, a_min=0, a_max=100)

def rh2vaporDens(temp, rh):
    # RH2VAPORDENS Convert relative humidity [#] to vapor density [kg{H2O} m^{-3}]

    # Usage:
    #   vaporDens = rh2vaporDens(temp, rh)
    # Inputs:
    #   temp        given temperatures [�C] (numeric vector)
    #   rh          relative humidity [#] between 0 and 100 (numeric vector)
    #   Inputs should have identical dimensions
    # Outputs:
    #   vaporDens   absolute humidity [kg{H20} m^{-3}] (numeric vector)

    # Calculation based on 
    #   http://www.conservationphysics.org/atmcalc/atmoclc2.pdf

    # David Katzin, Wageningen University
    # david.katzin@wur.nl
    # david.katzin1@gmail.com

    # constants
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    Mw = 18.01528e-3 # molar mass of water [kg mol^-{1}]
    
    # parameters used in the conversion
    p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
    # default value is [610.78 238.3 17.2694 -6140.4 273 28.916]
    
    satP = p[0]*np.exp(p[2]*temp/(temp+p[1]))
        # Saturation vapor pressure of air in given temperature [Pa]
    
    pascals=(rh/100)*satP # Partial pressure of vapor in air [Pa]
    
    # convert to density using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.
    
    return pascals*Mw/(R*(temp+C2K))

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

