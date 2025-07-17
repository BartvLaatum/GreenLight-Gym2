import casadi as ca
import numpy as np
import pandas as pd

from gl_gym.environments.models.ode import ODE

def define_model(nx: int, nu: int, nd: int, n_params: int, dt: float):
    """
    Defines a CasADi integrator model for a given system's ODE.

    Args:
        nx (int): Number of state variables.
        nu (int): Number of control input variables.
        nd (int): Number of disturbance variables.
        n_params (int): Number of model parameters.
        dt (float): Integration time step.

    Returns:
        casadi.integrator: A CasADi integrator object configured for the system ODE.
    """
    # Define the symbolic variables for CasADi
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    d = ca.SX.sym("d", nd)
    p = ca.SX.sym("p", n_params)

    dxdt = ODE(x, u, d, p)
    input_args_sym = ca.vertcat(d, p)

    int_opts = {"abstol": 1e-4, "reltol": 1e-4, "max_num_steps": 7e4}
    # int_opts = {}
    F = ca.integrator(
        "F", "cvodes",
        {"x": x, "u": u, "p": input_args_sym, "ode": dxdt},
        0.0, dt, int_opts
    )

    return F

def satVp_cpp(temp):
    """
    Calculates saturation vapor pressure.
    
    Args:
        temp (float): Temperature in degrees Celsius.
            
    Returns:
        float: Saturation vapor pressure in Pascals.
    """
    a = 610.78
    b = 17.2694
    c = 238.3
    return a * np.exp(b * temp / (temp + c))

def cond(hec, vp1, vp2):
    """
    Condensation function
    Args:
        hec (float): Heat exchange coefficient.
        vp1 (float): Vapor pressure of the first layer.
        vp2 (float): Vapor pressure of the second layer.

    Returns:
        float: Condensation value.
    """
    a = 6.4e-9
    return 1.0 / (1.0 + np.exp(-0.1 * (vp1 - vp2))) * a * hec * (vp1 - vp2)

def co2dens2ppm(temp, dens):
    """
    Convert CO2 density [kg m^{-3}] to CO2 concentration [ppm]
    Args:
        temp (float): Temperature in degrees Celsius.
        dens (float): CO2 density in kg m^{-3}.

    Returns:
        float: CO2 concentration in ppm.
    """
    R = 8.3144598        # Molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15         # Conversion from Celsius to Kelvin [K]
    M_CO2 = 44.01e-3     # Molar mass of CO2 [kg mol^{-1}]
    P = 101325           # Pressure (assumed to be 1 atm) [Pa]
    
    return 1e6 * R * (temp + C2K) * dens / (P * M_CO2)

def proportional_control(processVar, setPt, pBand, minVal, maxVal):
    """
    Proportional control function
    
    Args:
        processVar (float): Current process variable.
        setPt (float): Set point for the control.
        pBand (float): Proportional band.
        minVal (float): Minimum value of the control output.
        maxVal (float): Maximum value of the control output.

    Returns:
        float: Control output value.
    """
    return minVal + (maxVal - minVal) * (1. / (1. + np.exp(-2. / pBand * np.log(100.) * (processVar - setPt - pBand / 2.))))

def tau12(tau1, tau2, rho1Dn, rho2Up):
    """
    Transmission coefficient of a double layer [-]
    """
    return tau1 * tau2 / (1. - rho1Dn * rho2Up)

def rhoDn(tau2, rho1Dn, rho2Up, rho2Dn):
    """
    Reflection coefficient of the lower layer [-]
    """
    return rho2Dn + (tau2 * tau2 * rho1Dn) / (1. - rho1Dn * rho2Up)

def dli_check(lamp_input, dli):
    """
    Check daily light integral
    """
    if dli > 15.:
        return 0.
    return lamp_input

def init_state(d0, rhMax, time_in_days):
    """
    Initialize greenhouse state vector
    """
    state = np.zeros(28)
    state[0] = d0[3]        # co2Air
    state[1] = state[0]     # co2Top
    state[2] = 18.5         # tAir
    state[3] = state[2]     # tTop
    state[4] = state[2] + 4 # tCan
    state[5] = state[2]     # tCovIn
    state[6] = state[2]     # tCovE
    state[7] = state[2]     # tThScr
    state[8] = state[2]     # tFlr
    state[9] = state[2]     # tPipe
    state[10] = state[2]    # tSoil1
    state[11] = .25*(3.*state[2] + d0[6])   # tSoil2
    state[12] = .25*(2.*state[2] + 2*d0[6]) # tSoil3
    state[13] = .25*(state[2] + 3*d0[6])    # tSoil4
    state[14] = d0[6]       # tSoil5
    state[15] = rhMax / 100. * satVp_cpp(state[2])  # vpAir
    state[16] = state[15]   # vpTop
    state[17] = state[2]    # tLamp
    state[18] = state[2]    # tIntLamp
    state[19] = state[2]    # tGroPipe
    state[20] = state[2]    # tBlScr
    state[21] = state[4]    # tCan24
    state[22] = 1000.       # cBuf
    state[23] = 9.5283e4    # cLeaf
    state[24] = 2.5107e5    # cStem
    state[25] = 5.5338e4    # cFruit
    state[26] = 3.0978e3    # tCanSum
    state[27] = time_in_days # time
    return state

def load_dummy_weather(N, dt, month='june'):
    """
    Load weather data from CSV file

    Args:
        N: Number of timesteps to load
        month: Month to load data from (default: 'june')
    Returns:
        weather_data: List of weather data rows
    """
    try:
        df = pd.read_csv(f"weather/{month}/weather-{int(dt)}dt.csv")
        weather_data = df.head(N).values

    except FileNotFoundError:
        print(f"Error: Could not open weather data file for {month}")
        weather_data = []
    return weather_data

def vaporDens2rh(temp, vaporDens):
    """
    Convert vapor density to relative humidity.

    Args:
        temp (float): Temperature in degrees Celsius.
        vaporDens (float): Vapor density in kg m^{-3}.

    Returns:
        float: Relative humidity in percentage.
    """
    # Constants
    R = 8.3144598 # molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15 # conversion from Celsius to Kelvin [K]
    Mw = 18.01528e-3 # molar mass of water [kg mol^-{1}]

    # parameters used in the conversion
    c = np.array([610.78, 238.3, 17.2694, -6140.4, 273, 28.916])

    satP = c[0]*np.exp(c[2]*np.divide(temp,(temp+c[1]))) 
    # Saturation vapor pressure of air in given temperature [Pa]

    # convert to relative humidity using the ideal gas law pV=nRT => n=pV/RT 
    # so n=p/RT is the number of moles in a m^3, and Mw*n=Mw*p/(R*T) is the 
    # number of kg in a m^3, where Mw is the molar mass of water.    
    rh = np.divide(100*R*(temp+C2K),(Mw*satP))*vaporDens

    return rh

def satVp(temp):
    """
    Saturated vapor pressure (Pa) at temperature temp (C)
    Calculation based on 
    http://www.conservationphysics.org/atmcalc/atmoclc2.pdf
    Args:
        temp (float): Temperature in degrees Celsius.

    Returns:
        float: Saturated vapor pressure in Pascals.
    """
    # Saturation vapor pressure of air in given temperature [Pa]
    return 610.78* np.exp(17.2694*temp/(temp+238.3))


def vaporPres2rh(temp, vaporPres):
    """
    Convert vapor pressure to relative humidity.
    Args:
        temp (float): Temperature in degrees Celsius.
        vaporPres (float): Vapor pressure in Pascals.

    Returns:
        float: Relative humidity in percentage.
    """
    return ca.fmin(100*vaporPres/satVp(temp), 100)

def convert_rh_ppm(X: np.ndarray) -> np.ndarray:
    """
    Convert CO2 and vapor pressure to ppm and relative humidity.
    Takes 2D state array X as input, which contains all state variables.

    Args:
        X (np.ndarray): Input array with CO2 density and vapor pressure.

    Returns:
        np.ndarray: Converted array with CO2 concentration in ppm and relative humidity.
    """
    X[0, : ] = co2dens2ppm(X[2, :], X[0, :]*1e-6)
    X[15, :] = vaporPres2rh(X[2, :], X[15, :]).toarray().ravel()
    return X
