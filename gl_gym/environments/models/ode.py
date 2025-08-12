import casadi as ca
import numpy as np

def ODE(x: np.ndarray, u: np.ndarray, d: np.ndarray, p: np.ndarray):
    """
    Computes the time derivatives of the state variables
    for a greenhouse climate and crop model.
    
    Args:
        x (array-like): State vector containing the current values of all model variables.
        u (array-like): Control input vector.
        d (array-like): Disturbance vector (e.g., external weather conditions).
        p (array-like): Parameter vector containing model parameters and constants.
    Returns
        dxdt (casadi.SX): Vector of time derivatives for each state variable.
    """
    from gl_gym.environments.models.aux_states import update

    # Compute the auxiliary variables
    a = update(x, u, d, p)
    dxdt = ca.SX.zeros(x.shape[0])

    # Carbon concentration of main compartment [mg m^{-3} s^{-1}]
    dxdt[0] = (1./p[122]) * (a[223] + a[222] + a[224] - a[216] - a[217] - a[219])

    # Carbon concentration of top compartment [mg m^{-3} s^{-1}]
    dxdt[1] = (1./p[123]) * (a[217] - a[218])

    # Greenhouse air temperature [°C s^{-1}]
    dxdt[2] = (1./p[112]) * (a[146] + a[225] - a[235] + a[157]
        + a[226] + a[227] + a[79] - a[147] - a[148] - a[150]
        - a[151] - a[229] - a[230] - a[149]
        + a[165] + a[77]
        + a[166] + a[167] + a[78])

    # Air above screen temperature [°C s^{-1}]
    dxdt[3] = (1./p[120]) * (a[152] + a[151] - a[154] - a[155] + a[153])

    # Canopy temperature [°C s^{-1}]
    dxdt[4] = (1./a[32]) * (a[54] + a[68] + a[92] -
        a[146] - a[187] - a[84] - a[87] - a[85] - a[86] - a[108] +
        a[55] + a[69] + a[101] +
        a[105] + a[56] + a[70] + a[117])

    # Internal cover temperature [°C s^{-1}]
    dxdt[5] = (1./a[34]) * (a[154] + a[190] + a[84] +
        a[93] + a[88] + a[96] - a[164] +
        a[103] + a[110] + a[121])

    # External cover temperature [°C s^{-1}]
    dxdt[6] = (1./a[33]) * (a[80] + a[164] - a[156] - a[98])

    # Thermal screen temperature [°C s^{-1}]
    dxdt[7] = (1./p[119]) * (a[148] + a[188] + a[86] +
        a[95] + a[90] - a[152] - a[96] - a[97] + a[109] +
        a[102] + a[120])

    # Greenhouse floor temperature [°C s^{-1}]
    dxdt[8] = (1./p[113]) * (a[147] + a[74] + a[71] +
        a[87] + a[91] - a[158] - a[93] - a[94] - a[95] +
        a[75] + a[72] + a[99] - a[106] +
        a[76] + a[73] + a[115])

    # Pipe temperature [°C s^{-1}]
    dxdt[9] = (1./p[110]) * (a[220] + a[231] + a[232] - a[89] -
        a[88] - a[92] - a[91] - a[90] - a[157] +
        a[100] - a[107] + a[238] + a[116])

    # Soil layer temperatures [°C s^{-1}]
    dxdt[10] = (1./p[114]) * (a[158] - a[159])
    dxdt[11] = (1./p[115]) * (a[159] - a[160])
    dxdt[12] = (1./p[116]) * (a[160] - a[161])
    dxdt[13] = (1./p[117]) * (a[161] - a[162])
    dxdt[14] = (1./p[118]) * (a[162] - a[163])

    # Vapor pressures main compartment [Pa s^{-1}]
    dxdt[15] = (1./a[35]) * (a[176] + a[177] + a[178] + a[179] -
        a[181] - a[184] - a[186] - a[180] - a[236] - a[182])

    # Vapor pressures above screen  [Pa s^{-1}]
    dxdt[16] = (1./a[36]) * (a[184] - a[183] - a[185])

    # Top lamp temperature [°C s^{-1}]
    dxdt[17] = (1./p[184]) * (a[37] - a[165] - a[104] - a[103] -
        a[102] - a[100] - a[77] - a[112] -
        a[75] - a[72] - a[99] -
        a[55] - a[69] - a[101] - a[233] + a[118])

    # Inter lamp temperature [°C s^{-1}]
    dxdt[18] = (1./p[191]) * (a[38] - a[167] - a[122] - a[121] -
        a[120] - a[116] - a[78] - a[119] -
        a[76] - a[73] - a[115] -
        a[56] - a[70] - a[117] - a[118])

    # Grow pipe temperature [°C s^{-1}]
    dxdt[19] = (1./p[171]) * (a[221] - a[105] - a[166])

    # Blackout screen temperature [°C s^{-1}]
    dxdt[20] = (1./p[121]) * (a[149] + a[189] + a[108] +
        a[106] + a[107] - a[153] - a[110] - a[111] - a[109] +
        a[112] + a[119])

    # Average canopy temperature in last 24 hours
    dxdt[21] = (1./86400.) * (x[4] - x[21])

    # Carbohydrates in the buffer [mg{CH2O} m^{-2} s^{-1}]
    dxdt[22] = a[200] - a[208] - a[206] - a[207] - a[209]

    # Carbohydrates in the leaves [mg{CH2O} m^{-2} s^{-1}]
    dxdt[23] = a[206] - a[210] - a[214]

    # Carbohydrates in the stem [mg{CH2O} m^{-2} s^{-1}]
    dxdt[24] = a[207] - a[211]

    # Carbohydrates in fruit [mg{CH2O} m^{-2} s^{-1}]
    dxdt[25] = a[208] - a[212] - a[215]

    # # Crop development stage, aka temperature sum  [°C day s^{-1}]
    dxdt[26] = (1./86400.) * x[4]

    # time in days
    dxdt[27] = 1./86400.

    return dxdt
