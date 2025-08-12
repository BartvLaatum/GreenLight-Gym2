import casadi as ca
# import numpy as np

# from model.utils import init_state, load_dummy_weather
# from model.parameters import init_default_params

def satVP(temp):
    """Calculate the saturation vapor pressure [Pa]
    Equation 6 [1]
    """
    a = 610.78
    b = 17.2694
    c = 238.3
    return a * ca.exp(b * temp / (temp + c))

def co2dens2ppm(temp, dens):
    """Convert CO2 density to CO2 concentration [ppm]
    Equation 7 [1]
    """
    R = 8.3144598        # Molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15         # Conversion from Celsius to Kelvin [K]
    M_CO2 = 44.01e-3     # Molar mass of CO2 [kg mol^{-1}]
    P = 101325           # Pressure (assumed to be 1 atm) [Pa]
    
    return 1e6 * R * (temp + C2K) * dens / (P * M_CO2)

def tau12(tau1, tau2, rho1Dn, rho2Up):
    """Transmission coefficient of a double layer [-]
    Equation 14 [1], Equation A4 [5]
    """
    return tau1 * tau2 / (1. - rho1Dn * rho2Up)

def rhoUp(tau1, rho1Up, rho1Dn, rho2Up):
    """Reflection coefficient of the upper layer [-]
    Equation 15 [1], Equation A5 [5]
    """
    return rho1Up + (tau1 * tau1 * rho2Up) / (1. - rho1Dn * rho2Up)

def rhoDn(tau2, rho1Dn, rho2Up, rho2Dn):
    """Reflection coefficient of the lower layer [-]
    Equation 15 [1], Equation A5 [5]
    """
    return rho2Dn + (tau2 * tau2 * rho1Dn) / (1. - rho1Dn * rho2Up)

def degrees2rad(degrees):
    """Convert degrees to radians"""
    return degrees * ca.pi / 180.

def fir(a1, eps1, eps2, f12, t1, t2, sigma):
    """Net far infrared flux from 1 to 2 [W m^{-2}]
    Equation 37 [1]
    """
    return a1 * eps1 * eps2 * f12 * sigma * (ca.constpow(t1 + 273.15, 4.) - ca.constpow(t2 + 273.15, 4.))

def sensible(hec, t1, t2):
    """Sensible heat flux from 1 to 2 [W m^{-2}]
    Equation 38 [1]
    """
    return ca.fabs(hec) * (t1 - t2)

def cond(hec, vp1, vp2):
    """Condensation function"""
    a = 6.4e-9
    return 1.0 / (1.0 + ca.exp(-0.1 * (vp1 - vp2))) * a * hec * (vp1 - vp2)

def smoothHar(processVar, cutOff, smooth, maxRate):
    """Smooth harmonic function
    Args:
        processVar: Process variable
        cutOff: Cutoff value
        smooth: Smoothing parameter
        maxRate: Maximum rate
    """
    k = 2.0 * 4.6052 / smooth
    z = k * (processVar - cutOff) / 2.0
    return maxRate * (ca.tanh(z) + 1.0) / 2.0

def airMv(f12, vp1, vp2, t1, t2):
    """Vapor flux accompanying an air flux [kg m^{-2} s^{-1}]
    Equation 44 [1]
    """
    c2k = 273.15
    a = 0.002165
    return a * ca.fabs(f12) * (vp1 / (t1 + c2k) - vp2 / (t2 + c2k))

def airMc(f12, c1, c2):
    """Co2 flux accompanying an air flux [kg m^{-2} s^{-1}]
    Equation 45 [1]
    """
    return ca.fabs(f12) * (c1 - c2)

def update(x, u, d, p):
    """Update function for auxiliary variables
    Args:
        x: State vector (CasADi SX)
        u: Control vector with 6 elements (CasADi SX):
            u[0] = uBoil
            u[1] = uCo2
            u[2] = uThScr
            u[3] = uVent
            u[4] = uLamp
            u[5] = uBlScr
        d: Disturbance vector (CasADi SX)
        p: Parameter vector (CasADi SX)
    Returns:
        CasADi SX vector of auxiliary variables
    """
    # Initialize auxiliary vector with 239 elements as CasADi symbolic vector
    a = ca.SX.zeros(239)
    
    # PAR transmission coefficient of the thermal screen [-]
    # a.tauThScrPar = 1-u[2]*(1-p.tauThScrPar)
    a[0] = 1 - u[2] * (1 - p[80])
    
    # PAR reflection coefficient of the thermal screen [-]
    a[1] = u[2] * p[77]
    
    # PAR transmission coefficient of the thermal screen and roof [-]
    a[2] = tau12(p[69], a[0], p[66], a[1])

    # PAR reflection coefficient of the thermal screen and roof towards the top [-]
    a[3] = rhoUp(p[69], p[66], p[66], a[1])

    # PAR reflection coefficient of the thermal screen and roof towards the bottom [-]
    a[4] = rhoDn(a[0], p[66], a[1], a[1])

    # Thermal Screen and Roof 
    # NIR transmission coefficient of the thermal screen [-]
    a[5] = 1 - u[2] * (1 - p[79])

    # NIR reflection coefficient of the thermal screen [-]
    a[6] = u[2] * p[76]

    # NIR transmission coefficient of the thermal screen and roof [-]
    a[7] = tau12(p[68], a[5], p[65], a[6])

    # NIR reflection coefficient of the thermal screen and roof towards the top [-]
    a[8] = rhoUp(p[68], p[65], p[65], a[6])
    
    # NIR reflection coefficient of the thermal screen and roof towards the top [-]
    a[9] = rhoDn(a[5], p[65], a[6], a[6])

    # Vanthoor cover with blackout screen

    # PAR transmission coefficient of the blackout screen [-]
    # a.tauBlScrPar = u.blScrPar
    a[10] = 1 - u[5] * (1 - p[90])

    # PAR reflection coefficient of the blackout screen [-]
    # a.rhoBlScrPar
    a[11] = u[5] * p[88]

    # PAR transmission coefficient of the old cover and blackout screen [-]
    # Equation A9 [5]
    # a.tauCovBlScrPar
    a[12] = tau12(a[2], a[10], a[4], a[11])

    # PAR up reflection coefficient of the old cover and blackout screen [-]
    # Equation A10 [5]
    a[13] = rhoUp(a[2], a[3], a[4], a[11])

    # PAR down reflection coefficient of the old cover and blackout screen [-]
    # Equation A11 [5]
    a[14] = rhoDn(a[10], a[4], a[11], a[11])
    
    # NIR transmission coefficient of the blackout screen [-]
    a[15] = 1 - u[5] * (1 - p[89])
    
    # NIR reflection coefficient of the blackout screen [-]
    a[16] = u[5] * p[87]
    
    # NIR transmission coefficient of the old cover and blackout screen [-]
    a[17] = tau12(a[7], a[15], a[9], a[16])

    # NIR up reflection coefficient of the old cover and blackout screen [-]
    a[18] = rhoUp(a[7], a[8], a[9], a[16])
    
    # NIR down reflection coefficient of the old cover and blackout screen [-]
    a[19] = rhoDn(a[15], a[9], a[16], a[16])

    # ALL LAYERS OF GL MODEL

    # PAR transmission coefficient of the cover [-]
    # Equation A12 [5]
    a[20] = tau12(a[12], p[176], a[14], p[179])

    # PAR reflection coefficient of the cover [-]
    # Equation A13 [5]
    a[21] = rhoUp(a[12], a[13], a[14], p[179])

    # NIR transmission coefficient of the cover [-]
    a[22] = tau12(a[17], p[177], a[19], p[180])

    # NIR reflection coefficient of the cover [-]
    # a.rhoCovNir
    a[23] = rhoUp(a[17], a[18], a[19], p[180])

    # SINCE ONLY THE SHADING SCREEN AND THE ROOF HAVE AN EFFECT ON THE FIR TRANSMISSION AND REFLECTION
    # WE CAN SIMPLY SET THIS TO THE FIR TRANSMISSION OF THE ROOF
    # FIR transmission coefficient of the cover, excluding screens and lamps [-]
    a[24] = p[70]

    # FIR reflection coefficient of the cover, excluding screens and lamps [-]
    a[25] = p[67]

    # PAR absorption coefficient of the cover [-]
    a[26] = 1 - a[20] - a[21]
    
    # NIR absorption coefficient of the cover [-]
    a[27] = 1 - a[22] - a[23]

    # FIR absorption coefficient of the cover [-]
    a[28] = 1 - a[24] - a[25]

    # FIR emission coefficient of the cover [-]
    # See comment before equation 18 [1]
    a[29] = a[28]

    # Heat capacity of the lumped cover [J K^{-1} m^{-2}]
    # Equation 18 [1]

    # SINCE U[9] IS ALWAYS ZERO IF WE DONT USE PERMANENT SHADING SCREEN WE CAN EASILY CHANGE THIS TO
    # a.capCov = cos(degrees2rad(p.psi)) * (u[9] * p.hShScrPer * p.rhoShScrPer * p.cPShScrPer + p[73] * p.rhoRf * p.cPRf)
    a[30] = ca.cos(degrees2rad(p[45])) * p[73] * p[64] * p[72] # COULD THIS BE A PROBLEM

    # Capacities - Section 4 [1] ####

    # Leaf area index [m^2{leaf} m^{-2}]
    # Equation 5 [2]
    a[31] = p[142] * x[23]

    # Heat capacity of canopy [J K^{-1} m^{-2}]
    # Equation 19 [1]
    a[32] = p[16] * a[31]

    # Heat capacity of external and internal cover [J K^{-1} m^{-2}]
    # Equation 20 [1]
    a[33] = 0.1 * a[30]
    a[34] = 0.1 * a[30]

    # Vapor capacity of main compartment [kg m J^{-1}] 
    # Equation 24 [1]
    a[35] = p[38] * p[48] / (p[39] * (x[2] + 273.15))

    # Vapor capacity of top compartment [kg m J^{-1}] 
    a[36] = p[38] * (p[49] - p[48]) / (p[39] * (x[3] + 273.15))

    # Global, PAR, and NIR heat fluxes - Section 5.1 [1]

    # Lamp electrical input [W m^{-2}]
    # Equation A16 [5]
    a[37] = p[172] * u[4]

    # Interlight electrical input [W m^{-2}]
    # Equation A26 [5]
    # a[38] = p[196] * u[5]
    a[38] = 0

    # PAR above the canopy from the sun [W m^{-2}]
    # Equation 27 [1], Equation A14 [5]
    # a.rParSun
    a[39] = (1 - p[44]) * a[20] * p[6] * d[0]

    # PAR above the canopy from the lamps [W m^{-2}] 
    # Equation A15 [5]
    a[40] = p[174] * a[37]
    
    # PAR outside the canopy from the interlights [W m^{-2}] 
    # Equation 7.7, 7.14 [7]
    a[41] = p[192] * a[38]
    
    # Global radiation above the canopy from the sun [W m^{-2}]
    # (PAR+NIR, where UV is counted together with NIR)
    # Equation 7.24 [7]
    # a.rCanSun
    a[42] = (1 - p[44]) * d[0] * (p[6] * a[20] + p[5] * a[22])
    
    # Global radiation above the canopy from the lamps [W m^{-2}]
    # (PAR+NIR, where UV is counted together with NIR)
    # Equation 7.25 [7]
    a[43] = (p[174] + p[175]) * a[37]

    # Global radiation outside the canopy from the interlight lamps [W m^{-2}]
    # (PAR+NIR, where UV is counted together with NIR)
    # Equation 7.26 [7]
    a[44] = (p[192] + p[193]) * a[38]

    # Global radiation above and outside the canopy [W m^{-2}]
    # (PAR+NIR, where UV is counted together with NIR)
    # Equation 7.23 [7]
    a[45] = a[42] + a[43] + a[44]

    # PAR from the sun directly absorbed by the canopy [W m^{-2}]
    # Equation 26 [1]
    a[46] = a[39] * (1 - p[10]) * (1 - ca.exp(-p[32] * a[31]))

    # PAR from the lamps directly absorbed by the canopy [W m^{-2}]
    # Equation A17 [5]
    a[47] = a[40] * (1 - p[10]) * (1 - ca.exp(-p[32] * a[31]))


    # Fraction of PAR from the interlights reaching the canopy [-]
    # Equation 7.13 [7]
    a[48] = (1 - p[190] * ca.exp(-p[200] * p[189] * a[31]) + 
             (p[190] - 1) * ca.exp(-p[200] * (1 - p[189]) * a[31]))

    # Fraction of NIR from the interlights reaching the canopy [-]
    # Analogous to Equation 7.13 [7]
    a[49] = (1 - p[190] * ca.exp(-p[202] * p[189] * a[31]) + 
             (p[190] - 1) * ca.exp(-p[202] * (1 - p[189]) * a[31]))

    # PAR from the interlights directly absorbed by the canopy [W m^{-2}]
    # Equation 7.16 [7]
    a[50] = a[41] * a[48] * (1 - p[10])
    
    # PAR from the sun absorbed by the canopy after reflection from the floor [W m^{-2}]
    # Equation 28 [1]
    # addAux(gl, 'rParSunFlrCanUp', mulNoBracks(gl.a[39], ca.exp(-p[32]*gl.a[31])*p[98]* 
    #     (1-p[10]).*(1-exp(-p[33]*gl.a[31]))))
    a[51] = a[39] * ca.exp(-p[32] * a[31]) * p[98] * (1 - p[10]) * (1 - ca.exp(-p[33] * a[31]))

    # PAR from the lamps absorbed by the canopy after reflection from the floor [W m^{-2}]
    # Equation A18 [5]
    a[52] = a[40] * ca.exp(-p[32] * a[31]) * p[98] * (1 - p[10]) * (1 - ca.exp(-p[33] * a[31]))

    # PAR from the interlights absorbed by the canopy after reflection from the floor [W m^{-2}]
    # Equation 7.18 [7]
    a[53] = a[41] * p[190] * ca.exp(-p[200] * p[189] * a[31]) * p[98] * (1 - p[10]) * (1 - ca.exp(-p[201] * a[31]))
        # if p[189]==1, the lamp is above the canopy, light loses
        # ca.exp(-k*LAI) on its way to the floor.
        # if p[189]==0, the lamp is below the canopy, no light is
        # lost on the way to the floor
    
    # Total PAR from the sun absorbed by the canopy [W m^{-2}]
    # Equation 25 [1]
    #  a.rParSunCan = a.rParSunCanDown + a.rParSunFlrCanUp
    #  a[54] = a[46] + a[51]
    a[54] = a[46] + a[51]
    
    # Total PAR from the lamps absorbed by the canopy [W m^{-2}]
    # Equation A19 [5]
    a[55] = a[47] + a[52]

    # Total PAR from the interlights absorbed by the canopy [W m^{-2}]
    # Equation A19 [5], Equation 7.19 [7]
    a[56] = a[50] + a[53]

    # Virtual NIR transmission for the cover-canopy-floor lumped model [-]
    # Equation 29 [1]
    #  a.tauHatCovNir
    a[57] = 1 - a[23]
    a[58] = 1 - p[97]

    # NIR transmission coefficient of the canopy [-]
    # Equation 30 [1]
    #  a.tauHatCanNir 
    a[59] = ca.exp(-p[34] * a[31])

    # NIR reflection coefficient of the canopy [-]
    # Equation 31 [1]
    #  a.rhoHatCanNir
    a[60] = p[11] * (1 - a[59])

    # NIR transmission coefficient of the cover and canopy [-]
    #  a.tauCovCanNir
    a[61] = tau12(a[57], a[59], a[23], a[60])

    # NIR reflection coefficient of the cover and canopy towards the top [-]
    # a.rhoCovCanNirUp
    a[62] = rhoUp(a[57], a[23], a[23], a[60])

    # NIR reflection coefficient of the cover and canopy towards the bottom [-]
    #  a.rhoCovCanNirDn
    a[63] = rhoDn(a[59], a[23], a[60], a[60])

    # NIR transmission coefficient of the cover, canopy and floor [-]
    #  a.tauCovCanFlrNir
    a[64] = tau12(a[61], a[58], a[63], p[97])

    # NIR reflection coefficient of the cover, canopy and floor [-]
    #  a.rhoCovCanFlrNir
    a[65] = rhoUp(a[61], a[62], a[63], p[97])

    # The calculated absorption coefficient equals a[66] [-]
    #  a.aCanNir
    a[66] = 1 - a[64] - a[65]

    # The calculated transmission coefficient equals m.a[67] [-]
    # pg. 23 [1]
    # a.aFlrNir
    a[67] = a[64]

    # NIR from the sun absorbed by the canopy [W m^{-2}]
    # Equation 32 [1]
    # a.rNirSunCan = (1-p[44]).*gl.a[66].*p[5].*d.iGlob)
    a[68] = (1 - p[44]) * a[66] * p[5] * d[0]
    
    # NIR from the lamps absorbed by the canopy [W m^{-2}]
    # Equation A20 [5]
    # addAux(gl, 'rNirLampCan', p[175].*gl.a[37].*(1-p[11]).*(1-exp(-p[34]*gl.a[31])))
    a[69] = p[175] * a[37] * (1 - p[11]) * (1 - ca.exp(-p[34] * a[31]))

    # NIR from the interlights absorbed by the canopy [W m^{-2}]
    # Equation 7.20 [7]
    # addAux(gl, 'rNirIntLampCan', p[193].*gl.a[38].*gl.a[49].*(1-p[11]))
    a[70] = p[193] * a[38] * a[49] * (1 - p[11])

    # NIR from the sun absorbed by the floor [W m^{-2}]
    # Equation 33 [1]
    # a.rNirSunFlr = (1-p[44]).*gl.a[67].*p[5].*d.iGlob)
    a[71] = (1 - p[44]) * a[67] * p[5] * d[0]

    # NIR from the lamps absorbed by the floor [W m^{-2}]
    # Equation A22 [5]
    # addAux(gl, 'rNirLampFlr', (1-p[97]).*exp(-p[34]*gl.a[31]).*p[175].*gl.a[37])
    a[72] = (1 - p[97]) * ca.exp(-p[34] * a[31]) * p[175] * a[37]

    # NIR from the interlights absorbed by the floor [W m^{-2}]
    # Equation 7.21 [7]
    a[73] = p[190] * (1 - p[97]) * ca.exp(-p[202] * a[31] * p[189]) * p[193] * a[38]
    # if p[189]==1, the lamp is above the canopy, light loses
    # ca.exp(-k*LAI) on its way to the floor.
    # if p[189]==0, the lamp is below the canopy, no light is
    # lost on the way to the floor
    
    # PAR from the sun absorbed by the floor [W m^{-2}]
    # Equation 34 [1]
    # addAux(gl, 'rParSunFlr', (1-p[98]).*exp(-p[32]*gl.a[31]).*gl.a[39])
    a[74] = (1 - p[98]) * ca.exp(-p[32] * a[31]) * a[39]

    # PAR from the lamps absorbed by the floor [W m^{-2}]
    # Equation A21 [5]
    # addAux(gl, 'rParLampFlr', (1-p[98]).*exp(-p[32]*gl.a[31]).*gl.a[40])
    a[75] = (1 - p[98]) * ca.exp(-p[32] * a[31]) * a[40]

    # PAR from the interlights absorbed by the floor [W m^{-2}]
    # Equation 7.17 [7]
    # addAux(gl, 'rParIntLampFlr', gl.a[41].*p[190].*(1-p[98]).*
    #     ca.exp(-p[200]*gl.a[31].*p[189]))
    a[76] = a[41] * p[190] * (1 - p[98]) * ca.exp(-p[200] * a[31] * p[189])
	# PAR and NIR from the lamps absorbed by the greenhouse air [W m^{-2}]
    # Equation A23 [5]
	# addAux(gl, 'rLampAir', (p[174]+p[175])*gl.a[37] - gl.a[55] - 
	# 	gl.a[69] - gl.a[75] - gl.a[72])

    a[77] = (p[174] + p[175]) * a[37] - a[55] - a[69] - a[75] - a[72]
	
    # PAR and NIR from the interlights absorbed by the greenhouse air [W m^{-2}]
    # Equation 7.22 [7]
    a[78] = (p[192] + p[193]) * a[38] - a[56] - a[70] - a[76] - a[73]
    
    # Global radiation from the sun absorbed by the greenhouse air [W m^{-2}]
    # Equation 35 [1]
    #  a.rGlobSunAir
    a[79] = p[44] * d[0] * (a[20] * p[6] + (a[66] + a[67]) * p[5])

    # Global radiation from the sun absorbed by the cover [W m^{-2}]
    # Equation 36 [1]
    #  a.rGlobSunCovE = gl.a[26]*p[6]+gl.a[27]*p[5]).*d.iGlob
    a[80] = (a[26] * p[6] + a[27] * p[5]) * d[0]
    # FIR heat fluxes - Section 5.2 [1]

    # FIR transmission coefficient of the thermal screen
    # Equation 38 [1]
    # addAux(gl, 'tauThScrFirU', (1-u.thScr*(1-p[81])))
    a[81] = 1 - u[2] * (1 - p[81])
    # FIR transmission coefficient of the blackout screen
    # addAux(gl, 'tauBlScrFirU', (1-u.blScr*(1-p[91])))   
    a[82] = 1 - u[5] * (1 - p[91])

    # Surface of canopy per floor area [-]
    # Table 3 [1]
    # addAux(gl, 'aCan', 1-exp(-p[35]*gl.a[31]))
    a[83] = 1 - ca.exp(-p[35] * a[31])

    # FIR between greenhouse objects [W m^{-2}]
    # Table 7.4 [7]. Based on Table 3 [1] and Table A1 [5]

    # FIR between canopy and cover [W m^{-2}]
    # addAux(gl, 'rCanCovIn', fir(gl.a[83], p[3], gl.a[29], 
    #     p[178]*gl.a[81].*gl.a[82],
    #     x.tCan, x.tCovIn))
    a[84] = fir(a[83], p[3], a[29], p[178] * a[81] * a[82], x[4], x[5], p[2])

    # FIR between canopy and sky [W m^{-2}]
    # addAux(gl, 'rCanSky', fir(gl.a[83], p[3], p[4], 
    #     p[178]*gl.a[24].*gl.a[81].*gl.a[82],
    #     x.tCan, d.tSky))
    a[85] = fir(a[83], p[3], p[4], p[178] * a[24] * a[81] * a[82], x[4], d[5], p[2])

    # FIR between canopy and thermal screen [W m^{-2}]
    # addAux(gl, 'rCanThScr', fir(gl.a[83], p[3], p[74], 
    #     p[178]*u.thScr.*gl.a[82], x.tCan, x.tThScr))
    a[86] = fir(a[83], p[3], p[74], p[178] * u[2] * a[82], x[4], x[7], p[2])

    # FIR between canopy and floor [W m^{-2}]
    # addAux(gl, 'rCanFlr', fir(gl.a[83], p[3], p[95], 
    #     p[125], x.tCan, x.tFlr))
    a[87] = fir(a[83], p[3], p[95], p[125], x[4], x[8], p[2])

    # FIR between pipes and cover [W m^{-2}]
    # addAux(gl, 'rPipeCovIn', fir(p.aPipe, p[104], gl.a[29], 
    #     p[199]*p[178]*gl.a[81].*gl.a[82]*0.49.*
    #     ca.exp(-p[35]*gl.a[31]), x.tPipe, x.tCovIn))
    a[88] = fir(p[124], p[104], a[29], p[199] * p[178] * a[81] * a[82] * 0.49 * ca.exp(-p[35] * a[31]), x[9], x[5], p[2])

    # FIR between pipes and sky [W m^{-2}]
    # addAux(gl, 'rPipeSky', fir(p[124], p[104], p[4], 
    #     p[199]*p[178]*gl.a[24].*gl.a[81].*
    #     gl.a[82]*0.49.*exp(-p[35]*gl.a[31]), x.tPipe, d.tSky))
    a[89] = fir(p[124], p[104], p[4], p[199] * p[178] * a[24] * a[81] * 0.49 * ca.exp(-p[35] * a[31]), x[9], d[5], p[2])
    # FIR between pipes and thermal screen [W m^{-2}]
    # addAux(gl, 'rPipeThScr', fir(p[124], p[104], p[74], 
    #     p[199]*p[178]*u.thScr.*gl.a[82]*0.49.*
    #     ca.exp(-p[35]*gl.a[31]), x.tPipe, x.tThScr))
    a[90] = fir(p[124], p[104], p[74], p[199] * p[178] * u[2] * a[82] * 0.49 * ca.exp(-p[35] * a[31]), x[9], x[7], p[2])

    # FIR between pipes and floor [W m^{-2}]
    # addAux(gl, 'rPipeFlr', fir(p[124], p[104], p[95], 0.49, x.tPipe, x.tFlr))
    a[91] = fir(p[124], p[104], p[95], 0.49, x[9], x[8], p[2])
    # FIR between pipes and canopy [W m^{-2}]
    # addAux(gl, 'rPipeCan', fir(p[124], p[104], p[3], 
    #     0.49.*(1-exp(-p[35]*gl.a[31])), x.tPipe, x.tCan))
    a[92] = fir(p[124], p[104], p[3], 0.49 * (1 - ca.exp(-p[35] * a[31])), x[9], x[4], p[2])

    # FIR between floor and cover [W m^{-2}]
    # fir(1, p[95], gl.a[29], 
    #     p[199]*p[178]*gl.a[81].*gl.a[82]*
    #     (1-0.49*pi*p[107]*p[105]).*exp(-p[35]*gl.a[31]), x.tFlr, x.tCovIn)
    a[93] = fir(1, p[95], a[29], p[199] * p[178] * a[81] * a[82] * (1 - 0.49 * ca.pi * p[107] * p[105]) * ca.exp(-p[35] * a[31]), x[8], x[5], p[2])
    # FIR between floor and sky [W m^{-2}]
    # addAux(gl, 'rFlrSky', fir(1, p[95], p[4], 
    #     p[199]*p[178]*gl.a[24].*gl.a[81].*gl.a[82]*
    #     (1-0.49*pi*p[107]*p[105]).*exp(-p[35]*gl.a[31]), x.tFlr, d.tSky))
    a[94] = fir(1, p[95], p[4], p[199] * p[178] * a[24] * a[81] * a[82] * (1 - 0.49 * ca.pi * p[107] * p[105]) * ca.exp(-p[35] * a[31]), x[8], d[5], p[2])

    # FIR between floor and thermal screen [W m^{-2}]
    # addAux(gl, 'rFlrThScr', fir(1, p[95], p[74], 
    #     p[199]*p[178]*u.thScr.*gl.a[82]*(1-0.49*pi*p[107]*p[105]).*
    #     ca.exp(-p[35]*gl.a[31]), x.tFlr, x.tThScr))
    a[95] = fir(1, p[95], p[74], p[199] * p[178] * u[2] * a[82] * (1 - 0.49 * ca.pi * p[107] * p[105]) * ca.exp(-p[35] * a[31]), x[8], x[7], p[2])

    # FIR between thermal screen and cover [W m^{-2}]
    # addAux(gl, 'rThScrCovIn', fir(1, p[74], gl.a[29], 
    #     u.thScr, x.tThScr, x.tCovIn))
    a[96] = fir(1, p[74], a[29], u[2], x[7], x[5], p[2])

    # FIR between thermal screen and sky [W m^{-2}]
    # addAux(gl, 'rThScrSky', fir(1, p[74], p[4], 
    #     gl.a[24].*u.thScr, x.tThScr, d.tSky))
    a[97] = fir(1, p[74], p[4], a[24] * u[2], x[7], d[5], p[2])

    # FIR between cover and sky [W m^{-2}]
    # addAux(gl, 'rCovESky', fir(1, gl.a[28], p[4], 1, x.tCovE, d.tSky))
    a[98] = fir(1, a[28], p[4], 1, x[6], d[5], p[2])

    # FIR between lamps and floor [W m^{-2}]
    # addAux(gl, 'rFirLampFlr', fir(p[181], p[183], p[95], 
    #     p[199].*(1-0.49*pi*p[107]*p[105]).*exp(-p[35]*gl.a[31]), x.tLamp, x.tFlr))
    a[99] = fir(p[181], p[183], p[95], p[199] * (1 - 0.49 * ca.pi * p[107] * p[105]) * ca.exp(-p[35] * a[31]), x[17], x[8], p[2])

    # FIR between lamps and pipe [W m^{-2}]
    # addAux(gl, 'rLampPipe', fir(p[181], p[183], p[104], 
    #     p[199].*0.49*pi*p[107]*p[105].*exp(-p[35]*gl.a[31]), x.tLamp, x.tPipe))
    a[100] = fir(p[181], p[183], p[104], p[199] * 0.49 * ca.pi * p[107] * p[105] * ca.exp(-p[35] * a[31]), x[17], x[9], p[2])
    # FIR between lamps and canopy [W m^{-2}]
    # addAux(gl, 'rFirLampCan', fir(p[181], p[183], p[3], 
    #     gl.a[83], x.tLamp, x.tCan))
    a[101] = fir(p[181], p[183], p[3], a[83], x[17], x[4], p[2])

    # FIR between lamps and thermal screen [W m^{-2}]
    # addAux(gl, 'rLampThScr', fir(p[181], p[182], p[74], 
    #     u.thScr.*gl.a[82], x.tLamp, x.tThScr))
    a[102] = fir(p[181], p[182], p[74], u[2] * a[82], x[17], x[7], p[2])
    # FIR between lamps and cover [W m^{-2}]
    # addAux(gl, 'rLampCovIn', fir(p[181], p[182], gl.a[29], 
    #     gl.a[81].*gl.a[82], x.tLamp, x.tCovIn))
    a[103] = fir(p[181], p[182], a[29], a[81] * a[82], x[17], x[5], p[2])

    # FIR between lamps and sky [W m^{-2}]
    # addAux(gl, 'rLampSky', fir(p[181], p[182], p[4], 
    #     gl.a[24].*gl.a[81].*gl.a[82], x.tLamp, d.tSky))
    a[104] = fir(p[181], p[182], p[4], a[24] * a[81] * a[82], x[17], d[5], p[2])

    # FIR between grow pipes and canopy [W m^{-2}]
    # addAux(gl, 'rGroPipeCan', fir(p[169], p(165), p[3], 1, x.tGroPipe, x.tCan))
    a[105] = fir(p[169], p[165], p[3], 1, x[19], x[4], p[2])

    # FIR between blackout screen and floor [W m^{-2}]	
    # addAux(gl, 'rFlrBlScr', fir(1, p[95], p[85], 
    #     p[199]*p[178]*u.blScr*(1-0.49*pi*p[107]*p[105]).*
    #     ca.exp(-p[35]*gl.a[31]), x.tFlr, x.tBlScr))
    a[106] = fir(1, p[95], p[85], p[199] * p[178] * u[5] * (1 - 0.49 * ca.pi * p[107] * p[105]) * ca.exp(-p[35] * a[31]), x[8], x[20], p[2])

    # FIR between blackout screen and pipe [W m^{-2}]
    # addAux(gl, 'rPipeBlScr', fir(p[124], p[104], p[85], 
    #     p[199]*p[178]*u.blScr*0.49.*exp(-p[35]*gl.a[31]), x.tPipe, x.tBlScr))
    a[107] = fir(p[124], p[104], p[85], p[199] * p[178] * u[5] * 0.49 * ca.exp(-p[35] * a[31]), x[9], x[20], p[2])

    # FIR between blackout screen and canopy [W m^{-2}]
    # addAux(gl, 'rCanBlScr', fir(gl.a[83], p[3], p[85], 
    #     p[178]*u.blScr, x.tCan, x.tBlScr))
    a[108] = fir(a[83], p[3], p[85], p[178] * u[5], x[4], x[20], p[2])

    # FIR between blackout screen and thermal screen [W m^{-2}]
    # addAux(gl, 'rBlScrThScr', fir(u.blScr, p[85], 
    #     p[74], u.thScr, x.tBlScr, x.tThScr))
    a[109] = fir(u[5], p[85], p[74], u[2], x[20], x[7], p[2])

    # FIR between blackout screen and cover [W m^{-2}]
    # addAux(gl, 'rBlScrCovIn', fir(u.blScr, p[85], gl.a[29], 
    #     gl.a[81], x.tBlScr, x.tCovIn))
    a[110] = fir(u[5], p[85], a[29], a[81], x[20], x[5], p[2])

    # FIR between blackout screen and sky [W m^{-2}]
    # addAux(gl, 'rBlScrSky', fir(u.blScr, p[85], p[4], 
    #     gl.a[24].*gl.a[81], x.tBlScr, d.tSky))
    a[111] = fir(u[5], p[85], p[4], a[24] * a[81], x[20], d[5], p[2])

    # FIR between blackout screen and lamps [W m^{-2}]
    # addAux(gl, 'rLampBlScr', fir(p[181], p[182], p[85], 
    #     u.blScr, x.tLamp, x.tBlScr))
    a[112] = fir(p[181], p[182], p[85], u[5], x[17], x[20], p[2])

    # Fraction of radiation going up from the interlight to the canopy [-]
    # Equation 7.29 [7]
    # addAux(gl, 'fIntLampCanUp', 1-exp(-p[203]*(1-p[189]).*gl.a[31]))
    a[113] = 1 - ca.exp(-p[203] * (1 - p[189]) * a[31])

    # Fraction of radiation going down from the interlight to the canopy [-]
    # Equation 7.30 [7]
    # addAux(gl, 'fIntLampCanDown', 1-exp(-p[203]*p[189].*gl.a[31]))
    a[114] = 1 - ca.exp(-p[203] * p[189] * a[31])

    # FIR between interlights and floor [W m^{-2}]
    # addAux(gl, 'rFirIntLampFlr', fir(p[194], p[195], p[95], 
    #     (1-0.49*pi*p[107]*p[105]).*(1-gl.a[114]),
    #     x.tIntLamp, x.tFlr))
    a[115] = fir(p[194], p[195], p[95], (1 - 0.49 * ca.pi * p[107] * p[105]) * (1 - a[114]), x[18], x[8], p[2])

    # FIR between interlights and pipe [W m^{-2}]
    # addAux(gl, 'rIntLampPipe', fir(p[194], p[195], p[104], 
    #     0.49*pi*p[107]*p[105].*(1-gl.a[114]),
    #     x.tIntLamp, x.tPipe))
    a[116] = fir(p[194], p[195], p[104], 0.49 * ca.pi * p[107] * p[105] * (1 - a[114]), x[18], x[9], p[2])

    # FIR between interlights and canopy [W m^{-2}]
    # addAux(gl, 'rFirIntLampCan', fir(p[194], p[195], p[3], 
    #     gl.a[114]+gl.a[113], x.tIntLamp, x.tCan))
    a[117] = fir(p[194], p[195], p[3], a[114] + a[113], x[18], x[4], p[2])

    # FIR between interlights and toplights [W m^{-2}]
    # addAux(gl, 'rIntLampLamp', fir(p[194], p[195], p[183], 
    #     (1-gl.a[113]).*p[181], x.tIntLamp, x.tLamp))
    a[118] = fir(p[194], p[195], p[183], (1 - a[113]) * p[181], x[18], x[17], p[2])

    # FIR between interlights and blackout screen [W m^{-2}]
    # addAux(gl, 'rIntLampBlScr', fir(p[194], p[195], p[85], 
    #     u.blScr.*p[178].*(1-gl.a[113]), x.tIntLamp, x.tBlScr))
    a[119] = fir(p[194], p[195], p[85], u[5] * p[178] * (1 - a[113]), x[18], x[20], p[2])
        # if p[189]==0, the lamp is above the canopy, no light is
        # lost on its way up
        # if p[189]==1, the lamp is below the canopy, the light
        # loses ca.exp(-k*LAI) on its way up

    # FIR between interlights and thermal screen [W m^{-2}]
    # addAux(gl, 'rIntLampThScr', fir(p[194], p[195], p[74], 
    #     u.thScr.*gl.a[82].*p[178].*(1-gl.a[113]),
    #     x.tIntLamp, x.tThScr))
    a[120] = fir(p[194], p[195], p[74], u[2] * a[82] * p[178] * (1 - a[113]), x[18], x[7], p[2])

    # FIR between interlights and cover [W m^{-2}]
    # addAux(gl, 'rIntLampCovIn', fir(p[194], p[195], gl.a[29], 
    #     gl.a[81].*gl.a[82].*p[178].*(1-gl.a[113]),
    #     x.tIntLamp, x.tCovIn))
    a[121] = fir(p[194], p[195], a[29], a[81] * a[82] * p[178] * (1 - a[113]), x[18], x[5], p[2])

    # FIR between interlights and sky [W m^{-2}]
    # addAux(gl, 'rIntLampSky', fir(p[194], p[195], p[4], 
    #     gl.a[24].*gl.a[81].*gl.a[82].*p[178].*(1-gl.a[113]),
    #     x.tIntLamp, d.tSky))
    a[122] = fir(p[194], p[195], p[4], a[24] * a[81] * a[82] * p[178] * (1 - a[113]), x[18], d[5], p[2])

    #  Natural Ventilation

    # Aperature of the roof
    # Aperture of the roof [m^{2}]
    # Equation 67 [1]
    a[123] = u[3] * p[55]
    a[124] = p[55]
    a[125] = 0

    # Aperture of the sidewall [m^{2}]
    # Equation 68 [1] 
    # (this is 0 in the Dutch greenhouse)
    ## SINCE WE DON'T USE GREENHOUSE 
    # a[126] = u[10]*p.aSide
    a[126] = 0

    # Ratio between roof vent area and total ventilation area [-]
    # (not very clear in the reference [1], but always 1 if m.a[126] == 0)
    # a.etaRoof
    a[127] = 1
    # a.etaRoofNoSide
    a[128] = 1

    # Ratio between side vent area and total ventilation area [-]
    # (not very clear in the reference [1], but always 0 if m.a[126] == 0)    
    a[129] = 0

    # Discharge coefficient [-]
    # Equation 73 [1]
    # SINCE SHADING SCREEN IS ALWAYS = 0 WE CAN CHANGE cD = p[59]
    # a[130] = p[59] * (1 - p.etaShScrCd*u[8])
    a[130] = p[59]

    # Discharge coefficient [-]
    # Equation 74 [-]
    # addAux(gl, 'cW', p[61]*(1-p.etaShScrCw*u.shScr))
    # SINCE SHADING SCREEN IS ALWAYS = 0 WE CAN CHANGE cW = p[61]
    # a[131] = p[61] * (1 - p.etaShScrCw*u[8])
    a[131] = p[61]

    # Natural ventilation rate due to roof ventilation [m^{3} m^{-2} s^{-1}]
    # Equation 64 [1]
    # fVentRoof2
    a[132] = u[3] * p[55] * a[130] / (2. * p[46]) *\
        ca.sqrt(ca.fabs(p[26] * p[56] * (x[2] - d[1]) / (2. * (0.5 * x[2] + 0.5 * d[1] + 273.15)) + a[131] * (d[4] * d[4])))

    # BOTH UNUSED BY THE MODEL
    # a[136]2Max = p[55] * a[130]/(2*p[46]) * 
    #     sqrt(fabs(p[26]*p[56] * (x[2]-d[1]) / (2*(0.5*x[2] + 0.5*d[1] + 273.15)) + a[131]*d[4]**2))
    # a[136]2Min = 0

    # Ventilation rate through roof and side vents [m^{3} m^{-2} s^{-1}]
    # Equation 65 [1]
    a[133] = a[130] / p[46] * \
        ca.sqrt(
            1e-8 + ca.constpow((a[123] * a[126] / ca.sqrt(ca.fmax(a[123]*a[123] + a[126]*a[126], 0.01))), 2) *
            (2 * p[26] * p[62] * (x[2] - d[1]) / (0.5 * x[2] + 0.5 * d[1] + 273.15)) + 
            (ca.constpow((a[123] + a[126]/2.), 2) * a[131] * (d[4] * d[4]))
        )

    # Ventilation rate through sidewall only [m^{3} m^{-2} s^{-1}]
    # Equation 66 [1]
    # THIS COULD BE SET TO 0 SINCE a[126] = 0
    a[134] = a[130] * a[126] * d[4] / (2*p[46]) * ca.sqrt(a[131])

    # Leakage ventilation [m^{3} m^{-2} s^{-1}]
    # Equation 70 [1]
    # addAux(gl, 'fLeakage', ifElse('d.wind<p[205]',p[205]*p[60],p[60]*d.wind))
    a[135] = ca.if_else(
        d[4] < p[205],
        p[205] * p[60],
        p[60] * d[4]
    )

    # # Total ventilation through the roof [m^{3} m^{-2} s^{-1}]
    # # Equation 71 [1], Equation A42 [5]
    # addAux(gl, 'fVentRoof', ifElse([getDefStr(gl.a.etaRoof) '>= etaRoofThr'], p.etaInsScr*gl.a.fVentRoof2+p[204]*gl.a[135],
    #     p[57]*(max(u.thScr,u.blScr).*gl.a[132]+(1-max(u.thScr,u.blScr)).*gl.a[133].*gl.a[127])
    #     +p[204]*gl.a[135]))
    a[136] = ca.if_else(
        a[127] >= p[8],
        p[57] * a[132] + p[204] * a[135],
        p[57] * (ca.fmax(u[2], u[5]) * a[132] + (1 - ca.fmax(u[2], u[5])) * a[133] * a[127]) + p[204] * a[135]
    )

    # # Total ventilation through side vents [m^{3} m^{-2} s^{-1}]
    # # Equation 72 [1], Equation A43 [5]
    a[137] = ca.if_else(
        a[127] >= p[8],
        p[57] * a[134] + (1 - p[204]) * a[135],
        p[57] * (ca.fmax(u[2], u[5]) * a[134] + (1 - ca.fmax(u[2], u[5])) * a[133] * a[129]) + (1 - p[204]) * a[135]
    )

    # CO2 concentration in main compartment [ppm]
    a[138] = co2dens2ppm(x[2], 1e-6*x[0])

    #  Convection and conduction
    # density of air as it depends on pressure and temperature, see
    # https:# en.wikipedia.org/wiki/Density_of_air
    a[139] = p[36] * p[126] / ((x[3] + 273.15) * p[39])
    a[140] = p[36] * p[126] / ((x[2] + 273.15) * p[39])

    # See [4], where rhoMean is "the mean
    # density of air beneath and above the screen".
    a[141] = 0.5 * (a[139] + a[140])

    # Air flux through the thermal screen [m s^{-1}]
    # Equation 40 [1], Equation A36 [5]
    # There is a mistake in [1], see equation 5.68, pg. 91, [4]
    # tOut, rhoOut, should be tTop, rhoTop
    # There is also a mistake in [4], whenever sqrt is taken, abs should be included
    # addAux(gl, 'fThScr', u.thScr*p[84].*(abs((x.tAir-x.tTop)).^0.66) +  
    #     ((1-u.thScr)./gl.a[141]).*sqrt(0.5*gl.a[141].*(1-u.thScr).*p[26].*abs(gl.a[140]-gl.a[139])))
    a[142] = u[2] * p[84] * ca.constpow(ca.fabs(x[2] - x[3] + 1e-10), 0.66) + \
        ((1. - u[2]) / a[141]) * ca.sqrt(0.5 * a[141] * (1. - u[2]) * p[26] * ca.fabs(a[140] - a[139]) + 1e-10)

    # Air flux through the blackout screen [m s^{-1}]
    # Equation A37 [5]
    # addAux(gl, 'fBlScr', u.blScr*p[94].*(abs((x.tAir-x.tTop)).^0.66) +  
    #     ((1-u.blScr)./gl.a[141]).*sqrt(0.5*gl.a[141].*(1-u.blScr).*p[26].*abs(gl.a[140]-gl.a[139])))
    a[143] = u[5] * p[94] * ca.constpow(ca.fabs(x[2] - x[3] + 1e-10), 0.66) + \
        ((1. - u[5]) / a[141]) * ca.sqrt(0.5 * a[141] * (1. - u[5]) * p[26] * ca.fabs(a[140] - a[139]) + 1e-10)

    # Air flux through the screens [m s^{-1}]
    # Equation A38 [5]
    # addAux(gl, 'fScr', min(gl.a[142],gl.a[143]))
    a[144] = ca.fmin(a[142], a[143])

    #  Convective and conductive heat fluxes [W m^{-2}] ####

    # # Forced ventilation (doesn't exist in current gh)
    # addAux(gl, 'fVentForced', DynamicElement('0', 0))
    a[145] = 0

    # # Between canopy and air in main compartment [W m^{-2}]
    # addAux(gl, 'hCanAir', sensible(2*p.alfaLeafAir*gl.a[31], x.tCan, x.tAir))
    a[146] = sensible(2 * p[0] * a[31], x[4], x[2])

    # # Between air in main compartment and floor [W m^{-2}]
    a[147] = ca.if_else(
        x[8] > x[2],
        sensible(1.7 * ca.constpow(ca.fabs(x[8] - x[2] + 1e-10), (1./3.)), x[2], x[8]),
        sensible(1.3 * ca.constpow(ca.fabs(x[2] - x[8] + 1e-10), (1./4.)), x[2], x[8])
    )
    # # Between air in main compartment and thermal screen [W m^{-2}]
    # addAux(gl, 'hAirThScr', sensible(1.7.*u.thScr.*nthroot(abs(x.tAir-x.tThScr),3),
    #     x.tAir,x.tThScr))
    a[148] = sensible(1.7 * u[2] * ca.constpow(ca.fabs(x[2] - x[7] + 1e-10), (1./3.)), x[2], x[7])

    # # Between air in main compartment and blackout screen [W m^{-2}]
    # # Equations A28, A32 [5]
    # addAux(gl, 'hAirBlScr', sensible(1.7.*u.blScr.*nthroot(abs(x.tAir-x.tBlScr),3),
    #     x.tAir,x.tBlScr))

    a[149] = sensible(1.7 * u[5] * ca.constpow(ca.fabs(x[2] - x[20] + 1e-10), (1./3.)), x[2], x[20])
        
    # # Between air in main compartment and outside air [W m^{-2}]
    # addAux(gl, 'hAirOut', sensible(p[111]*p[23]*(gl.a[137]+gl.a[145]),
#     x.tAir, d.tOut))
    a[150] = sensible(p[111] * p[23] * (a[137] + a[145]), x[2], d[1])
        
    # # Between air in main and top compartment [W m^{-2}]
    # addAux(gl, 'hAirTop', sensible(p[111]*p[23]*gl.a[144], x.tAir, x.tTop))
    a[151] = sensible(p[111] * p[23] * a[144], x[2], x[3])

    # # Between thermal screen and top compartment [W m^{-2}]
    # addAux(gl, 'hThScrTop', sensible(1.7.*u.thScr.*nthroot(abs(x.tThScr-x.tTop),3),
    #     x.tThScr,x.tTop))
    a[152] = sensible(1.7 * u[2] * ca.constpow(ca.fabs(x[7] - x[3] + 1e-10), (1./3.)), x[7], x[3])

    # # Between blackout screen and top compartment [W m^{-2}]
    # addAux(gl, 'hBlScrTop', sensible(1.7.*u.blScr.*nthroot(abs(x.tBlScr-x.tTop),3),
    #     x.tBlScr,x.tTop))
    a[153] = sensible(1.7 * u[5] * ca.constpow(ca.fabs(x[20] - x[3] + 1e-10), (1./3.)), x[20], x[3])

    # # Between top compartment and cover [W m^{-2}]
    # addAux(gl, 'hTopCovIn', sensible(params[50]*nthroot(abs(x.tTop-x.tCovIn),3)*p[47]/p[46],
    #     x.tTop, x.tCovIn))
    a[154] = sensible(p[50] * ca.constpow(ca.fabs(x[3] - x[5] + 1e-10), (1./3.)) * p[47] / p[46], x[3], x[5])

    # # Between top compartment and outside air [W m^{-2}]
    # addAux(gl, 'hTopOut', sensible(p[111]*p[23]*gl.a[136], x.tTop, d.tOut))
    a[155] = sensible(p[111] * p[23] * a[136], x[3], d[1])

    # # Between cover and outside air [W m^{-2}]
    # addAux(gl, 'hCovEOut', sensible(
    #     p[47]/p[46]*(p[51]+p[52]*d.wind.^p[53]),
    #     x.tCovE, d.tOut))
    a[156] = sensible(p[47] / p[46] * (p[51] + p[52] * ca.constpow(d[4], p[53])), x[6], d[1])

    # # Between pipes and air in main compartment [W m^{-2}]
    # addAux(gl, 'hPipeAir', sensible(
    #     1.99*pi*p[105]*p[107]*(abs(x.tPipe-x.tAir)).^0.32,
    #     x.tPipe, x.tAir))
    a[157] = sensible(1.99 * ca.pi * p[105] * p[107] * ca.constpow(ca.fabs(x[9] - x[2] + 1e-10), 0.32), x[9], x[2])

    # # Between floor and soil layer 1 [W m^{-2}]
    # addAux(gl, 'hFlrSo1', sensible(
    #     2/(p[101]/p[99]+p[27]/p[103]),
    #     x.tFlr, x.tSo1))
    a[158] = sensible(2. / (p[101] / p[99] + p[27] / p[103]), x[8], x[10])

    # # Between soil layers 1 and 2 [W m^{-2}]
    # addAux(gl, 'hSo1So2', sensible(2*p[103]/(p[27]+p[28]),
    #     x.tSo1, x.tSo2))
    a[159] = sensible(2. * p[103]/(p[27]+p[28]), x[10], x[11])

    # # Between soil layers 2 and 3 [W m^{-2}]
    # addAux(gl, 'hSo2So3', sensible(2*p[103]/(p[28]+p[29]), x.tSo2, x.tSo3))
    a[160] = sensible(2. * p[103] / (p[28] + p[29]), x[11], x[12])

    # # Between soil layers 3 and 4 [W m^{-2}]
    # addAux(gl, 'hSo3So4', sensible(2*p[103]/(p[29]+p[30]), x.tSo3, x.tSo4))
    a[161] = sensible(2. * p[103] / (p[29] + p[30]), x[12], x[13])

    # # Between soil layers 4 and 5 [W m^{-2}]
    # addAux(gl, 'hSo4So5', sensible(2*p[103]/(p[30]+p[31]), x.tSo4, x.tSo5))
    a[162] = sensible(2. * p[103] / (p[30] + p[31]), x[13], x[14])

    # # Between soil layer 5 and the external soil temperature [W m^{-2}]
    # # See Equations 4 and 77 [1]
    # addAux(gl, 'hSo5SoOut', sensible(2*p[103]/(p[31]+p[37]), x.tSo5, d.tSoOut))
    a[163] = sensible(2. * p[103] / (p[31] + p[37]), x[14], d[6])

    # # Conductive heat flux through the lumped cover [W K^{-1} m^{-2}]
    # # See comment after Equation 18 [1]
    # SINCE U[8] IS ALWAYS 0 WE CAN CHANGE THIS TO THE FOLLOWING:
    # a[164] = sensible(
    #     1/(p[73]/p[71] + u[8]*p.hShScrPer/p.lambdaShScrPer),
    #     x[5], x[6])
    a[164] = sensible(1. / (p[73] / p[71]), x[5], x[6])

    # # Between lamps and air in main compartment [W m^{-2}]
    # # Equation A29 [5]
    # addAux(gl, 'hLampAir', sensible(p[185], x.tLamp, x.tAir))
    a[165] = sensible(p[185], x[17], x[2])

    # Between grow pipes and air in main compartment [W m^{-2}]
    # Equations A31, A33 [5]
    # addAux(gl, 'hGroPipeAir', sensible(
        # 1.99*pi*p[167]*p[166]*(abs(x.tGroPipe-x.tAir)).^0.32, 
    #     x.tGroPipe, x.tAir))
    a[166] = sensible(1.99 * ca.pi * p[167] * p[166] * ca.constpow(ca.fabs(x[19] - x[2] + 1e-10), 0.32), x[19], x[2])

    # # Between interlights and air in main compartment [W m^{-2}]
    # # Equation A30 [5]
    # addAux(gl, 'hIntLampAir', sensible(p[198], x.tIntLamp, x.tAir))
    a[167] = sensible(p[198], x[18], x[2])

    # Smooth switch between day and night [-]
    # Equation 50 [1]
    # addAux(gl, 'sRs', 1./(1+exp(p[43].*(gl.a[45]-p[40]))))
    a[168] = 1. / (1. + ca.exp(p[43] * (a[45] - p[40])))

    # Parameter for co2 influence on stomatal resistance [ppm{CO2}^{-2}]
    # Equation 51 [1]
    # addAux(gl, 'cEvap3', p[20]*(1-gl.a[168])+p[19]*gl.a[168])
    a[169] = p[20] * (1. - a[168]) + p[19] * a[168]
 
    # Parameter for vapor pressure influence on stomatal resistance [Pa^{-2}]
    # addAux(gl, 'cEvap4', p[22]*(1-gl.a[168])+p[21]*gl.a[168])
    a[170] = p[22] * (1. - a[168]) + p[21] * a[168]

    # Radiation influence on stomatal resistance [-]
    # Equation 49 [1]
    # addAux(gl, 'rfRCan', (gl.a[45]+p[17])./(gl.a[45]+p[18]))
    a[171] = (a[45] + p[17]) / (a[45] + p[18])

    # CO2 influence on stomatal resistance [-]
    # Equation 49 [1]
    # addAux(gl, 'rfCo2', min(1.5, 1 + gl.a[169].* (p[7]*x.co2Air-200).^2))
    a[172] = ca.fmin(1.5, 1. + a[169] * ca.constpow((p[7] * x[0] - 200), 2))
        # perhpas replace p[7]*x.co2Air with a[138]

    # Vapor pressure influence on stomatal resistance [-]
    # Equation 49 [1]
    # addAux(gl, 'rfVp', min(5.8, 1+gl.a[170].*(satVP(x.tCan)-x.vpAir).^2))
    a[173] = ca.fmin(5.8, 1. + a[170] * ca.constpow((satVP(x[4]) - x[15]), 2))

    # Stomatal resistance [s m^{-1}]
    # Equation 48 [1]
    # addAux(gl, 'rS', p[42].*gl.a[171].*gl.a[172].*gl.a[173])
    a[174] = p[42] * a[171] * a[172] * a[173]

    # Vapor transfer coefficient of canopy transpiration [kg m^{-2} Pa^{-1} s^{-1}]
    # Equation 47 [1]
    # addAux(gl, 'vecCanAir', 2*p[111]*p[23]*gl.a[31]./
    #     (p[1]*p[14]*(p[41]+gl.a[174])))
    a[175] = 2. * p[111] * p[23] * a[31] / (p[1] * p[14] * (p[41] + a[174]))

    # Canopy transpiration [kg m^{-2} s^{-1}]
    # Equation 46 [1]
    # addAux(gl, 'mvCanAir', (satVP(x.tCan)-x.vpAir).*gl.a[175]) 
    a[176] = (satVP(x[4]) - x[15]) * a[175]

    #  Vapor Fluxes

    # These are currently not used in the model..
    #  a.mvPadAir = 0
    a[177] = 0
    #  a.mvFogAir = 0
    a[178] = 0
    #  a.mvBlowAir = 0
    a[179] = 0
    #  a.mvAirOutPad = 0
    a[180] = 0

    # Condensation from main compartment on thermal screen [kg m^{-2} s^{-1}]
    # Table 4 [1], Equation 42 [1]
    # addAux(gl, 'mvAirThScr', cond(1.7*u.thScr.*nthroot(abs(x.tAir-x.tThScr),3), 
    #     x.vpAir, satVP(x.tThScr)))
    a[181] = cond(1.7 * u[2] * ca.constpow(ca.fabs(x[2] - x[7] + 1e-10), (1./3.)), x[15], satVP(x[7]))

    # Condensation from main compartment on blackout screen [kg m^{-2} s^{-1}]
    # Equatio A39 [5], Equation 7.39 [7]
    # addAux(gl, 'mvAirBlScr', cond(1.7*u.blScr.*nthroot(abs(x.tAir-x.tBlScr),3), 
    #     x.vpAir, satVP(x.tBlScr)))
    a[182] = cond(1.7 * u[5] * ca.constpow(ca.fabs(x[2] - x[20] + 1e-10), (1./3.)), x[15], satVP(x[20]))

    # Condensation from top compartment to cover [kg m^{-2} s^{-1}]
    # Table 4 [1]
    # addAux(gl, 'mvTopCovIn', cond(params[50]*nthroot(abs(x.tTop-x.tCovIn),3)*p[47]/p[46],
    #     x.vpTop, satVP(x.tCovIn)))
    a[183] = cond(p[50]* ca.constpow(ca.fabs(x[3] - x[5] + 1e-10), (1./3.)) * p[47]/p[46], x[16], satVP(x[5]))

    # Vapor flux from main to top compartment [kg m^{-2} s^{-1}]
    # addAux(gl, 'mvAirTop', airMv(gl.a[144], x.vpAir, x.vpTop, x.tAir, x.tTop))
    a[184] = airMv(a[144], x[15], x[16], x[2], x[3])

    # Vapor flux from top compartment to outside [kg  m^{-2} s^{-1}]
    # addAux(gl, 'mvTopOut', airMv(gl.a[136], x.vpTop, d.vpOut, x.tTop, d.tOut))
    a[185] = airMv(a[136], x[16], d[2], x[3], d[1])

    # Vapor flux from main compartment to outside [kg m^{-2} s^{-1}]
    # addAux(gl, 'mvAirOut', airMv(gl.a[137]+gl.a[145], x.vpAir, 
#     d.vpOut, x.tAir, d.tOut))
    a[186] = airMv(a[137]+a[145], x[15], d[2], x[2], d[1])

    #  Latent heat fluxes ####
    a[187] = p[1] * a[176]
    a[188] = p[1] * a[181]
    a[189] = p[1] * a[182]
    a[190] = p[1] * a[183]

    #  Canopy photosynthesis ####

    # PAR absorbed by the canopy [umol{photons} m^{-2} s^{-1}]
    # Equation 17 [2]
    # addAux(gl, 'parCan', p[187]*gl.a[55] + p[140]*gl.a[54] + 
    #     p[197]*gl.a[56])
    #  a.parCan = p.zetaLampPar*a.rParLampCan + p.parJtoUmolSun*a.rParSunCan + 
    #      p.zetaIntLampPar*a.rParIntLampCan
    #  a[191] = p[187] * a[55] + p[140] * a[54] + p[197] * a[56]
    a[191] = p[187] * a[55] + p[140] * a[54] + p[197] * a[56]

    # Maximum rate of electron transport rate at 25C [umol{e-} m^{-2} s^{-1}]
    # Equation 16 [2]
    # addAux(gl, 'j25CanMax', gl.a[31]*p[129])
    #  a[192] = a[31] * p[129]
    a[192] = a[31] * p[129]

    # CO2 compensation point [ppm]
    # Equation 23 [2]
    # addAux(gl, 'gamma', divNoBracks(p[129], (gl.a[192])*1) .*p[130].*x.tCan + 
    #     20*p[130].*(1-divNoBracks(p[129],(gl.a[192])*1)))
    a[193] = (p[129] / a[192]) * p[130] * x[4] + 20 * p[130] * (1 - (p[129] / a[192]))

    # CO2 concentration in the stomata [ppm]
    # Equation 21 [2]
    # addAux(gl, 'co2Stom', p[131]*gl.a[138])
    a[194] = p[131] * a[138]

    # # Potential rate of electron transport [umol{e-} m^{-2} s^{-1}]
    # # Equation 15 [2]
    # # Note that R in [2] is 8.314 and R in [1] is 8314
    # addAux(gl, 'jPot', gl.a[192].*exp(p[132]*(x.tCan+273.15-p[133])./(1e-3*p[39]*(x.tCan+273.15)*p[133])).*
    #     (1+exp((p[134]*p[133]-p[135])./(1e-3*p[39]*p[133])))./
    #     (1+exp((p[134]*(x.tCan+273.15)-p[135])./(1e-3*p[39]*(x.tCan+273.15)))))
    a[195] = a[192] * ca.exp(p[132] * (x[4] + 273.15 - p[133]) / (1e-3*p[39] * (x[4] + 273.15) * p[133])) * \
        (1 + ca.exp((p[134] * p[133] - p[135]) / (1e-3*p[39] * p[133]))) / \
        (1 + ca.exp((p[134] * (x[4] + 273.15) - p[135]) / (1e-3*p[39] * (x[4] + 273.15))))

    # # Electron transport rate [umol{e-} m^{-2} s^{-1}]
    # # Equation 14 [2]
    # addAux(gl, 'j', (1/(2*p[136]))*(gl.a[195]+p[137]*gl.a[191]-
    #     sqrt((gl.a[195]+p[137]*gl.a[191]).^2-4*p[136]*gl.a[195].*p[137].*gl.a[191])))
    #  a[196] = (1. / (2. * p[136])) * (a[195] + p[137] * a[191] -
    #      sqrt(ca.constpow((a[195] + p[137] * a[191]), 2) - 4*p[136] * a[195] * p[137] * a[191]))
    a[196] = (1. / (2. * p[136])) * (a[195] + p[137] * a[191] - \
        ca.sqrt(ca.constpow((a[195] + p[137] * a[191]), 2) - 4*p[136] * a[195] * p[137] * a[191] + 1e-10))

    # # Photosynthesis rate at canopy level [umol{co2} m^{-2} s^{-1}]
    # # Equation 12 [2]
    # addAux(gl, 'p', gl.a[196].*(gl.a[194]-gl.a[193])./(4*(gl.a[194]+2*gl.a[193])))
    a[197] = a[196] * (a[194]-a[193]) / (4*(a[194] + 2*a[193]))

    # # Photrespiration [umol{co2} m^{-2} s^{-1}]
    # # Equation 13 [2]
    # addAux(gl, 'r', gl.a[197].*gl.a[193]./gl.a[194])
    a[198] = a[197]*a[193] / a[194]

    # # Inhibition due to full carbohydrates buffer [-]
    # # Equation 11, Equation B.1, Table 5 [2]
    # addAux(gl, 'hAirBuf', 1./(1+exp(5e-4*(x.cBuf-p[157]))))
    a[199] = 1. / (1. + ca.exp(5e-4*(x[22] - p[157])))

    # # Net photosynthesis [mg{CH2O} m^{-2} s^{-1}]
    # # Equation 10 [2]
    # addAux(gl, 'mcAirBuf', p[138]*gl.a[199].*(gl.a[197]-gl.a[198]))
    a[200] = p[138] * a[199] * (a[197] - a[198])

    # ## Carbohydrate buffer
    # # Temperature effect on structural carbon flow to organs
    # # Equation 28 [2]
    # addAux(gl, 'gTCan24', 0.047*x.tCan24+0.06)
    a[201] = 0.047*x[21] + 0.06

    # # Inhibition of carbohydrate flow to the organs
    # # Equation B.3 [2]
    # addAux(gl, 'hTCan24', 1./(1+exp(-1.1587*(x.tCan24-p[160]))).* 
    #     1./(1+exp(1.3904*(x.tCan24-p[159]))))
    a[202] = 1. / (1. + ca.exp(-1.1587 * (x[21]-p[160]))) * \
        1. / (1. + ca.exp(1.3904*(x[21] - p[159])))

    # # Inhibition of carbohydrate flow to the fruit
    # # Equation B.3 [2]
    # addAux(gl, 'hTCan', 1./(1+exp(-0.869*(x.tCan-p[162]))).* 
    #     1./(1+exp(0.5793*(x.tCan-p[161]))))
    a[203] = 1. / (1. + ca.exp(-0.869*(x[4] - p[162]))) * \
        1. / (1. + ca.exp(0.5793*(x[4] - p[161])))

    # # Inhibition due to development stage 
    # # Equation B.6 [2]
    # gl, 'hTCanSum', 0.5*(x.tCanSum/p[163]+
    #     sqrt((x.tCanSum./p[163]).^2+1e-4)) - 
    #     0.5*((x.tCanSum-p[163])./p[163]+
    #     sqrt(((x.tCanSum-p[163])/p[163]).^2 + 1e-4))
    a[204] = 0.5 *(x[26] / p[163] + \
        ca.sqrt(ca.constpow((x[26] / p[163]), 2) + 1e-4)) - \
        0.5 * ((x[26] - p[163]) / p[163] + \
        ca.sqrt(ca.constpow(((x[26] - p[163]) / p[163]), 2) + 1e-4))

    # # Inhibition due to insufficient carbohydrates in the buffer [-]
    # # Equation 26 [2]
    # gl, 'hBufOrg', 1./(1+exp(-5e-3*(x.cBuf-p[158])))
    a[205] = 1. / (1. + ca.exp(-5e-3*(x[22] - p[158])))

    # # Carboyhdrate flow from buffer to leaves [mg{CH2O} m^{2} s^{-1}]
    # Equation 25 [2]
    # addAux(gl, 'mcBufLeaf', gl.a[205].*gl.a[202].*gl.a[201].*gl.p[155])
    a[206] = a[205] * a[202] * a[201] * p[155]

    # # Carboyhdrate flow from buffer to stem [mg{CH2O} m^{2} s^{-1}]
    # # Equation 25 [2]
    # addAux(gl, 'mcBufStem', gl.a[205].*gl.a[202].*gl.a[201].*gl.p[156])
    a[207] = a[205] * a[202] * a[201] * p[156]

    # # Carboyhdrate flow from buffer to fruit [mg{CH2O} m^{2} s^{-1}]
    # # Equation 24 [2]
    # addAux(gl, 'mcBufFruit', gl.a[205].*
    #     gl.a[203].*gl.a[202].*gl.a[204].*gl.a[201].*gl.p[154])
    a[208] = a[205] * a[203] * a[202] * a[204] * a[201] * p[154]

    # Growth respiration [mg{CH2O} m^{-2] s^{-1}]
    # Equations 43-44 [2]
    # addAux(gl, 'mcBufAir', p[147]*gl.a[206] + p[148]*gl.a[207] 
    #     +p[146]*gl.a[208])
    a[209] = p[147]*a[206] + p[148]*a[207] + p[146]*a[208]

    # Leaf maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    # Equation 45 [2]
    # addAux(gl, 'mcLeafAir', (1-exp(-p[149]*p[143])).*p[150].^(0.1*(x.tCan24-25)).* 
    #     x.cLeaf*p.cLeafM)
    a[210] = (1. - ca.exp(-p[149] * p[143])) * ca.constpow(p[150], 0.1*(x[21]-25)) * x[23] * p[152]

    # Stem maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    # Equation 45 [2]
    # addAux(gl, 'mcStemAir', (1-exp(-p[149]*p[143])).*p[150].^(0.1*(x.tCan24-25)).* 
    #     x.cStem*p[153])
    a[211] = (1. - ca.exp(-p[149] * p[143])) * ca.constpow(p[150], 0.1 * (x[21] - 25)) * x[24] * p[153]

    # Fruit maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    # Equation 45 [2]
    # addAux(gl, 'mcFruitAir', (1-exp(-p[149]*p[143])).*p[150].^(0.1*(x.tCan24-25)).* 
    #     x.cFruit*p.cFruitM)
    a[212] = (1. - ca.exp(-p[149] * p[143])) * ca.constpow(p[150], (0.1*(x[21] - 25))) * x[25] * p[151]

    # Total maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    # Equation 45 [2]
    # addAux(gl, 'mcOrgAir', gl.a[210]+gl.a[211]+gl.a[212])
    a[213] = a[210] + a[211] + a[212]

    # Leaf pruning and fruit harvest
    # A new smoothing function has been applied here to avoid stiffness
    # Leaf pruning [mg{CH2O} m^{-2] s^{-1}]
    # Equation B.5 [2]
    a[214] = smoothHar(x[23], p[144], 1e4, 5e4)

    #  Fruit harvest [mg{CH2O} m^{-2} s^{-1}]
    #  Equation A45 [5], Equation 7.45 [7]
    a[215] = smoothHar(x[25], p[145], 1e4, 5e4)

    #  Net crop assimilation [mg{CO2} m^{-2} s^{-1}]
    #  It is assumed that for every mol of CH2O in net assimilation, a mol
    #  of CO2 is taken from the air, thus the conversion uses molar masses
    #  addAux(gl, 'mcAirCan', (p.mCo2/p.mCh2o)*(gl.a.mcAirBuf-gl.a.mcBufAir-gl.a.mcOrgAir))
    a[216] = (p[139]/p[138]) * (a[200] - a[209] - a[213])

    #  Other CO2 flows [mg{CO2} m^{-2} s^{-1}]
    #  Equation 45 [1]

    #  From main to top compartment
    #  addAux(gl, 'mcAirTop', airMc(gl.a.fScr, x.co2Air, x.co2Top))
    a[217] = airMc(a[144], x[0], x[1])

    # From top compartment outside
    # addAux(gl, 'mcTopOut', airMc(gl.a.fVentRoof, x.co2Top, d.co2Out))
    a[218] = airMc(a[136], x[1], d[3])

    # From main compartment outside
    # addAux(gl, 'mcAirOut', airMc(gl.a[137]+gl.a[145], x.co2Air, d.co2Out))
    a[219] = airMc(a[137] + a[145], x[0], d[3])

    # Heat from boiler - Section 9.2 [1]

    # Heat from boiler to pipe rails [W m^{-2}]
    # Equation 55 [1]
    # addAux(gl, 'hBoilPipe', u.boil*p.pBoil/p[46])
    a[220] = u[0] * p[108] / p[46]

    # Heat from boiler to grow pipes [W m^{-2}]
    # addAux(gl, 'hBoilGroPipe', u.boilGro*p.pBoilGro/p[46])
    # 2 = u(6) * p(170) / p[46]
    a[221] = 0

    # External CO2 source - Section 9.9 [1]

    # CO2 injection [mg m^{-2} s^{-1}]
    # Equation 76 [1]
    # addAux(gl, 'mcExtAir', u.extCo2*p.phiExtCo2/p[46])
    a[222] = u[1] * p[109] / p[46]

    #  Objects not currently included in the model
    #  a.mcBlowAir = 0
    a[223] = 0
    #  a.mcPadAir = 0
    a[224] = 0
    #  a.hPadAir = 0
    a[225] = 0
    #  a.hPasAir = 0
    a[226] = 0
    #  a.hBlowAir = 0
    a[227] = 0
    #  a.hAirPadOut = 0
    a[228] = 0
    #  a.hAirOutPad = 0
    a[229] = 0
    #  a.lAirFog = 0
    a[230] = 0
    #  a.hIndPipe = 0
    a[231] = 0
    #  a.hGeoPipe = 0
    a[232] = 0

    #  Lamp cooling
    # Equation A34 [5], Equation 7.34 [7]
    #  a.hLampCool = p.etaLampCool * a[37]
    a[233] = p[186] * a[37]

    # Heat harvesting, mechanical cooling and dehumidification
    # By default there is no mechanical cooling or heat harvesting
    # see addHeatHarvesting.m for mechanical cooling and heat harvesting
    #  a.hecMechAir = 0
    a[234] = 0
    #  a.hAirMech = 0
    a[235] = 0
    #  a.mvAirMech = 0
    a[236] = 0
    #  a.lAirMech = 0
    a[237] = 0
    #  a.hBufHotPipe = 0
    a[238] = 0

    return a


# Np = 2
# d_values = load_dummy_weather(Np)
# x0_init = init_state(d_values[0], 90.0, 0.0)
# p_values = init_default_params(208)


# # Solve the NLP
# uk = np.ones(6)*0.5
# dk = d_values[0]
# a = update(x0_init, uk, dk, p_values)
# print(a)
