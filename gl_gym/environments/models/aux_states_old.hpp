#include <casadi/casadi.hpp>

using namespace casadi;

inline SX satVP(const SX& temp) {
    // Calculate the saturation vapor pressure [Pa]
    // Equation 6 [1]
    const double a = 610.78;
    const double b = 17.2694;
    const double c = 238.3;
    return a * exp(b * temp / (temp + c));
}

inline SX co2dens2ppm(const SX& temp, const SX& dens) {
    // Convert CO2 density to CO2 concentration [ppm]
    // Equation 7 [1]
    const double R = 8.3144598;        // Molar gas constant [J mol^{-1} K^{-1}]
    const double C2K = 273.15;         // Conversion from Celsius to Kelvin [K]
    const double M_CO2 = 44.01e-3;     // Molar mass of CO2 [kg mol^{-1}]
    const double P = 101325;           // Pressure (assumed to be 1 atm) [Pa]
    
    return 1e6 * R * (temp + C2K) * dens / (P * M_CO2);
}

inline SX tau12(const SX& tau1, const SX& tau2, const SX& rho1Dn, const SX& rho2Up) {
    // Transmission coefficient of a double layer [-]
    // Equation 14 [1], Equation A4 [5]
    return tau1 * tau2 / (1. - rho1Dn * rho2Up);
}

inline SX rhoUp(const SX& tau1, const SX& rho1Up, const SX& rho1Dn, const SX& rho2Up) {
    // Reflection coefficient of the upper layer [-]
    // Equation 15 [1], Equation A5 [5]
    return rho1Up + (tau1 * tau1 * rho2Up) / (1. - rho1Dn * rho2Up);
}

inline SX rhoDn(const SX& tau2, const SX& rho1Dn, const SX& rho2Up, const SX& rho2Dn) {
    // Reflection coefficient of the lower layer [-]
    // Equation 15 [1], Equation A5 [5]
    return rho2Dn + (tau2 * tau2 * rho1Dn) / (1. - rho1Dn * rho2Up);
}

inline SX degrees2rad(const SX& degrees) {
    // Convert degrees to radians
    return degrees * M_PI / 180.;
}

inline SX fir(const SX& a1, const SX& eps1, const SX& eps2, const SX& f12, const SX& t1, const SX& t2, const SX& sigma) {
    // Net far infrared flux from 1 to 2 [W m^{-2}]
    // Equation 37 [1]
    return a1 * eps1 * eps2 * f12 * sigma * (pow(t1 + 273.15, 4.) - pow(t2 + 273.15, 4.));
}

inline SX sensible(const SX& hec, const SX& t1, const SX& t2) {
    // Sensible heat flux from 1 to 2 [W m^{-2}]
    // Equation 38 [1]
    return fabs(hec) * (t1 - t2);
}

inline SX cond(const SX& hec, const SX& vp1, const SX& vp2) {
    const double a = 6.4e-9; 
    return 1.0 / (1.0 + exp(-0.1 * (vp1 - vp2))) * a * hec * (vp1 - vp2);
}

// inline SX smoothHar(const SX& processVar, const SX& cutOff, double smooth, double maxRate) {
//     // Define a smooth function for harvesting (leaves, fruit, etc)
//     // processVar - the DynamicElement to be controlled
//     // cutoff     - the value at which the processVar should be harvested
//     // smooth     - smoothing factor. The rate will go from 0 to max at
//     //              a range with approximately this width
//     // maxRate    - the maximum harvest rate
//     return maxRate / (1 + exp(-(processVar - cutOff) * 2. * 4.6052 / smooth));
// }

inline SX smoothHar(const SX& processVar, const SX& cutOff, double smooth, double maxRate) {
    double k = 2.0 * 4.6052 / smooth;
    SX z = k * (processVar - cutOff) / 2.0;
    return maxRate * (tanh(z) + 1.0) / 2.0;
}

inline SX airMv(const SX& f12, const SX& vp1, const SX& vp2, const SX& t1, const SX& t2) {
    // Vapor flux accompanying an air flux [kg m^{-2} s^{-1}]
    // Equation 44 [1]
    const float c2k = 273.15;
    const double a = 0.002165;
    return a * fabs(f12) * (vp1 / (t1 + c2k) - vp2 / (t2 + c2k));
}

inline SX airMc(const SX& f12, const SX& c1, const SX& c2) {
    // Co2 flux accompanying an air flux [kg m^{-2} s^{-1}]
    // Equation 45 [1]
    return fabs(f12) * (c1 - c2);
}

// Update function for auxiliary variables
SX update(const SX& x, const SX& u, const SX& d, const SX& p) {
    /*
        u is a control vector with 6 elements:
        u(0) = uBoil
        u(1) = uCo2
        u(2) = uThScr
        u(3) = uVent
        u(4) = uLamp
        u(5) = uBlScr
    */
    std::vector<SX> a(240);

    a[0] = 1 - u(2) * (1 - p(80));
    a[1] = u(2) * p(77);
    a[2] = tau12(p(69), a[0], p(66), a[1]);
    a[3] = rhoUp(p(69), p(66), p(66), a[1]);
    a[4] = rhoDn(a[0], p(66), a[1], a[1]);

    // Thermal Screen and Roof NIR
    a[5] = 1 - u(2) * (1 - p(79));
    a[6] = u(2) * p(76);
    a[7] = tau12(p(68), a[5], p(65), a[6]);
    a[8] = rhoUp(p(68), p(65), p(65), a[6]);
    a[9] = rhoDn(a[5], p(65), a[6], a[6]);

    // Blackout screen and Roof
    a[10] = 1 - u(5) * (1 - p(90));
    a[11] = u(5) * p(88);
    a[12] = tau12(a[2], a[10], a[4], a[11]);
    a[13] = rhoUp(a[2], a[3], a[4], a[11]);
    a[14] = rhoDn(a[10], a[4], a[11], a[11]);

    a[15] = 1 - u(5) * (1 - p(89));
    a[16] = u(5) * p(87);
    a[17] = tau12(a[7], a[15], a[9], a[16]);
    a[18] = rhoUp(a[7], a[8], a[9], a[16]);
    a[19] = rhoDn(a[15], a[9], a[16], a[16]);

    // Full cover model
    a[20] = tau12(a[12], p(176), a[14], p(179));
    a[21] = rhoUp(a[12], a[13], a[14], p(179));
    a[22] = tau12(a[17], p(177), a[19], p(180));
    a[23] = rhoUp(a[17], a[18], a[19], p(180));

    a[24] = p(70);
    a[25] = p(67);
    a[26] = 1 - a[20] - a[21];
    a[27] = 1 - a[22] - a[23];
    a[28] = 1 - a[24] - a[25];
    a[29] = a[28];
    a[30] = cos(degrees2rad(p(45))) * p(73) * p(64) * p(72);

    // Capacities
    a[31] = p(142) * x(23);
    a[32] = p(16) * a[31];
    a[33] = 0.1 * a[30];
    a[34] = 0.1 * a[30];
    a[35] = p(38) * p(48) / (p(39) * (x(2) + 273.15));
    a[36] = p(38) * (p(49) - p(48)) / (p(39) * (x(3) + 273.15));

    // Global, PAR, and NIR heat fluxes
    // qLampIn;         p.thetaLampMax * u[4];
    a[37] = p(172) * u(4);                      
    // a[38] = p(196) * u(5);
    // Excluded the inter lights from the model
    a[38] = 0;
      // rParGhSun
      a[39] = (1 - p(44)) * a[20] * p(6) * d(0);
    // rParGhLamp;      p.etaLampPar * a.qLampIn;
    a[40] = p(174) * a[37];                     
    a[41] = p(192) * a[38];
    a[42] = (1 - p(44)) * d(0) * (p(6) * a[20] + p(5) * a[22]);
    a[43] = (p(174) + p(175)) * a[37];
    a[44] = (p(192) + p(193)) * a[38];
    // a.rCan
    a[45] = a[42] + a[43] + a[44];

    a[46] = a[39] * (1 - p(10)) * (1 - exp(-p(32) * a[31]));
    a[47] = a[40] * (1 - p(10)) * (1 - exp(-p(32) * a[31]));
    a[48] = 1 - p(190) * exp(-p(200) * p(189) * a[31]) +
                       (p(190) - 1) * exp(-p(200) * (1 - p(189)) * a[31]);
    a[49] = 1 - p(190) * exp(-p(202) * p(189) * a[31]) +
                       (p(190) - 1) * exp(-p(202) * (1 - p(189)) * a[31]);
    a[50] = a[41] * a[48] * (1 - p(10));

    // a.rParSunFlrCanUp
    a[51] = a[39] * exp(-p(32) * a[31]) * p(98) * (1 - p(10)) * (1 - exp(-p(33) * a[31]));
    a[52] = a[40] * exp(-p(32) * a[31]) * p(98) * (1 - p(10)) * (1 - exp(-p(33) * a[31]));
    a[53] = a[41] * p(190) * exp(-p(200) * p(189) * a[31]) * p(98) * (1 - p(10)) * (1 - exp(-p(201) * a[31]));

    a[54] = a[46] + a[51];
    a[55] = a[47] + a[52];
    a[56] = a[50] + a[53];

    a[57] = 1 - a[23];
    a[58] = 1 - p(97);
    a[59] = exp(-p(34) * a[31]);
    a[60] = p(11) * (1 - a[59]);

    a[61] = tau12(a[57], a[59], a[23], a[60]);
    a[62] = rhoUp(a[59], a[23], a[23], a[60]);
    a[63] = rhoDn(a[59], a[23], a[60], a[60]);
    a[64] = tau12(a[61], a[58], a[63], p(97));
    a[65] = rhoUp(a[61], a[62], a[63], p(97));

    a[66] = 1 - a[64] - a[65];
    a[67] = a[64];
    a[68] = (1 - p(44)) * a[66] * p(5) * d(0);
    a[69] = p(175) * a[37] * (1 - p(11)) * (1 - exp(-p(34) * a[31]));
    a[70] = p(193) * a[38] * a[49] * (1 - p(11));
    a[71] = (1 - p(44)) * a[67] * p(5) * d(0);
    a[72] = (1 - p(97)) * exp(-p(34) * a[31]) * p(175) * a[37];
    a[73] = p(190) * (1 - p(97)) * exp(-p(202) * a[31] * p(189)) * p(193) * a[38];

    a[74] = (1 - p(98)) * exp(-p(32) * a[31]) * a[39];
    a[75] = (1 - p(98)) * exp(-p(32) * a[31]) * a[40];
    a[76] = a[41] * p(190) * (1 - p(98)) * exp(-p(200) * a[31] * p(189));

    a[77] = (p(174) + p(175)) * a[37] - a[55] - a[69] - a[75] - a[72];
    a[78] = (p(192) + p(193)) * a[38] - a[56] - a[70] - a[76] - a[73];
    a[79] = p(44) * d(0) * (a[20] * p(6) + (a[66] + a[67]) * p(5));
    a[80] = (a[26] * p(6) + a[27] * p(5)) * d(0);

    // FIR transmission coefficient of the Thermal screen
    a[81] = 1 - u(2) * (1 - p(81));

    // FIR transmission coefficient of the blackout screen
    a[82] = 1 - u(5) * (1 - p(91));

    // Surface of canopy per floor area
    a[83] = 1 - exp(-p(35) * a[31]);

    // FIR between canopy and cover [W m^{-2}]
    a[84] = fir(a[83], p(3), a[29], p(178) * a[81] * a[82], x(4), x(5), p(2));

    // FIR between canopy and sky [W m^{-2}]
    a[85] = fir(a[83], p(3), p(4), p(178) * a[24] * a[81] * a[82], x(4), d(5), p(2));

    // FIR between canopy and thermal screen [W m^{-2}]
    a[86] = fir(a[83], p(3), p(74), p(178) * u(2) * a[82], x(4), x(7), p(2));

    // FIR between canopy and floor [W m^{-2}]
    a[87] = fir(a[83], p(3), p(95), p(125), x(4), x(8), p(2));

    // FIR between pipes and cover [W m^{-2}]
    a[88] = fir(p(124), p(104), a[29], p(199) * p(178) * a[81] * a[82] * 0.49 * exp(-p(35) * a[31]), x(9), x(5), p(2));

    // FIR between pipes and sky [W m^{-2}]
    a[89] = fir(p(124), p(104), p(4), p(199) * p(178) * a[24] * a[81] * 0.49 * exp(-p(35) * a[31]), x(9), d(5), p(2));

    // FIR between pipes and thermal screen [W m^{-2}]
    a[90] = fir(p(124), p(104), p(74), p(199) * p(178) * u(2) * a[82] * 0.49 * exp(-p(35) * a[31]), x(9), x(7), p(2));

    // FIR between pipes and floor [W m^{-2}]
    a[91] = fir(p(124), p(104), p(95), 0.49, x(9), x(8), p(2));

    // FIR between pipes and canopy [W m^{-2}]
    a[92] = fir(p(124), p(104), p(3), 0.49 * (1 - exp(-p(35) * a[31])), x(9), x(4), p(2));

    // FIR between floor and cover [W m^{-2}]
    a[93] = fir(1, p(95), a[29], p(199) * p(178) * a[81] * a[82] * (1 - 0.49 * M_PI * p(107) * p(105)) * exp(-p(35) * a[31]), x(8), x(5), p(2));

    // FIR between floor and sky [W m^{-2}]
    a[94] = fir(1, p(95), p(4), p(199) * p(178) * a[24] * a[81] * a[82] * (1 - 0.49 * M_PI * p(107) * p(105)) * exp(-p(35) * a[31]), x(8), d(5), p(2));

    // FIR between floor and thermal screen [W m^{-2}]
    a[95] = fir(1, p(95), p(74), p(199) * p(178) * u(2) * a[82] * (1 - 0.49 * M_PI * p(107) * p(105)) * exp(-p(35) * a[31]), x(8), x(7), p(2));

    // FIR between thermal screen and cover [W m^{-2}]
    a[96] = fir(1, p(74), a[29], u(2), x(7), x(5), p(2));

    // FIR between thermal screen and sky [W m^{-2}]
    a[97] = fir(1, p(74), p(4), a[24] * u(2), x(7), d(5), p(2));

    // FIR between cover and sky [W m^{-2}]
    a[98] = fir(1, a[28], p(4), 1, x(6), d(5), p(2));

    // FIR between lamps and floor [W m^{-2}]
    a[99] = fir(p(181), p(183), p(95), p(199) * (1 - 0.49 * M_PI * p(107) * p(105)) * exp(-p(35) * a[31]), x(17), x(8), p(2));

    // FIR between lamps and pipe [W m^{-2}]
    a[100] = fir(p(181), p(183), p(104), p(199) * 0.49 * M_PI * p(107) * p(105) * exp(-p(35) * a[31]), x(17), x(9), p(2));

    // FIR between lamps and canopy [W m^{-2}]
    a[101] = fir(p(181), p(183), p(3), a[83], x(17), x(4), p(2));

    // FIR between lamps and thermal screen [W m^{-2}]
    a[102] = fir(p(181), p(182), p(74), u(2) * a[82], x(17), x(7), p(2));

    // FIR between lamps and cover [W m^{-2}]
    a[103] = fir(p(181), p(182), a[29], a[81] * a[82], x(17), x(5), p(2));

    // FIR between lamps and sky [W m^{-2}]
    a[104] = fir(p(181), p(182), p(4), a[24] * a[81] * a[82], x(17), d(5), p(2));

    // FIR between grow pipes and canopy [W m^{-2}]
    a[105] = fir(p(169), p(165), p(3), 1, x(19), x(4), p(2));

    // FIR between blackout screen and floor [W m^{-2}]	
    a[106] = fir(1, p(95), p(85), p(199) * p(178) * u(5) * (1 - 0.49 * M_PI * p(107) * p(105)) * exp(-p(35) * a[31]), x(8), x(20), p(2));

    // FIR between blackout screen and pipe [W m^{-2}]
    a[107] = fir(p(124), p(104), p(85), p(199) * p(178) * u(5) * 0.49 * exp(-p(35) * a[31]), x(9), x(20), p(2));

    // FIR between blackout screen and canopy [W m^{-2}]
    a[108] = fir(a[83], p(3), p(85), p(178) * u(5), x(4), x(20), p(2));

    // FIR between blackout screen and thermal screen [W m^{-2}]
    a[109] = fir(u(5), p(85), p(74), u(2), x(20), x(7), p(2));

    // FIR between blackout screen and cover [W m^{-2}]
    a[110] = fir(u(5), p(85), a[29], a[81], x(20), x(5), p(2));

    // FIR between blackout screen and sky [W m^{-2}]
    a[111] = fir(u(5), p(85), p(4), a[24] * a[81], x(20), d(5), p(2));

    // FIR between blackout screen and lamps [W m^{-2}]
    a[112] = fir(p(181), p(182), p(85), u(5), x(17), x(20), p(2));

    // Fraction of radiation going up from the interlight to the canopy [-]
    a[113] = 1 - exp(-p(203) * (1 - p(189)) * a[31]);

    // Fraction of radiation going down from the interlight to the canopy [-]
    a[114] = 1 - exp(-p(203) * p(189) * a[31]);

    // FIR between interlights and floor [W m^{-2}]
    a[115] = fir(p(194), p(195), p(95), (1 - 0.49 * M_PI * p(107) * p(105)) * (1 - a[114]), x(18), x(8), p(2));

    // FIR between interlights and pipe [W m^{-2}]
    a[116] = fir(p(194), p(195), p(104), 0.49 * M_PI * p(107) * p(105) * (1 - a[114]), x(18), x(9), p(2));
    
    // FIR between interlights and canopy [W m^{-2}]
    a[117] = fir(p(194), p(195), p(3), a[114] + a[113], x(18), x(4), p(2));
    
    // FIR between interlights and toplights [W m^{-2}]
    a[118] = fir(p(194), p(195), p(183), (1 - a[113]) * p(181), x(18), x(17), p(2));
    
    // FIR between interlights and blackout screen [W m^{-2}]
    a[119] = fir(p(194), p(195), p(85), u(5) * p(178) * (1 - a[113]), x(18), x(20), p(2));
    
    // FIR between interlights and thermal screen [W m^{-2}]
    a[120] = fir(p(194), p(195), p(74), u(2) * a[82] * p(178) * (1 - a[113]), x(18), x(7), p(2));

    // FIR between interlights and cover [W m^{-2}]
    a[121] = fir(p(194), p(195), a[29], a[81] * a[82] * p(178) * (1 - a[113]), x(18), x(5), p(2));

    // FIR between interlights and sky [W m^{-2}]
    a[122] = fir(p(194), p(195), p(4), a[24] * a[81] * a[82] * p(178) * (1 - a[113]), x(18), d(5), p(2));


    ///////////////////////////////
    ///// Natural Ventilation /////
    ///////////////////////////////

    // Aperature of the roof
    // Aperture of the roof [m^{2}]
    // Equation 67 [1]
    a[123] = u(3) * p(55);
    a[124] = p(55);
    a[125] = 0;
    
    // Aperture of the sidewall [m^{2}]
    // Equation 68 [1] 
    // (this is 0 in the Dutch greenhouse)
    a[126] = 0;


    // Ratio between roof vent area and total ventilation area [-]
    // (not very clear in the reference [1], but always 1 if m.a[126] == 0)
    a[127] = 1;
    a[128] = 1;

    // Ratio between side vent area and total ventilation area [-]
    // (not very clear in the reference [1], but always 0 if m.a[126] == 0)    
    a[129] = 0;

    // Discharge coefficient [-]
    // Equation 73 [1]
    // SINCE SHADING SCREEN IS ALWAYS = 0 WE CAN CHANGE cD = p(59)
    // a[130] = p(59) * (1 - p.etaShScrCd*u[8])
    a[130] = p(59);

    // Discharge coefficient [-]
    // Equation 74 [-]
    // addAux(gl, 'cW', p(61)*(1-p.etaShScrCw*u.shScr))
    //// SINCE SHADING SCREEN IS ALWAYS = 0 WE CAN CHANGE cW = p(61)
    // a[131] = p(61) * (1 - p.etaShScrCw*u[8])
    a[131] = p(61);

    a[132] = u(3) * p(55) * a[130] / (2. * p(46)) *
                   sqrt(fabs(p(26) * p(56) * (x(2) - d(1) + 1e-8) / (2. * (0.5 * x(2) + 0.5 * d(1) + 273.15)) + a[131] * (d(4) * d(4))));

    // a[136]2Max = p(55) * a[130]/(2*p(46)) * 
    //      sqrt(fabs(p(26)*p(56) * (x[()-d[()) / (2*(0.5*x(2) + 0.5*d(1) + 273.15)) + pow(a[131]*d(4), 2)));
    // a[136]2Min = 0;

    // a.fVentRoofSide2
    a[133] = a[130] / p(46) * sqrt(1e-8 +
                           pow((a[123] * a[126] / sqrt(fmax(a[123]*a[123] + a[126]*a[126], 0.01))), 2) *
                           (2 * p(26) * p(62) * (x(2) - d(1) + 1e-8) / (0.5 * x(2) + 0.5 * d(1) + 273.15)) +
                           ((a[123] + a[126]/2.) * (a[123] + a[126] / 2.) * a[131] * d(4) * d(4)));

    // Ventilation rate through sidewall only
    // a.fVentSide2
    a[134] = a[130] * a[126] * d(4) / (2*p(46)) * sqrt(a[131]);

    // Leakage ventilation
    // a.fLeakage
    a[135] = if_else(d(4) < p(205), p(205) * p(60), p(60) * d(4));

    // Total ventilation through the roof
    a[136] = if_else(
                a[127] >= p(8),
                p(57) * a[132] + p(204) * a[135],
                p(57) * (fmax(u(2), u(5)) * a[132] + (1 - fmax(u(2), u(5))) * a[133] * a[127]) + p(204) * a[135]
            );

    // Total ventilation through side vents
    // a.fVentSide
    a[137] = if_else(
                a[127] >= p(8),
                p(57) * a[134] + (1 - p(204)) * a[135],
                p(57) * (fmax(u(2), u(5)) * a[134] + (1 - fmax(u(2), u(5))) * a[133] * a[129]) + (1 - p(204)) * a[135]
            );


    // CO2 concentration in main compartment [ppm]
    a[138] = co2dens2ppm(x(2), 1e-6*x(0));

    // Density of air as it depends on pressure and temperature
    // a.rhoTop
    a[139] = p(36) * p(126) / ((x(3) + 273.15) * p(39));
    // a.rhoAir
    a[140] = p(36) * p(126) / ((x(2) + 273.15) * p(39));

    // Mean density of air beneath and above the screen
    // a.rhoMeanAir
    a[141] = 0.5 * (a[139] + a[140]);

    // Air flux through the thermal screen [m s^{-1}]
    // a.fThScr
    a[142] = u(2) * p(84) * pow(fabs(x(2) - x(3) + 1e-14), 0.66) +
        ((1. - u(2)) / a[141]) * sqrt(0.5 * a[141] * (1. - u(2)) * p(26) * fabs(a[140] - a[139]) + 1e-14);
    
    // Air flux through the blackout screen [m s^{-1}]
    // a.fBlScr
    a[143] = u(5) * p(94) * pow(fabs(x(2) - x(3) + 1e-14), 0.66) +
        ((1. - u(5)) / a[141]) * sqrt(0.5 * a[141] * (1. - u(5)) * p(26) * fabs(a[140] - a[139]) + 1e-14);

    // Air flux through the screens [m s^{-1}]
    // a.fScr
    a[144] = fmin(a[142], a[143]);

    //////////////////////////////////////////////////////////
    //// Convective and conductive heat fluxes [W m^{-2}] ////
    //////////////////////////////////////////////////////////

    // Forced ventilation (doesn't exist in current gh)
    a[145] = 0;

    // Between canopy and air in main compartment [W m^{-2}]
    a[146] = sensible(2 * p(0) * a[31], x(4), x(2));

    // Between air in main compartment and floor [W m^{-2}]
    a[147] = if_else(
                x(8) > x(2), 
                sensible(1.7 * pow(fabs(x(8) - x(2) + 1e-8), (1./3.)), x(2), x(8)),
                sensible(1.3 * pow(fabs(x(2) - x(8) + 1e-8), (1./4.)), x(2), x(8))
            );
    

    // Between air in main compartment and thermal screen [W m^{-2}]
    a[148] = sensible(1.7 * u(2) * pow(fabs(x(2) - x(7) + 1e-8), (1./3.)), x(2), x(7));

    // Between air in main compartment and blackout screen [W m^{-2}]
    a[149] = sensible(1.7 * u(5) * pow(fabs(x(2) - x(20) + 1e-8), (1./3.)), x(2), x(20));

    // Between air in main compartment and outside air [W m^{-2}]
    a[150] = sensible(p(111) * p(23) * (a[137] + a[145]), x(2), d(1));

    // Between air in main and top compartment [W m^{-2}]
    a[151] = sensible(p(111) * p(23) * a[144], x(2), x(3));

    // Between thermal screen and top compartment [W m^{-2}]
    a[152] = sensible(1.7 * u(2) * pow(fabs(x(7) - x(3) + 1e-8), 1./3.), x(7), x(3));

    // Between blackout screen and top compartment [W m^{-2}]
    a[153] = sensible(1.7 * u(5) * pow(fabs(x(20) - x(3) + 1e-8), 1./3.), x(20), x(3));

    // Between top compartment and cover [W m^{-2}]
    a[154] = sensible(p(50) * pow(fabs(x(3) - x(5) + 1e-8), (1./3.)) * p(47) / p(46), x(3), x(5));

    // Between top compartment and outside air [W m^{-2}]
    a[155] = sensible(p(111) * p(23) * a[136], x(3), d(1));

    // Between cover and outside air [W m^{-2}]
    a[156] = sensible(p(47) / p(46) * (p(51) + p(52) * pow(d(4), p(53))), x(6), d(1));

    // Between pipes and air in main compartment [W m^{-2}]
    a[157] = sensible(1.99 * M_PI * p(105) * p(107) * pow(fabs(x(9) - x(2) + 1e-8), 0.32), x(9), x(2));

    // Between floor and soil layer 1 [W m^{-2}]
    a[158] = sensible(2. / (p(101) / p(99) + p(27) / p(103)), x(8), x(10));

    // Between soil layers 1 and 2 [W m^{-2}]
    a[159] = sensible(2. * p(103) / (p(27) + p(28)), x(10), x(11));

    // Between soil layers 2 and 3 [W m^{-2}]
    a[160] = sensible(2. * p(103) / (p(28) + p(29)), x(11), x(12));

    // Between soil layers 3 and 4 [W m^{-2}]
    a[161] = sensible(2. * p(103) / (p(29) + p(30)), x(12), x(13));

    // Between soil layers 4 and 5 [W m^{-2}]
    a[162] = sensible(2 * p(103) / (p(30) + p(31)), x(13), x(14));

    // Between soil layer 5 and the external soil temperature [W m^{-2}]
    a[163] = sensible(2. * p(103) / (p(31) + p(37)), x(14), d(6));

    // Conductive heat flux through the lumped cover [W K^{-1} m^{-2}]
    // Since u[8] is always 0, we use the simplified expression
    a[164] = sensible(1. / (p(73) / p(71)), x(5), x(6));

    // Between lamps and air in main compartment [W m^{-2}]
    a[165] = sensible(p(185), x(17), x(2));

    // Between grow pipes and air in main compartment [W m^{-2}]
    a[166] = sensible(1.99 * M_PI * p(167) * p(166) * pow(fabs(x(19) - x(2) + 1e-8), 0.32), x(19), x(2));

    // Between interlights and air in main compartment [W m^{-2}]
    a[167] = sensible(p(198), x(18), x(2));

    // Smooth switch between day and night [-]
    // Equation 50 [1]
    a[168] = 1. / (1. + exp(p(43) * (a[45] - p(40))));

    // Parameter for CO2 influence on stomatal resistance [ppm{CO2}^{-2}]
    // Equation 51 [1]
    a[169] = p(20) * (1. - a[168]) + p(19) * a[168];

    // Parameter for vapor pressure influence on stomatal resistance [Pa^{-2}]
    a[170] = p(22) * (1. - a[168]) + p(21) * a[168];

    // Radiation influence on stomatal resistance [-]
    // Equation 49 [1]
    a[171] = (a[45] + p(17)) / (a[45] + p(18));

    // CO2 influence on stomatal resistance [-]
    // Equation 49 [1]
    a[172] = fmin(1.5, 1. + a[169] * pow((p(7) * x(0) - 200), 2));
    // Alternatively, you could use a[138] instead of p.etaMgPpm * x(0)

    // Vapor pressure influence on stomatal resistance [-]
    // Equation 49 [1]
    a[173] = fmin(5.8, 1. + a[170] * pow((satVP(x(4)) - x(15)), 2));

    // Stomatal resistance [s m^{-1}]
    // Equation 48 [1]
    a[174] = p(42) * a[171] * a[172] * a[173];
    
    // Vapor transfer coefficient of canopy transpiration [kg m^{-2} Pa^{-1} s^{-1}]
    // Equation 47 [1]
    a[175] = 2. * p(111) * p(23) * a[31] / (p(1) * p(14) * (p(41) + a[174]));

    // Canopy transpiration [kg m^{-2} s^{-1}]
    // Equation 46 [1]
    a[176] = (satVP(x(4)) - x(15)) * a[175];

    //////////////////////
    //// Vapor Fluxes ////
    //////////////////////

    // These are currently not used in the model..
    a[177] = 0;
    a[178] = 0;
    a[179] = 0;
    a[180] = 0;

    // Condensation from main compartment on thermal screen [kg m^{-2} s^{-1}]
    // Table 4 [1], Equation 42 [1]
    a[181] = cond(1.7 * u(2) * pow(fabs(x(2) - x(7) + 1e-8), (1./3.)), x(15), satVP(x(7)));

    // Condensation from main compartment on blackout screen [kg m^{-2} s^{-1}]
    // Equatio A39 [5], Equation 7.39 [7]
    a[182] = cond(1.7 * u(5) * pow(fabs(x(2) - x(20) + 1e-8), (1./3.)), x(15), satVP(x(20)));

    // Condensation from top compartment to cover [kg m^{-2} s^{-1}]
    // Table 4 [1]
    a[183] = cond(p(50)* pow(fabs(x(3) - x(5) + 1e-8), (1./3.)) * p(47)/p(46), x(16), satVP(x(5)));

    // Vapor flux from main to top compartment [kg m^{-2} s^{-1}]
    a[184] = airMv(a[144], x(15), x(16), x(2), x(3));

    // Vapor flux from top compartment to outside [kg  m^{-2} s^{-1}]
    a[185] = airMv(a[136], x(16), d(2), x(3), d(1));

    // Vapor flux from main compartment to outside [kg m^{-2} s^{-1}]
    a[186] = airMv(a[137]+a[145], x(15), d(2), x(2), d(1));

    ////////////////////////////
    //// Latent heat fluxes ////
    ////////////////////////////

    a[187] = p(1) * a[176];
    a[188] = p(1) * a[181];
    a[189] = p(1) * a[182];
    a[190] = p(1) * a[183];

    //////////////////////////////////////
    //////// Canopy photosynthesis ///////
    //////////////////////////////////////

    // PAR absorbed by the canopy [umol{photons} m^{-2} s^{-1}]
    // Equation 17 [2]
    a[191] = p(187) * a[55] + p(140) * a[54] + p(197) * a[56];

    // Maximum rate of electron transport rate at 25C [umol{e-} m^{-2} s^{-1}]
    // Equation 16 [2]
    a[192] = a[31] * p(129);

    // CO2 compensation point [ppm]
    // Equation 23 [2]
    a[193] = (p(129) / a[192]) * p(130) * x(4) + 20 * p(130) * (1 - (p(129) / a[192]));

    // CO2 concentration in the stomata [ppm]
    // Equation 21 [2]
    a[194] = p(131) * a[138];

    // // Potential rate of electron transport [umol{e-} m^{-2} s^{-1}]
    // // Equation 15 [2]
    // // Note that R in [2] is 8.314 and R in [1] is 8314
    a[195] = a[192] * exp(p(132) * (x(4) + 273.15 - p(133)) / (1e-3*p(39) * (x(4) + 273.15) * p(133))) *
        (1 + exp((p(134) * p(133) - p(135)) / (1e-3*p(39) * p(133)))) /
        (1 + exp((p(134) * (x(4) + 273.15) - p(135)) / (1e-3*p(39) * (x(4) + 273.15))));

    // // Electron transport rate [umol{e-} m^{-2} s^{-1}]
    // // Equation 14 [2]
    a[196] = (1. / (2. * p(136))) * (a[195] + p(137) * a[191] -
        sqrt(pow((a[195] + p(137) * a[191]), 2) - 4*p(136) * a[195] * p(137) * a[191] + 1e-8));

    // // Photosynthesis rate at canopy level [umol{co2} m^{-2} s^{-1}]
    // // Equation 12 [2]
    a[197] = a[196] * (a[194]-a[193]) / (4*(a[194] + 2*a[193]));

    // // Photrespiration [umol{co2} m^{-2} s^{-1}]
    // // Equation 13 [2]
    a[198] = a[197]*a[193] / a[194];

    // // Inhibition due to full carbohydrates buffer [-]
    // // Equation 11, Equation B.1, Table 5 [2]
    a[199] = 1. / (1. + exp(5e-4*(x(22) - p(157))));

    // // Net photosynthesis [mg{CH2O} m^{-2} s^{-1}]
    // // Equation 10 [2]
    a[200] = p(138) * a[199] * (a[197] - a[198]);

    // //// Carbohydrate buffer
    // // Temperature effect on structural carbon flow to organs
    // // Equation 28 [2]
    a[201] = 0.047*x(21) + 0.06;

    // // Inhibition of carbohydrate flow to the organs
    // // Equation B.3 [2]
    a[202] = 1. / (1. + exp(-1.1587 * (x(21)-p(160)))) *
        1. / (1. + exp(1.3904*(x(21) - p(159))));

    // // Inhibition of carbohydrate flow to the fruit
    // // Equation B.3 [2]
    a[203] = 1. / (1. + exp(-0.869*(x(4) - p(162)))) * // hTcan
        1. / (1. + exp(0.5793*(x(4) - p(161))));

    // // Inhibition due to development stage 
    // // Equation B.6 [2]
    a[204] = 0.5 *(x(26) / p(163) +
        sqrt(pow((x(26) / p(163)), 2) + 1e-4)) -
        0.5 * ((x(26) - p(163)) / p(163) +
        sqrt(pow(((x(26) - p(163)) / p(163)), 2) + 1e-4));

    // Inhibition due to development stage
    // hTcan
    // a[205] = 1 / (1 + exp(0.01 * (a(26) - p.tEndSumGrowth)));
    a[205] = 1;

    // // Inhibition due to insufficient carbohydrates in the buffer [-]
    // // Equation 26 [2]
    a[206] = 1. / (1. + exp(-5e-3*(x(22) - p(158))));

    // // Carboyhdrate flow from buffer to leaves [mg{CH2O} m^{2} s^{-1}]
    // Equation 25 [2]
    a[207] = a[206] * a[202] * a[201] * p(155);

    // // Carboyhdrate flow from buffer to stem [mg{CH2O} m^{2} s^{-1}]
    // // Equation 25 [2]
    a[208] = a[206] * a[202] * a[201] * p(156);

    // // Carboyhdrate flow from buffer to fruit [mg{CH2O} m^{2} s^{-1}]
    // // Equation 24 [2]
    a[209] = a[206] * a[203] * a[202] * a[204] * a[201] * p(154) * a[205];

    // Growth respiration [mg{CH2O} m^{-2] s^{-1}]
    // Equations 43-44 [2]
    a[210] = p(147)*a[207] + p(148)*a[208] + p(146)*a[209];

    // Leaf maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    // Equation 45 [2]
    a[211] = (1. - exp(-p(149) * p(143))) * pow(p(150), 0.1*(x(21)-25)) * x(23) * p(152);

    // Stem maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    a[212] = (1. - exp(-p(149) * p(143))) * pow(p(150), 0.1 * (x(21) - 25)) * x(24) * p(153);

    // Fruit maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    // Equation 45 [2]
    a[213] = (1. - exp(-p(149) * p(143))) * pow(p(150), (0.1*(x(21) - 25))) * x(25) * p(151);

    // Total maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    // Equation 45 [2]
    a[214] = a[211] + a[212] + a[213];

    //// Leaf pruning and fruit harvest
    // A new smoothing function has been applied here to avoid stiffness
    // Leaf pruning [mg{CH2O} m^{-2] s^{-1}]
    // Equation B.5 [2]
    a[215] = smoothHar(x(23), p(144), 1e4, 5e4);

    // Fruit harvest [mg{CH2O} m^{-2} s^{-1}]
    // Equation A45 [5], Equation 7.45 [7]
    a[216] = smoothHar(x(25), p(145), 1e4, 5e4);

    // Net crop assimilation [mg{CO2} m^{-2} s^{-1}]
    // It is assumed that for every mol of CH2O in net assimilation, a mol
    // of CO2 is taken from the air, thus the conversion uses molar masse
    a[217] = (p(139)/p(138)) * (a[200]-a[210]-a[214]);

    // Other CO2 flows [mg{CO2} m^{-2} s^{-1}]

    // From main to top compartment 
    a[218] = airMc(a[144], x(0), x(1)); // a.mcAirTop

    // From top compartment outside
    a[219] = airMc(a[136], x(1), d(3)); 

    // From main compartment outside
    a[220] = airMc(a[137] + a[145], x(0), d(3)); // a.mcAirOut

    //// Heat from boiler - Section 9.2 [1]

    // Heat from boiler to pipe rails [W m^{-2}]
    // Equation 55 [1]
    a[221] = u(0) * p(108);

    // Heat from boiler to grow pipes [W m^{-2}]
    a[222] = 0; // Excluded the grow pipes from the model
    // a[222] = u(6) * p(170);

    //  CO2 injection [mg m^{-2} s^{-1}]
    a[223] = u(1) * p(109);

    // Objects not currently included in the model
    a[224] = 0;
    a[225] = 0;
    a[226] = 0;
    a[227] = 0;
    a[228] = 0;
    a[229] = 0;
    a[230] = 0;
    a[231] = 0;
    a[232] = 0;
    a[233] = 0;

    //  Lamp cooling
    // Equation A34 [5], Equation 7.34 [7]
    a[234] = p(186) * a[37];

    // Heat harvesting, mechanical cooling and dehumidification
    // By default there is no mechanical cooling or heat harvesting
    // see addHeatHarvesting.m for mechanical cooling and heat harvesting
    a[235] = 0;
    a[236] = 0;
    a[237] = 0;
    a[238] = 0;
    a[239] = 0;

    return vertcat(a);
}