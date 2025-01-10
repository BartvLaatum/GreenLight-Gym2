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
    a[10] = 1 - u(7) * (1 - p(90));
    a[11] = u(7) * p(88);
    a[12] = tau12(a[2], a[10], a[4], a[11]);
    a[13] = rhoUp(a[2], a[3], a[4], a[11]);
    a[14] = rhoDn(a[10], a[4], a[11], a[11]);

    a[15] = 1 - u(7) * (1 - p(89));
    a[16] = u(7) * p(87);
    a[17] = tau12(a[7], a[15], a[9], a[16]);
    a[18] = rhoUp(a[7], a[8], a[9], a[16]);
    a[19] = rhoDn(a[15], a[9], a[16], a[16]);

    // Full cover model
    a[20] = tau12(a[12], p(176), a[14], p(179));
    a[21] = rhoUp(a[12], a[13], a[14], p(179));
    a[22] = tau12(a[17], p(177), a[19], p(180));
    a[23] = rhoUp(a[17], a[18], a[19], p(180));

    // // Thermal Screen and Roof PAR
    // a[0] = 1 - u(2) * (1 - p(80));              // tauThScrPar;         1 - u[2] * (1 - p.tauThScrPar);
    // a[1] = u(2) * p(77);                        // rhoThScrPar;         u[2] * p.rhoThScrPar;
    // a[2] = tau12(p(69), a[0], p(66), a[1]);     // tauCovThScrPar;      tau12(p.tauRfPar, a.tauThScrPar, p.rhoRfPar, a[1]);
    // a[3] = rhoUp(p(69), p(66), p(66), a[1]);    // rhoCovThScrParUp;    rhoUp(p.tauRfPar, p.rhoRfPar, p.rhoRfPar, a.rhoThScrPar);
    // a[4] = rhoDn(a[0], p(66), a[1], a[1]);      // rhoCovThScrParDn;    rhoDn(a.tauThScrPar, p.rhoRfPar, a.rhoThScrPar, a.rhoThScrPar);

    // // Thermal Screen and Roof NIR
    // a[5] = 1 - u(2) * (1 - p(79));              // a[5];       1 - u[2] * (1 - p.tauThScrNir);
    // a[6] = u(2) * p(76);                        // a.rhoThScrNir;       u[2] * p.rhoThScrNir;
    // a[7] = tau12(p(68), a[5], p(65), a[6]);     // a.tauCovThScrNir;    tau12(p.tauRfNir, a.tauThScrNir, p.rhoRfNir, a.rhoThScrNir);
    // a[8] = rhoUp(p(68), p(65), p(65), a[6]);    // rhoCovThScrNirUp;    rhoUp(p.tauRfNir, p.rhoRfNir, p.rhoRfNir, a.rhoThScrNir);
    // a[9] = rhoDn(a[5], p(65), a[6], a[6]);     // rhoCovThScrNirDn;    rhoDn(a.tauThScrNir, p.rhoRfNir, a.rhoThScrNir, a.rhoThScrNir);

    // // Blackout screen PAR
    // a[10] = 1 - u(7) * (1 - p(90));             // tauBlScrPar;         1 - u[7] * (1 - p(90));
    // a[11] = u(7) * p(88);                       // rhoBlScrPar;         u[7] * p.rhoBlScrPar;
    // a[12] = tau12(a[2], a[10], a[4], a[11]);    // tauCovBlScrPar;      tau12(a[2], a.tauBlScrPar, a[4], a.rhoBlScrPar);
    // a[13] = rhoUp(a[2], a[3], a[4], a[11]);     // rhoCovBlScrParUp;    rhoUp(a.tauCovThScrPar, a[3], a.rhoCovThScrParDn, a.rhoBlScrPar);
    // a[14] = rhoDn(a[10], a[4], a[11], a[11]);   // rhoCovBlScrParDn;    rhoDn(a.tauBlScrPar, a.rhoCovThScrParDn, a.rhoBlScrPar, a.rhoBlScrPar);
    
    // // Blackout screen NIR
    // a[15] = 1 - u(7) * (1 - p(89));             // tauBlScrNir;         1 - u[7] * (1 - p.tauBlScrNir);
    // a[16] = u(7) * p(87);                       // rhoBlScrNir;         u[7] * p.rhoBlScrNir;
    // a[17] = tau12(a[7], a[15], a[9], a[16]);   // tauCovBlScrNir;      tau12(a.tauCovThScrNir, a.tauBlScrNir, a.rhoCovThScrNirDn, a.rhoBlScrNir);
    // a[18] = rhoUp(a[7], a[8], a[9], a[16]);    // rhoCovBlScrNirUp;    rhoUp(a.tauCovThScrNir, a[8], a.rhoCovThScrNirDn, a.rhoBlScrNir);
    // a[19] = rhoDn(a[15], a[9], a[16], a[16]);  // rhoCovBlScrNirDn;    rhoDn(a.tauBlScrNir, a.rhoCovThScrNirDn, a.rhoBlScrNir, a.rhoBlScrNir);

    // // Full cover model
    // a[20] = tau12(a[12], p(176), a[14], p(179));    // tauCovPar;       tau12(a.tauCovBlScrPar, p.tauLampPar, a.rhoCovBlScrParDn, p.rhoLampPar);
    // a[21] = rhoUp(a[12], a[13], a[14], p(179));     // rhoCovPar;       rhoUp(a.tauCovBlScrPar, a.rhoCovBlScrParUp, a.rhoCovBlScrParDn, p.rhoLampPar);
    // a[22] = tau12(a[17], p(177), a[19], p(180));    // tauCovNir;       tau12(a.tauCovBlScrNir, p.tauLampNir, a.rhoCovBlScrNirDn, p.rhoLampNir);
    // a[23] = rhoUp(a[17], a[18], a[19], p(180));     // rhoCovNir;       rhoUp(a.tauCovBlScrNir, a.rhoCovBlScrNirUp, a.rhoCovBlScrNirDn, p.rhoLampNir);

    a[24] = p(70);                                  // tauCovFir;       p.tauRfFir;
    a[25] = p(67);                                  // rhoCovFir;       p.rhoRfFir;
    a[26] = 1 - a[20] - a[21];                      // aCovPar;         1 - a.tauCovPar - a.rhoCovPar;
    a[27] = 1 - a[22] - a[23];                      // aCovNir;         1 - a.tauCovNir - a.rhoCovNir;
    a[28] = 1 - a[24] - a[25];                      // aCovFir;         1 - a.tauCovFir - a.rhoCovFir;
    a[29] = a[28];                                  // epsCovFir;       a.aCovFir;
    a[30] = cos(degrees2rad(p(45))) * p(73) * p(64) * p(72);    // capCov;  std::cos(rad2degrees(p.psi)) * p.hRf * p.rhoRf * p.cPRf;

    // Capacities
    a[31] = p(142) * x(23);                         // lai;             p.sla * x[23];
    a[32] = p(16) * a[31];                          // capCan;          p.capLeaf * a.lai;
    a[33] = 0.1 * a[30];                            // capCovE;         0.1 * a.capCov;
    a[34] = 0.1 * a[30];                            // capCovIn;        0.1 * a.capCov;
    a[35] = p(38) * p(48) / (p(39) * (x(2) + 273.15)); // capVpAir;      p.mWater * p.hAir / (p.R * (x[2] + 273.15));
    a[36] = p(38) * (p(49) - p(48)) / (p(39) * (x(3) + 273.15)); // capVpTop;  p.mWater * (p.hGh - p.hAir) / (p.R * (x[3] + 273.15));

    // PAR radiation
    a[37] = p(172) * u(4);                          // qLampIn;         p.thetaLampMax * u[4];
    a[38] = p(196) * u(5);                          // qIntLampIn;      p.thetaIntLampMax * u[5];
    a[39] = (1 - p(44)) * a[20] * p(6) * d(0);      // rParGhSun;       (1 - p.etaGlobAir) * a.tauCovPar * p.etaGlobPar * d[0];
    a[40] = p(174) * a[37];                         // rParGhLamp;      p.etaLampPar * a.qLampIn;
    a[41] = p(192) * a[38];                         // rParGhIntLamp;   p.etaIntLampPar * a.qIntLampIn;
    a[42] = (1 - p(44)) * d(0) * (p(6) * a[20] + p(5) * a[22]); // rCanSun; (1 - p.etaGlobAir) * d[0] * (p.etaGlobPar * a.tauCovPar + p.etaGlobNir * a.tauCovNir);
    a[43] = (p(174) + p(175)) * a[37];              // rCanLamp;        (p.etaLampPar + p.etaLampNir) * a.qLampIn;
    a[44] = (p(192) + p(193)) * a[38];              // rCanIntLamp;     (p.etaIntLampPar + p.etaIntLampNir) * a.qIntLampIn;
    a[45] = a[42] + a[43] + a[44];                  // rCan;            a.rCanSun + a.rCanLamp + a.rCanIntLamp;

    a[46] = a[39] * (1 - p(10)) * (1 - exp(-p(32) * a[31]));    // rParSunCanDown;     a.rParGhSun * (1 - p.rhoCanPar) * (1 - std::exp(-p.k1Par * a.lai));
    a[47] = a[40] * (1 - p(10)) * (1 - exp(-p(32) * a[31]));    // rParLampCanDown;    a.rParGhLamp * (1 - p.rhoCanPar) * (1 - std::exp(-p.k1Par * a.lai));
    a[48] = 1 - p(190) * exp(-p(200) * p(189) * a[31]) +        // fIntLampCanPar;     1 - p.fIntLampDown * std::exp(-p.k1IntPar * p.vIntLampPos * a.lai) +
            (p(190)-1) * exp(-p(200) * (1-p(189)) *a[31]);      //                     (p.fIntLampDown - 1) * std::exp(-p.k1IntPar * (1 - p.vIntLampPos) * a.lai);
    a[49] = 1 - p(190) * exp(-p(202) * p(189) * a[31]) +        // fIntLampCanNir;     1 - p.fIntLampDown * std::exp(-p.kIntNir * p.vIntLampPos * a.lai) +
            (p(190)-1) * exp(-p(202) * (1-p(189)) * a[31]);     //                     (p.fIntLampDown - 1) * std::exp(-p.kIntNir * (1 - p.vIntLampPos) * a.lai);
    a[50] = a[41] * a[48] * (1 - p(10));                        // rParIntLampCanDown; a.rParGhIntLamp * a.fIntLampCanPar * (1 - p.rhoCanPar);

    a[51] = a[39] * exp(-p(32) * a[31]) * p(98) * (1 - p(10)) * (1 - exp(-p(33) * a[31]));    // rParSunFlrCanUp;     a.rParGhSun * std::exp(-p.k1Par * a.lai) * p.rhoFlrPar * (1 - p.rhoCanPar) * (1 - std::exp(-p.k2Par * a.lai));
    a[52] = a[40] * exp(-p(32) * a[31]) * p(98) * (1 - p(10)) * (1 - exp(-p(33) * a[31]));    // rParLampFlrCanUp;    a.rParGhLamp * std::exp(-p.k1Par * a.lai) * p.rhoFlrPar * (1 - p.rhoCanPar) * (1 - std::exp(-p.k2Par * a.lai));
    a[53] = a[41] * p(190) * exp(-p(200) * p(189) * a[31]) * p(98) * (1 - p(10)) *          // rParIntLampFlrCanUp; a.rParGhIntLamp * p.fIntLampDown *std::exp(-p.k1IntPar * p.vIntLampPos * a.lai) * p.rhoFlrPar * (1 - p.rhoCanPar) *
            (1-exp(-p(201)*a[31]));                                                         //                      (1 - std::exp(-p.k2IntPar * a.lai));

    a[54] = a[46] + a[51];  // rParSunCan; a.rParSunCanDown + a.rParSunFlrCanUp;
    a[55] = a[47] + a[52];  // rParLampCan; a.rParLampCanDown + a.rParLampFlrCanUp;
    a[56] = a[50] + a[53];  // rParIntLampCan; a.rParIntLampCanDown + a.rParIntLampFlrCanUp;    

    // NIR radiation
    a[57] = 1 - a[23];              // tauHatCovNir;    1 - a.rhoCovNir;
    a[58] = 1 - p(97);              // tauHatFlrNir;    1 - p.rhoFlrNir;
    a[59] = exp(-p(34) * a[31]);    // tauHatCanNir;    exp(-p.kNir * a.lai);
    a[60] = p(11) * (1 - a[59]);    // rhoHatCanNir;    p.rhoCanNir * (1 - a.tauHatCanNir);

    a[61] = tau12(a[57], a[59], a[23], a[60]);  // tauCovCanNir;    tau12(a.tauHatCovNir, a.tauHatCanNir, a.rhoCovNir, a.rhoHatCanNir);
    a[62] = rhoUp(a[59], a[23], a[23], a[60]);  // rhoCovCanNirUp;  rhoUp(a.tauHatCanNir, a.rhoCovNir, a.rhoCovNir, a.rhoHatCanNir);
    a[63] = rhoDn(a[59], a[23], a[60], a[60]);  // rhoCovCanNirDn;  rhoDn(a.tauHatCanNir, a.rhoCovNir, a.rhoHatCanNir, a.rhoHatCanNir);
    a[64] = tau12(a[61], a[58], a[63], p(97));  // tauCovCanFlrNir; tau12(a.tauCovCanNir, a.tauHatFlrNir, a.rhoCovCanNirDn, p.rhoFlrNir);
    a[65] = rhoUp(a[61], a[62], a[63], p(97));  // rhoCovCanFlrNir; rhoUp(a.tauCovCanNir, a.rhoCovCanNirUp, a.rhoCovCanNirDn, p.rhoFlrNir);

    a[66] = 1 - a[64] - a[65];                   // aCanNir;     1 - a.tauCovCanFlrNir - a.rhoCovCanFlrNir;
    a[67] = a[64];                              // aFlrNir;     a.tauCovCanFlrNir;
    a[68] = (1 - p(44)) * a[66] * p(5) * d(0);  // rNirSunCan;  (1 - p.etaGlobAir) * a.aCanNir * p.etaGlobNir * d[0];
    a[69] = p(175) * a[37] * (1 - p(11)) * (1 - exp(-p(34) * a[31]));   // rNirLampCan; p.etaLampNir * a.qLampIn * (1 - p.rhoCanNir) * (1 - std::exp(-p.kNir * a.lai));
    a[70] = p(193) * a[38] * a[49] * (1 - p(11));   // rNirIntLampCan; p.etaIntLampNir * a.qIntLampIn * a.fIntLampCanNir * (1 - p.rhoCanNir);

    a[71] = (1 - p(44)) * a[67] * p(5) * d(0);                                      // rNirSunFlr;      (1 - p.etaGlobAir) * a.aFlrNir * p.etaGlobNir * d[0];
    a[72] = (1 - p(97)) * exp(-p(34) * a[31]) * p(175) * a[37];                     // rNirLampFlr;     (1 - p.rhoFlrNir) * std::exp(-p.kNir * a.lai) * p.etaLampNir * a.qLampIn;
    a[73] = p(190) * (1 - p(97)) * exp(-p(201)  * a[31]* p(189)) * p(193) * a[38];  // rNirIntLampFlr;  p.fIntLampDown * (1 - p.rhoFlrNir) * std::exp(-p.kIntNir * a.lai * p.vIntLampPos) * p.etaIntLampNir * a.qIntLampIn;

    a[74] = (1 - p(98)) * exp(-p(32) * a[31]) * a[39];                      // rParSunFlr; (1 - p.rhoFlrPar) * std::exp(-p.k1Par * a.lai) * a.rParGhSun;
    a[75] = (1 - p(98)) * exp(-p(32) * a[31]) * a[40];                      // rParLampFlr; (1 - p.rhoFlrPar) * std::exp(-p.k1Par * a.lai) * a.rParGhLamp;
    a[76] = a[41] * p(190) * (1 - p(98)) * exp(-p(200) * a[31] * p(189));   // rParIntLampFlr; a.rParGhIntLamp * p.fIntLampDown * (1 - p.rhoFlrPar) * std::exp(-p.k1IntPar * a.lai * p.vIntLampPos);

    a[77] = (p(174) + p(175)) * a[37] - a[55] - a[69] - a[75] - a[72];  // rLampAir;    (p.etaLampPar + p.etaLampNir) * a.qLampIn - a.rParLampCan - a.rNirLampCan - a.rParLampFlr - a.rNirLampFlr;
    a[78] = (p(192) + p(193)) * a[38] - a[56] - a[70] - a[76] - a[73];  // rIntLampAir; (p.etaIntLampPar + p.etaIntLampNir) * a.qIntLampIn - a.rParIntLampCan - a.rNirIntLampCan - a.rParIntLampFlr - a.rNirIntLampFlr;
    a[79] = p(44) * d(0) * (a[20] * p(6) + (a[66] + a[67]) * p(5));     // rGlobSunAir; p.etaGlobAir * d[0] * (a.tauCovPar * p.etaGlobPar + (a.aCanNir + a.aFlrNir) * p.etaGlobNir);
    a[80] = (a[26] * p(6) + a[27] * p(5)) * d(0);                       // rGlobSunCovE; (a.aCovPar * p.etaGlobPar + a.aCovNir * p.etaGlobNir) * d[0];

    // FIR radiation
    a[81] = 1 - u(2) * (1 - p(81));             // tauThScrFirU;    1 - u[2] * (1 - p.tauThScrFir);
    a[82] = 1 - u(7) * (1 - p(91));             // tauBlScrFirU;    1 - u[7] * (1 - p.tauBlScrFir);
    a[83] = 1 - exp(-p(35) * a[31]);            // aCan;            1 - std::exp(-p.kFir * a.lai);

    // from canopy
    a[84] = fir(a[83], p(3), a[29], p(178) * a[81] * a[82], x(4), x(5), p(2));          // rCanCovIn;   fir(a.aCan, p.epsCan, a.epsCovFir, p.tauLampFir * a.tauThScrFirU * a.tauBlScrFirU, x[4], x[5], p.sigma);
    a[85] = fir(a[83], p(3), p(4), p(178) * a[24] * a[81] * a[82], x(4), d(5), p(2));   // rCanSky;     fir(a.aCan, p.epsCan, p.epsSky, p.tauLampFir * a.tauCovFir * a.tauThScrFirU * a.tauBlScrFirU, x[4], d[5], p.sigma);
    a[86] = fir(a[83], p(3), p(74), p(178) * u(2) * a[82], x(4), x(7), p(2));             // rCanThScr;   fir(a.aCan, p.epsCan, p.epsThScrFir, p.tauLampFir * u[2] * a.tauBlScrFirU, x[4], x[7], p.sigma);
    a[87] = fir(a[83], p(3), p(95), p(125), x(4), x(8), p(2));  	                    // rCanFlr;     fir(a.aCan, p.epsCan, p.epsFlr, p.fCanFlr, x[4], x[8], p.sigma);

    // from heating pipes
    a[88] = fir(p(124), p(104), a[29], p(199) * p(178) * a[81] * a[82] * 0.49 * exp(-p(35) * a[31]), x(9), x(5), p(2)); // rPipeCovIn;  fir(p.aPipe, p.epsPipe, a.epsCovFir, p.tauIntLampFir * p.tauLampFir * a.tauThScrFirU * a.tauBlScrFirU * 0.49 * std::exp(-p.kFir * a.lai), x[9], x[5], p.sigma);
    a[89] = fir(p(124), p(104), p(4), p(199)  * p(178) * a[24] * a[81] * 0.49 * exp(-p(35) * a[31]), x(9), d(5), p(2)); // rPipeSky;    fir(p.aPipe, p.epsPipe, p.epsSky, p.tauIntLampFir * p.tauLampFir * a.tauCovFir * a.tauThScrFirU * 0.49 * std::exp(-p.kFir * a.lai), x[9], d[5], p.sigma);
    a[90] = fir(p(124), p(104), p(74), p(199)  * p(178) * u(2) * a[82] * 0.49 * exp(-p(35) * a[31]), x(9), x(7), p(2)); // rPipeThScr;  fir(p.aPipe, p.epsPipe, p.epsThScrFir, p.tauIntLampFir * p.tauLampFir * u[2] * a.tauBlScrFirU * 0.49 * std::exp(-p.kFir * a.lai), x[9], x[7], p.sigma);
    a[91] = fir(p(124), p(104), p(95), 0.49, x(9), x(8), p(2));  	                                                    // rPipeFlr;    fir(p.aPipe, p.epsPipe, p.epsFlr, 0.49, x[9], x[8], p.sigma);
    a[92] = fir(p(124), p(104), p(3), 0.49 * (1 - exp(-p(35) * a[31])), x(9), x(4), p(2));                              // rPipeCan;    fir(p.aPipe, p.epsPipe, p.epsCan, 0.49 * (1 - std::exp(-p.kFir * a.lai)), x[9], x[4], p.sigma);

    // from floor
    a[93] = fir(1, p(95), a[29], p(199) * p(178) * a[81] * a[82] * (1 - 0.49 * M_PI * p(107) * p(105)) * exp(-p(35) * a[31]), x(8), x(5), p(2)); // rFlrCovIn;  fir(1, p.epsFlr, a.epsCovFir, p.tauIntLampFir * p.tauLampFir * a.tauThScrFirU * a.tauBlScrFirU * (1 - 0.49 * M_PI * p.lPipe * p.phiPipeE) * std::exp(-p.kFir * a.lai), x[8], x[5], p.sigma);
    a[94] = fir(1, p(95), p(4), p(199) * p(178) * a[24] * a[81] * a[82] * (1 - 0.49 * M_PI * p(107) * p(105)) * exp(-p(35) * a[31]), x(8), d(5), p(2)); // rFlrSky; fir(1, p.epsFlr, p.epsSky, p.tauIntLampFir * p.tauLampFir * a.tauCovFir * a.tauThScrFirU * a.tauBlScrFirU * (1 - 0.49 * M_PI * p.lPipe * p.phiPipeE) * std::exp(-p.kFir * a.lai), x[8], d[5], p.sigma); 
    a[95] = fir(1, p(95), p(74), p(199) * p(178) * u(2) * a[82] * (1 - 0.49 * M_PI * p(107) * p(105)) * exp(-p(35) * a[31]), x(8), x(7), p(2)); // rFlrThScr; fir(1, p.epsFlr, p.epsThScrFir, p.tauIntLampFir * p.tauLampFir * u[2] * a.tauBlScrFirU * (1 - 0.49 * M_PI * p.lPipe * p.phiPipeE) * std::exp(-p.kFir * a.lai), x[8], x[7], p.sigma);

    // from cover
    a[96] = fir(1, p(74), a[29], u(2), x(7), x(5), p(2));           // rThScrCovIn; fir(1, p.epsThScrFir, a.epsCovFir, u[2], x[7], x[5], p.sigma);
    a[97] = fir(1, p(74), p(4), a[24] * u(2), x(7), d(5), p(2));    // rThScrSky;   fir(1, p.epsThScrFir, p.epsSky, a.tauCovFir * u[2], x[7], d[5], p.sigma);

    // from cover
    a[98] = fir(1, a[28], p(4), 1, x(6), d(5), p(2));              // rCovESky; fir(1, a.aCovFir, p.epsSky, 1, x[6], d[5], p.sigma);

    // from lamps
    a[99] = fir(p(181), p(183), p(95), p(199) * (1 - 0.49 * M_PI * p(107) * p(105)) * exp(-p(35) * a[31]), x(17), x(8), p(2)); // rFirLampFlr; fir(p.aLamp, p.epsLampBottom, p.epsFlr, p.tauIntLampFir * (1 - 0.49 * M_PI * p.lPipe * p.phiPipeE) * std::exp(-p.kFir * a.lai), x[17], x[8], p.sigma);
    a[100] = fir(p(181), p(183), p(104), p(199) * 0.49 * M_PI * p(107) * p(105) * exp(-p(35) * a[31]), x(17), x(9), p(2));      // rLampPipe;   fir(p.aLamp, p.epsLampBottom, p.epsPipe, p.tauIntLampFir * 0.49 * M_PI * p.lPipe * p.phiPipeE * std::exp(-p.kFir * a.lai), x[17], x[9], p.sigma);
    a[101] = fir(p(181), p(183), p(3), a[83], x(17), x(4), p(2));           // rFirLampCan; fir(p.aLamp, p.epsLampBottom, p.epsCan, a.aCan, x[17], x[4], p.sigma);
    a[102] = fir(p(181), p(182), p(74), u(2) * a[82], x(17), x(7), p(2));   // rLampThScr;  fir(p.aLamp, p.epsLampTop, p.epsThScrFir, u[2] * a.tauBlScrFirU, x[17], x[7], p.sigma);
    a[103] = fir(p(181), p(182), a[29], a[81] * a[82], x(17), x(5), p(2));   // rLampCovIn; fir(p.aLamp, p.epsLampTop, a.epsCovFir, a.tauThScrFirU * a.tauBlScrFirU * a.aCan, x[17], x[5], p.sigma);
    a[104] = fir(p(181), p(182), p(4), a[24] * a[81] * a[82], x(17), d(5), p(2)); // rLampSky; fir(p.aLamp, p.epsLampTop, p.epsSky, a.tauCovFir * a.tauThScrFirU * a.tauBlScrFirU, x[17], d[5], p.sigma);

    // grow pipes
    a[105] = fir(p(169), p(165), p(3), 1, x(19), x(4), p(2)); // rGroPipeCan; fir(p.aGroPipe, p.epsGroPipe, p.epsCan, 1, x[19], x[4], p.sigma);

    // blackout screen
    a[106] = fir(1, p(95), p(85), p(199) * p(178) * u(7) * (1 - 0.49 * M_PI * p(107) * p(105)) * exp(-p(35) * a[31]), x(8), x(20), p(2)); // rFlrBlScr; fir(1, p.epsFlr, p.epsBlScrFir, p.tauIntLampFir * p.tauLampFir * u[7] * (1 - 0.49 * M_PI * p.lPipe * p.phiPipeE) * std::exp(-p.kFir * a.lai), x[8], x[20], p.sigma);
    a[107] = fir(p(124), p(104), p(85), p(199) * p(178) * u(7) * 0.49 * exp(-p(35) * a[31]), x(9), x(20), p(2));    // rPipeBlScr;  fir(p.aPipe, p.epsPipe, p.epsBlScrFir, p.tauIntLampFir * p.tauLampFir * u[7] * 0.49 * std::exp(-p.kFir * a.lai), x[9], x[20], p.sigma);
    a[108] = fir(a[83], p(3), p(85), p(178) * u(7), x(4), x(20), p(2));     // rCanBlScr;   fir(a.aCan, p.epsCan, p.epsBlScrFir, p.tauLampFir * u[7], x[4], x[20], p.sigma);
    a[109] = fir(u(7), p(85), p(74), u(2), x(20), x(7), p(2));              // rBlScrThScr; fir(u[7], p.epsBlScrFir, p.epsThScrFir, u[2], x[20], x[7], p.sigma);
    a[110] = fir(u(7), p(85), a[29], a[81], x(20), x(5), p(2));              // rBlScrCovIn; fir(u[7], p.epsBlScrFir, a.epsCovFir, a.tauThScrFirU, x[20], x[5], p.sigma);
    a[111] = fir(u(7), p(85), p(4), a[24] * a[81], x(20), d(5), p(2));       // rBlScrSky;   fir(u[7], p.epsBlScrFir, p.epsSky, a.tauCovFir * a.tauThScrFirU, x[20], d[5], p.sigma);
    a[112] = fir(p(181), p(182), p(85), u(7), x(17), x(20), p(2));           // rLampBlScr;  fir(p.aLamp, p.epsLampTop, p.epsBlScrFir, u[7], x[17], x[20], p.sigma);

    // interlights
    a[113] = 1 - exp(-p(203) * (1 - p(189)) * a[31]);   // fIntLampCanUp;   1 - std::exp(-p.kIntFir * (1 - p.vIntLampPos) * a.lai);
    a[114] = 1 - exp(-p(203) * p(189) * a[31]);         // fIntLampCanDown  1 - std::exp(-p.kIntFir * p.vIntLampPos * a.lai);
    a[115] = fir(p(194), p(195), p(95), (1 - 0.49 * M_PI * p(107) * p(105)) * (1 - a[114]), x(18), x(8), p(2)); // rFirIntLampFlr;  fir(p.aIntLamp, p.epsIntLamp, p.epsFlr, (1 - 0.49 * M_PI * p.lPipe * p.phiPipeE) * (1 - a.fIntLampCanDown), x[18], x[8], p.sigma);
    a[116] = fir(p(194), p(195), p(104), 0.49 * M_PI * p(107) * p(105) * (1 - a[114]), x(18), x(8), p(2));      // rIntLampPipe;    fir(p.aIntLamp, p.epsIntLamp, p.epsPipe, 0.49 * M_PI * p.lPipe * p.phiPipeE * (1 - a.fIntLampCanDown), x[18], x[9], p.sigma);
    a[117] = fir(p(194), p(195), p(3), a[113] + a[114], x(18), x(4), p(2));                         // rFirIntLampCan;  fir(p.aIntLamp, p.epsIntLamp, p.epsCan, a.fIntLampCanDown + a.fIntLampCanUp, x[18], x[4], p.sigma);
    a[118] = fir(p(194), p(195), p(183), (1 - a[113]) * p(181), x(18), x(17), p(2));                // rIntLampLamp;    fir(p.aIntLamp, p.epsIntLamp, p.epsLampBottom, (1 - a.fIntLampCanUp) * p.aLamp, x[18], x[17], p.sigma);
    a[119] = fir(p(194), p(195), p(85), u(7) * p(178) * (1 - a[113]), x(18), x(20), p(2));          // rIntLampBlScr;   fir(p.aIntLamp, p.epsIntLamp, p.epsBlScrFir, u[7] * p.tauLampFir * (1 - a.fIntLampCanUp), x[18], x[20], p.sigma);
    a[120] = fir(p(194), p(195), p(74), u(2) * a[82] * p(178) * (1 - a[113]), x(18), x(7), p(2));   // rIntLampThScr; fir(p.aIntLamp, p.epsIntLamp, p.epsThScrFir, u[2] * a.tauBlScrFirU * p.tauLampFir * (1 - a.fIntLampCanUp), x[18], x[7], p.sigma);
    a[121] = fir(p(194), p(195), a[29], a[81] *a[82] * p(178) * (1 - a[113]), x(18), x(5), p(2));   // rIntLampCovIn; fir(p.aIntLamp, p.epsIntLamp, a.epsCovFir, a.tauThScrFirU * a.tauBlScrFirU * p.tauLampFir * (1 - a.fIntLampCanUp), x[18], x[5], p.sigma);
    a[122] = fir(p(194), p(195), p(4), a[24] * a[81] * a[82] * p(178) * (1 - a[113]), x(18), d(5), p(2)); // rIntLampSky; fir(p.aIntLamp, p.epsIntLamp, p.epsSky, a.tauCovFir * a.tauThScrFirU * a.tauBlScrFirU * p.tauLampFir * (1 - a.fIntLampCanUp), x[18], d[5], p.sigma);

    // Natural ventilation
    a[123] = u(3) * p(55);  // aRoofU;      u[3] * p.aRoof;
    a[124] = p(55);         // aRoofUMax;   p.aRoof;
    a[125] = 0;             // aRoofMin;    0;

    a[126] = 0;             // aSideU;      0;
    a[127] = 1;             // etaRoof;     1;
    a[128] = 1;             // etaRoofNoSide;   1;
    a[129] = 0;             // etaSide;     0;
    a[130] = p(59);         // cD;          p.cDgh;         Discharge coefficient [-]
    a[131] = p(61);         // cW;          p.cWgh;         Discharge coefficient [-]

    a[132] = u(3) * p(55) * a[130] / (2.* p(46)) *      // fVentRoof2;  u[3] * p.aRoof * a.cD / (2. * p.aFlr) * std::sqrt(std::abs(p.g * p.hVent * (x[2] - d[1]) / (2. * (0.5 * x[2] + 0.5 * d[1] + 273.15)) + a.cW * (d[4] * d[4])));
                sqrt(fabs(p(26) * p(56) * (x(2) - d(1) + p(207)) / (2. * (0.5 * x(2) + 0.5 * d(1) + 273.15)) + a[131] * (d(4) * d(4))));


    a[133] = a[130] / p(46) * sqrt(p(207) +                     // fVentRoofSide2
            pow((a[123] * a[126] / sqrt(fmax(a[123] * a[123] + a[126] * a[126], 0.01))), 2) *
            (2. * p(26) * p(62) * (x(2) - d(1)) / (0.5 * x(2) + 0.5 * d(1) + 273.15)) +
            (pow((a[123] + a[126]) / 2., 2) * a[131] * d(4) * d(4)));

    a[134] = a[130] * a[126] * d(4) / (2. * p(46) * sqrt(a[131])); // fVentSide2;   a.cD * a.aSideU * d[4] / (2*p.aFlr) * std::sqrt(a.cW);

    a[135] = if_else(d(4) < p(205), p(205) * p(60), p(60) * d(4));  // fLeakage; 

    a[136] = if_else(a[127] >= p(8),                                // fVentRoof;
                        p(57) * a[132] + p(204) * a[135],
                        p(57) * (fmax(u(2), u(7)) * a[132] + (1 - fmax(u(2), u(7))) * a[133] * a[127])  + p(204) * a[135]);

    a[137] = if_else(a[127] >= p(8),                             // fVentSide;
                        p(57) * a[134] + (1 - p(204)) * a[135],
                        p(57) * (fmax(u(2), u(7)) * a[134] + (1- fmax(u(2), u(7))) * a[133] * a[129]) + (1 - p(204)) * a[135]);

    // CO2 concentration in parts per million (ppm) 
    a[138] = co2dens2ppm(x(2), 1e-6*x(0));              // co2InPpm;    co2dens2ppm(x[2], 1e-6 * x[0]);

    // Density of air [kg m^{-3}]
    a[139] = p(36) * p(126) / ((x(3) + 273.15) * p(39)); // rhoTop;      p.mAir * p.pressure / ((x[3] + 273.15) * p.R);
    a[140] = p(36) * p(126) / ((x(2) + 273.15) * p(39)); // rhoAir;      p.mAir * p.pressure / ((x[2] + 273.15) * p.R);
    a[141] = 0.5 * (a[139] + a[140]);                    // rhoAirMean;  0.5 * (a.rhoTop + a.rhoAir);

    // Air flux through screens [m s^{-1}]
    a[142] = u(2) * p(84) * pow(fabs(x(2) - x(3) + p(207)), 0.66) + // fThScr;    u[2] * p.uThScr * std::pow(std::abs(x[2] - x[3]), 0.66) + ((1 - u[2]) / a.rhoAirMean) * std::sqrt(0.5 * a.rhoAirMean * (1 - u[7]) * p.g * std::fabs(a.rhoAir - a.rhoTop));
                ((1. - u(2)) / a[141]) * sqrt(0.5 * a[141] * (1. - u(7)) * p(26) * fabs(a[140] - a[139]) + p(207));

    a[143] = u(7) * p(94) * pow(fabs(x(2) - x(3) + p(207)), 0.66) + // fBlScr;
                ((1. - u(7)) / a[141]) * sqrt(0.5 * a[141] * (1. - u(7)) * p(26) * fabs(a[140] - a[139]) + p(207));

    a[144] = fmin(a[142], a[143]);                        // fScr;   	min(a.fThScr, a.fBlScr);

    // Convective and conductive heat fluxes [W m^{-2}]
    a[145] = 0;                                 // fVentForced (no mechanical ventilation)
    a[146] = sensible(2. * p(0) * a[31], x(4), x(2));   // hCanAir  sensible(2 * p.alfaLeafAir, x[4], x[2]);
    a[147] = if_else(x(8) > x(2),               // hAirFlr
                sensible(1.7 * pow(fabs(x(8) - x(2) + p(207)), 1./3.), x(2), x(8)),
                sensible(1.3 * pow(fabs(x(2) - x(8) + p(207)), 1./4.), x(2), x(8))
           );

    a[148] = sensible(1.7 * u(2) * pow(fabs(x(2) - x(7) + p(207)), 1./3.), x(2), x(7));               // hAirThScr    sensible(1.7 * u[2] * std::pow(std::abs(x[2] - x[7]), 1./3.), x[2], x[7]);
    a[149] = sensible(1.7 * u(7) * pow(fabs(x(2) - x(20) + p(207)), 1./3.), x(2), x(20));             // hAirBlScr    sensible(1.7 * u[7] * std::pow(std::abs(x[2] - x[20]), 1./3.), x[2], x[20]);
    a[150] = sensible(p(111) * p(23) * (a[137] + a[145]), x(2), d(1));                      // hAirOut      sensible(p.rhoAir * p.cPAir * (a.fVentSide + a.fVentForced), x[2], d[1]);
    a[151] = sensible(p(111) * p(23) * a[144], x(2), x(3));                                 // hAirTop      sensible(p.rhoAir * p.cPAir * a.fScr, x[2], x[3]);
    a[152] = sensible(1.7 * u(2) * pow(fabs(x(7) - x(3) + p(207)),  1./3.), x(7), x(3));              // hThScrTop    sensible(1.7 * u[2] * std::pow(std::abs(x[7] - x[3]), 1./3.), x[7], x[3]);
    a[153] = sensible(1.7 * u(7) * pow(fabs(x(20) - x(3) + p(207)), 1./3.), x(20), x(3));             // hBlScrTop    sensible(1.7 * u[7] * std::pow(std::abs(x[20] - x[3]), 1./3.), x[20], x[3]);
    a[154] = sensible(p(50) * pow(fabs(x(3) - x(5) + p(207)), 1./3.) * p(47) / p(46), x(3), x(5));    // hTopCovIn    sensible(p.cHecIn * std::pow(std::abs(x[3] - x[5]), 1./3.) * p.aCov / p.aFlr, x[3], x[5]);
    a[155] = sensible(p(111) * p(23) * a[136], x(3), d(1));                                 // hTopOut      sensible(p.rhoAir * p.cPAir * a.fVentRoof, x[3], d[1]);
    a[156] = sensible(p(47) / p(46) * (p(51) + p(52) * pow(d(4), p(53))), x(6), d(1));      // hCovEOut     sensible(p.aCov / p.aFlr * (p.cHecOut1 + p.cHecOut2 * std::pow(d[4], p.cHecOut3)), x[6], d[1]);
    a[157] = sensible(1.99 * M_PI * p(105) * p(107) * pow(fabs(x(9) - x(2) + p(207)), 0.32), x(9), x(2)); // hPipeAir    sensible(1.99 * M_PI * p.phiPipeE * p.lPipe * std::pow(std::abs(x[9] - x[2]), 0.32), x[9], x[2]);
    a[158] = sensible(2. / (p(101)/p(99) + p(27)/p(103)), x(8), x(10));                     // hFlrSo1      sensible(2. / (p.hFlr / p.lambdaFlr + p.hSo1 / p.lambdaSo), x[8], x[10]);
    a[159] = sensible(2. * p(103) / (p(27) + p(28)), x(10), x(11));                         // hSo1So2      sensible(2. * p.lambdaSo / (p.hSo1 + p.hSo2), x[10], x[11]);
    a[160] = sensible(2. * p(103) / (p(28) + p(29)), x(11), x(12));                         // hSo2So3      sensible(2. * p.lambdaSo / (p.hSo2 + p.hSo3), x[11], x[12]);
    a[161] = sensible(2. * p(103) / (p(29) + p(30)), x(12), x(13));                         // hSo3So4      sensible(2. * p.lambdaSo / (p.hSo3 + p.hSo4), x[12], x[13])
    a[162] = sensible(2. * p(103) / (p(30) + p(31)), x(13), x(14));                         // hSo4So5      sensible(2. * p.lambdaSo / (p.hSo4 + p.hSo5), x[13], x[14]);
    a[163] = sensible(2. * p(103) / (p(31) + p(37)), x(14), d(6));                          // hSo5SoOut    sensible(2. * p.lambdaSo / (p.hSo5 + p.hSoOut), x[14], d[6]);

    a[164] = sensible(1. / (p(73)/p(71)), x(5), x(6));                                       // hCovInCovE   sensible(1. / (p.hRf / p.lambdaRf), x[5], x[6]);
    a[165] = sensible(p(185), x(17), x(2));                                                 // hLampAir     sensible(p.cHecLampAir, x[17], x[2]);
    a[166] = sensible(1.99 * M_PI * p(167) * p(166) * pow(fabs(x(19) - x(2) + p(207)), 0.32), x(19), x(2)); // hGroPipeAir sensible(1.99 * M_PI * p.phiGroPipeE * p.lGroPipe * std::pow(std::abs(x[19] - x[2]), 0.32), x[19], x[2]);
    a[167] = sensible(p(198), x(18), x(2));                                                 // hIntLampAir  sensible(p.cHecIntLampAir, x[18], x[2]);

    // Model of stomatal resistance

    // smooth transition between night and day
    a[168] = 1./ (1. + exp(p(43) * (a[45] - p(40))));                                       // sRs   1. / (1. + std::exp(p.sRs * (a.rCan + p.rCanSp)));

    a[169] = p(20) * (1 - a[168]) + p(19) * a[168];                                         // cEvap3;  p.cEvap3Night * (1 - a.sRs) + p.cEvap3Day * a.sRs; (parameter for effect of co2 on stomatal resistance)
    a[170] = p(22) * (1 - a[168]) + p(21) * a[168];                                         // cEvap4;  p.cEvap4Night * (1 - a.sRs) + p.cEvap4Day * a.sRs; (parameter for effect of VP on stomatal resistance)
    a[171] = (a[45] + p(17)) / (a[45] + p(18));                                             // rfRCan;  (a.rCan + p.cEvap1) / (a.rCan + p.cEvap2); (effect of radiation on stomatal resistance)
    a[172] = fmin(1.5, 1 + a[169] * pow(a[138] - 200, 2));                                   // rfCo2;   std::min(1.5, 1 + a.cEvap3 * std::pow(a.co2InPpm - 200, 2)); (effect of co2 on stomatal resistance)
    a[173] = fmin(5.8, 1 + a[170] * pow(satVP(x(4))- x(15), 2));                            // rfVP;    std::min(5.8, 1 + a.cEvap4 * std::pow(satVP(x[4] - x[15]), 2)); (effect of VP on stomatal resistance)
    // a[174] = p(42) * a[171] * a[172] * a[173];                                              // rS;      p.rSMin * a.rfRCan * a.rfCo2 * a.rfVP; (stomatal resistance)
    a[174] = p(42) * a[171];// * a[172] * a[173];                                              // rS;      p.rSMin * a.rfRCan * a.rfCo2 * a.rfVP; (stomatal resistance)

    a[175] = 2. * p(111) * p(23) * a[31] / (p(1) * p(14) * (p(41) + a[174]));       // vecCanAir;   2. * p.rhoAir * p.cPAir * a.lai / (p.L * p.gamma * (p.rB + a.rS)); (VP coef of canopy transpiration)
    a[176] = (satVP(x(4)) -x(15)) *  a[175];                                         // mvCanAir;    (satVP(x[4]) - x[15]) * a.vecCanAir; (canopy transpiration)


    // Vapour Fluxes
    a[177] = 0;  // mvPadAir
    a[178] = 0;  // mvFogAir
    a[179] = 0;  // mvBlowAir
    a[180] = 0;  // mvAirOutPad

    // condensations
    a[181] = cond(1.7 * u(2) * pow(fabs(x(2) - x(7) + p(207)), 1./3.), x(15), satVP(x(7)));           // mvAirThScr;  cond(1.7 * u[2] * std::pow(fabs(x[2]-x[7]), (1/3)), x[15], satVp(x[7]));
    a[182] = cond(1.7 * u(7) * pow(fabs(x(2) - x(20) + p(207)), 1./3.), x(15), satVP(x(20)));         // mvAirBlScr;  cond(1.7 * u[7] * std::pow(fabs(x[2]-x[20]), (1/3)), x[15], satVp(x[20]));
    a[183] = cond(p(50) * pow(fabs(x(3) - x(5) + p(207)), 1./3.) * p(47)/p(46), x(16), satVP(x(5)));  // mvTopCovIn; cond(p.cHecIn * std::pow(fabs(x[3] - x[5]), (1/3)) * p.aCov / p.aFlr, x[16], satVp(x[5]));

    // Vapour fluxes
    a[184] = airMv(a[144], x(15), x(16), x(2), x(3));                                       // mvAirTop;   airMv(a.fScr, x[15], x[16], x[2], x[3]);
    a[185] = airMv(a[136], x(16), d(2), x(3), d(1));                                        // mvTopOut;   airMv(a.fVentRoof, x[15], x[16], x[2], d[1]);
    a[186] = airMv(a[137] + a[145], x(15), d(2), x(2), d(1));                               // mvAirOut;   airMv(a.fVentSide + a.fVentForced, x[15], d[2], x[2], d[1]);

    // Latent heat fluxes
    a[187] = p(1) * a[176];   // lCanAir
    a[188] = p(1) * a[181];   // lAirThScr
    a[189] = p(1) * a[182];   // lAirBlScr
    a[190] = p(1) * a[183];   // lTopCovIn

    // Canopy photosynthesis
    // PAR absorbed by the canopy [umol{photons} m^{-2} s^{-1}]
    a[191] = p(187) * a[55] + p(140) * a[54] + p(197) * a[56];      // parCan;      p.zetaLampPar*a.rParLampCan + p.parJtoUmolSun*a.rParSunCan + p.zetaIntLampPar*a.rParIntLampCan;

    // Maximum rate of electron transport rate at 25C [umol{e-} m^{-2} s^{-1}]
    a[192] = a[31] * p(129);                                        // j25CanMax;   a.lai * p.j25CanMax;

    // CO2 compensation point [ppm]
    a[193] = (p(129) / a[192]) * p(130) * x(4) + 20. * p(130) * (1 - (p(129)/a[192])); // gamma; (p.j25CanMax / a.j25CanMax) * p.co2Comp25 * x[4] + 20. * p.co2Comp25 * (1 - (p.j25CanMax / a.j25CanMax));
    
    // CO2 concentration in the stomata [ppm]
    a[194] = p(131) * a[138];    // co2Stom; p.etaCo2AirStom * a.co2InPpm;

    // Potential rate of electron transport [umol{e-} m^{-2} s^{-1}]
    a[195] = a[192] * exp(p(132) * (x(4) + 273.15 - p(133)) / (1e-3*p(39)*(x(4)+273.15)*p(133))) *  // jPot;
        (1 + exp((p(134)*p(133)-p(135)) / (1e-3*p(39)*p(133)))) /
        (1 + exp((p(134)*(x(4)+273.15)-p(135)) / (1e-3*p(39)*(x(4)+273.15))));

    // Electron transport rate [umol{e-} m^{-2} s^{-1}]
    a[196] = (1./(2.*p(136))) * (a[195] + p(137) * a[191] -         // j
        sqrt(pow((a[195] + p(137) * a[191]), 2) - 4. * p(136) * a[195] * p(137) * a[191] + p(207)));

    // Photosynthesis rate at canopy level [umol{co2} m^{-2} s^{-1}]
    a[197] = a[196] * (a[194] - a[193]) / (4.*(a[194] + 2.*a[193])); // p

    // Photrespiration [umol{co2} m^{-2} s^{-1}]
    a[198] = a[197]*a[193] / a[194];                                // r;  a.p*a.gamma / a.co2Stom;

    // Inhibition due to full carbohydrates buffer [-]
    a[199] = 1. /(1. + exp(5e-4 * (x(22) - p(157))));           // hAirBuf;  1. / (1. + std::exp(5e-4 * (x[22] - p.cBufMax));

    // Net photosynthesis [mg{CH2O} m^{-2} s^{-1}]
    a[200] = p(138) * a[199] * (a[197] - a[198]);               // a.mcAirBuf;  p.mCh2o * a.hAirBuf * (a.p - a.r);

    // Carbohydrate buffer
    // Temperature effect on structural carbon flow to organs
    a[201] = 0.047*x(21) + 0.06;                                // gTCan24;  0.047*x[21] + 0.06;

    // Inhibition of carbohydrate flow to the organs due to mean temperature
    a[202] = 1. / (1. + exp(-1.1587*(x(21)-p(160)))) *          // hTCan24;
        1. / (1. + exp(1.3904*(x(21)-p(159))));


    // Inhibition of carbohydrate flow to the fruit due to current temperature 
    a[203] = 1. / (1. + exp(-0.869*(x(4)-p(162)))) *              // hTCan;
        1. / (1. + exp(0.5793*(x(4)-p(161))));

    // Inhibition due to development stage 
    a[204] = 0.5 *(x(26) / p(163) +                             // hTCanSum;
        sqrt(pow((x(26) / p(163)), 2) + 1e-4)) -
        0.5 * ((x(26) - p(163)) / p(163) +
        sqrt(pow(((x(26) - p(163)) / p(163)), 2) + 1e-4));

    // Inhibition due to development stage
    // a.hTCanSumEnd = 1 / (1 + std::exp(0.01 * (x[26] - p.tEndSumGrowth)));
    a[205] = 1.;

    // Inhibition due to insufficient carbohydrates in the buffer [-]
    a[206] = 1. / (1. + exp(-5e-3*(x(22) - p(158))));  // hBufOrg;     1 / (1 + std::exp(-5e-3*(x[22] - p.cBufMin)));

    // Carboyhdrate flow from buffer to leaves [mg{CH2O} m^{2} s^{-1}]
    a[207] = a[206] * a[202] * a[201] * p(155);         // mcBufLeaf;   a.hBufOrg * a.hTCan24 * a.gTCan24 * p.rgLeaf;

    // Carboyhdrate flow from buffer to stem [mg{CH2O} m^{2} s^{-1}]
    a[208] = a[206] * a[202] * a[201] * p(156);         // mcBufStem;   a.hBufOrg * a.hTCan24 * a.gTCan24 * p.rgStem;

    // Carboyhdrate flow from buffer to fruit [mg{CH2O} m^{2} s^{-1}]
    a[209] = a[206] * a[203] * a[202] * a[204] * a[201] * p(154) * a[205];  // mcBufFruit;  a.hBufOrg * a.hTCan * a.hTCan24 * a.hTCanSum * a.gTCan24 * p.rgFruit * a.hTCanSumEnd;

    // Growth respiration [mg{CH2O} m^{-2] s^{-1}]
    a[210] = p(147) * a[207] + p(148) * a[208] +  p(146) * a[209];   // mcBufAir;    p.cLeafG*a.mcBufLeaf + p.cStemG*a.mcBufStem + p.cFruitG*a.mcBufFruit;

    // Leaf maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    a[211] = (1 - exp(-p(149)*p(143))) * pow(p(150), 0.1*(x(21)-25)) * x(23) * p(152);  // mcLeafAir;   (1- std::exp(-p.cRgr*p.rgr)) * std::pow(p.q10m, 0.1*(x[21]-25)) * x[23] * p.cLeafM;

    // Stem maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    a[212] = (1 - exp(-p(149)*p(143))) * pow(p(150), 0.1*(x(21)-25)) * x(24) * p(153);  // mcStemAir;   (1 - exp(-p.cRgr * p.rgr)) * pow(p.q10m, 0.1 * (x[21] - 25)) * x[24] * p.cStemM;

    // Fruit maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    a[213] = (1- exp(-p(149)*p(143))) * pow(p(150), 0.1*(x(21)-25)) * x(25) * p(151);   // mcFruitAir;  (1- std::exp(-p.cRgr*p.rgr)) * std::pow(p.q10m,(0.1*(x[21]-25))) * x[25] * p.cFruitM;

    // Total maintenance respiration [mg{CH2O} m^{-2} s^{-1}]
    a[214] = a[211] + a[212] + a[213];              // mcOrgAir;    a.mcLeafAir + a.mcStemAir + a.mcFruitAir;

    // Leaf pruning
    a[215] = smoothHar(x(23), p(144), 1e4, 5e4);    // mcLeafHar = smoothHar(x[23], p.cLeafMax, 1e4, 5e4);

    // Fruit harvest [mg{CH2O} m^{-2} s^{-1}]
    a[216] = smoothHar(x(25), p(145), 1e4, 5e4);    // mcFruitHar = smoothHar(x[25], p.cFruitMax, 1e4, 5e4);

    // Net crop assimilation [mg{CO2} m^{-2} s^{-1}]
    // It is assumed that for every mol of CH2O in net assimilation, a mol
    // of CO2 is taken from the air, thus the conversion uses molar masses
    a[217] = (p(139)/p(138))* (a[200]-a[210]-a[214]);    // mcAirCan;      (p.mCo2/p.mCh2o) * (a.mcAirBuf-a.mcBufAir-a.mcOrgAir);

    // Other CO2 flows [mg{CO2} m^{-2} s^{-1}]

    // From main to top compartment 
    a[218] = airMc(a[144], x(0), x(1));             // mcAirTop;    airMc(a.fScr, x[0], x[1]);

    // From top compartment outside 
    a[219] = airMc(a[136], x(1), d(3));             // mcTopOut;    airMc(a.fVentRoof, x[1], d[3]);

    // From main compartment outside
    a[220] = airMc(a[137] + a[145], x(0), d(3));    // mcAirOut;   airMc(a.fVentSide + a.fVentForced, x[0], d[3]);

    // Heat from boiler - Section 9.2 [1]

    // Heat from boiler to pipe rails [W m^{-2}]
    // Equation 55 [1]
    a[221] = u(0) * p(108);                     // hBoilPipe;   u[0] * p.pBoil;

    // Heat from boiler to grow pipes [W m^{-2}]
    a[222] = u(6) * p(170);                     // hBoilGroPipe;    u[6] * p.pBoilGro;

    //  CO2 injection [mg m^{-2} s^{-1}]
    a[223] = u(1) * p(109);                     // mcExtAir;    u[1] * p.phiExtCo2;

    // Objects not currently included in the model
    a[224] = 0;  // mcBlowAir
    a[225] = 0;  // mcPadAir
    a[226] = 0;  // hPadAir
    a[227] = 0;  // hPasAir
    a[228] = 0;  // hBlowAir
    a[229] = 0;  // hAirPadOut
    a[230] = 0;  // hAirOutPad
    a[231] = 0;  // lAirFog
    a[232] = 0;  // hIndPipe
    a[233] = 0;  // hGeoPipe

    //  Lamp cooling
    // Equation A34 [5], Equation 7.34 [7]
    a[234] = p(186) * a[37];        // hLampCool; p.etaLampCool * a.qLampIn;

    // Heat harvesting, mechanical cooling and dehumidification
    // By default there is no mechanical cooling or heat harvesting
    // see addHeatHarvesting.m for mechanical cooling and heat harvesting
    a[235] = 0;  // hecMechAir
    a[236] = 0;  // hAirMech
    a[237] = 0;  // mvAirMech
    a[238] = 0;  // lAirMech
    a[239] = 0;  // hBufHotPipe

    return vertcat(a);
}
