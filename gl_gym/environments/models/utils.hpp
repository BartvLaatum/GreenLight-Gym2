// utils_casadi.h
#ifndef UTILS_H
#define UTILS_H

#include <casadi/casadi.hpp>
#include <cmath>
#include <vector>

inline double satVp_cpp(double temp) {
    const float a = 610.78;
    const float b = 17.2694;
    const float c = 238.3;
    return a * std::exp(b * temp / (temp + c));
}

inline double cond(double hec, double vp1, double vp2) {
    const double a = 6.4e-9; 
    return 1.0 / (1.0 + std::exp(-0.1 * (vp1 - vp2))) * a * hec * (vp1 - vp2);
}

inline double co2dens2ppm_cpp(double temp, double dens) {
    const double R = 8.3144598;        // Molar gas constant [J mol^{-1} K^{-1}]
    const double C2K = 273.15;         // Conversion from Celsius to Kelvin [K]
    const double M_CO2 = 44.01e-3;     // Molar mass of CO2 [kg mol^{-1}]
    const double P = 101325;           // Pressure (assumed to be 1 atm) [Pa]
    
    return 1e6 * R * (temp + C2K) * dens / (P * M_CO2);
}

inline double proportional_control(double processVar, double setPt, double pBand, double minVal, double maxVal) {
    return minVal + (maxVal - minVal) * (1. / (1. + std::exp(-2. / pBand * std::log(100.) * (processVar - setPt - pBand / 2.))));
}

inline double tau12(double tau1, double tau2, double rho1Dn, double rho2Up) {
    // Transmission coefficient of a double layer [-]
    // Equation 14 [1], Equation A4 [5]
    return tau1 * tau2 / (1. - rho1Dn * rho2Up);
}

inline double rhoDn(double tau2, double rho1Dn, double rho2Up, double rho2Dn) {
    // Reflection coefficient of the lower layer [-]
    // Equation 15 [1], Equation A5 [5]
    return rho2Dn + (tau2 * tau2 * rho1Dn) / (1. - rho1Dn * rho2Up);
}


double dli_check(double lamp_input, float dli)
{
    if (dli > 15.) {
        return 0.;
    } 
    return lamp_input;
}

#endif  // UTILS_H
