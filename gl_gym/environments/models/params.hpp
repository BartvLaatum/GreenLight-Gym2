#include <cmath>  // For exp, INFINITY, M_PI
#include <casadi/casadi.hpp>

using namespace casadi;

std::vector<double> init_default_params(uint32_t np)
{
    // Initialize parameters
    // std::vector<double> params[np]; // Initialize params as an empty vector
    std::vector<double> params = std::vector<double>(np);

    // Physical constants and climate parameters
    params[0] = 5.;             // alfaLeafAir;     convective heat transfer coefficient leaf-air [W m-2 K-1]
    params[1] = 2.45e6;     	// L;               latent heat of vaporization of water [J kg-1]
    params[2] = 5.67e-8;        // sigma;           Stefan-Boltzmann constant [W m-2 K-4]
    params[3] = 1.;             // epsCan;          FIR emissivity of the canopy []
    params[4] = 1.;             // epsSky;          FIR emissivity of the sky []
    params[5] = 0.5;            // etaGlobNir;      Ratio of NIR in global radiation []
    params[6] = 0.5;            // etaGlobPar;      Ratio of PAR in global radiation []

    params[7] = 0.554;          // etaMgPpm;        CO2 conversion factor from mg/m^{3} to ppm []
    params[8] = 0.9;            // etaRoofThr;      Ratio between roof vent area and total vent area where no chimney effects is assumed
    params[9] = 1.2;            // rhoAir0;         Density of air at 0 degrees Celsius [kg m-3]
    params[10] = 0.07;          // rhoCanPar;       PAR reflection coefficient of the top of the canopy
    params[11] = 0.35;          // rhoCanNir;       NIR reflection coefficient of the top of the canopy
    params[12] = 7850.;         // rhoSteel;        Density of steel
    params[13] = 1000.;         // rhoWater;        Density of water
    params[14] = 65.8;          // gamma;           Psychrometric constant [Pa K-1]
    params[15] = 1.99e-7;       // omega;           Yearly frequency to calculate soil temperature [s-1]
    
    params[16] = 1200.;         // capLeaf;         Heat capacity of the leaf [J m-2 K-1]
    params[17] = 4.3;           // cEvap1;          Coefficient for radiation effect on stomatal resistance
    params[18] = 0.54;          // cEvap2;          Coefficient for vapour pressure deficit effect on stomatal resistance
    params[19] = 6.1e-7;        // cEvap3Day;       Coefficient for CO2 effect on stomatal resistance during day
    params[20] = 1.1e-11;       // cEvap3Night;     Coefficient for CO2 effect on stomatal resistance during night
    params[21] = 4.3e-6;        // cEvap4Day;       Coefficient for vapour pressure deficit effect on stomatal resistance during day
    params[22] = 5.2e-6;        // cEvap4Night;     Coefficient for vapour pressure deficit effect on stomatal resistance during night
    params[23] = 1000.;         // cPAir;           Specific heat capacity of air [J kg-1 K-1]
    params[24] = 640.;          // cPSteel;         Specific heat capacity of steel [J kg-1 K-1]
    params[25] = 4180.;         // cPWater;         Specific heat capacity of water [J kg-1 K-1]
    params[26] = 9.81;          // g;               Gravitational acceleration [m s-2]

    // SOIL PARAMETERS
    params[27] = 0.04;          // hSo1;            Thickness of soil layer 1 [m]
    params[28] = 0.08;          // hSo2;            Thickness of soil layer 2 [m]
    params[29] = 0.16;          // hSo3;            Thickness of soil layer 3 [m]
    params[30] = 0.32;          // hSo4;            Thickness of soil layer 4 [m]
    params[31] = 0.64;          // hSo5;            Thickness of soil layer 5 [m]

    params[32] = 0.7;           // k1Par;           PAR extinction coefficient of canopy [m2 m-2]
    params[33] = 0.7;           // k2Par;           PAR extinction coefficient of canopy [m2 m-2]
    params[34] = 0.27;          // kNir;            NIR extinction coefficient of canopy [m2 m-2]
    params[35] = 0.94;          // kFir;            FIR extinction coefficient of canopy [m2 m-2]
    params[36] = 28.96;         // mAir;            Molar mass of air [g mol-1]
    params[37] = 1.28;          // hSoOut;          Thickness of external soil layer [m]
    params[38] = 18.;           // mWater;          Molar mass of water [g mol-1]
    params[39] = 8314.;         // R;               Universal gas constant [J mol-1 K-1]

    params[40] = 5.;            // rCanSp;          Radiation value above the canopy when night becomes day
    params[41] = 275.;           // rB;             Ball-Berry model parameter [s m-1]
    params[42] = 82.;           // rSMin;           Minimum stomatal resistance [s m-1]
    params[43] = -1.;            // sRs;            Stomatal resistance parameter []

    // GREENHOUSE CONSTRUCTION PARAMETERS
    params[44] = 0.1;           // etaGlobAir;      Ratio of global radiation absorbed by the greenhouse construction []
    params[45] = 23.;           // psi;             mean greenhouse cover slope []
    params[46] = 96.;           // aFlr;            Floor area of greenhouse [m2]
    params[47] = 224.;          // aCov;            Surface of the cover including side walls [m2]
    params[48] = 5.7;           // hAir;            Height of the main greenhouse compartment [m]
    params[49] = 6.2;           // hGh;             Mean height of the greenhouse [m]
    params[50] = 3.84;          // cHecIn;          Convective heat exchange paramater between cover and indoor air [W m-2 K-1]
    params[51] = 2.8;           // cHecOut1;        Convective heat exchange paramater between cover and outdoor air [W m-2 K-1]
    params[52] = 1.2;           // cHecOut2;        Convective heat exchange paramater between cover and outdoor air [W m-2 K-1]
    params[53] = 1.;            // cHecOut3;        Convective heat exchange paramater between cover and outdoor air [W m-2 K-1]
    params[54] = 0.;            // hElevation;      Altitude of the greenhouse [m]
    params[55] = 40.;           // aRoof;           Roof area of the greenhouse [m2]
    params[56] = 0.87;          // hVent;           Height of the ventilation opening [m]
    params[57] = 1.;            // etaInsScr;       Insulation factor of the screen []
    params[58] = 0.;            // aSide;           Side wall area of the greenhouse [m2]
    params[59] = 0.72;          // cDgh;            Discharge coefficient for the greenhouse [W m-2 K-1]
    params[60] = 1.2e-4;        // cLeakage;        Ventilation leakage coefficient [m3 s-1 m-2 Pa-1]
    params[61] = 0.09;          // cWgh;            Wind shelter factor of the greenhouse []
    params[62] = 0.;            // hSideRoof;       Height of the side roof [m]

    // ROOF PARAMETERS
    params[63] = 0.7;           // epsRfFir;        FIR emissivity of the roof []
    params[64] = 2600.;         // rhoRf;           Density of the roof [kg m-3]
    params[65] = 0.13;          // rhoRfNir;        NIR reflection coefficient of the roof []
    params[66] = 0.13;          // rhoRfPar;        PAR reflection coefficient of the roof []
    params[67] = 0.15;          // rhoRfFir;        FIR reflection coefficient of the roof []
    params[68] = 0.57;          // tauRfNir;        NIR transmission coefficient of the roof []
    params[69] = 0.57;          // tauRfPar;        PAR transmission coefficient of the roof []
    params[70] = 0.;            // tauRfFir;        FIR transmission coefficient of the roof []
    params[71] = 1.05;          // lambdaRf;        Thermal conductivity of the roof [W m-1 K-1]
    params[72] = 840.;          // cPRf;            Specific heat capacity of the roof [J kg-1 K-1]
    params[73] = 4e-3;          // hRf;             Thickness of the roof [m]

    // THERMAL SCREEN PARAMETERS
    params[74] = 0.67;          // epsThScrFir;     FIR emissivity of the thermal screen []
    params[75] = 127.5;         // rhoThScr;        Density of the thermal screen [kg m-3]
    params[76] = 0.35;          // rhoThScrNir;     NIR reflection coefficient of the thermal screen []
    params[77] = 0.35;          // rhoThScrPar;     PAR reflection coefficient of the thermal screen []
    params[78] = 0.18;          // rhoThScrFir;     FIR reflection coefficient of the thermal screen []
    params[79] = 0.6;           // tauThScrNir;     NIR transmission coefficient of the thermal screen []
    params[80] = 0.6;           // tauThScrPar;     PAR transmission coefficient of the thermal screen []
    params[81] = 0.15;          // tauThScrFir;     FIR transmission coefficient of the thermal screen []
    params[82] = 1800.;         // cPThScr;         Specific heat capacity of the thermal screen [J kg-1 K-1]
    params[83] = 4e-3;          // hThScr;          Thickness of the thermal screen [m]
    params[84] = 3.5e-4;        // kThScr;          Thermal screen flux coefficient

    // BLACKOUT SCREEN PARAMETERS
    params[85] = 0.67;          // epsBlScrFir;     FIR emissivity of the blackout screen []
    params[86] = 167.5;         // rhoBlScr;        Density of the blackout screen [kg m-3]
    params[87] = 0.35;          // rhoBlScrNir;     NIR reflection coefficient of the blackout screen []
    params[88] = 0.35;          // rhoBlScrPar;     PAR reflection coefficient of the blackout screen []
    params[89] = 0.01;          // tauBlScrNir;     NIR transmission coefficient of the blackout screen []
    params[90] = 0.01;          // tauBlScrPar;     PAR transmission coefficient of the blackout screen []
    params[91] = 0.7;           // tauBlScrFir;     FIR transmission coefficient of the blackout screen []
    params[92] = 1800.;         // cPBlScr;         Specific heat capacity of the blackout screen [J kg-1 K-1]
    params[93] = 4e-3;          // hBlScr;          Thickness of the blackout screen [m]
    params[94] = 5e-4;          // kBlScr;          Blackout screen flux coefficient

    // FLOOR PARAMETERS
    params[95] = 1.;            // epsFlr;          FIR emissivity of the floor []
    params[96] = 2300.;         // rhoFlr;          Density of the floor [kg m-3]
    params[97] = 0.5;           // rhoFlrNir;       NIR reflection coefficient of the floor []
    params[98] = 0.65;          // rhoFlrPar;       PAR reflection coefficient of the floor []
    params[99] = 1.7;           // lambdaFlr;       Thermal conductivity of the floor [W m-1 K-1]
    params[100] = 880.;         // cPFlr;           Specific heat capacity of the floor [J kg-1 K-1]
    params[101] = 0.02;         // hFlr;            Thickness of the floor [m]

    params[102] = 1'730'000.;   // rhoCpSo;         Volumetric heat capacity of the soil
    params[103] = 0.85;         // lambdaSo;        Thermal heat conductivity of the soil [W m-1 K-1]

    // HEATING PIPE PARAMETERS
    params[104] = 0.88;         // epsPipe;         FIR emission coefficient of the heating pipes
    params[105] = (51e-3) - (2.25e-3); // phiPipeE;     External diameter of the heating pipes [m]
    params[106] = 48.75e-3;     // phiPipeI;        Internal diameter of the heating pipes [m]
    params[107] = 1.3375;       // lPipe;           Length of the heating pipes [m]
    params[108] = 80.;          // pBoil;           Max energy input from boiler into the heating system [W/m2]

    params[109] = 2.083;        // phiExtCo2;       Capacity of external CO2 source [mg m-2 s-1]
    // capPipe; Heat capacity of the heating pipes [J m-2 K-1]
    params[110] = 0.25 * M_PI * params[107] * ((params[105] * params[105] - params[106] * params[106]) * params[12] * params[24] + params[106] * params[106] * params[13] * params[25]); 

    // HEAT CAPACITIES
    params[111] = params[9] * exp(params[26] * params[36] * params[54] / (params[39] * 293.15)); // rhoAir; Density of air [kg m-3]
    params[112] = params[48] * params[111] *  params[23]; // capAir;    Heat capacity of the air [J m-3 K-1]
    params[113] = params[101] * params[96] * params[100]; // capFlr;    Heat capacity of floor [J m-2 K-1]
    params[114] = params[27] * params[102];                // capSo1;   Heat capacity of soil layer 1 [J m-2 K-1]
    params[115] = params[28] * params[102];                // capSo2;   Heat capacity of soil layer 2 [J m-2 K-1]
    params[116] = params[29] * params[102];                // capSo3;   Heat capacity of soil layer 3 [J m-2 K-1]
    params[117] = params[30] * params[102];                // capSo4;   Heat capacity of soil layer 4 [J m-2 K-1]
    params[118] = params[31] * params[102];                // capSo5;   Heat capacity of soil layer 5 [J m-2 K-1]
    params[119] = params[83] * params[75] * params[82];    // capThScr; Heat capacity of thermal screen [J m-2 K-1]
    params[120] = (params[49] - params[48]) * params[111] * params[23]; // capTop; Heat capacity of the air in the top compartment [J m-3 K-1]
    params[121] = params[93] * params[86] * params[92];    // capBlScr; Heat capacity of blackout screen [J m-2 K-1]

    // CO2 CAPACITIES
    params[122] = params[48];                                // capCo2Air;  Capacity of CO2 of the air in the main compartment [m]
    params[123] = params[49]-params[48];                     // capCo2Top;  Capacity of CO2 of the air in the top compartment [m]

    params[124] = M_PI * params[107] * params[105];         // aPipe;       Surface of the heating pipers per floor area [m2 m-2]
    params[125] = 1 - 0.49 * M_PI * params[107] *params[105];  // fCanFlr;     View factor from canopy to floor; Emission coefficient of the heating pipes []

    params[126] = 101325 * pow((1 - 2.5577e-5 * params[54]), 5.25588); // pressure; Air pressure [Pa]
    params[127] = 31.65;        // energyContentGas; Energy content of gas [MJ m-3]

    // CROP PARAMETERS
    params[128] = 2.3;          // globJtUmol;      Conversion factor from global radiation to PAR
    params[129] = 210.;         // j25LeafMax;      Maximum rate of electron transport at 25 degrees [umol m-2 s-1]
    params[130] = 1.7;          // cGamma;          Effect of canopy temperature on CO2 compensation point
    params[131] = 0.67;         // etaCo2AirStom;   Conversion from greenhouse air co2 concentration and stomatal co2 concentration
    params[132] = 37000;        // eJ;              Activation energy for Jpot calcualtion
    params[133] = 298.15;       // t25k;            Temperature at which Jpot is 25 degrees
    params[134] = 710;          // S;               Entropy factor for Jpot calculation
    params[135] = 220'000;      // H;               Deactivation energy for Jpot calculation
    params[136] = 0.7;          // theta;           Degree of curvature of the electron transport rate
    params[137] = 0.385;        // alpha;           Conversion factor from photons to electrons including efficiency term
    params[138] = 30e-3;        // mCh2o;           Molar mass of CH2O [kg mol-1]
    params[139] = 44e-3;        // mCo2;            Molar mass of CO2 [kg mol-1]
    params[140] = 4.6;          // parJtoUmolSun;   Conversion factor from PAR to umol m-2 s-1
    params[141] = 3.0;          // laiMax;          Maximum leaf area index
    params[142] = 2.66e-5;      // sla;             Specific leaf area [m2 kg-1]
    params[143] = 3e-6;         // rgr;             Relative growth rate [kg m-2 s-1]
    params[144] = 3'000'000;    // cLeafMax;        Maximum leaf carbon content [kg m-2]
    params[145] = 3'000'000;    // cFruitMax;       Maximum Fruit carbon content [kg m-2]
    params[146] = 0.27;         // cFruitG;         Growth respiration coefficient for fruit
    params[147] = 0.28;         // cLeafG;          Growth respiration coefficient for leaf
    params[148] = 0.3;          // cStemG;          Growth respiration coefficient for stem
    params[149] = 2'850'000;    // cRgr;            Regression coefficient in maintenance respiration function	    
    params[150] = 2.;           // q10m;            Q10 value of temperature effect on maintenance respiration

    params[151] = 1.16e-7;      // cFruitM;         Maintenance respiration coefficient for fruit
    params[152] = 3.47e-7;      // cLeafM;          Maintenance respiration coefficient for leaf
    params[153] = 1.47e-7;      // cStemM;          Maintenance respiration coefficient for stem

    params[154] = 0.328;        // rgFruit;         Growth respiration coefficient for fruit
    params[155] = 0.095;        // rgLeaf;          Growth respiration coefficient for leaf
    params[156] = 0.074;        // rgStem;          Growth respiration coefficient for stem

    params[157] = 5000;         // cBufMax;         Maximum buffer capacity [J m-2 K-1]
    params[158] = 250;          // cBufMin;         Minimum buffer capacity [J m-2 K-1]
    params[159] = 23;           // tCan24Max;       Maximum canopy temperature at 24 hours [°C]
    params[160] = 18;           // tCan24Min;       Minimum canopy temperature at 24 hours [°C]
    params[161] = 28;           // tCanMax;         Maximum canopy temperature [°C]
    params[162] = 14;           // tCanMin;         Minimum canopy temperature [°C]
    params[163] = 250;          // tEndSum;         Temperature sum for reaching max potential growth [day °C]
    params[164] = 1250;         // tEndSumGrowth;   End of temperature sum for growth [day °C]

    // GROWTH PIPE PARAMETERS
    params[165] = 0;            // epsGroPipe;      FIR emission coefficient of the growth pipes
    params[166] = 1.655;        // lGroPipe;        Length of the growth pipes [m]
    params[167] = 35e-3;        // phiGroPipeE;     External diameter of the growth pipes [m]
    params[168] = 35e-3 - 1.2e-3; // phiGroPipeI;   Internal diameter of the growth pipes [m]
    params[169] = M_PI * params[166] * params[167]; // aGroPipe; Surface of the growth pipes per floor area [m2 m-2]
    params[170] = 0;            // pBoilGro;        Max energy input from boiler into the growth system [W/m2]
    params[171] = 0.25 * M_PI * params[166] * ((params[167] * params[167] - params[168] * params[168]) * params[12] * params[24] + params[168] * params[168] * params[13] * params[25]); // capGroPipe; Heat capacity of the growth pipes [J m-2 K-1]


    // LED LAMP PARAMETERS
    params[172] = 62.5f;        // thetaLampMax;    Max energy input of the lamps [W/m2]
    params[173] = 0;            // heatCorrection;  Heat correction factor for the lamp []

    params[174] = 3.2/4.6;      // etaLampPar;      PAR efficiency of the lamp []
    params[175] = 0.02f;        // etaLampNir;      NIR efficiency of the lamp []

    params[176] = 0.98f;        // tauLampPar;      PAR transmission coefficient of the lamp []
    params[177] = 0.98f;        // tauLampNir;      NIR transmission coefficient of the lamp []
    params[178] = 0.98f;        // tauLampFir;      FIR transmission coefficient of the lamp []
    params[179] = 0.;           // rhoLampPar;      PAR reflection coefficient of the lamp []
    params[180] = 0.;           // rhoLampNir;      NIR reflection coefficient of the lamp []
    params[181] = 0.02;         // aLamp;           Surface area of the lamp per floor area [m2 m-2]

    params[182] = 0.88f;        // epsLampTop;      FIR emission coefficient of the top lamp []
    params[183] = 0.88f;        // epsLampBottom;   FIR emission coefficient of the bottom lamp []
    params[184] = 10.;          // capLamp;         Heat capacity of the lamp [J m-2 K-1]
    params[185] = 2.3f;         // cHecLampAir;     Heat exchange coefficient between the lamp and the air [W m-2 K-1]
    params[186] = 0;            // etaLampCool;     Cooling efficiency of the lamp []
    params[187] = 3.2f;         // zetaLampPar;     PAR emission coefficient of the lamp []

    // INTERNAL LAMP PARAMETERS	
    params[188] = 0;            // intLamps;        Are internal lamps present
    params[189] = 0.5f;         // vIntLampPos      Vertical position of the interlights within the canopy [0-1, 0 is above canopy and 1 is below]
    params[190] = 0.5f;         // fIntLampDown     Fraction of interlight light output of lamps that is directed downwards
    params[191] = 10;           // capIntLamp       Capacity of interlight lamps
    params[192] = 0;            // etaIntLampPar    fraction of interlight lamp input converted to PAR
    params[193] = 0;            // etaIntLampNir    fraction of interlight lamp input converted to NIR
    params[194] = 0;            // aIntLamp         interlight lamp area
    params[195] = 0;            // epsIntLamp       emissivity of interlight lamp
    params[196] = 0;            // thetaIntLampMax  Maximum intensity of interlight lamps
    params[197] = 0;            // zetaIntLampPar   J to umol conversion of PAR output of interlight lamp
    params[198] = 0;            // cHecIntLampAir   heat exchange coefficient of interlight lamp
    params[199] = 1;            // tauIntLampFir    transmissivity of interlight lamp later to FIR
    params[200] = 1.4f;         // k1IntPar         PAR extinction coefficient of the canopy
    params[201] = 1.4f;         // k2IntPar         PAR extinction coefficient of the canopy for light reflected from the floor
    params[202] = 0.54f;        // kIntNir          NIR extinction coefficient of the canopy
    params[203] = 1.88f;        // kIntFir          FIR extinction coefficient of the canopy


    params[204] = 0.9f;         // cLeakTop         Fraction of leakage ventilation going from the top 
    params[205] = 0.25f;        // minWind          wind speed where the effect of wind on leakage begins
    params[206] = 0.0627f;      // dmfm             Dry matter to Fresh matter conversion rate
    params[207] = 1e-6;         // eps              Epsilon for numerical stability
    return params;
}
