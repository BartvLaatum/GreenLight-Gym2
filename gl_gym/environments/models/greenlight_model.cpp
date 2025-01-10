#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.h"
#include "params.h"
#include "aux_states.h"

namespace py = pybind11;

std::vector<double> control_signal(const std::vector<double>& x, const std::vector<double>& d, const std::vector<double>& setpoints, std::vector<double>& u, float dli) 
{
    // the control parameters are removed from the parameters vector.
    // Thus we need another vector/struct to hold those    
    double co2InPpm;
    double ventHeat;
    double ventCold;
    double rhIn;
    double ventRh;
    double blScr;
    u_int8_t lampsOn;

    // CO2 concentration in main compartment [ppm]
    co2InPpm = co2dens2ppm_cpp(x[2], 1e-6 * x[0]);

    // Ventilation control due to excess heating set point [°C]
    ventHeat = proportional_control(x[2], setpoints[3], 6., 0., 1.); // ventHeatBand 6.

    // Relative humidity [%]
    rhIn = 100 * x[15] / satVp_cpp(x[2]);

    // Ventilation setpoint due to excess humidity [°C]
    ventRh = proportional_control(rhIn, 90., 5., 0., 1.); // rhmax = 90., ventRhPband = 5.

    // Ventilation setpoint due to cold [°C]
    ventCold = proportional_control(x[2], setpoints[0] - 1., -1., 1, 0); // tVentOff = 1.	ventColdPband = -1.

    // // use blackout screen if outside is dark and cold
    blScr = (d[9] > 0.) ? 0. : ((1 - d[9] > 0. && d[1] < 13.) ? .9 : 0.);

    // turn lamps off if DLI is too high
    lampsOn = dli_check(setpoints[4], dli);


    u[0] = proportional_control(x[2], setpoints[0], -1., 0., 1.); // tHeatBand = -1.
    u[1] = proportional_control(co2InPpm, setpoints[1], -50., 0, 1); // co2Band = -50.
    u[2] = setpoints[2];
    u[3] = fmin(ventCold, fmax(ventHeat, ventRh));

    u[4] = setpoints[4] * lampsOn;
    u[5] = 0;                                                   // grow pipes unused in this case
    u[6] = 0;                                                   // inter lamps unused in this case 
    u[7] = fmax((1.-d[9])*lampsOn, blScr);                      // lampsOn

    return u;
}

SX ODE(const SX& x, const SX& u, const SX& d, const SX& p)
{
    // Compute the auxiliary variables
    SX a = update(x, u, d, p);
    SX dxdt = SX::zeros(x.size());

    // Carbon concentration of main compartment [mg m^{-3} s^{-1}]
    dxdt(0) = (1./p(122))  * (a(224) + a(223) + a(225) - a(217) - a(218) - a(220));
    
    // Carbon concentration of top compartment [mg m^{-3} s^{-1}]
    dxdt(1) = (1./p(123)) * (a(218) - a(219));

    // Greenhouse air temperature [°C s^{-1}]
    dxdt(2) = (1./p(112)) * (a(146)+a(226)-a(236)+a(157)
        +a(227)+a(228)+a(79)-a(147)-a(148)-a(150)
        -a(151)-a(230)-a(231)-a(149)
        +a(165)+a(77)
        +a(166)+a(167)+a(78));

    // Air above screen temperature [°C s^{-1}]
    dxdt(3) = (1./p(120)) * (a(152) + a(151) - a(154) - a(155) + a(153));

   // Canopy temperature [°C s^{-1}]
    dxdt(4) = (1./a(32)) * (a(54)+a(68)+a(92)-
        a(146)-a(187)-a(84)-a(87)-a(85)-a(86)-a(108)+
        a(55)+a(69)+a(101)+
        a(105)+a(56)+a(70)+a(117));

    // Internal cover temperature [°C s^{-1}]
    dxdt(5) = (1./a(34)) * (a(154)+a(190)+a(84)+
        a(93)+a(88)+a(96)-a(164)+
        a(103)+a(110)+a(121));

    // // External cover temperature [�C s^{-1}]
    dxdt(6) = (1./a(33)) * (a(80)+a(164)-a(156)-a(98));

    // Thermal screen temperature [°C s^{-1}]
    dxdt(7) = (1./p(119)) * (a(148)+a(188)+a(86)+
        a(95)+a(90)-a(152)-a(96)-a(97)+ a(109)+
        a(102)+a(120));

    // Greenhouse floor temperature [°C s^{-1}]
    dxdt(8) = (1./p(113)) * (a(147)+a(74)+a(71)+
        a(87)+a(91)-a(158)-a(93)-a(94)-a(95)+
        a(75)+a(72)+a(99)-a(106)+
        a(76)+a(73)+a(115));

    // Pipe temperature [°C s^{-1}]
    dxdt(9) = (1./p(110)) * (a(221)+a(232)+a(233)-a(89)-
        a(88)-a(92)-a(91)-a(90)-a(157)+
        a(100)-a(107)+a(239)+a(116));

    // Soil layer 1 temperature [°C s^{-1}]
    dxdt(10) = (1./p(114)) * (a(158)-a(159));

    // Soil layer 2 temperature [°C s^{-1}]
    dxdt(11) = (1./p(115)) * (a(159)-a(160));

    // Soil layer 3 temperature [°C s^{-1}]
    dxdt(12) = (1./p(116)) * (a(160)-a(161));

    // Soil layer 4 temperature [°C s^{-1}]
    dxdt(13) = (1./p(117)) * (a(161)-a(162));

    // Soil layer 5 temperature [°C s^{-1}]
    dxdt(14) = (1./p(118)) * (a(162)-a(163));

    // Vapor pressure of greenhouse air [Pa s^{-1}] = [kg m^{-1} s^{-3}]
    dxdt(15) = (1./a(35)) * (a(176)+a(177)+a(178)+a(179) -
        a(181)-a(184)-a(186)-a(180)-a(237)-a(182));

    // Vapor pressure of above screen air [Pa s^{-1}] = [kg m^{-1} s^{-3}]
    dxdt(16) = (1./a(36)) * (a(184)-a(183)-a(185));

    // Lamp temperature [°C s^{-1}]
    dxdt(17) = (1./p(184)) * (a(37)-a(165)-a(104)-a(103)-
        a(102)-a(100)-a(77)-a(112)-
        a(75)-a(72)-a(99)-
        a(55)-a(69)-a(101)-a(234)+a(118));

    // Inter lamp temperature [°C s^{-1}]
    dxdt(18) = (1./p(191)) * (a(38)-a(167)-a(122)-a(121)-
        a(120)-a(116)-a(78)-a(119)-
        a(76)-a(73)-a(115)-
        a(56)-a(70)-a(117)-a(118));

    // Grow pipes temperature [°C s^{-1}]
    dxdt(19) = (1./p(171)) * (a(222)-a(105)-a(166));

    // Blackout screen temperature [°C s^{-1}]
    dxdt(20) = (1./p(121)) * (a(149)+a(189)+a(108)+
        a(106)+a(107)-a(153)-a(110)-a(111)-a(109)+
        a(112)+a(119));

    // Average canopy temperature in last 24 hours
    dxdt(21) = (1./86400.) * (x(4)-x(21));

    // Carbohydrates in buffer [mg{CH2O} m^{-2} s^{-1}]
    dxdt(22) = a(200)-a(209)-a(207)-a(208)-a(210);

    // Carbohydrates in leaves [mg{CH2O} m^{-2} s^{-1}]
    dxdt(23) = a(207)-a(211)-a(215);

    // Carbohydrates in stem [mg{CH2O} m^{-2} s^{-1}]
    dxdt(24) = a(208)-a(212);

    // Carbohydrates in fruit [mg{CH2O} m^{-2} s^{-1}]
    dxdt(25) = a(209)-a(213)-a(216);

    // Crop development stage [°C day s^{-1}]
    dxdt(26) = (1./86400.) * x(4);

    // time in days since 00-00-0000
    dxdt(27) = 1./86400.;

    return dxdt;
}


struct GreenLight
{
    SX x;
    SX u;
    SX d;
    SX a;
    SX p;

    DM x_state;                                 // Numeric state variable
    std::vector<double> u_inputs;               // Numeric control inputs
    std::vector<double> u_hourly;             // Numeric control inputs
    std::vector<double> params;                 // Numerical parameter vector
    std::vector<double> setpoints;              // Numerical setpoints
    std::vector<std::vector<double>> weather;   // Numerical weather data

    uint8_t nx;
    uint8_t nu = 8;
    uint8_t nd = 10;
    uint16_t np = 208;
    uint16_t na = 240;
    uint32_t timestep = 0;
    uint32_t d_index = 0;
    uint8_t solver_steps;
    float h;                        // Resolution between control input for the integrator
    float solver_steps_div;         // Number of steps to divide the solver steps by

    double co2_dosing;
    double heating_pipe_energy;
    double elec_use;
    double prev_fruit_weight;
    double fruit_growth;
    float dli;


    // Function ODE_func;
    Function integrator_func;
    MXDict ode;

    GreenLight(u_int8_t nx, uint8_t solver_steps, float h, float solver_steps_div)
    : nx(nx), solver_steps(solver_steps), h(h), solver_steps_div(solver_steps_div)
    {
        // Define the symbolic variables for CasADi
        p = SX::sym("p", np);
        x = SX::sym("x", nx);
        u = SX::sym("u", nu);
        d = SX::sym("d", nd);
        a = SX::sym("a", na);

        // the control signal vector.
        u_inputs = std::vector<double>(nu);
        u_hourly = std::vector<double>(nu);
        setpoints = std::vector<double>(5);

        // define update function as symbolic function in CasADi
        Function update_aux = Function("update_aux", {x, u, d, p}, {update(x, u, d, p)});
        SX dxdt = ODE(x, u, d, p);
        SX input_args_sym = SX::vertcat({u, d, p});

        Dict opts;
        Dict jit_options;                       // Nested dictionary for JIT options
        
        opts["jit"] = true;
        opts["compiler"] = "shell";
        opts["abstol"] = 1e-6;
        opts["reltol"] = 1e-6;
        jit_options["flags"] = "-Ofast -march=native";        // Use specific compiler flags (e.g., -Ofast for aggressive optimization)
        opts["jit_options"] = jit_options;      // Add JIT options to the main options dictionary        // define numerical integrator

        integrator_func = integrator(
            "integrator_func", "cvodes",
            {{"x", x}, {"p", input_args_sym}, {"ode", dxdt}},
            0., h, opts
        );

        // set the parameters to theirdefault values
        params = init_default_params(np);
    }

    ~GreenLight() {
        std::cout << "GreenLight destructor called" << std::endl;
    }

    void reset(const double time_in_days, const std::vector<std::vector<double>>& weather_data)
    {
        timestep = 0;
        d_index = 0;
        dli = 0;
        reset_consumptions();

        weather = std::move(weather_data);
        x_state = init_state(weather_data[d_index], 90., time_in_days);
        prev_fruit_weight = x_state(25).scalar();
    }

    DM init_state(const std::vector<double>& d0, float rhMax, double time_in_days) 
    {
        DM state = DM::zeros(nx);
        state(0) = d0[3];       // co2Air
        state(1) = state(0);    // co2Top
        state(2) = 18.5;        // tAir
        state(3) = state(2);    // tTop
        state(4) = state(2) + 2; // tCan
        state(5) = state(2);    // tCovIn
        state(6) = state(2);    // tCovE
        state(7) = state(2);    // tThScr
        state(8) = state(2);    // tFlr
        state(9) = state(2);    // tPipe
        state(10) = state(2);   // tSoil1
        state(11) = .25*(3.*state(2) + d0[6]);  // tSoil2
        state(12) = .25*(2.*state(2) + 2*d0[6]);// tSoil3
        state(13) = .25*(state(2) + 3*d0[6]);   // tSoil4
        state(14) = d0[6];      // tSoil5
        state(15) = rhMax / 100. * satVP(state(2)); // vpAir
        state(16) = state(15);  // vpTop
        state(17) = state(2);   // tLamp
        state(18) = state(2);   // tIntLamp
        state(19) = state(2);   // tGroPipe
        state(20) = state(2);   // tBlScr
        state(21) = state(4);   // tCan24
        state(22) = 1000.;      // cBuf
        state(23) = 26000.;     // cLeaf
        state(24) = 18000.;     // cStem
        state(25) = 0.;         // cFruit
        state(26) = 0.;         // tCanSum
        state(27) = time_in_days; // time
        return state;
    }

    // Step function with setpoints vector as input
    void step(const std::vector<double>& action)
    {
        setpoints = action;
        static std::vector<double> running_sum(nu, 0.0);

        for (u_int8_t current_step = 0; current_step < solver_steps; ++current_step)
        {
            d_index = solver_steps * timestep + current_step;

            // Combine integrator input arguments (controls, disturbance, parameters) in numerically DM
            u_inputs = control_signal(get_state(), weather[d_index], setpoints, u_inputs, dli);
            // Compute running average over inputs

            for (size_t i = 0; i < nu; ++i) {
                u_hourly[i] += (u_inputs[i] / (solver_steps));
            }

            DM input_args = DM::vertcat({u_inputs, weather[d_index], params});

            // Prepare inputs for the integrator
            std::map<std::string, DM> integrator_in;
            integrator_in["x0"] = x_state;
            integrator_in["p"] = input_args;

            // Call the integrator
            DMDict result = integrator_func(integrator_in);
            // Extract the new state
            x_state = result["xf"];

            // Compute consumption of resources
            resource_consumptions();
        }
        calc_fruit_growth();
        timestep += 1;
    }

    std::vector<double> get_state() const
    {
        std::vector<double> state_vec = static_cast<std::vector<double>>(x_state);
        return state_vec;
    }

    std::vector<double> get_controls() const
    // returns the the average control inputs (uBoil, uCo2, uVent, uBlScr) for the previous hour
    {
        return u_hourly;
    }

    std::vector<double> get_weather() const
    {
        return weather[d_index];
    }

    std::vector<double> get_weather_pred(const u_int32_t pred) const
    {
        return weather[d_index + pred*solver_steps];
    }

    std::vector<double> get_setpoints() const
    {
        return setpoints;
    }

    void resource_consumptions()
    // Function that computes heat consumptions in kWh
    {
        // params 108 is the heat capacity of boiler to heating pipes [W/m2]
        heating_pipe_energy += (u_inputs[0] * params[108] / 1000./ solver_steps_div);
        // params 109 is the max CO2 injection rate [mg m^{-2} s^{-1}]
        co2_dosing += (u_inputs[1] * params[109] * 1e-6*h); // kg/m2
        elec_use += (u_inputs[4] * params[172]/1000./solver_steps_div);

        // compute the the daily light integral in mol/m2/day
        // convert the light intensity from W/m2 to mol/m2/s
        // and integrate over the time step
        dli += (sun_dli() + u_inputs[4]*params[172]*params[174]) * 4.6 * h*1e-6;
        dli += 0;
    }

    float sun_dli() const
    {
        float tauThScrPar = 1 - u_inputs[2] * (1 - params[80]);  // tauThScrPar
        float rhoThScrPar = u_inputs[2] * params[77];
        float tauCovThScrPar = tau12(params[69], tauThScrPar, params[66], rhoThScrPar);
        float rhoCovThScrParDn = rhoDn(tauThScrPar, params[66], rhoThScrPar, rhoThScrPar);

        float tauBlScrPar = 1 - u_inputs[7] * (1 - params[90]);
        float rhoBlScrPar = u_inputs[7] * params[88];
        float tauCovBlScrPar = tau12(tauCovThScrPar, tauBlScrPar, rhoCovThScrParDn, rhoBlScrPar);
        
        float rhoCovBlScrParDn = rhoDn(tauBlScrPar, rhoCovThScrParDn, rhoBlScrPar, rhoBlScrPar);
        float tauCovPar = tau12(tauCovBlScrPar, params[176], rhoCovBlScrParDn, params[179]);

        return (1 - params[44]) * tauCovPar * params[6] * weather[d_index][0];
    }

    double get_time() const
    //  we assume that the maximum fruit growth is equal to the difference between the potential fruit growth rate and the fruit maintenance rate
    // converted to g/m2/day by multiplying by 3600*24*1e-3
    {
        return x_state(27).scalar();
    }

    double get_can_temp_sum() const
    {
        return x_state(26).scalar();
    }

    double get_end_temp_sum() const
    {
        return params[164];
    }

    void calc_fruit_growth()
    // Function that calculates the fruit growth in mg/m2.
    {
        fruit_growth += x_state(25).scalar() - prev_fruit_weight;
        prev_fruit_weight = x_state(25).scalar();
    }

    void reset_consumptions()
    {
        heating_pipe_energy = 0.;
        co2_dosing = 0.;
        elec_use = 0.;
        fruit_growth = 0.;
        u_hourly = std::vector<double>(nu, 0.);
    }

    void reset_dli()
    {
        dli = 0.;
    }
};

PYBIND11_MODULE(greenlight_model, m) {
    m.doc() = "Pybind11 plugin for stiff ODE system with Boost Odeint and Eigen AutoDiff";

    py::class_<GreenLight>(m, "GreenLight")
        .def(py::init<uint8_t, uint8_t, float, float>())
        .def("get_state", &GreenLight::get_state)
        .def("get_controls", &GreenLight::get_controls)
        .def("get_weather", &GreenLight::get_weather)
        .def("step", &GreenLight::step)
        .def("reset", &GreenLight::reset)
        .def("reset_consumptions", &GreenLight::reset_consumptions)
        .def("reset_dli", &GreenLight::reset_dli)
        .def("get_time", &GreenLight::get_time)
        .def("get_end_temp_sum", &GreenLight::get_end_temp_sum)
        .def("get_can_temp_sum", &GreenLight::get_can_temp_sum)
        .def("get_setpoints", &GreenLight::get_setpoints)
        .def("get_weather_pred", &GreenLight::get_weather_pred)
        .def_readwrite("timestep", &GreenLight::timestep)
        .def_readwrite("heating_pipe_energy", &GreenLight::heating_pipe_energy)
        .def_readwrite("co2_dosing", &GreenLight::co2_dosing)
        .def_readwrite("elec_use", &GreenLight::elec_use)
        .def_readwrite("fruit_growth", &GreenLight::fruit_growth)
        .def_readwrite("dli", &GreenLight::dli);
}
