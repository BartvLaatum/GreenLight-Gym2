#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils.hpp"
#include "params.hpp"
#include "aux_states.hpp"
#include "ode.hpp"

namespace py = pybind11;

struct GreenLight
{
    SX x;
    SX u;
    SX d;
    SX a;
    SX p;

    // Reuse buffers to minimize overhead:
    std::vector<DM> inputs_;        // F has 4 inputs: x, u, d, p
    std::vector<DM> result_;        // F returns 1 output: x_next
    std::vector<double> x_next_;    // Reusable output buffer

    // Function ODE_func;
    Function integrator_func;
    Function F;
    MXDict ode;

    GreenLight(u_int8_t nx, u_int8_t nu, u_int8_t nd, u_int16_t np, float dt)
    {
        // Define the symbolic variables for CasADi
        SX x = SX::sym("x", nx);
        SX u = SX::sym("u", nu);
        SX d = SX::sym("d", nd);
        SX p = SX::sym("p", np);

        // Define ODE right-hand side
        SX dxdt = ODE(x, u, d, p);

        // Concatenate control inputs, disturbances and parameters
        SX input_args_sym = SX::vertcat({u, d, p});

        // Set up integrator options
        Dict opts;
        Dict jit_options;

        opts["jit"] = true;
        opts["compiler"] = "shell";
        opts["abstol"] = 1e-6;
        opts["reltol"] = 1e-6;
        opts["jit_cleanup"] = true;
        jit_options["flags"] = "-O3";
        jit_options["cleanup"] = true;   // Some CasADi versions recognize this
        opts["jit_options"] = jit_options;
        SXDict input_output{{"x", x}, {"p", input_args_sym}, {"ode", dxdt}};
        // Create the integrator
        integrator_func = integrator(
            "integrator_func", "cvodes",
            input_output,
            0.0, dt, opts
        );

        // Create a wrapper function that takes (x, u, d, p) as input
        // and returns the next state
        MX x_mx = MX::sym("x", nx);
        MX u_mx = MX::sym("u", nu);
        MX d_mx = MX::sym("d", nd);
        MX p_mx = MX::sym("p", np);

        MX input_args_mx = MX::vertcat({u_mx, d_mx, p_mx});

        // Call integrator symbolically
        std::map<std::string, MX> integrator_in;
        integrator_in["x0"] = x_mx;
        integrator_in["p"] = input_args_mx;
        
        auto result = integrator_func(integrator_in);

        // Create the final function
        F = Function(
            "F",                            // name
            {x_mx, u_mx, d_mx, p_mx},       // inputs
            {result.at("xf")},              // outputs
            {"x", "u", "d", "p"},           // input names
            {"x_next"}                      // output names
        );

        // Pre-allocate buffers
        inputs_.resize(4); // x, u, d, p
        result_.resize(1); // x_next
        x_next_.resize(nx, 0.0);
    }

    std::vector<double> evalF(
        const std::vector<double>& x_np,
        const std::vector<double>& u_np,
        const std::vector<double>& d_np,
        const std::vector<double>& p_np
    )
    {
        // Reuse input buffer
        inputs_[0] = x_np;  // x
        inputs_[1] = u_np;  // u
        inputs_[2] = d_np;  // d
        inputs_[3] = p_np;  // p

        // Evaluate CasADi function
        result_ = F(inputs_);

        // result_[0] is DM holding x_nexts
        DM& x_next_dm = result_[0];

        // Copy to x_next_ buffer
        auto data = x_next_dm.nonzeros();
        std::copy(data.begin(), data.end(), x_next_.begin());

        return x_next_;
    }

    ~GreenLight() {
        std::cout << "GreenLight destructor called" << std::endl;
        F = Function();
        integrator_func = Function();
    }

};

PYBIND11_MODULE(greenlight_model, m) {
    m.doc() = "Pybind11 plugin for stiff ODE system with Boost Odeint and Eigen AutoDiff";

    py::class_<GreenLight>(m, "GreenLight")
        .def(py::init<uint8_t, uint8_t, uint8_t, uint16_t, float>())
        .def("evalF", &GreenLight::evalF);
}
