import casadi as ca
import numpy as np
import pandas as pd

from gl_gym.models.GreenLight.ode import ODE

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

