import numpy as np

def parametric_crop_uncertainty(parameters, uncertainty, RNG):
    """
    Adds uncertainty to a vector of parameters based on the uncertainty parameter.
    
    Args:
        parameters (np.ndarray): The original parameter vector.
        uncertainty_param (float): The level of uncertainty to add.
        RNG: random number generator.

    Returns:
        np.ndarray: Parameter vector with added uncertainty.
    """
    # The following crop parameters in the parameter vector are perturbed
    indices = np.arange(128, 162)
    parameters = np.array(parameters)
    noise = RNG.uniform(-uncertainty/2, uncertainty/2, size=indices.shape)
    parameters[indices] += noise*parameters[indices]

    # cLeafMax is dependent of laiMax and sla
    parameters[144] = parameters[141] /parameters[142]
    return parameters
