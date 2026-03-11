from gl_gym.components.parameters import ParameterDef, ParameterRegistry

TOMATO_PARAMETER_DEFS = [
    ParameterDef("floor_area", 46, low=0, high=2000, unit="m^2"),
    ParameterDef("max_heating_power", 108, low=0.0, high=1e6, unit="W"),
    ParameterDef("max_co2_dosing", 109, low=0.0, high=1e5, unit="mg/s"),
    ParameterDef("max_fruit_dw_growth_rate", 154, low=0.2, high=0.5, unit="mg/m^2/s"),
    ParameterDef("lamp_power", 172, low=50.0, high=400, unit="W/m^2"),
]

TOMATO_PARAMETER_REGISTRY = ParameterRegistry(TOMATO_PARAMETER_DEFS)