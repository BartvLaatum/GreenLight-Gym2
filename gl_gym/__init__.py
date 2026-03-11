from gymnasium.envs.registration import register

register(
    id="gl_gym/GreenLightTomato-v0",
    entry_point="gl_gym.environments.greenlight_env:GreenLightEnv",
    kwargs={
        "num_params": 208,
        "nx": 28,
        "nu": 6,
        "nd": 10,
        "dt": 900,
        "u_min": [0, 0, 0, 0, 0, 0],
        "u_max": [1, 1, 1, 1, 1, 1],
        "delta_u_max": 0.1,
        "season_length": 60,
        "pred_horizon": 0.5,
        "observation_modules": [
            "BasicCropObservations",
            "ControlObservations",
            "IndoorClimateObservations",
            "WeatherObservations",
            "TimeObservations",
        ],
        "constraints": {
            "co2_min": 300.0,
            "co2_max": 1600.0,
            "temp_min": 15.0,
            "temp_max": 34.0,
            "rh_min": 50.0,
            "rh_max": 85.0,
        },
        "reward_fn": "GreenhouseReward",
        "reward_kwargs": {
            "elec_price": 0.3,
            "heating_price": 0.09,
            "co2_price": 0.3,
            "fruit_price": 1.6,
            "dmfm": 0.065,
            "pen_lamp": 0.1,
        },
        "weather_scenario_sampler": "fixed",
        "weather_scenario_sampler_kwargs": {
            "location": "Amsterdam",
            "growth_year": 2010,
            "start_day": 59,
        },
        "controlled_inputs": [
            "uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr",
        ],
        "normalize_actions": True,
        "parameter_provider": "fixed",
        "parameter_provider_kwargs": {},
    },
)
