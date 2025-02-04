from abc import ABC, abstractmethod
from typing import SupportsFloat, List, Optional

import numpy as np

class BaseReward(ABC):
    profit: float
    fixed_costs: float
    variable_costs: float
    gains: float
    
    # def _scale(self, r: float) -> SupportsFloat:
    #     return (r - self.rmin)/(self.rmax - self.rmin)

    @abstractmethod
    def compute_reward(self) -> SupportsFloat:
        pass

    def scale_reward(self, r: float, min_r, max_r) -> SupportsFloat:
        return (r - min_r)/(max_r - min_r)

class GreenhouseReward(BaseReward):
    """	
    Economic reward function for the GreenLight environment.
    The reward is computed as the difference between the gains and the costs.
    The gains are computed as the fruit growth per pot per day multiplied by the fruit price.
    The costs are computed as the sum of the heating, co2, off peak and on peak electricity costs.
    The fixed costs for the greenhouse, co2, lamps, screens and spacing are also taken into account.

    Args:
        fixed_greenhouse_cost (float): fixed costs for the greenhouse [€/m2/year]
        fixed_co2_cost (float): fixed costs for the co2 [€/m2/year]
        fixed_lamp_cost (float): fixed costs for the lamps [€/m2/year]
        fixed_screen_cost (float): fixed costs for the screens [€/m2/year]
        fixed_spacing_cost (float): fixed costs for the spacing [€/m2/year]
        off_peak_price (float): price for off peak electricity [€/kWh]
        on_peak_price (float): price for on peak electricity [€/kWh]
        heating_price (float): price for heating [€/kWh]
        co2_price (float): price for co2 [€/kg]
        fruit_price (float): price for the fruit [€/kg]
        dmfm (float): ration of dry matter to fresh matter
        max_fruit_weight_pot (float): maximum fruit weight per pot
        gl_model (GreenLight): GreenLight model object
    """

    def __init__(
                self,
                env,
                fixed_greenhouse_cost: float,
                fixed_co2_cost: float,
                fixed_lamp_cost: float,
                fixed_screen_cost: float,
                elec_price: float,
                heating_price: float,
                co2_price: float,
                fruit_price: float,
                pen_weights: List[float],
                pen_lamp: float,
                dmfm: float,
                ) -> None:
        super(GreenhouseReward, self).__init__()

        # fixed costs for the greenhouse
        self.env = env
        self.fixed_greenhouse_cost = fixed_greenhouse_cost
        self.fixed_co2_cost = fixed_co2_cost
        self.fixed_lamp_cost = fixed_lamp_cost * 116 # the max intensity per lamp
        self.fixed_screen_cost = fixed_screen_cost
        yearly_fixed_costs = sum([self.fixed_greenhouse_cost, self.fixed_co2_cost, self.fixed_lamp_cost, self.fixed_screen_cost])
        self.fixed_costs = self._fixed_costs_timestep(yearly_fixed_costs)
        # variable prices for the electricity, heating co2
        self.elec_price = elec_price                # €/kWh
        self.heating_price = heating_price          # €/kWh
        self.co2_price = co2_price                  # €/kg

        self.fruit_price = fruit_price              # €/kg
        self.dmfm = dmfm                            # ratio of dry matter to fresh matter; Assumption
        self.pen_weights = np.array(pen_weights)
        self.pen_lamp = pen_lamp
        self._init_costs()
        self._init_violations()
        self.max_profit = self.max_profit_reward()
        self.min_profit = self.min_profit_reward()
        self.min_state_violations = self.min_violations()
        self.max_state_violations = self.max_violations()

    def min_violations(self):
        return np.zeros(3)
    
    def max_violations(self):
        co2_violation = 2500
        temp_violation = 15
        rh_violation = 15
        return np.array([co2_violation, temp_violation, rh_violation])

    def max_profit_reward(self):
        """
        Computes the maximum possible reward for the current timestep.
        The maximum reward is computed as the maximum possible gains minus the minimum possible costs.
        The maximum possible gains are computed as the maximum fruit growth per pot per day multiplied by the fruit price.
        The minimum possible costs are computed as the sum of the heating, co2, off peak and on peak electricity costs.
        Returns:
            float: The maximum possible reward.
        """
        max_gains = self.env.p[154] * self.env.dt * 1e-6 /self.dmfm * self.fruit_price
        return max_gains

    def min_profit_reward(self):
        """
        Computes the minimum possible reward for the current timestep.
        The minimum reward is computed as the minimum possible gains minus the maximum possible costs.
        The minimum possible gains are computed as the minimum fruit growth per pot per day multiplied by the fruit price.
        The maximum possible costs are computed as the sum of the heating, co2, off peak and on peak electricity costs.
        Returns:
            float: The minimum possible reward.
        """

        # min_gains = self.env.p[155] * self.env.dt * 1e-6 /self.dmfm * self.fruit_price
        max_heating = self.env.p[108] / self.env.p[46] * self.env.dt/3600*1e-3 * self.heating_price     # convert W/aFlr to kWh/m2
        max_elec = self.env.p[172] * self.env.dt/3600*1e-3 * self.elec_price                          # convert W/aFlr to kWh/m2
        max_cost = self.env.p[109] / self.env.p[46] * self.env.dt * 1e-6 * self.co2_price        # convert to kg/m2
        max_costs = sum([max_heating, max_elec, max_cost])
        max_costs = -max_costs
        return max_costs

    def _init_violations(self):
        self.temp_violation = 0
        self.co2_violation = 0
        self.rh_violation = 0
        self.lamp_violation = 0

    def _init_costs(self):
        self.variable_costs = 0
        self.gains = 0
        self.profit = 0
        self.heat_costs = 0
        self.co2_costs = 0
        self.elec_costs = 0

    def _fixed_costs_daily(self):
        """
        Computes the daily fixed costs.
        These costs refelct the daily fixed costs for the greenhouse, co2, lamps, screens and spacing.
        The unit is converted from €/m2/year to €/m2/day.
        """
        return self.yearly_fixed_costs/365.

    def _fixed_costs_timestep(self, yearly_fixed_costs):
        """
        Computes the fixed costs per timestep.
        These costs refelct the hourly fixed costs for the greenhouse, co2, lamps, screens and spacing.
        The unit is converted from €/m2/year to €/m2/timestep.
        """
        return yearly_fixed_costs/365/(86400//self.env.dt)

    def _variable_costs(self):
        """
        Calculate the variable costs based on the given GreenLight model.
        These costs reflect the daily variable costs for heating, co2, off peak and on peak electricity.
        Has the same unit as the gains, which are computed as €/m2/day.
        Returns:
            float: The total variable costs.
        """
        heating_energy = self.env.u[0] * self.env.p[108] / self.env.p[46] * self.env.dt/3600*1e-3   # convert W/aFlr to kWh/m2
        elec_use = self.env.u[4] * self.env.p[172] * self.env.dt/3600*1e-3                          # convert W/aFlr to kWh/m2
        co2_dosing = self.env.u[1] * self.env.p[109] / self.env.p[46] * self.env.dt * 1e-6          # convert to kg/m2
        self.heat_costs = heating_energy * self.heating_price
        self.co2_costs = co2_dosing * self.co2_price
        self.elec_costs = elec_use * self.elec_price
        return sum([self.heat_costs, self.co2_costs, self.elec_costs])

    def _gains(self):
        """
        Computes the daily gains based on the given GreenLight model.
        These gains are computed as the gains per pot per day.
        Does the following steps:
        1. Computes the fruit growth in dry weight (DW) in (mg/m2)
        2. Converts the fruit DW to fruit fresh weight (FFW) in (kg/m2) using dmfm conversion factor
        3. Multiplies the daily FFW growth by the fruit price, which resembles €/kg.
        """
        fruit_growth_dm = self.env.x[25] - self.env.x_prev[25]
        fruit_growth_ffw = fruit_growth_dm * 1e-6 / self.dmfm
        return fruit_growth_ffw * self.fruit_price

    def output_violations(self):
        """
        Function that computes the absolute penalties for violating system constraints.
        System constraints are currently non-dynamical, and based on observation bounds of gym environment.
        We do not look at dry mass bounds, since those are non-existent in real greenhouse.
        """
        lowerbound = self.env.constraints_low[:] - self.env.obs[[0, 1, 2]]
        lowerbound[lowerbound < 0] = 0
        upperbound = self.env.obs[[0, 1, 2]] - self.env.constraints_high[:]
        upperbound[upperbound < 0] = 0
        self.co2_violation = lowerbound[0] + upperbound[0]
        self.temp_violation = lowerbound[1] + upperbound[1]
        self.rh_violation = lowerbound[2] + upperbound[2]
        return lowerbound+upperbound

    def output_penalty_reward(self, violations):
        return np.dot(self.pen_weights, violations)

    def control_violation(self):
        """
        Checks if lamps are used during night hours (after 8 PM).
        Sets lamp_violation to 1 if lamps are on after 20:00,
        otherwise sets it to 0.
        """
        if self.env.hour_of_day >= 20:
            if self.env.u[4] > 0:
                self.lamp_violation = 1
        self.lamp_violation = 0

    def control_penalty(self):
        self.control_violation()
        return self.lamp_violation * self.pen_lamp

    def compute_reward(self) -> SupportsFloat:
        self.variable_costs = self._variable_costs()
        self.gains = self._gains()
        # self.profit = self.gains - self.variable_costs - self.fixed_costs
        self.profit = self.gains - self.variable_costs

        violations = self.output_violations()
        self.penalty = self.output_penalty_reward(violations)
        self.control_pen = self.control_penalty()

        scaled_profit = self.scale_reward(self.profit, self.min_profit, self.max_profit)
        scaled_pen = np.sum(self.scale_reward(violations, self.min_state_violations, self.max_state_violations))
        # r_pen = self.scale_reward(self.penalty, 0, 1)
        return scaled_profit - scaled_pen - self.control_pen
