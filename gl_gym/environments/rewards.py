from abc import ABC, abstractmethod
from typing import SupportsFloat, List, Optional

import numpy as np

from gl_gym.environments.models.greenlight_model import GreenLight

class BaseReward(ABC):
    profit: float
    fixed_costs: float
    variable_costs: float
    gains: float
    
    # def _scale(self, r: float) -> SupportsFloat:
    #     return (r - self.rmin)/(self.rmax - self.rmin)

    @abstractmethod
    def compute_reward(self, gl_model: GreenLight) -> SupportsFloat:
        pass

class EconomicReward(BaseReward):
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
                fixed_greenhouse_cost: float,
                fixed_co2_cost: float,
                fixed_lamp_cost: float,
                fixed_screen_cost: float,
                fixed_spacing_cost: float,
                off_peak_price: float,
                on_peak_price: float,
                heating_price: float,
                co2_price: float,
                fruit_price: float,
                dmfm: float,
                max_fruit_weight_pot: float,
                resolution: str,
                gl_model: GreenLight,
                timesteps_in_day: int,
                ) -> None:
        super(EconomicReward, self).__init__()

        # fixed costs for the greenhouse
        self.fixed_greenhouse_cost = fixed_greenhouse_cost
        self.fixed_co2_cost = fixed_co2_cost
        self.fixed_lamp_cost = fixed_lamp_cost * 200 # the max intensity per lamp
        self.fixed_screen_cost = fixed_screen_cost
        self.fixed_spacing_cost = fixed_spacing_cost

        # variable prices for the electricity, heating co2
        self.off_peak_price = off_peak_price    # €/kWh
        self.on_peak_price = on_peak_price      # €/kWh
        self.heating_price = heating_price      # €/kWh
        self.co2_price = co2_price              # €/kg

        self.fruit_price = fruit_price/max_fruit_weight_pot # (€/pot) / (g/pot) * (pot/m2) = €.g^{-1}.m^{-2}
        self.red_fruit_fraction = 1./3.         # Fraction of red fruit in the total fruit production; Assumption
        self.dmfm = dmfm                        # ration of dry matter to fresh matte; Assumption
        self.timesteps_in_day = timesteps_in_day

        self.resolution = resolution

        self.yearly_fixed_costs = sum([self.fixed_greenhouse_cost, self.fixed_co2_cost, self.fixed_lamp_cost, self.fixed_screen_cost, self.fixed_spacing_cost])

        self.reward_fns = {
            'daily': self.daily_reward,
            'hourly': self.hourly_reward
        }

        # self.rmin = self._min_reward(gl_model)
        # self.rmax = self._max_reward(gl_model)

        self._init_costs()

    def _init_costs(self):
        self.fixed_costs = 0
        self.variable_costs = 0
        self.gains = 0
        self.profit = 0
        self.heat_costs = 0
        self.co2_costs = 0
        self.off_peak_costs = 0
        self.on_peak_costs = 0

    def _fixed_costs_daily(self):
        """
        Computes the daily fixed costs.
        These costs refelct the daily fixed costs for the greenhouse, co2, lamps, screens and spacing.
        The unit is converted from €/m2/year to €/m2/day.
        """
        return self.yearly_fixed_costs/365.

    def _fixed_costs_hourly(self):
        """
        Computes the hourly fixed costs.
        These costs refelct the hourly fixed costs for the greenhouse, co2, lamps, screens and spacing.
        The unit is converted from €/m2/year to €/m2/hour.
        """
        return self.yearly_fixed_costs/365./self.timesteps_in_day

    def _variable_costs(self, gl_model):
        """
        Calculate the variable costs based on the given GreenLight model.
        These costs reflect the daily variable costs for heating, co2, off peak and on peak electricity.
        Has the same unit as the gains, which are computed as €/m2/day.
        Args:
            GLModel (greenlight_cy.GreenLight): The GreenLight model object.

        Returns:
            float: The total variable costs.
        """
        self.heat_costs = gl_model.heating_pipe_energy * self.heating_price
        self.co2_costs = gl_model.co2_dosing * self.co2_price
        self.on_peak_costs = gl_model.elec_use * self.on_peak_price
        # self.off_peak_costs = gl_model.elec_off_peak * self.off_peak_price
        # self.on_peak_costs = gl_model.elec_peak * self.on_peak_price
        return sum([self.heat_costs, self.co2_costs, self.on_peak_costs])

    def _gains(self, gl_model):
        """
        Computes the daily gains based on the given GreenLight model.
        These gains are computed as the gains per pot per day.
        Does the following steps:
        1. Converts the fruit growth in dry weight (DW) from (mg/m2) to (g/2)
        2. Converts the fruit DW to fruit fresh weight (FFW) using dmfm conversion factor
        3. Multiplies the daily FFW growth by the fruit price, which resembles €/g.
        """
        return max(gl_model.fruit_growth, 0) * 1e-3 / self.dmfm * self.fruit_price * self.red_fruit_fraction

    def _max_reward(self, gl_model):
        """
        Computes the maximum reward based on the given GreenLight model.
        """
        return gl_model.get_max_fruit_growth() * 1e-3 / self.dmfm * self.fruit_price * self.red_fruit_fraction

    def _min_reward(self, gl_model):
        """
        Computes the minimum reward based on the given GreenLight model.
        """
        min_var_costs = -(gl_model.get_max_pheat() * self.heating_price)\
            - (gl_model.get_max_co2dosing() * self.co2_price)\
            - (gl_model.get_max_elec() * (self.off_peak_price/3 + self.on_peak_price*2/3))
        return (min_var_costs - self._fixed_costs_hourly())

    # def compute_reward(self, gl_model) -> SupportsFloat:
    #     self.fixed_costs = self._fixed_costs()
    #     self.variable_costs = self._variable_costs(gl_model)
    #     self.gains = self._gains(gl_model)
    #     self.profit = self.gains - self.variable_costs - self.fixed_costs
    #     return self.profit



    def daily_reward(self, gl_model) -> SupportsFloat:
        if gl_model.timestep % self.timesteps_in_day== 0:
            self.fixed_costs = self._fixed_costs_daily()
            self.variable_costs = self._variable_costs(gl_model)
            self.gains = self._gains(gl_model)
            self.profit = self.gains - self.variable_costs - self.fixed_costs
            gl_model.reset_consumptions()
            gl_model.reset_dli()
        else:
            self.fixed_costs = 0
            self.variable_costs = 0
            self.gains = 0
            self.profit = 0
        return self.profit

    def hourly_reward(self, gl_model) -> SupportsFloat:
        self.fixed_costs = self._fixed_costs_hourly()
        self.variable_costs = self._variable_costs(gl_model)
        self.gains = self._gains(gl_model)
        self.profit = self.gains - self.variable_costs - self.fixed_costs
        gl_model.reset_consumptions()

        if gl_model.timestep % self.timesteps_in_day== 0:
            gl_model.reset_dli()

        return self.profit        

    def compute_reward(self, gl_model) -> SupportsFloat:
        return self.reward_fns[self.resolution](gl_model)

