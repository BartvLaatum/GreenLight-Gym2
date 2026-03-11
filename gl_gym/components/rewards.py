from abc import ABC, abstractmethod
from typing import SupportsFloat, Dict
import numpy as np
from gl_gym.core.types import RewardContext
from gl_gym.components.price_model import FixedPrice

class BaseReward(ABC):
    @abstractmethod
    def compute_reward(self, ctx: RewardContext) -> tuple[SupportsFloat, Dict[str, float]]:
        ...

    @staticmethod
    def scale_reward(r: float, min_r: float, max_r: float) -> float:
        return (r - min_r) / (max_r - min_r)

class GreenhouseReward(BaseReward):
    """	
    Economic reward function for the GreenLight environment.
    The reward is computed as the difference between the gains and the costs.
    The gains are computed as the fruit growth per pot per day multiplied by the fruit price.
    The costs are computed as the sum of the heating, co2, off peak and on peak electricity costs.

    Args:
        elec_price (float): price for off peak electricity [€/kWh]
        heating_price (float): price for heating [€/kWh]
        co2_price (float): price for co2 [€/kg]
        fruit_price (float): price for the fruit [€/kg]
        dmfm (float): ration of dry matter to fresh matter
        pen_lamp (float): weight for the lamp violation
    """

    def __init__(
                self,
                elec_price: float,
                heating_price: float,
                co2_price: float,
                fruit_price: float,
                pen_lamp: float,
                dmfm: float,
                dt: int,
                p: np.ndarray,
                ) -> None:
        super(GreenhouseReward, self).__init__()


        ctx = RewardContext(
            t=0,
            dt=dt,
            Np=0,
            x_prev=np.zeros(26),
            x=np.zeros(26),
            u=np.zeros(6),
            p=p,
            d=np.zeros(10),
            obs={},
            day_of_year=0,
            hour_of_day=0,
        )

        # variable prices for the electricity, heating co2
        self.elec_price_model = FixedPrice(elec_price)
        self.heating_price_model = FixedPrice(heating_price)
        self.co2_price_model = FixedPrice(co2_price)
        self.fruit_price_model = FixedPrice(fruit_price)

        self.dmfm = dmfm            # ratio of dry matter to fresh matter; Assumption
        self.pen_lamp = pen_lamp
        self._init_costs()
        self._init_violations()
        self.max_profit = self.max_profit_reward(ctx)
        self.min_profit = self.min_profit_reward(ctx)
        self.min_state_violations = self.min_violations()
        self.max_state_violations = self.max_violations()

    def min_violations(self):
        return np.zeros(3)

    def max_violations(self):
        co2_violation = 2500
        temp_violation = 15
        rh_violation = 15
        return np.array([co2_violation, temp_violation, rh_violation])

    def max_profit_reward(self, ctx: RewardContext) -> float:
        """
        Computes the maximum possible reward for the current timestep.
        The maximum reward is computed as the maximum possible gains minus the minimum possible costs.
        The maximum possible gains are computed as the maximum fruit growth per pot per day multiplied by the fruit price.
        The minimum possible costs are computed as the sum of the heating, co2, off peak and on peak electricity costs.
        Returns:
            float: The maximum possible reward.
        """
        max_gains = ctx.p[154] * ctx.dt * 1e-6 /self.dmfm * self.fruit_price_model.get_price(ctx)
        return max_gains

    def min_profit_reward(self, ctx: RewardContext) -> float:
        """
        Computes the minimum possible reward for the current timestep.
        The minimum reward is computed as the minimum possible gains minus the maximum possible costs.
        The minimum possible gains are computed as the minimum fruit growth per pot per day multiplied by the fruit price.
        The maximum possible costs are computed as the sum of the heating, co2, off peak and on peak electricity costs.
        Returns:
            float: The minimum possible reward.
        """
        max_heating = ctx.p[108] / ctx.p[46] * ctx.dt/3600*1e-3 * self.heating_price_model.get_price(ctx)     # convert W/aFlr to kWh/m2
        max_elec = ctx.p[172] * ctx.dt/3600*1e-3 * self.elec_price_model.get_price(ctx)                          # convert W/aFlr to kWh/m2
        max_cost = ctx.p[109] / ctx.p[46] * ctx.dt * 1e-6 * self.co2_price_model.get_price(ctx)        # convert to kg/m2
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

    def _variable_costs(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        """
        Calculate the variable costs based on the given GreenLight model.
        These costs reflect the daily variable costs for heating, co2, off peak and on peak electricity.
        Has the same unit as the gains, which are computed as €/m2/day.
        Returns:
            float: The total variable costs.
        """
        heating_energy = ctx.u[0] * ctx.p[108] / ctx.p[46] * ctx.dt/3600 * 1e-3   # convert W/aFlr to kWh/m2
        elec_use = ctx.u[4] * ctx.p[172] * ctx.dt/3600*1e-3                          # convert W/aFlr to kWh/m2
        co2_dosing = ctx.u[1] * ctx.p[109] / ctx.p[46] * ctx.dt * 1e-6          # convert to kg/m2
        self.heat_cost = heating_energy * self.heating_price_model.get_price(ctx)
        self.co2_cost = co2_dosing * self.co2_price_model.get_price(ctx)
        self.elec_cost = elec_use * self.elec_price_model.get_price(ctx)
        return sum([self.heat_cost, self.co2_cost, self.elec_cost]), {
            "heat_cost": self.heat_cost,
            "co2_cost": self.co2_cost,
            "elec_cost": self.elec_cost,
        }

    def _gains(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        """
        Computes the daily gains based on the given GreenLight model.
        These gains are computed as the gains per pot per day.
        Does the following steps:
        1. Computes the fruit growth in dry weight (DW) in (mg/m2)
        2. Converts the fruit DW to fruit fresh weight (FFW) in (kg/m2) using dmfm conversion factor
        3. Multiplies the daily FFW growth by the fruit price, which resembles €/kg.
        """
        fruit_growth_dm = ctx.x[25] - ctx.x_prev[25]
        fruit_growth_ffw = fruit_growth_dm * 1e-6 / self.dmfm
        fruit_price = float(self.fruit_price_model.get_price(ctx))
        gains = fruit_growth_ffw * fruit_price
        return gains, {
            "fruit_growth_dm": float(fruit_growth_dm),
            "fruit_growth_ffw": float(fruit_growth_ffw),
            "fruit_price": fruit_price,
            "revenue": gains,
        }

    def _output_violations(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        """
        Function that computes the absolute penalties for violating system constraints.
        System constraints are currently non-dynamical, and based on observation bounds of gym environment.
        We do not look at dry mass bounds, since those are non-existent in real greenhouse.
        """
        indoor_climate_obs = np.asarray(ctx.obs["IndoorClimateObservations"][:3], dtype=np.float64)
        lowerbound = ctx.constraints_low[:] - indoor_climate_obs
        lowerbound[lowerbound < 0] = 0
        upperbound = indoor_climate_obs - ctx.constraints_high[:]
        upperbound[upperbound < 0] = 0
        co2_violation = lowerbound[0] + upperbound[0]
        temp_violation = lowerbound[1] + upperbound[1]
        rh_violation = lowerbound[2] + upperbound[2]
        return lowerbound+upperbound

    def _control_violation(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        """
        Checks if lamps are used during night hours (after 8 PM).
        Sets lamp_violation to 1 if lamps are on after 20:00,
        otherwise sets it to 0.
        """
        if ctx.hour_of_day >= 20:
            if ctx.u[4] > 0:
                self.lamp_violation = 1
        lamp_violation = 0
        return lamp_violation

    def _control_penalty(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        lamp_violation = self._control_violation(ctx)
        return lamp_violation * self.pen_lamp, {
            "lamp_penalty": lamp_violation * self.pen_lamp,
        }

    def compute_reward(self, ctx: RewardContext) -> SupportsFloat:
        variable_costs, variable_costs_info = self._variable_costs(ctx)
        gains, gains_info = self._gains(ctx)
        profit = gains - variable_costs

        violations = self._output_violations(ctx)
        control_penalty, control_penalty_info = self._control_penalty(ctx)

        scaled_profit = self.scale_reward(profit, self.min_profit, self.max_profit)
        scaled_pen = self.scale_reward(violations, self.min_state_violations, self.max_state_violations)
        reward = scaled_profit - sum(scaled_pen) - control_penalty
        info = {
            "reward": reward,
            "profit": profit,
            "control_penalty": control_penalty,
            "penalty": sum(scaled_pen),
            "variable_costs": variable_costs,
            "gains": gains,
            "temp_penalty": scaled_pen[0],
            "co2_penalty": scaled_pen[1],
            "rh_penalty": scaled_pen[2],
            **variable_costs_info,
            **gains_info,
            **control_penalty_info,
        }
        return reward, info


REWARDS_MODULES = {"GreenhouseReward": GreenhouseReward}
