import numpy as np
from gl_gym.environments.utils import co2dens2ppm, satVp

class RuleBasedController:
    def __init__(
        self,
        lamps_on,
        lamps_off,
        lamps_day_start,
        lamps_day_stop,
        lamps_off_sun,
        lamp_rad_sum_limit,
        temp_setpoint_day,
        temp_setpoint_night,
        heat_correction,
        heat_deadzone,
        co2_day,
        vent_heat_Pband,
        rh_max,
        mech_dehumid_Pband,
        vent_rh_Pband,
        t_vent_off,
        vent_cold_Pband,
        thScrSpDay,
        thScrSpNight,
        thScrPband,
        thScrDeadZone,
        thScrRh,
        thScrRhPband,
        lampExtraHeat,
        blScrExtraRh,
        rhMax,
        tHeatBand,
        co2Band,
        useBlScr
    ):
        self.lamps_on = lamps_on
        self.lamps_off = lamps_off
        self.lamps_day_start = lamps_day_start
        self.lamps_day_stop = lamps_day_stop
        self.lamps_off_sun = lamps_off_sun
        self.lamp_rad_sum_limit = lamp_rad_sum_limit
        self.temp_setpoint_day = temp_setpoint_day
        self.temp_setpoint_night = temp_setpoint_night
        self.heat_correction = heat_correction
        self.heat_deadzone = heat_deadzone
        self.co2_day = co2_day
        self.vent_heat_Pband = vent_heat_Pband
        self.rh_max = rh_max
        self.mech_dehumid_Pband = mech_dehumid_Pband
        self.vent_rh_Pband = vent_rh_Pband
        self.t_vent_off = t_vent_off
        self.vent_cold_Pband = vent_cold_Pband
        self.thScrSpDay = thScrSpDay
        self.thScrSpNight = thScrSpNight
        self.thScrPband = thScrPband
        self.thScrDeadZone = thScrDeadZone
        self.thScrRh = thScrRh
        self.thScrRhPband = thScrRhPband
        self.lampExtraHeat = lampExtraHeat
        self.blScrExtraRh = blScrExtraRh
        self.rhMax = rhMax
        self.tHeatBand = tHeatBand
        self.co2Band = co2Band
        self.useBlScr = useBlScr
        
    
    def predict(self, x, d, env):
        u = np.zeros(env.nu)

        # Control of the lamp according to the time of day [0/1]
        # if p.lampsOn < p.lampsOff, lamps are on from p.lampsOn to p.lampsOff each day
        # if p.lampsOn > p.lampsOff, lamps are on from p.lampsOn until p.lampsOff the next day
        # if p.lampsOn == p.lampsOff, lamps are always off
        # for continuous light, set p.lampsOn = -1, p.lampsOff = 25
        lampTimeOfDay = ((self.lamps_on <= self.lamps_off) * (self.lamps_on < env.hour_of_day and env.hour_of_day < self.lamps_off) + \
                            (1-(self.lamps_on <= self.lamps_off)) * (self.lamps_on < env.hour_of_day or env.hour_of_day < self.lamps_off))

        # CURRENTLY UNUSED...
        # Control of the lamp according to the day of year [0/1]
        # if p.dayLampStart < p.dayLampStop, lamps are on from p.dayLampStart to p.dayLampStop
        # if p.dayLampStart > p.dayLampStop, lamps are on from p.lampsOn until p.lampsOff the next year
        # if p.dayLampStart == p.dayLampStop, lamps are always off
        # for no influence of day of year, set p.dayLampStart = -1, p.dayLampStop > 366
        lampDayOfYear = ((self.lamps_day_start <= self.lamps_day_stop) * (self.lamps_day_start < env.day_of_year and env.day_of_year < self.lamps_day_stop) + \
                            (1-(self.lamps_day_start <= self.lamps_day_stop)) * (self.lamps_day_start < env.day_of_year or env.day_of_year < self.lamps_day_stop))


        # THIS VARIABLE MAINLY REPRESENTS WHETHER WE ARE IN LIGHT PERIOD OF THE GREENHOUSE
        # Control for the lamps disregarding temperature and humidity constraints
        # Chapter 4 Section 2.3.2, Chapter 5 Section 2.4 [7]
        # Section 2.3.2 [8]
        # This variable is used to decide if the greenhouse is in the light period
        # ("day inside"), needed to set the climate setpoints. 
        # However, the lamps may be switched off if it is too hot or too humid
        # in the greenhouse. In this case, the greenhouse is still considered
        # to be in the light period
        lampNoCons = (d[0] < self.lamps_off_sun) * (d[7] < self.lamp_rad_sum_limit) * lampTimeOfDay * lampDayOfYear

        ## Smoothing of control of the lamps
        # To allow smooth transition between day and night setpoints

        # Linear version of lamp switching on: 
        # 1 at lampOn, 0 one hour before lampOn, with linear transition
        # Note: this current function doesn't do a linear interpolation if
        # lampOn == 0
        linearLampSwitchOn = max(0, min(1, env.hour_of_day-self.lamps_on + 1))

        # Linear version of lamp switching on: 
        # 1 at lampOff, 0 one hour after lampOff, with linear transition
        # Note: this current function doesn't do a linear interpolation if
        # lampOff == 24
        linearLampSwitchOff = max(0, min(1, self.lamps_off - env.hour_of_day + 1))

        # Combination of linear transitions above
        # if p.lampsOn < p.lampsOff, take the minimum of the above
        # if p.lampsOn > p.lampsOn, take the maximum
        # if p.lampsOn == p.lampsOff, set at 0
        linearLampBothSwitches = (self.lamps_on!=self.lamps_off)*((self.lamps_on<self.lamps_off)*min(linearLampSwitchOn,linearLampSwitchOff)
            + (1-(self.lamps_on<self.lamps_off))*max(linearLampSwitchOn,linearLampSwitchOff))

        # Smooth (linear) approximation of the lamp control
        # To allow smooth transition between light period and dark period setpoints
        # 1 when lamps are on, 0 when lamps are off, with a linear
        # interpolation in between
        # Does not take into account the lamp switching off due to 
        # instantaenous sun radiation, excess heat or humidity
        smoothLamp = linearLampBothSwitches * (d[7] < self.lamp_rad_sum_limit) * lampDayOfYear

        # Indicates whether daytime climate settings should be used, i.e., if
        # the sun is out or the lamps are on
        # 1 if day, 0 if night. If lamps are on it is considered day
        isDayInside = max(smoothLamp, d[8])

        # Heating set point [°C]
        heatSetPoint = isDayInside*self.temp_setpoint_day + (1-isDayInside)* self.temp_setpoint_night + self.heat_correction*lampNoCons

        #% Ventilation setpoint due to excess heating set point [°C]
        heatMax = heatSetPoint + self.heat_deadzone

        # CO2 set point [ppm]
        co2SetPoint = isDayInside* self.co2_day

        # CO2 concentration in main compartment [ppm]
        co2InPpm = co2dens2ppm(x[2], 1e-6*x[0])

        # Ventilation setpoint due to excess heating set point [°C]
        ventHeat = self.proportional_control(x[2], heatMax, self.vent_heat_Pband, 0, 1)
    
        # Relative humidity [%]
        rhIn = 100*x[15]/satVp(x[2])

        # Ventilation setpoint due to excess humidity [°C]
        # mechallowed = 1 if mechanical ventilation is allowed, 0 otherwise We have have it at zero
        ventRh = self.proportional_control(rhIn, self.rh_max + 0 * self.mech_dehumid_Pband, self.vent_rh_Pband, 0, 1)

        # Ventilation closure due to too cold temperatures 
        ventCold = self.proportional_control(x[2], heatSetPoint-self.t_vent_off, self.vent_cold_Pband, 1, 0)

        # Setpoint for closing the thermal screen [°C]
        thScrSp = (d[8]) * self.thScrSpDay + (1- (d[8])) * self.thScrSpNight

        # Closure of the thermal screen based on outdoor temperature [0-1, 0 is fully open]
        thScrCold = self.proportional_control(d[1], thScrSp, self.thScrPband, 0, 1)

        # Opening of thermal screen closure due to too high temperatures 
        thScrHeat = self.proportional_control(x[2], heatSetPoint+ self.thScrDeadZone, -self.thScrPband, 1, 0)

        # Opening of thermal screen due to high humidity [0-1, 0 is fully open]
        thScrRh = max(self.proportional_control(rhIn, self.rhMax+self.thScrRh, self.thScrRhPband, 1, 0), 1-ventCold)

            # if 1-ventCold == 0 (it's too cold inside to ventilate)
            # don't force to open the screen (even if RH says it should be 0)
            # Better to reduce RH by increasing temperature

        # Control for the top lights: 
        # 1 if lamps are on, 0 if lamps are off
        # addAux(gl, 'lampOn', gl.lampNoCons.* ... # Lamps should be on
        #     self.proportional_control(gl.x.tAir, gl.heatMax+gl.p.lampExtraHeat, -0.5, 0, 1).* ... # Switch lamp off if too hot inside
        #     ...                                            # Humidity: only switch off if blackout screen is used 
        #     (gl.d.isDaySmooth + (1-gl.d.isDaySmooth).* ... # Blackout sceen is only used at night 
        #         max(self.proportional_control(gl.rhIn, gl.p.rhMax+gl.p.blScrExtraRh, -0.5, 0, 1),... # Switch lamp off if too humid inside
        #                     1-gl.ventCold))); # Unless ventCold == 0
                            # if ventCold is 0 it's too cold inside to ventilate, 
                            # better to raise the RH by heating. 
                            # So don't open the blackout screen and 
                            # don't stop illuminating in this case. 
        lampOn = lampNoCons * self.proportional_control(x[2], heatMax + self.lampExtraHeat, -0.5, 0, 1) *\
                    (d[9] + (1-d[9])) *\
                    max(self.proportional_control(rhIn, self.rhMax + self.blScrExtraRh, -0.5, 0, 1), 1-ventCold)

        # Control for the interlights: 
        # 1 if interlights are on, 0 if interlights are off
        # Lamps should be on
        # Switch lamp off if too hot inside
        # Humidity: only switch off if blackout screen is used 
        # Blackout screen is only used at night 
        # Switch lamp off if too humid inside
        # 1-gl.ventCold))); 
        # if ventCold is 0 it's too cold inside to ventilate, 
        # better to raise the RH by heating. 
        # So don't open the blackout screen and 
        # don't stop illuminating in this case. 
        # intLampOn = lampNoCons * self.proportional_control(x[2], heatMax + self.lampExtraHeat, -0.5, 0, 1) * \
        #             (d[9] + (1-d[9])) *\
        #             fmax(self.proportional_control(rhIn, self.rhMax + self.blScrExtraRh, -0.5, 0, 1), 1-ventCold)


        # boiler, co2, thscr, roof, lamps, intlamps, boilgro, blscr
        u[0] = self.proportional_control(x[2], heatSetPoint, self.tHeatBand, 0, 1)
        u[1] = self.proportional_control(co2InPpm, co2SetPoint, self.co2Band, 0, 1)
        u[2] = min(thScrCold, max(thScrHeat, thScrRh))
        u[3] = min(ventCold, max(ventHeat, ventRh))
        u[4] = lampOn
        u[5] = self.useBlScr * (1-d[9]) * lampOn

        # u[5] = intLampOn * self.intLamps
        # u[6] = self.proportional_control(x[2], heatSetPoint, self.tHeatBand, 0, 1) * self.pBoilGro
        # UNUSED shading screen, permanent shading screen, side ventilation
        # u[8] = 0
        # u[9] = 0
        # u[10] = 0
        return u


    def proportional_control(self, processVar, setPt, pBand, minVal, maxVal):
        return minVal + (maxVal - minVal)*(1/(1+np.exp(-2/pBand*np.log(100)*(processVar - setPt - pBand/2))))
