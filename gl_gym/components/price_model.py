from abc import ABC, abstractmethod


class BasePriceModel(ABC):
    @abstractmethod
    def get_price(self, ctx) -> float:
        ...

class FixedPrice(BasePriceModel):
    def __init__(self, price: float):
        self.price = float(price)

    def get_price(self, ctx) -> float:
        return self.price

class DailyFruitPriceTrajectory(BasePriceModel):
    def __init__(self, prices_by_day: dict[int, float], default_price: float | None = None):
        self.prices_by_day = prices_by_day

    def get_price(self, ctx) -> float:
        day = int(ctx.day_of_year)
        return self.prices_by_day[day]

class HourlyPriceTrajectory(BasePriceModel):
    def __init__(self, prices_by_hour: dict[int, float], default_price: float | None = None):
        self.prices_by_hour = prices_by_hour

    def get_price(self, ctx) -> float:
        hour = int(ctx.hour_of_day)
        return self.prices_by_hour[hour]

