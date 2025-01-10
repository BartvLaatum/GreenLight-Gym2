from typing import Callable

def linear_schedule(initial_learning_rate: float, final_learning_rate: float, final_progress: float) -> Callable[[float], float]:
    """
    Creates a function that returns a linearly interpolated value between 'initial_learning_rate' and 'final_learning_rate'
    up to 'final_progress' fraction of the total training duration, and then 'final_learning_rate' onwards.
    
    :param initial_learning_rate: The initial learning rate.
    :param final_learning_rate: The final learning rate.
    :param final_progress: The fraction of the total timesteps at which the final value is reached.
    :return: A function that takes a progress (0 to 1) and returns the learning rate.
    """
    def func(progress: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress > final_progress:
            return initial_learning_rate + (1.0 - progress) * (final_learning_rate - initial_learning_rate) / (1.0-final_progress)
        else:
            return final_learning_rate
    
    return func
