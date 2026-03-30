import time
from typing import Callable


class Timer:
    """
    Timer context manager to measure elapsed wall-clock time.

    Example:
        with Timer("Execution time:"):
            ...

    Args:
        prefix: Prefix text printed before elapsed time.
        postfix: Suffix text printed after elapsed time.
        print_func: Output callable, default is print.
    """

    def __init__(
        self, prefix: str = "", postfix: str = "", print_func: Callable[[str], None] = print
    ) -> None:
        self.prefix = prefix
        self.postfix = postfix
        self.print_func = print_func
        self.start_time = 0.0

    def __enter__(self):
        # perf_counter provides better precision for duration measurement.
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed_time = time.perf_counter() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
        self.print_func(f"{self.prefix} {time_str} {self.postfix}".strip())
