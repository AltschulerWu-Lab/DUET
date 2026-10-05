import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Stopwatch:
    """Context manager to measure execution time

    Can be used to measure the execution time of a code block or to accumulate
    total execution time by multiple code blocks with named timers.

    Usage
    -----
    ```
    with Stopwatch('Task name'):
        <code block>
    ```
    ```
    # Create a stopwatch instance
    t = Stopwatch('Performance Test')

    # Accumulate total execution time for myfunc
    for i in range(100):
        <code block>
        with t.time('myfunc'):
            myfunc()
        <code block>

    t.total('myfunc')
    ```
    """
    def __init__(self, label=None):
        self.start_time = None
        self.end_time = None
        self.label = label
        self._accumulated_times = {}

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        execution_time = self.end_time - self.start_time
        label = self.label or 'Execution time'
        logger.info("[time] %s: %s", label, self._format_time(execution_time))

    @contextmanager
    def time(self, name):
        """Accumulate execution time for a named code block."""
        if name not in self._accumulated_times:
            self._accumulated_times[name] = 0

        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self._accumulated_times[name] += end - start

    def get_time(self, name):
        """Get the accumulated time for a named code block."""
        return self._accumulated_times.get(name, 0)

    def print_time(self, name=None):
        """Print all accumulated times for named code blocks in descending order."""
        if not self._accumulated_times:
            logger.info("[time] No accumulated times recorded.")
            return

        if name is None:
            # Sort by time in descending order
            sorted_times = sorted(self._accumulated_times.items(), key=lambda x: x[1], reverse=True)
            logger.info("[time] All accumulated times (descending order):")
            for name, time_value in sorted_times:
                logger.info("[time]   %s: %s", name, self._format_time(time_value))
        else:
            logger.info("[time] %s total execution time: %s", name, self._format_time(self.get_time(name)))

    def reset(self, name=None):
        """Reset accumulated time for a specific name or all timings if name is None."""
        if name is None:
            self._accumulated_times = {}
        elif name in self._accumulated_times:
            self._accumulated_times[name] = 0

    def _format_time(self, seconds):
        """Format time in appropriate units."""
        if seconds < 0.001:  # Less than 1ms
            return f'{seconds * 1000000:.2f} μs'
        elif seconds < 1:  # Less than 1s
            return f'{seconds * 1000:.2f} ms'
        else:
            # Extract hours, minutes, seconds
            hours, remainder = divmod(int(seconds), 3600)
            minutes, seconds = divmod(remainder, 60)

            if hours > 0:
                return f"{hours} hour{'s' if hours > 1 else ''}, {minutes} minute{'s' if minutes > 1 else ''}, {seconds} second{'s' if seconds > 1 else ''}"
            elif minutes > 0:
                return f"{minutes} minute{'s' if minutes > 1 else ''}, {seconds} second{'s' if seconds > 1 else ''}"
            else:
                return f"{seconds} second{'s' if seconds > 1 else ''}"