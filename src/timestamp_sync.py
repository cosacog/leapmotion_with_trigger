# -*- coding: utf-8 -*-
"""
Timestamp Synchronization Utilities
Synchronizes Leap Motion hardware timer with system time (time.time()).
"""

import time
import threading
from typing import Optional, Tuple


class HighPrecisionTimer:
    """
    High-precision timer combining time.time() and time.perf_counter().

    Uses perf_counter for high precision while maintaining absolute time reference.
    """

    def __init__(self):
        # Calibrate offset between time.time() and time.perf_counter()
        self._time_base = time.time()
        self._perf_base = time.perf_counter()

    def now(self) -> float:
        """
        Get current time with high precision.

        Returns:
            Time in seconds (Unix epoch compatible)
        """
        elapsed_perf = time.perf_counter() - self._perf_base
        return self._time_base + elapsed_perf

    def perf_to_time(self, perf_counter: float) -> float:
        """
        Convert perf_counter value to time.time() equivalent.

        Args:
            perf_counter: Value from time.perf_counter()

        Returns:
            Equivalent time.time() value
        """
        elapsed = perf_counter - self._perf_base
        return self._time_base + elapsed


class TimestampConverter:
    """
    Converts between Leap Motion hardware timestamps and system time.

    Leap Motion timestamps are in microseconds since device startup.
    System time is in seconds since Unix epoch (time.time()).

    This class calculates the offset between the two clocks and provides
    conversion methods.
    """

    def __init__(self, use_high_precision: bool = True):
        """
        Initialize timestamp converter.

        Args:
            use_high_precision: Use time.perf_counter() for higher precision (default: True)
        """
        self._offset_us = None  # Offset in microseconds
        self._offset_calculated = False
        self._lock = threading.Lock()
        self._use_high_precision = use_high_precision

        # High precision timer
        if use_high_precision:
            self._timer = HighPrecisionTimer()
        else:
            self._timer = None

        # Statistics for offset stability
        self._offset_samples = []
        self._max_samples = 100

    def calibrate(self, leap_timestamp_us: int) -> float:
        """
        Calibrate the converter using a Leap timestamp.
        Should be called with the first received Leap frame.

        Args:
            leap_timestamp_us: Leap Motion timestamp in microseconds

        Returns:
            Calculated system time for this Leap timestamp
        """
        with self._lock:
            # Use high precision timer if available
            if self._timer:
                system_time = self._timer.now()
            else:
                system_time = time.time()

            system_time_us = int(system_time * 1_000_000)

            # Calculate offset: system_time_us - leap_timestamp_us
            offset = system_time_us - leap_timestamp_us

            if not self._offset_calculated:
                # First calibration
                self._offset_us = offset
                self._offset_calculated = True
                print(f"Timestamp sync: Initial calibration")
                print(f"  Leap timestamp: {leap_timestamp_us} μs")
                print(f"  System time: {system_time_us} μs ({system_time:.6f} s)")
                print(f"  Offset: {offset} μs ({offset/1_000_000:.6f} s)")
            else:
                # Update with moving average
                self._offset_samples.append(offset)
                if len(self._offset_samples) > self._max_samples:
                    self._offset_samples.pop(0)

                # Use median for robustness against outliers
                sorted_samples = sorted(self._offset_samples)
                median_offset = sorted_samples[len(sorted_samples) // 2]

                # Detect drift
                drift = abs(median_offset - self._offset_us)
                if drift > 1000:  # >1ms drift
                    print(f"Warning: Clock drift detected: {drift/1000:.3f} ms")

                self._offset_us = median_offset

            return self.leap_to_system(leap_timestamp_us)

    def leap_to_system(self, leap_timestamp_us: int) -> float:
        """
        Convert Leap Motion timestamp to system time.

        Args:
            leap_timestamp_us: Leap Motion timestamp in microseconds

        Returns:
            System time in seconds (compatible with time.time())
        """
        if not self._offset_calculated:
            raise RuntimeError("Converter not calibrated. Call calibrate() first.")

        with self._lock:
            system_time_us = leap_timestamp_us + self._offset_us
            return system_time_us / 1_000_000

    def system_to_leap(self, system_time: float) -> int:
        """
        Convert system time to Leap Motion timestamp.

        Args:
            system_time: System time in seconds (from time.time())

        Returns:
            Leap Motion timestamp in microseconds
        """
        if not self._offset_calculated:
            raise RuntimeError("Converter not calibrated. Call calibrate() first.")

        with self._lock:
            system_time_us = int(system_time * 1_000_000)
            return system_time_us - self._offset_us

    def get_offset_ms(self) -> Optional[float]:
        """
        Get current offset in milliseconds.

        Returns:
            Offset in milliseconds, or None if not calibrated
        """
        with self._lock:
            if self._offset_us is None:
                return None
            return self._offset_us / 1000

    def is_calibrated(self) -> bool:
        """Check if converter has been calibrated."""
        with self._lock:
            return self._offset_calculated

    def get_stats(self) -> dict:
        """
        Get synchronization statistics.

        Returns:
            Dictionary with offset statistics
        """
        with self._lock:
            if not self._offset_calculated:
                return {'calibrated': False}

            stats = {
                'calibrated': True,
                'offset_us': self._offset_us,
                'offset_ms': self._offset_us / 1000,
                'offset_s': self._offset_us / 1_000_000,
                'num_samples': len(self._offset_samples)
            }

            if len(self._offset_samples) > 1:
                samples = sorted(self._offset_samples)
                stats['min_offset_us'] = samples[0]
                stats['max_offset_us'] = samples[-1]
                stats['range_us'] = samples[-1] - samples[0]
                stats['range_ms'] = stats['range_us'] / 1000

            return stats


class DualTimestampRecorder:
    """
    Records events with both Leap and system timestamps for post-hoc synchronization.
    Useful when you want to analyze synchronization quality later.
    """

    def __init__(self):
        self.events = []
        self._lock = threading.Lock()

    def record_leap_event(self, leap_timestamp_us: int, event_type: str, **kwargs):
        """Record an event with Leap timestamp."""
        with self._lock:
            self.events.append({
                'type': event_type,
                'leap_timestamp_us': leap_timestamp_us,
                'system_time': time.time(),
                **kwargs
            })

    def record_system_event(self, event_type: str, **kwargs):
        """Record an event with system timestamp."""
        with self._lock:
            self.events.append({
                'type': event_type,
                'system_time': time.time(),
                'leap_timestamp_us': None,
                **kwargs
            })

    def estimate_offset(self) -> Tuple[float, float]:
        """
        Estimate offset between Leap and system time from recorded events.

        Returns:
            Tuple of (offset_us, uncertainty_us)
        """
        with self._lock:
            # Find events with both timestamps
            dual_events = [e for e in self.events if e.get('leap_timestamp_us') is not None]

            if not dual_events:
                raise ValueError("No events with both timestamps")

            offsets = []
            for event in dual_events:
                system_us = int(event['system_time'] * 1_000_000)
                leap_us = event['leap_timestamp_us']
                offset = system_us - leap_us
                offsets.append(offset)

            # Use median for robustness
            offsets_sorted = sorted(offsets)
            median_offset = offsets_sorted[len(offsets_sorted) // 2]

            # Estimate uncertainty as range
            uncertainty = (offsets_sorted[-1] - offsets_sorted[0]) / 2

            return median_offset, uncertainty

    def clear(self):
        """Clear all recorded events."""
        with self._lock:
            self.events = []


# Convenience function for simple use cases
def create_timestamp_converter() -> TimestampConverter:
    """
    Create a new TimestampConverter instance.

    Returns:
        TimestampConverter instance
    """
    return TimestampConverter()


# Example usage
if __name__ == "__main__":
    # Simulate Leap Motion timestamps
    print("=== Timestamp Synchronization Demo ===\n")

    converter = TimestampConverter()

    # Simulate first Leap frame (assume device started 10 seconds ago)
    fake_leap_start = 0
    fake_system_start = time.time() - 10.0

    # First frame at t=0 of Leap device
    leap_ts_1 = 0
    print("Frame 1:")
    sys_time_1 = converter.calibrate(leap_ts_1)
    print(f"  Converted system time: {sys_time_1:.6f}\n")

    # Second frame at t=0.01s
    leap_ts_2 = 10_000  # 10ms in microseconds
    print("Frame 2 (10ms later):")
    sys_time_2 = converter.leap_to_system(leap_ts_2)
    print(f"  Leap timestamp: {leap_ts_2} μs")
    print(f"  Converted system time: {sys_time_2:.6f}")
    print(f"  Delta: {(sys_time_2 - sys_time_1)*1000:.3f} ms\n")

    # Reverse conversion
    print("Reverse conversion:")
    current_sys_time = time.time()
    leap_equivalent = converter.system_to_leap(current_sys_time)
    print(f"  Current system time: {current_sys_time:.6f}")
    print(f"  Leap equivalent: {leap_equivalent} μs\n")

    # Statistics
    print("Statistics:")
    stats = converter.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
