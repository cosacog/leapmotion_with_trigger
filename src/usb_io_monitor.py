# -*- coding: utf-8 -*-
"""
USB-IO Event-Driven Monitor
Thread-based non-blocking pulse detection with callback support.
"""

import hid
import time
import threading
from typing import Callable, Optional, Dict, Any


class USBIOMonitor:
    """
    Event-driven USB-IO monitor with thread-based polling.
    Detects rising/falling edges and triggers callbacks.
    """

    # USB-IO Configuration
    VENDOR_ID = 0x1352
    PRODUCT_ID = 0x0121
    CMD_READ_SEND = 0x20

    def __init__(self,
                 pin_mask: int = 0x01,  # J2-0 pin
                 poll_interval: float = 0.0001,  # 100μs polling
                 edge_callback: Optional[Callable] = None,
                 use_high_precision: bool = True):
        """
        Initialize USB-IO monitor.

        Args:
            pin_mask: Bit mask for target pin (default 0x01 for J2-0)
            poll_interval: Polling interval in seconds (default 100μs)
            edge_callback: Callback function called on edge detection
                          Signature: callback(edge_type: str, timestamp: float, pulse_width: float)
                          edge_type: 'rising' or 'falling'
            use_high_precision: Use time.perf_counter() for higher precision timing
        """
        self.pin_mask = pin_mask
        self.poll_interval = poll_interval
        self.edge_callback = edge_callback
        self._use_high_precision = use_high_precision

        # Timer function
        self._time_func = time.perf_counter if use_high_precision else time.time

        # Device handle
        self.device = None

        # Thread control
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._running = False

        # State tracking
        self._prev_state = None
        self._pulse_start_time = None
        self._pulse_count = 0

        # Statistics
        self.stats = {
            'total_pulses': 0,
            'rising_edges': 0,
            'falling_edges': 0,
            'min_pulse_width': float('inf'),
            'max_pulse_width': 0,
            'avg_pulse_width': 0,
            'total_pulse_width': 0
        }

    def open(self) -> bool:
        """
        Open USB-IO device.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.device = hid.device()
            self.device.open(self.VENDOR_ID, self.PRODUCT_ID)
            product_name = self.device.get_product_string()
            print(f"USB-IO Monitor: Connected to {product_name}")
            return True
        except Exception as e:
            print(f"USB-IO Monitor: Failed to open device: {e}")
            return False

    def close(self):
        """Close USB-IO device."""
        if self.device:
            self.device.close()
            self.device = None
            print("USB-IO Monitor: Device closed")

    def _send_command(self, cmd: int, p1: int = 0, p2: int = 0) -> list:
        """
        Send command to USB-IO device.

        Args:
            cmd: Command byte
            p1: Parameter 1
            p2: Parameter 2

        Returns:
            Received data (64 bytes)
        """
        data = [0] * 64
        data[0] = 0
        data[1] = cmd
        data[2] = 1
        data[3] = p1
        data[4] = 2
        data[5] = p2
        data[63] = 0
        self.device.write(data)
        return self.device.read(64)

    def _read_port(self) -> int:
        """
        Read port state.

        Returns:
            Port 2 state byte
        """
        rcvd = self._send_command(self.CMD_READ_SEND, 0, 0)
        p1 = rcvd[1]
        p2 = rcvd[2]
        self._send_command(self.CMD_READ_SEND, p1, p2)
        return p2

    def _monitor_loop(self):
        """Main monitoring loop running in separate thread."""
        print(f"USB-IO Monitor: Started monitoring (poll interval: {self.poll_interval*1000:.2f}ms)")

        while not self._stop_event.is_set():
            try:
                # Read current state
                p2 = self._read_port()
                current_state = (p2 & self.pin_mask) != 0  # True if high, False if low
                current_time = self._time_func()  # High precision timer

                # Edge detection
                if current_state != self._prev_state and self._prev_state is not None:
                    if not current_state:  # Falling edge (High → Low)
                        self._pulse_start_time = current_time
                        self.stats['falling_edges'] += 1

                        if self.edge_callback:
                            self.edge_callback('falling', current_time, 0.0)

                    elif current_state and self._pulse_start_time is not None:  # Rising edge (Low → High)
                        pulse_width = (current_time - self._pulse_start_time)
                        self._pulse_count += 1
                        self.stats['rising_edges'] += 1
                        self.stats['total_pulses'] += 1

                        # Update statistics
                        self.stats['min_pulse_width'] = min(self.stats['min_pulse_width'], pulse_width)
                        self.stats['max_pulse_width'] = max(self.stats['max_pulse_width'], pulse_width)
                        self.stats['total_pulse_width'] += pulse_width
                        self.stats['avg_pulse_width'] = self.stats['total_pulse_width'] / self.stats['total_pulses']

                        if self.edge_callback:
                            self.edge_callback('rising', current_time, pulse_width)

                        self._pulse_start_time = None

                self._prev_state = current_state

                # Sleep for polling interval
                time.sleep(self.poll_interval)

            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"USB-IO Monitor: Error in monitoring loop: {e}")
                break

        print("USB-IO Monitor: Monitoring stopped")

    def start(self) -> bool:
        """
        Start monitoring in background thread.

        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            print("USB-IO Monitor: Already running")
            return False

        if not self.device:
            print("USB-IO Monitor: Device not opened")
            return False

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self._running = True
        return True

    def stop(self):
        """Stop monitoring thread."""
        if not self._running:
            return

        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        self._running = False

    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    def get_stats(self) -> Dict[str, Any]:
        """
        Get monitoring statistics.

        Returns:
            Dictionary containing statistics
        """
        stats = self.stats.copy()
        if stats['min_pulse_width'] == float('inf'):
            stats['min_pulse_width'] = 0
        return stats

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_pulses': 0,
            'rising_edges': 0,
            'falling_edges': 0,
            'min_pulse_width': float('inf'),
            'max_pulse_width': 0,
            'avg_pulse_width': 0,
            'total_pulse_width': 0
        }
        self._pulse_count = 0


# Context manager support
class USBIOMonitorContext:
    """Context manager for USB-IO monitor."""

    def __init__(self, **kwargs):
        self.monitor = USBIOMonitor(**kwargs)

    def __enter__(self):
        if self.monitor.open():
            self.monitor.start()
            return self.monitor
        else:
            raise RuntimeError("Failed to open USB-IO device")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop()
        self.monitor.close()
        return False
