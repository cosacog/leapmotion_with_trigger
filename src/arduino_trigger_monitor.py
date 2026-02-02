# -*- coding: utf-8 -*-
"""
Arduino-based TTL Trigger Monitor
High-precision TTL pulse detection using Arduino hardware interrupt.
"""

import serial
import serial.tools.list_ports
import time
import threading
from typing import Callable, Optional, Dict, Any, List


class ArduinoTriggerMonitor:
    """
    Monitor TTL triggers via Arduino with hardware interrupt precision.

    The Arduino detects TTL pulses using hardware interrupts and records
    timestamps with micros(). This class handles communication and provides
    callbacks for trigger events.

    Timing synchronization:
        - Call sync_start() at recording start
        - Call sync_end() at recording end
        - Use convert_to_pc_time() to convert Arduino timestamps to PC time
    """

    BAUD_RATE = 115200
    ARDUINO_ID_STRING = "TTL_TRIGGER_READY"

    def __init__(self,
                 port: Optional[str] = None,
                 trigger_callback: Optional[Callable] = None,
                 auto_connect: bool = True):
        """
        Initialize Arduino trigger monitor.

        Args:
            port: COM port (e.g., 'COM9'). If None, auto-detect.
            trigger_callback: Callback function called on trigger detection
                             Signature: callback(arduino_time_us: int, pc_time: float)
            auto_connect: Automatically connect on initialization
        """
        self.port = port
        self.trigger_callback = trigger_callback
        self.serial: Optional[serial.Serial] = None

        # Thread control
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Sync data
        self.sync_start_arduino_us: Optional[int] = None
        self.sync_start_pc_time: Optional[float] = None
        self.sync_end_arduino_us: Optional[int] = None
        self.sync_end_pc_time: Optional[float] = None

        # Trigger data (raw Arduino timestamps in microseconds)
        self.trigger_times_us: List[int] = []
        self.trigger_times_pc: List[float] = []  # PC receive times

        # Statistics
        self.stats = {
            'total_triggers': 0,
            'sync_drift_us': 0,
        }

        if auto_connect:
            self.connect()

    @staticmethod
    def find_arduino_port() -> Optional[str]:
        """
        Auto-detect Arduino port.

        Returns:
            COM port string or None if not found
        """
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # Check for common Arduino identifiers
            desc_lower = port.description.lower()
            if any(x in desc_lower for x in ['arduino', 'ch340', 'usb serial', 'usb-serial']):
                print(f"[Arduino] Found potential Arduino at {port.device}: {port.description}")
                return port.device
        return None

    def connect(self, port: Optional[str] = None) -> bool:
        """
        Connect to Arduino.

        Args:
            port: COM port. If None, use self.port or auto-detect.

        Returns:
            True if connected successfully
        """
        if port:
            self.port = port
        elif not self.port:
            self.port = self.find_arduino_port()
            if not self.port:
                print("[Arduino] No Arduino found. Please specify port manually.")
                return False

        try:
            self.serial = serial.Serial(
                self.port,
                self.BAUD_RATE,
                timeout=1.0
            )
            # Wait for Arduino to reset
            time.sleep(2.0)

            # Clear buffer and check for ready message
            self.serial.reset_input_buffer()

            # Send ping to verify connection
            self.serial.write(b'P')
            response = self._read_line(timeout=2.0)

            if response == "PONG":
                print(f"[Arduino] Connected to {self.port}")
                return True
            elif self.ARDUINO_ID_STRING in response:
                print(f"[Arduino] Connected to {self.port}")
                return True
            else:
                print(f"[Arduino] Unexpected response: {response}")
                self.serial.close()
                self.serial = None
                return False

        except Exception as e:
            print(f"[Arduino] Connection failed: {e}")
            if self.serial:
                self.serial.close()
                self.serial = None
            return False

    def disconnect(self):
        """Disconnect from Arduino."""
        self.stop()
        if self.serial:
            self.serial.close()
            self.serial = None
            print("[Arduino] Disconnected")

    def _read_line(self, timeout: float = 1.0) -> str:
        """Read a line from serial with timeout."""
        if not self.serial:
            return ""

        old_timeout = self.serial.timeout
        self.serial.timeout = timeout
        try:
            line = self.serial.readline().decode('utf-8').strip()
            return line
        except:
            return ""
        finally:
            self.serial.timeout = old_timeout

    def _send_command(self, cmd: str) -> str:
        """Send command and read response."""
        if not self.serial:
            return ""
        self.serial.write(cmd.encode('utf-8'))
        return self._read_line()

    def reset(self):
        """Reset Arduino trigger buffer."""
        if not self.serial:
            return

        self._send_command('R')
        self.trigger_times_us = []
        self.trigger_times_pc = []
        self.sync_start_arduino_us = None
        self.sync_start_pc_time = None
        self.sync_end_arduino_us = None
        self.sync_end_pc_time = None
        self.stats['total_triggers'] = 0
        print("[Arduino] Reset complete")

    def sync_start(self) -> bool:
        """
        Perform sync at recording start.

        Returns:
            True if sync successful
        """
        if not self.serial:
            return False

        pc_time = time.perf_counter()
        self.serial.write(b'S')
        response = self._read_line(timeout=2.0)

        if response.startswith("S,"):
            try:
                arduino_us = int(response.split(",")[1])
                self.sync_start_arduino_us = arduino_us
                self.sync_start_pc_time = pc_time
                print(f"[Arduino] Sync start: Arduino={arduino_us}us, PC={pc_time:.6f}s")
                return True
            except:
                pass

        print(f"[Arduino] Sync start failed: {response}")
        return False

    def sync_end(self) -> bool:
        """
        Perform sync at recording end.

        Returns:
            True if sync successful
        """
        if not self.serial:
            return False

        pc_time = time.perf_counter()
        self.serial.write(b'E')
        response = self._read_line(timeout=2.0)

        if response.startswith("E,"):
            try:
                arduino_us = int(response.split(",")[1])
                self.sync_end_arduino_us = arduino_us
                self.sync_end_pc_time = pc_time

                # Calculate drift
                if self.sync_start_arduino_us is not None and self.sync_start_pc_time is not None:
                    arduino_elapsed_us = arduino_us - self.sync_start_arduino_us
                    pc_elapsed_us = (pc_time - self.sync_start_pc_time) * 1_000_000
                    drift_us = arduino_elapsed_us - pc_elapsed_us
                    self.stats['sync_drift_us'] = drift_us
                    print(f"[Arduino] Sync end: drift={drift_us:.1f}us over {pc_time - self.sync_start_pc_time:.1f}s")

                # Read remaining trigger data (COUNT, T lines, END)
                self._read_end_data()
                return True
            except:
                pass

        print(f"[Arduino] Sync end failed: {response}")
        return False

    def _read_end_data(self):
        """Read trigger data sent after sync_end."""
        while True:
            line = self._read_line(timeout=1.0)
            if not line or line == "END":
                break
            if line.startswith("T,"):
                try:
                    arduino_us = int(line.split(",")[1])
                    if arduino_us not in self.trigger_times_us:
                        self.trigger_times_us.append(arduino_us)
                except:
                    pass
            elif line.startswith("COUNT,"):
                try:
                    count = int(line.split(",")[1])
                    self.stats['total_triggers'] = count
                except:
                    pass

    def convert_to_pc_time(self, arduino_us: int) -> float:
        """
        Convert Arduino timestamp to PC time using linear interpolation.

        Args:
            arduino_us: Arduino micros() value

        Returns:
            PC time (perf_counter seconds)
        """
        if (self.sync_start_arduino_us is None or self.sync_start_pc_time is None):
            raise ValueError("Sync not performed. Call sync_start() first.")

        # If we have end sync, use linear interpolation for drift correction
        if self.sync_end_arduino_us is not None and self.sync_end_pc_time is not None:
            arduino_total_us = self.sync_end_arduino_us - self.sync_start_arduino_us
            pc_total = self.sync_end_pc_time - self.sync_start_pc_time

            if arduino_total_us > 0:
                arduino_elapsed_us = arduino_us - self.sync_start_arduino_us
                scale = pc_total / (arduino_total_us / 1_000_000)
                return self.sync_start_pc_time + (arduino_elapsed_us / 1_000_000) * scale

        # Fallback: simple conversion without drift correction
        arduino_elapsed_us = arduino_us - self.sync_start_arduino_us
        return self.sync_start_pc_time + (arduino_elapsed_us / 1_000_000)

    def get_trigger_times_pc(self) -> List[float]:
        """
        Get all trigger times converted to PC time.

        Returns:
            List of trigger times in PC perf_counter seconds
        """
        return [self.convert_to_pc_time(t) for t in self.trigger_times_us]

    def _reader_loop(self):
        """Background thread for reading serial data."""
        print("[Arduino] Reader thread started")

        while not self._stop_event.is_set():
            try:
                if self.serial and self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8').strip()
                    pc_time = time.perf_counter()

                    if line.startswith("T,"):
                        try:
                            arduino_us = int(line.split(",")[1])
                            self.trigger_times_us.append(arduino_us)
                            self.trigger_times_pc.append(pc_time)
                            self.stats['total_triggers'] = len(self.trigger_times_us)

                            if self.trigger_callback:
                                self.trigger_callback(arduino_us, pc_time)

                        except ValueError:
                            pass

                else:
                    time.sleep(0.001)  # 1ms sleep when no data

            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"[Arduino] Reader error: {e}")
                break

        print("[Arduino] Reader thread stopped")

    def start(self) -> bool:
        """
        Start monitoring in background thread.

        Returns:
            True if started successfully
        """
        if self._running:
            print("[Arduino] Already running")
            return False

        if not self.serial:
            print("[Arduino] Not connected")
            return False

        # Reset Arduino buffer
        self.reset()

        # Perform initial sync
        if not self.sync_start():
            return False

        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._running = True
        print("[Arduino] Monitoring started")
        return True

    def stop(self):
        """Stop monitoring thread."""
        if not self._running:
            return

        # Perform end sync
        self.sync_end()

        self._stop_event.set()
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)
        self._running = False

    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        stats = self.stats.copy()
        stats['trigger_count'] = len(self.trigger_times_us)
        return stats


# Context manager support
class ArduinoTriggerMonitorContext:
    """Context manager for Arduino trigger monitor."""

    def __init__(self, **kwargs):
        self.monitor = ArduinoTriggerMonitor(**kwargs, auto_connect=False)

    def __enter__(self):
        if self.monitor.connect():
            self.monitor.start()
            return self.monitor
        else:
            raise RuntimeError("Failed to connect to Arduino")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.stop()
        self.monitor.disconnect()
        return False


# Test function
def test_arduino_monitor(port: Optional[str] = None, duration: float = 10.0):
    """
    Test Arduino trigger monitor.

    Args:
        port: COM port (auto-detect if None)
        duration: Test duration in seconds
    """
    def on_trigger(arduino_us, pc_time):
        print(f"  Trigger: Arduino={arduino_us}us, PC={pc_time:.6f}s")

    monitor = ArduinoTriggerMonitor(port=port, trigger_callback=on_trigger, auto_connect=False)

    if not monitor.connect():
        print("Failed to connect")
        return

    print(f"\nMonitoring for {duration} seconds...")
    print("Send TTL pulses to Pin 2\n")

    monitor.start()
    time.sleep(duration)
    monitor.stop()

    print(f"\n=== Results ===")
    print(f"Total triggers: {len(monitor.trigger_times_us)}")
    print(f"Sync drift: {monitor.stats['sync_drift_us']:.1f} us")

    if monitor.trigger_times_us:
        pc_times = monitor.get_trigger_times_pc()
        print(f"\nTrigger times (PC):")
        for i, t in enumerate(pc_times[:10]):
            print(f"  {i+1}: {t:.6f}s")
        if len(pc_times) > 10:
            print(f"  ... and {len(pc_times) - 10} more")

    monitor.disconnect()


if __name__ == "__main__":
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else None
    test_arduino_monitor(port=port)
