# Leap Motion Hand Tracking with USB-IO Integration

Leap Motion hand tracking system integrated with USB-IO 2.0 trigger device for synchronized data recording.

## Project Overview

This project combines Leap Motion hand tracking with USB-IO 2.0 device for precise timestamp synchronization. The system records hand tracking data with external trigger signals, enabling synchronized data collection for experimental setups.

## Key Features

- **High-precision timestamp synchronization** (~100ns resolution using `time.perf_counter()`)
- **Event-driven USB-IO monitoring** with edge detection callbacks
- **Thread-safe implementation** with 4 concurrent threads:
  - Leap Motion listener thread
  - USB-IO monitor thread
  - HDF5 writer thread
  - Main visualization thread
- **Real-time visualization** with OpenCV
- **HDF5 data format** for efficient storage with chunking
- **Frame drop prevention** with queue-based buffering

## Project Structure

```
leapmotion_handtracking/
├── README.md                    # Ultraleap official README
├── README_PROJECT.md           # This file (project-specific documentation)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── record_with_trigger.py      # Main launcher script (run from project root)
│
├── src/                        # Main source code
│   ├── __init__.py
│   ├── record_with_trigger.py  # Main recording application
│   ├── usb_io_monitor.py       # USB-IO event-driven monitoring
│   └── timestamp_sync.py       # High-precision timestamp synchronization
│
├── tests/                      # Test scripts
│   ├── __init__.py
│   └── test_usb_io_monitor.py  # USB-IO monitor unit test
│
├── archive/                    # Archived development files
│   ├── README_ARCHIVE.md       # Archive documentation
│   ├── record_and_visualize.py
│   ├── record_simple.py
│   └── test_record_trigger.py
│
├── docs/                       # Technical documentation
│   ├── README_PROJECT_STRUCTURE.md  # Detailed project structure (Japanese)
│   ├── USB_IO_INTEGRATION.md        # USB-IO integration technical doc
│   └── TIMESTAMP_SYNC.md            # Timestamp synchronization details
│
├── data/                       # Recorded data (gitignored)
│   └── leap_recording_trigger_YYYYMMDD_HHMMSS.h5
│
├── examples/                   # Example scripts
│   └── visualiser.py
│
├── leapc-cffi/                 # Leap Motion C FFI bindings
└── leapc-python-api/           # Leap Motion Python API
```

## Installation

### Prerequisites

1. **Ultraleap Gemini SDK** (5.17+)
   - Download from [Ultraleap Developer Site](https://developer.leapmotion.com/tracking-software-download)
   - Install to default location or set `LEAPSDK_INSTALL_LOCATION` environment variable

2. **USB-IO 2.0 Device** (optional, for trigger functionality)
   - Driver installation required for Windows

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd leapmotion_handtracking

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install Leap Python API
pip install -e leapc-python-api
```

## Usage

### Main Recording Application

Run from the project root directory:

```bash
python record_with_trigger.py
```

**Controls:**
- **SPACE key**: Mark task status = 1 (for manual event marking)
- **'q' or ESC**: Stop recording
- **Close window**: Stop recording

**Output:**
- Data saved to `data/leap_recording_trigger_YYYYMMDD_HHMMSS.h5`
- HDF5 file contains:
  - `leap_timestamp`: Original Leap timestamps (microseconds)
  - `system_timestamp`: Synchronized system time (seconds, perf_counter base)
  - `task_status`: SPACE key state (0/1)
  - `trigger_status`: USB-IO trigger state (0/1)
  - `right/left`: Hand tracking data (palm, wrist, elbow, fingers)

### Testing USB-IO Monitor

```bash
cd tests
python test_usb_io_monitor.py
```

This will test the USB-IO device connectivity and edge detection.

## Technical Specifications

- **Leap Motion sampling rate**: 90 Hz
- **USB-IO polling interval**: 100 μs (0.0001 seconds)
- **Timestamp precision**: ~100 nanoseconds (Windows)
- **HDF5 save interval**: 0.5 seconds
- **Frame queue size**: 10,000 frames
- **USB-IO pin**: J2-0 (configurable in `src/record_with_trigger.py`)

## Architecture

### Timestamp Synchronization

All timestamps use `time.perf_counter()` as a common base:
- **Leap Motion**: Hardware timer converted to system time
- **USB-IO**: Direct perf_counter timestamps
- **Precision**: ~100ns on Windows (vs 15.6ms for `time.time()`)

### Threading Model

1. **Leap Listener Thread**: Captures hand tracking events
2. **USB-IO Monitor Thread**: Polls USB-IO device at 100μs intervals
3. **Writer Thread**: Saves buffered data to HDF5 every 0.5s
4. **Main Thread**: Handles OpenCV visualization and user input

### Thread Safety

- `LatestFrameContainer`: Uses `threading.Lock()` for safe frame sharing
- `queue.Queue`: Thread-safe buffer between Leap listener and writer
- No shared state between USB-IO monitor and other threads (callback-based)

## Development

### Running Tests

```bash
# USB-IO monitor test
cd tests
python test_usb_io_monitor.py
```

### Archived Files

Development history files are preserved in `archive/` directory. See `archive/README_ARCHIVE.md` for details.

**Note**: Archived files are for reference only. Use `src/record_with_trigger.py` for production.

## Troubleshooting

### Issue: High precision timer fails to initialize

**Symptom**: Program exits with "High precision timer initialization failed"

**Solution**: This is expected behavior. The timer is critical for USB-IO synchronization. If it fails, data integrity cannot be guaranteed.

### Issue: USB-IO device not found

**Symptom**: "Warning: Failed to open USB-IO device"

**Solution**:
- Check USB-IO device connection
- Install USB-IO drivers
- Program will continue without trigger functionality

### Issue: Leap Motion not detected

**Symptom**: No frames received, empty visualization

**Solution**:
- Ensure Ultraleap Tracking software is running
- Check device connection
- Verify Leap Motion is working with official Ultraleap Visualizer

## Documentation

- **English**:
  - `docs/USB_IO_INTEGRATION.md` - USB-IO integration details
  - `docs/TIMESTAMP_SYNC.md` - Timestamp synchronization explanation

- **Japanese**:
  - `docs/README_PROJECT_STRUCTURE.md` - 詳細なプロジェクト構造説明

## License

See LICENSE.md for details.

## Support

For issues related to:
- **Leap Motion SDK**: [Ultraleap Support](mailto:support@ultraleap.com)
- **This project**: Open an issue on GitHub

## Version History

- **v1.0.0** (2026-01-21): Initial structured release
  - Event-driven USB-IO integration
  - High-precision timestamp synchronization
  - Stable multi-threaded architecture
