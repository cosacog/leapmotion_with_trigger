# Recording Analysis Tools

This directory contains utility scripts for analyzing and visualizing recorded Leap Motion data.

## Tools

### `analyze_recording.py`

Analyzes HDF5 recording files to detect frame drops, timing issues, and generate statistics.

**Usage:**
```bash
python tools/analyze_recording.py data/leap_recording_YYYYMMDD_HHMMSS.h5
```

**Features:**
- Recording metadata display
- Frame rate statistics (average fps, interval std dev)
- Frame drop detection from timestamp gaps
- Task status statistics
- Hand detection rate analysis

**Output Example:**
```
=== Recording Metadata ===
Total frames recorded: 60000
Frames dropped (during recording): 15

=== Frame Rate Analysis ===
Average FPS: 89.95
...

=== Frame Drop Detection ===
Suspected drop events: 5
Total estimated frames lost: 12
```

### `visualize_recording.py`

Visualizes recorded hand tracking data with OpenCV playback.

**Usage:**
```bash
python tools/visualize_recording.py data/leap_recording_YYYYMMDD_HHMMSS.h5
```

**Features:**
- Playback of recorded hand tracking data
- Visual representation of hand skeleton
- Task status and trigger status display
- Frame-by-frame navigation

**Controls:**
- `SPACE`: Pause/Resume playback
- `q` or `ESC` or `Ctrl+C`: Exit
- Arrow keys: Frame navigation (when paused)

## Requirements

Both tools require the same dependencies as the main recording application:
- numpy
- h5py
- opencv-python (for visualize_recording.py)

## Related Documentation

See `docs/RECORDING_IMPROVEMENTS.md` for technical details about frame drop prevention and detection strategies used in the recording system.
