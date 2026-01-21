# Leap Motion Data Recorder Implementation Plan

## Goal
Create a Python script `record_handtracking.py` that records Leap Motion hand tracking data continuously with high-precision timing, non-blocking file I/O, and event marking.

## User Review Required
> [!IMPORTANT]
> **Dependencies**: The script requires `h5py` for HDF5 saving and `pynput` for keyboard input. It also attempts to use `psychopy` for timing if available, falling back to `time.perf_counter()`.
> **File Format**: Data will be saved in HDF5 (`.h5`) format for speed and efficiency.
> **Data Scope**: The script will record Palm Position, Palm Orientation, and Fingertip Positions for both hands.

## Proposed Changes

### [New Script]
#### [NEW] [record_handtracking.py](file:///y:/python/leapmotion_handtracking/record_handtracking.py)
- **Architecture**:
    - **Producer (Leap Listener)**: Runs on Leap's callback thread. Captures `frame` data, timestamps, and task status. Pushes to a `queue.Queue`.
    - **Consumer (Writer Thread)**: Runs on a separate thread. Pulls data from `Queue`. Buffers data in memory. Writes to HDF5 file every ~0.5-1.0 seconds (chunked).
    - **Main Thread**: Handles `Ctrl+C` to stop recording safely.
    - **Input Listener**: Uses `pynput` to toggle a `task_status` flag (0/1) on Space key press.
- **Timing**:
    - Uses `psychopy.core.Clock` or `time.perf_counter()` for ms precision.
    - Records both "System Time" (acquisition time) and "Leap Time" (frame timestamp).
- **Data Structure (HDF5)**:
    - `timestamps`: [N] (float64)
    - `leap_timestamps`: [N] (int64)
    - `task_status`: [N] (int8)
    - `right_hand`:
        - `valid`: [N] (bool)
        - `palm_pos`: [N, 3] (float32)
        - `palm_ori`: [N, 4] (float32)
        - `fingers`: [N, 5, 3] (float32) (Tip positions)
    - `left_hand`: (Same as right)

## Verification Plan

### Automated Tests
- None (Hardware dependent).

### Manual Verification
1.  **Dry Run**: Run the script without a Leap Motion connected (if possible, or with one).
2.  **Keyboard Test**: Press Spacebar during recording. Verify `task_status` changes in the output file.
3.  **Timing Test**: Check if timestamps are monotonically increasing and have expected intervals (~10ms for 100Hz).
4.  **File Check**: Open the generated `.h5` file using a simple script or `h5dump` (if available) to verify structure and data integrity.
5.  **Performance**: Ensure no lag in data acquisition (check for dropped frames or queue size warnings).
