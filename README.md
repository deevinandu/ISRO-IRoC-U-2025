# ISRO IRoC-U 2025 - Autonomous UAV

**Finalist Project for the ISRO Robotics Challenge - U 2025**

This project implements an autonomous UAV system capable of navigating GPS-denied environments using Visual Inertial Odometry (VIO) and Simultaneous Localization and Mapping (SLAM). Designed for high-precision operations, the system leverages sensor fusion and onboard edge computing to detect safe terrain and perform autonomous landings.

## Key Features

*   **GPS-Denied Navigation**: Utilizes a custom VIO-SLAM pipeline to estimate position and velocity without GPS.
*   **Onboard Vision Processing**: Offloads heavy computer vision tasks (feature tracking, depth estimation, edge detection) to the OAK-D camera's Myriad X VPU, ensuring real-time performance on low-power companion computers.
*   **Sensor Fusion**: Fuses optical flow data with high-frequency IMU gyroscope readings for robust state estimation.
*   **Autonomous Safe Landing**: Analyzes terrain in real-time to identify flat, safe landing zones and executes precision landing sequences.
*   **Mission Planning**: Performs node-based surveys and autonomous decision-making for payload delivery or exploration tasks.

## Hardware Requirements

*   **Vision Sensor**: Luxonis OAK-D or OAK-D Lite (DepthAI compatible).
*   **Companion Computer**: Raspberry Pi 4, Nvidia Jetson Nano, or similar.
*   **Flight Controller**: MAVLink-compatible controller (e.g., Pixhawk, Cube) running ArduPilot or PX4.
*   **Drone Platform**: Quadcopter frame with appropriate propulsion system.

## Software Dependencies

The system is built on Python 3 and relies on the following key libraries:

*   `depthai`: For interfacing with the OAK-D camera and building the vision pipeline.
*   `opencv-python`: For image processing and computer vision algorithms.
*   `pymavlink`: For MAVLink communication with the flight controller.
*   `numpy`: For numerical operations and matrix transformations.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ISRO-IRoC-U-2025.git
    cd ISRO-IRoC-U-2025
    ```

2.  Install dependencies:
    ```bash
    pip install depthai opencv-python pymavlink numpy
    ```

## Configuration

Configuration parameters are located in `config.py`. Key settings include:

*   **MAVLink**: `MAVLINK_DEVICE` (e.g., `/dev/ttyACM0`) and `MAVLINK_BAUD`.
*   **Vision**: `IMG_WIDTH`, `IMG_HEIGHT`, `HFOV_DEG`.
*   **Mission**: `TAKEOFF_ALTITUDE_M`, `SURVEY_SPEED_MPS`, `MAX_SAFE_SPOTS_TO_LAND`.

## Usage

1.  Ensure the flight controller is connected via USB/UART and the OAK-D camera is plugged in.
2.  Power on the drone and companion computer.
3.  Run the main application:
    ```bash
    python main.py
    ```
4.  **Operation**:
    *   The system will initialize the vision pipeline and wait for a heartbeat from the flight controller.
    *   **Arming**: The mission is triggered via RC Channel 10 (PWM > 1900).
    *   The drone will take off, perform the survey pattern, scan for landing spots, and land autonomously upon detection.

## Project Structure

*   `main.py`: Entry point. Manages threads for vision and navigation.
*   `algorithms.py`: Core logic for VIO, SLAM backend, and Safe Landing Detection.
*   `navigation.py`: Mission control state machine (Takeoff, Survey, Land, Return).
*   `mavlink_manager.py`: Handles MAVLink communication (Telemetry, Commands).
*   `config.py`: Central configuration file.
*   `frame_saver.py` & `frames.py`: Utilities for debugging and recording.

## Safety & Disclaimer

This software controls physical hardware. Always conduct tests in a safe, controlled environment (e.g., simulation or netted area) before outdoor field trials. Ensure you have a manual override (RC transmitter) available at all times.
