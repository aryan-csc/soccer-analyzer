# Olympic Weightlifting Biomechanics Pipeline

A Python-based computer vision pipeline that analyzes an Olympic Snatch lift. Built to provide technical feedback from standard 2D video footage without the need for physical sensors.

The system tracks both the athlete and barbell, processes the raw coordinates through a custom physics engine, and renders an analytical overlay back onto the original video. Coaches and athletes can pinpoint weakness in their technique with this tool.

## Demo

<video autoplay loop muted playsinline width="100%">
  <source src="assets/lift4_analyzed.mp4" type="video/mp4">
</video>

## Key Features

* **Hybrid Computer Vision Tracking:** Combines MediaPipe's pose estimation for human joint tracking with OpenCV's CSRT tracker for barbell tracking.
* **Algorithmic Rep Detection:** Automatically isolates the active phases of the lift (the pull, turnover, and catch) by analyzing the barbell's vertical velocity and position.
* **Dynamic Ideal Path Generation:** Uses SciPy Cubic Splines to mathematically model an idealized "S-Curve" barbell trajectory. The curve is dynamically generated per video based on the specific lifter's bodily dimensions.
* **Kinematic Physics Engine:** Converts native top-down image coordinates into a standard Cartesian plane to accurately calculate true upward velocity, interior joint angles (via arctangent geometry), and applies Savitzky-Golay filtering to smooth tracking jitter.
* **Rendering:** Renders a data overlay directly onto the video. Features include a real-time velocity graphing, active joint angle arcs, and a color-coded barbell trail that turns red if the lifter deviates from the ideal vertical corridor.

## Tech Stack

* **Language:** Python
* **Infra:** Snakemake
* **Computer Vision:** OpenCV, MediaPipe
* **Data & Math:** Pandas, NumPy, SciPy (Interpolation & Signal Filtering)

## Architecture

The program works like a data pipeline. There are a collection of scripts which call utility classes to create intermediate outputs leading to the final render. I used Snakemake which acts like Make but for Python. It means that as I developed and changed the code only the updated scripts would need to be re-run. This was important since some stages (like extarcting the landmarks) where compute intensive and it was impractical to reload those each time.

I used three primary utility files for the core logic:

1. **`Extractor.py`**
   Handles the ingestion of the video file. Pauses on the first frame to allow the user to draw a bounding box around the barbell plate, initializes the CSRT tracker, and loops through the video to extract normalized spatial coordinates for the barbell and key body joints. Outputs raw data to a CSV.

2. **`Physics_utils.py`**
   The core mathematical engine. Contains three classes:
   * `KinematicsProcessor`: Inverts the Y-axis for standard physics calculations, smooths coordinate data, and calculates velocities and interior joint angles.
   * `RepDetector`: Scans the velocity data to find the exact frames of the explosive pull and the turnover, isolating the lift from the setup and drop.
   * `IdealPathCalculator`: Generates the target S-Curve trajectory using natural boundary cubic splines mapped to the lifter's proportions.

3. **`VideoRenderer.py`**
   The visualization layer. Vectorizes all coordinate data via Pandas before the drawing loop to ensure fast rendering. Overlays the dynamic velocity graph, the ideal path corridor, the neon-style skeletal tracking, and the reactive barbell trail.

## Usage

*Ensure all dependencies are installed via `pip install -r requirements.txt`*

1. Put all videos in data/input directory. Ensure they are side profile videos with a stable camera for best results.
   `python main.py --extract input.mp4`

2. Run the snakemake utility to start the pipeline.
   `snakemake --cores all`
   *Note: A window will appear on frame 1 of each input. Draw a box around the barbell plate and press ENTER to begin tracking.*

3. Renders appear in data/output directory. Additional videos can be added to input folder and only new/updated files will be processed.
