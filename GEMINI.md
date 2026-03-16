# Gemini Code Guide: weightlifting-analyzer

This document provides a comprehensive overview of the `weightlifting-analyzer` project, designed to facilitate understanding and collaboration for Gemini.

## Project Overview

The `weightlifting-analyzer` is a Python-based tool that uses computer vision to analyze the biomechanics of the snatch Olympic weightlifting movement. It takes a side-profile video of a lift as input and generates an annotated video overlaying the actual bar path against a calculated ideal bar path, along with key performance metrics.

The project is structured as a [Snakemake](https://snakemake.readthedocs.io/en/stable/) pipeline, which automates the entire analysis workflow from raw video to final report.

### Core Technologies

*   **Workflow Management:** Snakemake
*   **Computer Vision:** OpenCV, MediaPipe
*   **Data Manipulation:** Pandas, NumPy, SciPy
*   **Visualization:** Matplotlib, Seaborn

### Architecture

The analysis pipeline consists of the following stages:

1.  **Pose Extraction (`extract_pose`):** MediaPipe is used to extract 2D pose landmarks (wrists, shoulders, hips, etc.) from each frame of the input video. The raw landmark coordinates are saved to a CSV file.
2.  **Kinematic Processing (`process_kinematics`):** The raw landmark data is cleaned and processed. This involves:
    *   Calculating a proxy for the barbell's position.
    *   Smoothing the coordinate data to reduce jitter.
    *   Calculating key kinematics like vertical velocity and hip angle.
3.  **Repetition Detection (`detect_reps`):** The single "snatch" repetition is isolated by analyzing the barbell's vertical velocity and position to find the start, peak, and end of the lift.
4.  **Ideal Path Calculation (`calculate_ideal`):** A mathematically ideal "S-shaped" bar path is generated based on biomechanical principles and the lifter's specific movement range.
5.  **Video Rendering (`render_overlay`):** A new video is created, overlaying the following information onto the original footage:
    *   A simplified skeleton of the lifter.
    *   The actual bar path (a trailing red line).
    *   The ideal bar path (a static green line).
    *   Key metrics like vertical velocity and hip extension angle.

## Building and Running

The project is managed by the `Snakefile` and can be run using the `snakemake` command-line tool.

### Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Place Input Video:**
    Add your input video file (e.g., `my_lift.mp4`) into the `data/raw/` directory.

### Running the Analysis

To run the full pipeline for all videos in `data/raw/`:

```bash
snakemake --cores 1
```

This command will execute the entire workflow defined in the `Snakefile`, generating the final analyzed video in the `results/videos/` directory. For example, an input file named `data/raw/my_lift.mp4` will produce `results/videos/my_lift_analyzed.mp4`.

### Cleaning Up

To remove all generated files and start fresh:

```bash
snakemake --delete-all-output
```

## Development Conventions

*   **Workflow Definition:** The main analysis pipeline is defined in the `Snakefile`. New processing steps should be added as new `rule`s in this file.
*   **Core Logic:** The core scientific and computer vision logic is implemented in Python classes within the `src/` directory.
*   **Scripts:** The `scripts/` directory contains simple command-line wrappers that instantiate and run the classes from `src/`. These scripts are called by the rules in the `Snakefile`.
*   **Data Flow:**
    *   `data/raw/`: Input videos.
    *   `data/interim/`: Intermediate data files (e.g., raw landmarks).
    *   `data/processed/`: Cleaned and processed data files.
    *   `results/`: Final output files (annotated videos, reports).
*   **Python:** The project uses standard Python 3. Type hints are used in the source code.
