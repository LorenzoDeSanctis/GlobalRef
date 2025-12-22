# GlobalRef

**GlobalRef** is a Python framework for temporal synchronization, rigid alignment, and hand–eye calibration between heterogeneous motion capture systems, with a specific focus on combining marker-based tracking and video-based pose estimation.

The pipeline is intended for **research-grade validation**, with an emphasis on reproducibility when analyzing temporal synchronization, spatial alignment, and accuracy. Validation metrics follow the formulation proposed by Shah et al., allowing direct and transparent comparison across sensing modalities.

---

## Project Overview

This repository implements a complete workflow to:

- Load synchronized or unsynchronized pose trajectories from different sensing modalities
- Estimate temporal offsets using invariant rotational motion
- Refine synchronization via continuous-time interpolation (ScLERP)
- Estimate rigid transformations using robot–world / hand–eye calibration
- Quantify spatial consistency using:
  - Geodesic rotation and translation errors
  - Orientation and position accuracy metrics
- Analyze the sensitivity of validation metrics with respect to temporal offset

---

## License

### Code

The source code in this repository is released under the **MIT License**.

### Dataset

The dataset is released under the  
**Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)**.
