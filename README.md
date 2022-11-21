This repository contains Code, that was used to analyse 2D-positional Data (Walkthrough.ipynb, Analysis). It takes DLC data derived from animal videos, a camera calibration based on a checkerboard video (see calibrate.py), metadata describing the ROI around the area of the experimental paradigm. Converting the positional data to the reference space and data smoothening is then performed. Step Detection and parameter calculation (angles, distances, areas between bodyparts) results in single values reflecting the median of the parameters over all steps. 

The Code to select the ROI is based on Code from @DSegebarth and @MSchellenberger and can be found under this link: https://github.com/DSegebarth/BSc_MS/blob/32c940778c648770f63776760033b404b9a00aa5/GUI_annotate_maze_corners.ipynb.

For filtering and interpolation of the tracking data Code from @DSegebarth was copied into this analysis. The original Code can be found under this link: https://github.com/Defense-Circuits-Lab/Gait_Analysis/blob/8510de88e42069259981654f7adc8315d282527d/gait3d/gaitanalysis_with_top_cam.py.

For markerless position tracking of the animals, we used DeepLabCut (v2.1.8) with a resnet-50 model.

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
