# ptv-aquatron2019

## video processing workflow

- decimate video files into images with ffmpeg. E.g.:
`ffmpeg -i pi71_1554994358.h264 -qscale:v 2 C:\Projects\Aquatron2019_April\data\interim\April11\images\pi71_1554994358_%05d.jpg -hide_banner`
- select range of images with high cork density (usually ~2000 images) for processing, and put in 'select' images folder
- In `float-tracking-ptv.ipynb`, update video filename and run entire notebook. Update: run `float-tracking-ptv.py` -- greatly trimmed down.
- run `plot_trajectories_rotated.py` to produce plots of velocity, etc.

### Aside: calibration/ginput steps (to be done first)
- extract images from high light videos from each picam, where the vectrino mounting bracket can be seen. Use one of each of these images in `extract_jetangle.py` to determine the angle of the jet relative to the camera frame.
- extract images from the last videos logged -- containing the calibration broomstick (135 cm length). Run `extract_scaling.py` to come up with a scale factor (m/pixel).

## Import conda environment `environment.yml` if starting fresh