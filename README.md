# Object detection using color segmentation

| Build status | [![Build Status](https://github.com/ctu-mrs/object_detect/workflows/Noetic/badge.svg)](https://github.com/ctu-mrs/object_detect/actions) |
|--------------|------------------------------------------------------------------------------------------------------------------------------------------|

## Description

This repository contains the `object_detect` package, which is developed at the MRS group for detection and position estimation of round objects with consistent color, such as the ones that were used as targets for the [MBZIRC 2020 Challenge 1](http://mbzirc.com/challenge/2020).
It contains a detector that takes RGB images as an input, detects the targets using color segmentation, and estimates positions of the targets relative to the camera based on their known physical dimensions and their apparent sizes in the image.
Optionally, the detector can also use a depth image (such as that from a [Realsense D435](https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d435.html) camera) to improve the position estimation of the target.
The algorithm is implemented using LUT and optional HW acceleration using OpenCL and is described in [[1]](#references).
<p align="center">
  <img src="config/obd_balloon.png" />
</p>

## Requirements

This repository contains a ROS package and therefore requires [ROS](https://www.ros.org/) to be installed (tested with ROS Noetic on Ubuntu 20.04).

The package requires the [`mrs_lib`](https://github.com/ctu-mrs/mrs_lib) library, which can be installed separately or together with the [MRS UAV System](https://github.com/ctu-mrs/mrs_uav_system) [[2]](#references).

If you want to utilize hardware acceleration using OpenCL, you need to have OpenCL installed (should be straight-forward using the `apt` package manager).

## Usage

To use this program, you have to specify the set of valid color (either in HSV or L*a*b) and the physical dimensions of the target.
An example is provided in the [`Red.yaml` config file](config/balls/Red.yaml).
The format of the config file is as follows:

```yaml
# compulsory parameter namespace
ball:
  # this name can be anything, but it should describe the color being segmented
  segment_color_name: Green
  # this parameter specifies the physical diameter of the object for distance estimation
  # from its apparent size in the image
  physical_diameter: 0.65
  # which method to use for specification of the set of valid color
  # possible values are: 'HSV', 'LAB', 'HS_LUT', 'AB_LUT'
  binarization_method_name: HSV
  # these values will be used if the 'HSV' binarization method is selected
  hsv: {hue_center: 52.654596807119056, hue_range: 22.608414373005395, sat_center: 99.92753623188406,
    sat_range: 160.25069282432074, val_center: 117.91439449766642, val_range: 925.6251866270163}
  # these values will be used if the 'LAB' binarization method is selected
  lab: {a_center: 106.46094325718497, a_range: 3.75860395761193, b_center: 149.30987472365513,
    b_range: 11.129277064718911, l_center: 117.29145173176124, l_range: 923.0698135303048}
  # these values will be used if the 'HS_LUT' or 'AB_LUT' binarization method is selected
  lut: {data: {...}, subsampling: {x: 1, y: 1, z: 1}}
```

There are four methods of binarizations that algorithm can utilize:
 - `HSV` - uses the [Hue, Saturation, Value model](https://en.wikipedia.org/wiki/HSL_and_HSV) of the color space. Valid colors are specified as ranges in this color space which are used to threshold the input image.
 - `LAB` - uses the [L*a*b* (also reffered to as CIELAB) model](https://en.wikipedia.org/wiki/CIELAB_color_space) of the color space. Valid colors are specified as ranges in this color space which are used to threshold the input image.
 - `HS_LUT` - uses a Look-Up Table (LUT) from RGB to 0/1, where 0 indicates invalid color and 1 indicates valid color. The fastest method.
 - `AB_LUT` - same as `HS_LUT` (these two are separate for historical reasons).

For selecting the sets of valid colors (parameters of the above mentioned methods), it is recommended to use our dedicated semi-automatic color selection tool [Color Picker](https://github.com/ctu-mrs/color_picker), which provides live color range selection and LUT creation (using a camera topic from ROS).
The tool is integrated with this package, allowing on-the-fly tuning.

## References
If you use this work, please consider citing [1].

 * [1] Y. Stasinchuk, M. Vrba, M. Petrlík, T. Báča, V. Spurný, D. Heřt, D. Žaitlík, T. Nascimento and M. Saska, **"A Multi-UAV System for Detection and Elimination of Multiple Targets,"** accepted to ICRA 2021.
 * [2]: T. Báča, M. Petrlík, M. Vrba, V. Spurný, R. Pěnička, D. Heřt and M. Saska, **"The MRS UAV System: Pushing the Frontiers of Reproducible Research, Real-world Deployment, and Education with Autonomous Unmanned Aerial Vehicles"**, eprint arXiv: 2008.08050, accepted to JINT 2021 (https://arxiv.org/abs/2008.08050).

## Authors & Contact
-----------------
```
Matouš Vrba  <vrbamato@fel.cvut.cz> 
Yurii Stasinchuk <stasiyur@fel.cvut.cz>
Multi-Robot Systems mrs.felk.cvut.cz
Faculty of electrical engineering,
Czech Technical University in Prague
```
