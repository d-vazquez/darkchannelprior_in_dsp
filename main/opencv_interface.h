

#ifndef OPENCV_INTERFACE_H
#define OPENCV_INTERFACE_H

/**
* @file opencv_interface.h
* @brief Container for OpenCV headers, between ESP32 and OpenCV there is a symbol overlap, this symbol needs to be
* undefined and defined again, this file needs to be included in every place that OpenCV is required
* @author Dario Vazquez
*/

#undef EPS
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#define EPS 192

#endif // OPENCV_INTERFACE_H
