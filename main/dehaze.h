
#ifndef DEHAZE_H
#define DEHAZE_H

#undef EPS
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#define EPS 192

typedef struct
{
    unsigned char *data;
    size_t        lenght;
} image_header;



void dehaze(cv::Mat &src, cv::Mat &dst);

#endif // DEHAZE_H
