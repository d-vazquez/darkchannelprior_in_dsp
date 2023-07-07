
#ifndef DEHAZE_H
#define DEHAZE_H

#include "opencv_interface.h"
#include <iostream>
#include <vector>
 
typedef enum
{
    NO_OP = 0,
    DARK_CHANNEL_OP,
    ATMLIGHT_OP,
    TRANSMITION_ESTIMATE_OP,
    TRANSMITION_REFINE_OP,
    RECOVER_OP,
} dehaze_op;

void dehaze   (cv::Mat &src, cv::Mat &dst);
void mat_split(cv::Mat &src, cv::Mat &top, cv::Mat &bot, uint32_t overlap = 0);
void DarkChannel(cv::Mat &img, int sz,  cv::Mat &dst);
const cv::Scalar AtmLight(cv::Mat &img, cv::Mat &dark);
void TransmissionEstimate(cv::Mat &im, cv::Scalar A, int sz, cv::Mat &dst);
void Recover(cv::Mat &im, cv::Mat &t, cv::Mat &dst, cv::Scalar A, int tx);
void TransmissionRefine(cv::Mat &im, cv::Mat &et);
void Guidedfilter(cv::Mat &im_grey, cv::Mat &transmission_map, int r, float eps);



#endif // DEHAZE_H
