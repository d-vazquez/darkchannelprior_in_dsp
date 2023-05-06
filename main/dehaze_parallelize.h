
#ifndef DEHAZE_PARALLEL_H
#define DEHAZE_PARALLEL_H

#include "opencv_interface.h"
#include <iostream>
#include <vector>
 
void         parallel_DarkChannel(cv::Mat &src_T, cv::Mat &src_B, int sz, cv::Mat &dst_T, cv::Mat &dst_B);
cv::Scalar   parallel_AtmLight(cv::Mat &src_T, cv::Mat &src_B, cv::Mat &dark_T, cv::Mat &dark_B);
void         parallel_TransmissionEstimate(cv::Mat &src_T, cv::Mat &src_B, cv::Scalar A, int sz, cv::Mat &dst_T, cv::Mat &dst_B);
void         parallel_TransmissionRefine(cv::Mat &src_T, cv::Mat &src_B, cv::Mat &dst_T, cv::Mat &dst_B);
void         parallel_Recover(cv::Mat &src_T, cv::Mat &src_B, cv::Mat &te_T, cv::Mat &te_B, cv::Mat &dst_T, cv::Mat &dst_B, cv::Scalar A,  int tx);

extern long stop_darkc, start_darkc, stop_atml, start_atml, stop_tranEst, start_tranEst;
extern long stop_tranRef, start_tranRef, stop_recover, start_recover;
extern long darkc_time, atml_time, transEst_time, transRef_time, recover_time, dehaze_time;  

#endif // DEHAZE_PARALLEL_H
