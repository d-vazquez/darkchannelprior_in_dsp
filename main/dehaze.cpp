

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "freertos/event_groups.h"

#include <esp_log.h>
#include <esp_heap_caps.h>
#include <esp_task_wdt.h>

#include <vector>

#include "dehaze.h"

using namespace cv;
using namespace std;

static char TAG[] = "dehaze";

#define printf(...) ESP_LOGI(TAG, ##__VA_ARGS__)

extern void write_MAT_to_file(const char *file_dst, cv::Mat &src);

void DarkChannel(cv::Mat &img, int sz,  cv::Mat &dst);
const cv::Scalar AtmLight(cv::Mat &img, cv::Mat &dark);
void TransmissionEstimate(cv::Mat &im, cv::Scalar A, int sz, cv::Mat &dst);
void Recover(Mat &im, Mat &t, Mat &dst, Scalar A, int tx);
void TransmissionRefine(cv::Mat &im, cv::Mat &et);
void Guidedfilter(cv::Mat &im_grey, cv::Mat &transmission_map, int r, float eps);

void dehaze(cv::Mat &src, cv::Mat &dst)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("Start dehaze, Free heap: %d bytes", start_heap);
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %d bytes", uxTaskGetStackHighWaterMark(NULL));
    
    printf("");
    printf("+++ Calculating dark channel");
    Mat te;
    DarkChannel(src,15,te);
    printf("--- Returning dark...");

    // printf("Storing Dark channel");
    // write_MAT_to_file("/sdcard/darkc.bin", te);

    printf("");
    printf("+++ Calculating AtmLight");
    Scalar A = AtmLight(src,te);
    printf("--- Returning AtmLight...");

    printf("");
    printf("+++ Calculating TransmissionEstimate");
    TransmissionEstimate(src,A,15, te);
    printf("--- Returning TransmissionEstimate...");

    // printf("Storing TransmissionEstimate");
    // write_MAT_to_file("/sdcard/t_est.bin", te);

    printf("");
    printf("+++ Calculating TransmissionRefine");
    TransmissionRefine(src, te);
    printf("--- Returning TransmissionRefine...");

    // printf("Storing TransmissionRefine");
    // write_MAT_to_file("/sdcard/t_ref.bin", te);

    printf("");
    printf("+++ Calculating Recover");
    Recover(src, te, dst, A, 1);
    printf("--- Returning Recover...");

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("End dehaze, Free heap: %d bytes", end_heap);
    printf("Total memory used by dehaze: %d bytes", (start_heap - end_heap));

    return; 
}

const Scalar AtmLight(Mat &im, Mat &dark)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("Start AtmLight, Free heap: %d bytes", start_heap);
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %d bytes", uxTaskGetStackHighWaterMark(NULL));

    int _rows = im.rows;
    int _cols = im.cols;
    int imsz  = _rows*_cols;
    int numpx = (int)MAX(imsz/1000, 1);
    
    Mat darkvec = dark.reshape(0, 1);
    Mat imvec = im.reshape(0, 1);
    
    Mat indices;
    cv::sortIdx(darkvec, indices, SORT_DESCENDING);

    Scalar atmsum(0, 0, 0, 0);
    for(int ind = 0; ind < numpx; ind++)
    {
        atmsum.val[0] += imvec.at<Vec3b>(0, indices.at<int>(0,ind))[0];
        atmsum.val[1] += imvec.at<Vec3b>(0, indices.at<int>(0,ind))[1];
        atmsum.val[2] += imvec.at<Vec3b>(0, indices.at<int>(0,ind))[2];
    }
    
    atmsum.val[0] = atmsum.val[0]/ numpx;
    atmsum.val[1] = atmsum.val[1]/ numpx;
    atmsum.val[2] = atmsum.val[2]/ numpx;
    
    darkvec.release();
    imvec.release();
    indices.release();

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("End AtmLight, Free heap: %d bytes", end_heap);
    printf("Total memory used by AtmLight: %d bytes", (start_heap - end_heap));

    return atmsum;
}

void DarkChannel(Mat &img, int sz,  Mat &dst)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("Start DarkChannel, Free heap: %d bytes", start_heap);
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %d bytes", uxTaskGetStackHighWaterMark(NULL));

    dst = Mat::zeros(img.rows, img.cols, CV_8UC1);
    
    // Reduce memory
    for(int row = 0; row < img.rows; row++)
    {
        for(int col = 0; col < img.cols; col++)
        {
            dst.at<uchar>(row,col) = cv::min(cv::min(img.at<Vec3b>(row,col)[0], img.at<Vec3b>(row,col)[1]), img.at<Vec3b>(row,col)[2]);
        }
    }

    printf("Start spltting, Free heap: %d bytes",  heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

   
   



    // 'erode' image, so calculate the minimun value in the window given by sz
    Mat kernel = getStructuringElement(cv::MorphShapes::MORPH_RECT, Size(sz,sz));
    
    cv::erode(dst, dst, kernel);

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("End DarkChannel, Free heap: %d bytes", end_heap);
    printf("Total memory used by DarkChannel: %d bytes", (start_heap - end_heap));
}

void TransmissionEstimate(Mat &im, Scalar A, int sz, Mat &dst)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("Start TransmissionEstimate, Free heap: %d bytes", start_heap);
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %d bytes", uxTaskGetStackHighWaterMark(NULL));

    float omega = 0.95;
    Mat im3;

    vector<Mat> im_ch(3);
    cv::split(im, im_ch);

    im_ch[0] = (im_ch[0] / A.val[0]) * 255;
    im_ch[1] = (im_ch[1] / A.val[1]) * 255;
    im_ch[2] = (im_ch[2] / A.val[2]) * 255;

    cv::merge(im_ch, im3);

    for(int i = 0; i < im_ch.size(); i++)
    {
        im_ch[i].release();
    }

    Mat _dark;
    DarkChannel(im3,sz,_dark);
    dst = 255 - omega*_dark;

    _dark.release();
    im3.release();

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("End TransmissionEstimate, Free heap: %d bytes", end_heap);
    printf("Total memory used by TransmissionEstimate: %d bytes", (start_heap - end_heap));
}



void TransmissionRefine(Mat &im, Mat &et)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("Start TransmissionRefine, Free heap: %d bytes", start_heap);
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %d bytes", uxTaskGetStackHighWaterMark(NULL));
 
    Mat gray;
    cvtColor(im, gray, cv::COLOR_BGR2GRAY);

    // downscale
    printf("Before pyrdown, gray rows: %d, cols: %d", gray.rows, gray.cols);
    printf("Before pyrdown, et rows: %d, cols: %d", et.rows, et.cols);
    cv::pyrDown(et,et);
    cv::pyrDown(gray,gray);
    printf("After pyrdown, gray rows: %d, cols: %d", gray.rows, gray.cols);
    printf("After pyrdown, et rows: %d, cols: %d", et.rows, et.cols);
    

    Guidedfilter(gray, et, 60, 0.0001);


    // upscale
    printf("Before pyrUp, gray rows: %d, cols: %d", gray.rows, gray.cols);
    printf("Before pyrUp, et rows: %d, cols: %d", et.rows, et.cols);
    cv::pyrUp(et,et);
    cv::pyrUp(gray,gray);
    printf("After pyrUp, gray rows: %d, cols: %d", gray.rows, gray.cols);
    printf("After pyrUp, et rows: %d, cols: %d", et.rows, et.cols);


    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("End TransmissionRefine, Free heap: %d bytes", end_heap);
    printf("Total memory used by TransmissionRefine: %d bytes", (start_heap - end_heap));
}

void Guidedfilter(Mat &im_grey, Mat &transmission_map, int r, float eps)
{ 
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("Start Guidedfilter, Free heap: %d bytes", start_heap);
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %d bytes", uxTaskGetStackHighWaterMark(NULL));
 
    // Conver to float
    transmission_map.convertTo(transmission_map, CV_32FC1);
    transmission_map = transmission_map/255;

    printf("after converting transmission_map to float, Free heap: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    im_grey.convertTo(im_grey, CV_32FC1);
    im_grey = im_grey/255;
    
    printf("after converting im input to float, Free heap: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    
    Mat mean_I;
    Mat mean_Ip;
    Mat mean_II;
    
    // Mean
    mean_Ip = im_grey.mul(transmission_map);
    
    cv::boxFilter(mean_Ip, mean_Ip, CV_32F, Size(r,r));
    cv::boxFilter(im_grey, mean_I, CV_32F, Size(r,r));
    cv::boxFilter(transmission_map, transmission_map, CV_32F, Size(r,r));
    
    // cov_Ip
    // Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    mean_Ip = mean_Ip - mean_I.mul(transmission_map);
    
    // Mean
    mean_II = im_grey.mul(im_grey);
    cv::boxFilter(mean_II, mean_II,CV_32F,Size(r,r));
    
    // var_I
    // Mat var_I = mean_II - mean_I.mul(mean_I);
    mean_II = mean_II - mean_I.mul(mean_I);
    
    // a
    //  Mat a = cov_Ip/(var_I + eps);
    mean_II = cv::max(mean_II, eps);
    mean_Ip = mean_Ip/mean_II;
    // b
    // Mat b = mean_p - a.mul(mean_I);
    mean_I = mean_Ip.mul(mean_I);
    mean_I = transmission_map - mean_I;
    
    // Mean
    cv::boxFilter(mean_Ip, mean_Ip, CV_32F, Size(r,r));
    cv::boxFilter(mean_I, mean_I, CV_32F, Size(r,r));
    
    mean_Ip = im_grey.mul(mean_Ip);
    transmission_map = mean_Ip + mean_I;
    
    // Go back to uint8
    transmission_map = transmission_map * 255;
    transmission_map.convertTo(transmission_map, CV_8UC1);

    // Release memory
    mean_I.release();
    mean_Ip.release();
    mean_II.release();

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("End Guidedfilter, Free heap: %d bytes", end_heap);
    printf("Total memory used by Guidedfilter: %d bytes", (start_heap - end_heap));
}


void Recover(Mat &im, Mat &t, Mat &dst, Scalar A, int tx)
{
    dst = Mat::zeros(im.rows, im.cols, im.type());
    
    for(int _row = 0; _row < dst.rows; _row++)
    {
        for(int _col = 0; _col < dst.cols; _col++)
        {
            int   div    = MAX(t.at<uchar>(_row, _col), tx);
            float factor = 255.f/div;
            
            dst.at<Vec3b>(_row, _col)[0] = cv::abs((im.at<Vec3b>(_row, _col)[0] - A.val[0])*factor + A.val[0]);
            dst.at<Vec3b>(_row, _col)[1] = cv::abs((im.at<Vec3b>(_row, _col)[1] - A.val[1])*factor + A.val[1]);
            dst.at<Vec3b>(_row, _col)[2] = cv::abs((im.at<Vec3b>(_row, _col)[2] - A.val[2])*factor + A.val[2]);
        }
    }
}

