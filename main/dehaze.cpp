

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "freertos/event_groups.h"

#include <esp_log.h>
#include <esp_heap_caps.h>
#include <esp_task_wdt.h>
#include <esp_ipc.h>

#include <vector>

#include "dehaze.h"
#include "shared_rtos.h"
#include "offload_task.h"
#include "dehaze_parallelize.h"


QueueHandle_t       xDehazeToOffload_Queue;
EventGroupHandle_t  xMatEvents;

using namespace cv;
using namespace std;

static char TAG[] = "dehaze";

#define printf(...) ESP_LOGI(TAG, ##__VA_ARGS__)

extern void  write_MAT_to_file(const char *file_dst, cv::Mat &src);

static xQMatMessage _message;
xQMatMessage *message_tx = &_message;

// Measure time
long stop_darkc, start_darkc, stop_atml, start_atml, stop_tranEst, start_tranEst = 0;
long stop_tranRef, start_tranRef, stop_recover, start_recover = 0;
long darkc_time, atml_time, transEst_time, transRef_time, recover_time, dehaze_time = 0;  

void dehaze(cv::Mat &src, cv::Mat &dst)
{
    message_tx->id      = 0;
    message_tx->opcode  = (dehaze_op)0;
    message_tx->src      = NULL;
    message_tx->dst      = NULL;
    message_tx->aux      = NULL;
    message_tx->atmlight= NULL;
    // static xQMatMessage *message_rx;

    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("Start dehaze, Free heap: %d bytes", start_heap);
    
    
    Mat te  = Mat(src.rows, src.cols, CV_8UC1);
#ifdef PARALLELIZE
    Mat te_T, te_B, src_T, src_B, dst_T, dst_B;
    mat_split(src, src_T, src_B, 15);
    mat_split(te, te_T, te_B, 15);
#endif
    
    printf("");
    printf("+++ Calculating dark channel");    
#if !defined(PARALLELIZE)
    start_darkc = esp_timer_get_time(); 
    DarkChannel(src,15,te);
    stop_darkc = esp_timer_get_time();
#else
    parallel_DarkChannel(src_T, src_B, 15, te_T, te_B); 
#endif
    printf("start: %07li us stop: %07li us", start_darkc, stop_darkc);
    printf("Total time used for darkchannel: %07li us", stop_darkc - start_darkc);

#if defined(STORE_MAT_FILE)
    printf("Storing Dark channel");
    write_MAT_to_file("/sdcard/darkc.bin", te);
#endif

    printf("");
    printf("+++ Calculating AtmLight");
#if !defined(PARALLELIZE)
    start_atml = esp_timer_get_time();
    Scalar A = AtmLight(src,te);
    stop_atml = esp_timer_get_time();
#else
    Scalar A = parallel_AtmLight(src_T, src_B, te_T, te_B);
#endif
    printf("Atmlight = [%f %f %f]", A.val[0], A.val[1], A.val[2]);
    
    printf("");
    printf("+++ Calculating TransmissionEstimate");
    

#if !defined(PARALLELIZE)
    start_tranEst = esp_timer_get_time();
    TransmissionEstimate(src,A,15, te);
    stop_tranEst = esp_timer_get_time();
#else
    parallel_TransmissionEstimate(src_T, src_B, A, 15, te_T, te_B);
#endif

#if defined(STORE_MAT_FILE)
    printf("Storing TransmissionEstimate");
    write_MAT_to_file("/sdcard/t_est.bin", te);
#endif

    printf("");
    printf("+++ Calculating TransmissionRefine");
    start_tranRef = esp_timer_get_time();
    
#if !defined(PARALLELIZE)
    TransmissionRefine(src, te);
#else
    parallel_TransmissionRefine(src_T, src_B, te_T, te_B);
#endif

    stop_tranRef = esp_timer_get_time();
    printf("--- Returning TransmissionRefine...");

#if defined(STORE_MAT_FILE)
    printf("Storing TransmissionRefine");
    write_MAT_to_file("/sdcard/t_ref.bin", te);
#endif

    printf("");
    printf("+++ Calculating Recover");
#if !defined(PARALLELIZE)
    start_recover = esp_timer_get_time();
    Recover(src, te, dst, A, 1);
    stop_recover = esp_timer_get_time();
#else
    dst = Mat(src.rows, src.cols, CV_8UC3);
    printf("Splitting dst Mat");
    mat_split(dst, dst_T, dst_B, 15);

    parallel_Recover(src_T, src_B, te_T, te_B, dst_T, dst_B, A, 1);
#endif


    // int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    // printf("End dehaze, Free heap: %d bytes", end_heap);
    // ESP_LOGW(TAG,"Total memory used by dehaze: %d bytes", (start_heap - end_heap));

    darkc_time    = (stop_darkc - start_darkc);
    atml_time     = (stop_atml - start_atml);
    transEst_time = (stop_tranEst - start_tranEst);
    transRef_time = (stop_tranRef - start_tranRef);
    recover_time  = (stop_recover - start_recover);
    dehaze_time   = darkc_time + atml_time + transEst_time + transRef_time + recover_time;

    printf("Total time used for darkchannel: %07li us", darkc_time);
    printf("Total time used for atmLight:    %07li us", atml_time);
    printf("Total time used for TransEst:    %07li us", transEst_time);
    printf("Total time used for TransRef:    %07li us", transRef_time);
    printf("Total time used for Recover:     %07li us", recover_time);
    printf("Total time used for dehaze:      %07li us", dehaze_time);

    return; 
}

void mat_split(Mat &src, Mat &top, Mat &bot, uint32_t overlap)
{
    printf("Mat :: rows = %d, cols = %d", src.rows, src.cols);
    printf("Mat :: rows/2 = %d", src.rows/2);
    top = src(Range(0,src.rows/2 + overlap), Range(0,src.cols));
    bot = src(Range(src.rows/2 - overlap,src.rows), Range(0,src.cols));
}

void DarkChannel(Mat &img, int sz,  Mat &dst)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    long now = esp_timer_get_time();
    printf("Start DarkChannel im[%d,%d], Free heap: %d bytes Core %d ts: %li micro-seconds", img.rows, img.cols ,start_heap, xPortGetCoreID(), now);
    
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
    
    cv::erode(dst, dst, kernel, Point((int)(sz/2),(int)(sz/2)), 1, cv::BORDER_REFLECT_101);

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("End DarkChannel, Free heap: %d bytes", end_heap);
    printf("Total memory used by DarkChannel: %d bytes", (start_heap - end_heap));
}

const Scalar AtmLight(Mat &im, Mat &dark)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    long now = esp_timer_get_time();
    printf("Start AtmLight im[%d,%d], Free heap: %d bytes Core %d ts: %li micro-seconds", im.rows, im.cols ,start_heap, xPortGetCoreID(), now);

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
    ESP_LOGW(TAG,"Total memory used by AtmLight: %d bytes", (start_heap - end_heap));

    return atmsum;
}

void TransmissionEstimate(Mat &im, Scalar A, int sz, Mat &dst)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("Start TransmissionEstimate im[%d,%d], Free heap: %d bytes", im.rows, im.cols, start_heap);
    
    float omega = 0.95;

    Mat iaux;

    vector<Mat> im_ch(3);
    cv::split(im, im_ch);

    im_ch[0] = (im_ch[0] / A.val[0]) * 255;
    im_ch[1] = (im_ch[1] / A.val[1]) * 255;
    im_ch[2] = (im_ch[2] / A.val[2]) * 255;

    cv::merge(im_ch, iaux);

    for(int i = 0; i < im_ch.size(); i++)
    {
        im_ch[i].release();
    }

    Mat _dark;
    DarkChannel(iaux,sz,_dark);
    dst = 255 - omega*_dark;

    _dark.release();
    iaux.release();

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("End TransmissionEstimate, Free heap: %d bytes", end_heap);
    ESP_LOGW(TAG,"Total memory used by TransmissionEstimate: %d bytes", (start_heap - end_heap));
}

void TransmissionRefine(Mat &im, Mat &et)
{

    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("++ Start TransmissionRefine im[%d,%d], Free heap: %d bytes", im.rows, im.cols, start_heap);
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes, Core: %d", uxTaskGetStackHighWaterMark(NULL), xPortGetCoreID());
    
    Mat gray;
    cvtColor(im, gray, cv::COLOR_BGR2GRAY);

    // downscale
    cv::pyrDown(et,et);
    cv::pyrDown(gray,gray);
    

    Guidedfilter(gray, et, 60, 0.0001);


    // upscale
    cv::pyrUp(et,et);
    gray.release();

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("++ End TransmissionRefine, Free heap: %d bytes", end_heap);
    ESP_LOGW(TAG,"++ Total memory used by TransmissionRefine: %d bytes", (start_heap - end_heap));
}

void Guidedfilter(Mat &im_grey, Mat &transmission_map, int r, float eps)
{ 
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("++++ Start Guidedfilter, Free heap: %d bytes", start_heap);
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes, Core: %d", uxTaskGetStackHighWaterMark(NULL), xPortGetCoreID());
    
    Mat mean_I = Mat(im_grey.rows, im_grey.cols, CV_32FC1);
    Mat mean_Ip = Mat(im_grey.rows, im_grey.cols, CV_32FC1);
    Mat mean_II = Mat(im_grey.rows, im_grey.cols, CV_32FC1);

    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes, Core: %d", uxTaskGetStackHighWaterMark(NULL), xPortGetCoreID());
 
    // Conver to float
    transmission_map.convertTo(transmission_map, CV_32FC1);
    transmission_map = transmission_map/255;

    printf("after converting transmission_map to float, Free heap: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    im_grey.convertTo(im_grey, CV_32FC1);
    im_grey = im_grey/255;
    
    printf("after converting im input to float, Free heap: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    
    // Mean
    mean_Ip = im_grey.mul(transmission_map);
    
    cv::boxFilter(mean_Ip, mean_Ip, CV_32F, Size(r,r));
    cv::boxFilter(im_grey, mean_I, CV_32F, Size(r,r));
    cv::boxFilter(transmission_map, transmission_map, CV_32F, Size(r,r));
    
    // cov_Ip
    // Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    mean_Ip = mean_Ip - (mean_I).mul(transmission_map);
    
    // Mean
    mean_II = im_grey.mul(im_grey);
    cv::boxFilter(mean_II, mean_II,CV_32F,Size(r,r));
    
    // var_I
    // Mat var_I = mean_II - mean_I.mul(mean_I);
    mean_II = mean_II - (mean_I).mul(mean_I);
    
    // a
    //  Mat a = cov_Ip/(var_I + eps);
    mean_II = cv::max(mean_II, eps);
    mean_Ip = (mean_Ip)/(mean_II);
    // b
    // Mat b = mean_p - a.mul(mean_I);
    mean_I = (mean_Ip).mul(mean_I);
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
    mean_I.release();
    mean_I.release();

    // printf("Before CopyTo, Free heap: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    Mat temp;

    // printf("After CopyTo, Free heap: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("++++ End Guidedfilter, Free heap: %d bytes", end_heap);
    ESP_LOGW(TAG,"++++ Total memory used by Guidedfilter: %d bytes", (start_heap - end_heap));
}

void Recover(Mat &im, Mat &t, Mat &dst, Scalar A, int tx)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("++++ Start Recover im[%d,%d], Free heap: %d bytes", im.rows, im.cols, start_heap);
    
#if !defined(PARALLELIZE)
    dst = Mat::zeros(im.rows, im.cols, im.type());
#endif
    
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
    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("++++ End Recover, Free heap: %d bytes", end_heap);
    ESP_LOGW(TAG,"++++ Total memory used by Recover: %d bytes", (start_heap - end_heap));
}
