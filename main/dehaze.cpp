
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "freertos/event_groups.h"
#include <esp_log.h>
#include <esp_heap_caps.h>
#include <esp_task_wdt.h>
#include <vector>
#include "dehaze.h"
#include "shared_rtos.h"
#include "offload_task.h"

/**
* @file dehaze.cpp
* @brief Dehazing API Source file
* @author Dario Vazquez
*/

QueueHandle_t       xDehazeToOffload_Queue;
EventGroupHandle_t  xMatEvents;

using namespace cv;
using namespace std;

static char TAG[] = "dehaze";

#define printf(...) ESP_LOGI(TAG, ##__VA_ARGS__)

extern void  write_MAT_to_file(const char *file_dst, cv::Mat &src);
void         parallel_DarkChannel(Mat &src_T, Mat &src_B, int sz, Mat &dst_T, Mat &dst_B);
const Scalar parallel_AtmLight(Mat &src_T, Mat &src_B, Mat &dark_T, Mat &dark_B);
void         parallel_TransmissionEstimate(Mat &src_T, Mat &src_B, Scalar A, int sz, Mat &dst_T, Mat &dst_B);
void         parallel_TransmissionRefine(Mat &src_T, Mat &src_B, Mat &dst_T, Mat &dst_B);
void         parallel_Recover(Mat &src_T, Mat &src_B, Mat &te_T, Mat &te_B, Mat &dst_T, Mat &dst_B, Scalar A,  int tx);

static xQMatMessage _message;
static xQMatMessage *message_tx = &_message;
static EventBits_t  uxBits = 0x00;

// Measure time
long stop_darkc, start_darkc, stop_atml, start_atml, stop_tranEst, start_tranEst = 0;
long stop_tranRef, start_tranRef, stop_recover, start_recover = 0;

/**
 * Dehazing main function, it runs the image thru the steps of the dark channel prior algorithm
 * @author Dario Vazquez
 * @param src Input hazy Image in Mat format
 * @param dst Output dehazed Image in Mat format
 */
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
    
    Mat te;
    te  = Mat(src.rows, src.cols, CV_8UC1);
    
    Mat te_T, te_B, src_T, src_B, dst_T, dst_B;

    printf("Splitting Src Mat");
    mat_split(src, src_T, src_B);
    printf("Splitting te Mat");
    mat_split(te, te_T, te_B);

    printf("");
    printf("+++ Calculating dark channel");
    start_darkc = esp_timer_get_time();
    
#if !defined(PARALELLIZE)
    DarkChannel(src,15,te);
#else
    parallel_DarkChannel(src_T, src_B, 15, te_T, te_B); 
#endif

    stop_darkc = esp_timer_get_time();

    printf("Storing Dark channel");
    write_MAT_to_file("/sdcard/darkc.bin", te);

    printf("");
    printf("+++ Calculating AtmLight");
    start_atml = esp_timer_get_time();

#if !defined(PARALELLIZE)
    Scalar A = AtmLight(src,te);
#else
    Scalar A = parallel_AtmLight(src_T, src_B, te_T, te_B);
#endif
    
    printf("Atmlight = [%f %f %f]", A.val[0], A.val[1], A.val[2]);
    stop_atml = esp_timer_get_time();
    printf("--- Returning AtmLight...");


    printf("");
    printf("+++ Calculating TransmissionEstimate");
    start_tranEst = esp_timer_get_time();

#if !defined(PARALELLIZE)
    TransmissionEstimate(src,A,15, te);
#else
    parallel_TransmissionEstimate(src_T, src_B, A, 15, te_T, te_B);
#endif

    stop_tranEst = esp_timer_get_time();
    printf("--- Returning TransmissionEstimate...");

    printf("Storing TransmissionEstimate");
    write_MAT_to_file("/sdcard/t_est.bin", te);


    printf("");
    printf("+++ Calculating TransmissionRefine");
    start_tranRef = esp_timer_get_time();
    
#if !defined(PARALELLIZE)
    TransmissionRefine(src, te);
#else
    parallel_TransmissionRefine(src_T, src_B, te_T, te_B);
#endif

    stop_tranRef = esp_timer_get_time();
    printf("--- Returning TransmissionRefine...");

    printf("Storing TransmissionRefine");
    write_MAT_to_file("/sdcard/t_ref.bin", te);

    printf("");
    printf("+++ Calculating Recover");
    start_recover = esp_timer_get_time();
    
#if !defined(PARALELLIZE)
    Recover(src, te, dst, A, 1);
#else
    dst = Mat(src.rows, src.cols, CV_8UC3);
    printf("Splitting dst Mat");
    mat_split(dst, dst_T, dst_B);

    parallel_Recover(src_T, src_B, te_T, te_B, dst_T, dst_B, A, 1);
#endif

    stop_recover = esp_timer_get_time();
    printf("--- Returning Recover...");

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("End dehaze, Free heap: %d bytes", end_heap);
    ESP_LOGW(TAG,"Total memory used by dehaze: %d bytes", (start_heap - end_heap));

    long darkc_time    = (stop_darkc - start_darkc);
    long atml_time     = (stop_atml - start_atml);
    long transEst_time = (stop_tranEst - start_tranEst);
    long transRef_time = (stop_tranRef - start_tranRef);
    long recover_time  = (stop_recover - start_recover);
    long dehaze_time   = darkc_time + atml_time + transEst_time + transRef_time + recover_time;

    printf("Total time used for darkchannel: %07li us", darkc_time);
    printf("Total time used for atmLight:    %07li us", atml_time);
    printf("Total time used for TransEst:    %07li us", transEst_time);
    printf("Total time used for TransRef:    %07li us", transRef_time);
    printf("Total time used for Recover:     %07li us", recover_time);
    printf("Total time used for dehaze:      %07li us", dehaze_time);

    return; 
}

/**
 * Image splitting function, the image is not actually splitted, but a MAT structure is created
 * referencing the top and bottom part as if they were different MAT images, this is done to
 * parallelize the image processing
 * @author Dario Vazquez
 * @param src Input Image in Mat format
 * @param top Output ROI structure pointing to the top half of the image
 * @param top Output ROI structure pointing to the bottom half of the image
 */
void mat_split(Mat &src, Mat &top, Mat &bot)
{
    printf("Mat :: rows = %d, cols = %d", src.rows, src.cols);
    printf("Mat :: rows/2 = %d", src.rows/2);
    top = src(Range(0,src.rows/2), Range(0,src.cols));
    bot = src(Range(src.rows/2,src.rows), Range(0,src.cols));
}

/**
 * Handles a parallel processing of the Atmospheric light process, it loads information from one
 * half of the image and sends it to Core1, the other half is processed in Core0 and then halts
 * waiting for Core1 to finish, once it finishes function returns
 * @author Dario Vazquez
 * @param src_T Input image top part
 * @param src_B Input image bottom part
 * @param dark_T Output image top part
 * @param dark_B Output image bottom part
 * @retval Atmospheric light in Scalar
 */
const Scalar parallel_AtmLight(Mat &src_T, Mat &src_B, Mat &dark_T, Mat &dark_B)
{
    Scalar A_B; 
    Scalar A_T;
    Scalar A; 
    printf("Sending message");
    message_tx->id      = 420;
    message_tx->opcode  = ATMLIGHT_OP;
    message_tx->src      = &src_B;
    message_tx->dst      = &dark_B;
    message_tx->atmlight= &A_B;
    xQueueSend( xDehazeToOffload_Queue, ( void * ) &message_tx, ( TickType_t ) 0 );

    // Process half Mat
    A_T = AtmLight(src_T, dark_T);

    printf("Waiting Rendezvous");   
    uxBits = xEventGroupWaitBits(xMatEvents,       // Event group handler
                                 MAT_SPLIT_EVENT,  // Event to wait for
                                 pdTRUE,           // clear bits
                                 pdFALSE,          // do not wait for all, either event suffice
                                 portMAX_DELAY);   // wait forever
    
    printf("Rendezvous received");   
    
    // consolidate atmospheric light
    A.val[0] = cv::max(A_T.val[0], A_B.val[0]);
    A.val[1] = cv::max(A_T.val[1], A_B.val[1]);
    A.val[2] = cv::max(A_T.val[2], A_B.val[2]);
    
    printf("A_T           = [%f %f %f]", A_T.val[0], A_T.val[1], A_T.val[2]);
    printf("message_rx->A = [%f %f %f]", A_B.val[0], A_B.val[1], A_B.val[2]);
    printf("A             = [%f %f %f]", A.val[0], A.val[1], A.val[2]);

    return A;
}

/**
 * Atmospheric light process, it calculates atmospheric light for the scene
 * @author Dario Vazquez
 * @param im input image
 * @param dark Dark channel image
 * @retval Atmospheric light in Scalar
 */
const Scalar AtmLight(Mat &im, Mat &dark)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("Start AtmLight, Free heap: %d bytes", start_heap);
    

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

/**
 * Handles a parallel processing of the Dark channel process, it loads information from one
 * half of the image and sends it to Core1, the other half is processed in Core0 and then halts
 * waiting for Core1 to finish, once it finishes function returns
 * @author Dario Vazquez
 * @param src_T Input image top part
 * @param src_B Input image bottom part
 * @param sz Kernel size for Dark channel erode
 * @param dst_T Output image top part
 * @param dst_B Output image bottom part
 */
void parallel_DarkChannel(Mat &src_T, Mat &src_B, int sz, Mat &dst_T, Mat &dst_B)
{
    printf("Sending message");
    message_tx->id      = 911;
    message_tx->opcode  = DARK_CHANNEL_OP;
    message_tx->src      = &src_B;
    message_tx->dst      = &dst_B;
    xQueueSend( xDehazeToOffload_Queue, ( void * ) &message_tx, ( TickType_t ) 0 );

    // Process half Mat
    DarkChannel(src_T, sz, dst_T);

    printf("Waiting Rendezvous");   
    uxBits = xEventGroupWaitBits(xMatEvents,       // Event group handler
                                 MAT_SPLIT_EVENT,  // Event to wait for
                                 pdTRUE,           // clear bits
                                 pdFALSE,          // do not wait for all, either event suffice
                                 portMAX_DELAY);   // wait forever
    
    printf("Rendezvous received");   
}

/**
 * Dark channel process, calculates the dark channel of the input image
 * @author Dario Vazquez
 * @param im input image
 * @param sz kernel size for erode
 * @param dst Dark channel output image
 */
void DarkChannel(Mat &img, int sz,  Mat &dst)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("Start DarkChannel, Free heap: %d bytes", start_heap);
    
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

/**
 * Handles a parallel processing of the Transmission estimate process, it loads information from one
 * half of the image and sends it to Core1, the other half is processed in Core0 and then halts
 * waiting for Core1 to finish, once it finishes function returns
 * @author Dario Vazquez
 * @param src_T Input image top part
 * @param src_B Input image bottom part
 * @param A Atmospheric light array
 * @param sz kernel size for Dark channel
 * @param dst_T Output image top part
 * @param dst_B Output image bottom part
 */
void parallel_TransmissionEstimate(Mat &src_T, Mat &src_B, Scalar A, int sz, Mat &dst_T, Mat &dst_B)
{
    printf("Sending message");
    message_tx->id      = 690;
    message_tx->opcode  = TRANSMITION_ESTIMATE_OP;
    message_tx->src      = &src_B;
    message_tx->dst      = &dst_B;
    message_tx->atmlight= &A;
    message_tx->ksize   = sz;
    xQueueSend( xDehazeToOffload_Queue, ( void * ) &message_tx, ( TickType_t ) 0 );

    // Process half Mat
    TransmissionEstimate(src_T, A, sz, dst_T);

    printf("Waiting Rendezvous");   
    uxBits = xEventGroupWaitBits(xMatEvents,       // Event group handler
                                 MAT_SPLIT_EVENT,  // Event to wait for
                                 pdTRUE,           // clear bits
                                 pdFALSE,          // do not wait for all, either event suffice
                                 portMAX_DELAY);   // wait forever
    
    printf("Rendezvous received");  
}

/**
 * Transmission estimate process, it calculates the scene transmission map using
 * an eroded dark channel
 * @author Dario Vazquez
 * @param im input image
 * @param A atmospheric light image
 * @param sz Dark channel kernel size
 * @param dst Transmission map output
 */
void TransmissionEstimate(Mat &im, Scalar A, int sz, Mat &dst)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("Start TransmissionEstimate, Free heap: %d bytes", start_heap);
    
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

/**
 * Handles a parallel processing of the Transmission refine process, it loads information from one
 * half of the image and sends it to Core1, the other half is processed in Core0 and then halts
 * waiting for Core1 to finish, once it finishes function returns
 * @author Dario Vazquez
 * @param src_T Input image top part
 * @param src_B Input image bottom part
 * @param dst_T Output image top part
 * @param dst_B Output image bottom part
 */
void parallel_TransmissionRefine(Mat &src_T, Mat &src_B, Mat &dst_T, Mat &dst_B)
{
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes, Core: %d", uxTaskGetStackHighWaterMark(NULL), xPortGetCoreID());

    printf("Sending message");
    message_tx->id      = 123;
    message_tx->opcode  = TRANSMITION_REFINE_OP;
    message_tx->src      = &src_B;
    message_tx->dst      = &dst_B;
    message_tx->atmlight= NULL;
    message_tx->ksize   = 0;
    xQueueSend( xDehazeToOffload_Queue, ( void * ) &message_tx, ( TickType_t ) 0 );

    // Process half Mat
    TransmissionRefine(src_T, dst_T);

    printf("Waiting Rendezvous");   
    uxBits = xEventGroupWaitBits(xMatEvents,       // Event group handler
                                 MAT_SPLIT_EVENT,  // Event to wait for
                                 pdTRUE,           // clear bits
                                 pdFALSE,          // do not wait for all, either event suffice
                                 portMAX_DELAY);   // wait forever
    
    printf("Rendezvous received");  

}

/**
 * Transmission refine process, the transmission estimate is precessed by the dark channel kernel size
 * so its degradated, a guided filter is calculated using the original image, and its applied to the 
 * transmission estimate, the output transmission refined map is written on top of the transmission estimate
 * to save memory
 * @author Dario Vazquez
 * @param im input image input
 * @param et Transmission map input, the refined tranmission map is written on top of this to save memory
 */
void TransmissionRefine(Mat &im, Mat &et)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("++ Start TransmissionRefine, Free heap: %d bytes", start_heap);
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

/**
 * Transmission refine process, the transmission estimate is precessed by the dark channel kernel size
 * so its degradated, a guided filter is calculated using the original image, and its applied to the 
 * transmission estimate, the output transmission refined map is written on top of the transmission estimate
 * to save memory
 * @author Dario Vazquez
 * @param im_grey input image input in grey scale (guide)
 * @param transmission_map Image to be filtered, the output is written on top of this to save memory
 * @param r kernel width for the box filter
 * @param eps small number to avoid div by 0
 */
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

/**
 * Handles a parallel processing of the recovery process, it loads information from one
 * half of the image and sends it to Core1, the other half is processed in Core0 and then halts
 * waiting for Core1 to finish, once it finishes function returns
 * @author Dario Vazquez
 * @param src_T Input image top part
 * @param src_B Input image bottom part
 * @param te_T Transmission map top part
 * @param te_B Transmission map bottom part
 * @param dst_T Output image top part
 * @param dst_B Output image bottom part
 * @param A Atmospheric light array (Scalar)
 * @param tx Small value to avoid div by 0
 */
void parallel_Recover(Mat &src_T, Mat &src_B, Mat &te_T, Mat &te_B, Mat &dst_T, Mat &dst_B, Scalar A, int tx)
{
    printf("Sending message");
    message_tx->id      = 369;
    message_tx->opcode  = RECOVER_OP;
    message_tx->src      = &src_B;
    message_tx->dst      = &dst_B;
    message_tx->aux      = &te_B;
    message_tx->atmlight= &A;
    message_tx->ksize   = tx;
    xQueueSend( xDehazeToOffload_Queue, ( void * ) &message_tx, ( TickType_t ) 0 );

    // Process half Mat
    Recover(src_T, te_T, dst_T, A, tx);

    printf("Waiting Rendezvous");   
    uxBits = xEventGroupWaitBits(xMatEvents,       // Event group handler
                                 MAT_SPLIT_EVENT,  // Event to wait for
                                 pdTRUE,           // clear bits
                                 pdFALSE,          // do not wait for all, either event suffice
                                 portMAX_DELAY);   // wait forever
    
    printf("Rendezvous received");  

}

/**
 * Transmission refine process, the transmission estimate is precessed by the dark channel kernel size
 * so its degradated, a guided filter is calculated using the original image, and its applied to the 
 * transmission estimate, the output transmission refined map is written on top of the transmission estimate
 * to save memory
 * @author Dario Vazquez
 * @param src Input hazy image
 * @param te Transmission map 
 * @param dst Output dehazed image 
 * @param A Atmospheric light array (Scalar)
 * @param tx Small value to avoid div by 0
 */
void Recover(Mat &im, Mat &t, Mat &dst, Scalar A, int tx)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("++++ Start Recover, Free heap: %d bytes", start_heap);
    
#if !defined(PARALELLIZE)
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

