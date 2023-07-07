

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "freertos/event_groups.h"


#include "offload_task.h"
#include "shared_rtos.h"
#include <esp_log.h>

#include "opencv_interface.h"
#include "dehaze.h"

// QueueHandle_t       xDehazeToOffload_Queue;
// QueueHandle_t       xOffloadToDehaze_Queue;
// EventGroupHandle_t  xMatEvents;


using namespace std;
using namespace cv;

static char TAG[] = "Mat_offload";

void DarkChannel2(Mat &img, int sz,  Mat &dst);
const Scalar AtmLight2(Mat &im, Mat &dark);
void TransmissionEstimate2(Mat &im, Scalar A, int sz, Mat &dst);
void TransmissionRefine2(Mat &im, Mat &et);
void Guidedfilter2(Mat &im_grey, Mat &transmission_map, int r, float eps);
void Recover2(Mat &im, Mat &t, Mat &dst, Scalar A, int tx);

void dehaze_offload_task(void *arg)
{
    // Task memory
    // static xQMatMessage _message;
    // static xQMatMessage *message_tx = &_message;
    static xQMatMessage *message_rx;
    ESP_LOGW(TAG, "Starting task in Core %d", xPortGetCoreID());
    
    while(1)
    {
        // ESP_LOGW(TAG, "waiting for message...");
        
        // if (pdTRUE == xTaskGenericNotifyWait(0x00,          // UBaseType_t uxIndexToWaitOn, 
        //                                      0xff,          // uint32_t ulBitsToClearOnEntry, 
        //                                      0xff,          // uint32_t ulBitsToClearOnExit, 
        //                                      (uint32_t *)&message_rx,   // uint32_t *pulNotificationValue, 
        //                                      portMAX_DELAY  // TickType_t xTicksToWait
        //                                     ) ) 
        if (xQueueReceive(xDehazeToOffload_Queue, (void *)&message_rx, OFFLOAD_EVENT_WAIT_MS) == pdTRUE) 
        {
            long now = esp_timer_get_time();
            ESP_LOGW(TAG, "message received...Core %d ts: %li micro-seconds", xPortGetCoreID(), now);
            
            ESP_LOGW(TAG, "Value received %p", message_rx);

            // offload message
            Mat         *src   = message_rx->src;
            Mat         *dst   = message_rx->dst;
            Mat         *aux   = message_rx->aux;
            Scalar      *AtmL  = message_rx->atmlight;
            int         size   = message_rx->ksize;
            dehaze_op   opcode = message_rx->opcode;

            ESP_LOGW(TAG, "id:        %d", message_rx->id);

            switch(opcode)
            {
                case DARK_CHANNEL_OP:
                {
                    DarkChannel2(*src, 15, *dst);
                    break;
                }
                case ATMLIGHT_OP:
                {
                    *AtmL = AtmLight2(*src, *dst);
                    break;
                }
                case TRANSMITION_ESTIMATE_OP:
                {
                    TransmissionEstimate2(*src, *AtmL, size, *dst);
                    break;
                }
                case TRANSMITION_REFINE_OP:
                {
                    TransmissionRefine2(*src, *dst);
                    break;
                }
                case RECOVER_OP:
                {
                    Recover2(*src, *aux, *dst, *AtmL, size);
                    break;
                }
                default:
                {
                    ESP_LOGW(TAG, "Invalid opcode");
                    abort();
                    break;
                }
            }
            now = esp_timer_get_time();
            ESP_LOGW(TAG, "Sending event, %li", now);
            xEventGroupSetBits(xMatEvents, MAT_SPLIT_EVENT);
            now = esp_timer_get_time();
            ESP_LOGW(TAG, "Event sent %li", now);
        }
        else
        {
            // ESP_LOGW(TAG, "timeout waiting for message");
        }
    }
}


void DarkChannel2(Mat &img, int sz,  Mat &dst)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    long now = esp_timer_get_time();
    ESP_LOGW(TAG,"Start DarkChannel im[%d,%d], Free heap: %d bytes Core %d ts: %li micro-seconds", img.rows, img.cols ,start_heap, xPortGetCoreID(), now);
    
    dst = Mat::zeros(img.rows, img.cols, CV_8UC1);
    
    // Reduce memory
    for(int row = 0; row < img.rows; row++)
    {
        for(int col = 0; col < img.cols; col++)
        {
            dst.at<uchar>(row,col) = cv::min(cv::min(img.at<Vec3b>(row,col)[0], img.at<Vec3b>(row,col)[1]), img.at<Vec3b>(row,col)[2]);
        }
    }

    ESP_LOGW(TAG,"Start spltting, Free heap: %d bytes",  heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    // 'erode' image, so calculate the minimun value in the window given by sz
    Mat kernel = getStructuringElement(cv::MorphShapes::MORPH_RECT, Size(sz,sz));
    
    cv::erode(dst, dst, kernel, Point((int)(sz/2),(int)(sz/2)), 1, cv::BORDER_REFLECT_101);

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGW(TAG,"End DarkChannel, Free heap: %d bytes", end_heap);
    ESP_LOGW(TAG,"Total memory used by DarkChannel: %d bytes", (start_heap - end_heap));
}

const Scalar AtmLight2(Mat &im, Mat &dark)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    long now = esp_timer_get_time();
    ESP_LOGW(TAG,"Start AtmLight im[%d,%d], Free heap: %d bytes Core %d ts: %li micro-seconds", im.rows, im.cols ,start_heap, xPortGetCoreID(), now);

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
    ESP_LOGW(TAG,"End AtmLight, Free heap: %d bytes", end_heap);
    ESP_LOGW(TAG,"Total memory used by AtmLight: %d bytes", (start_heap - end_heap));

    return atmsum;
}

void TransmissionEstimate2(Mat &im, Scalar A, int sz, Mat &dst)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGW(TAG,"Start TransmissionEstimate im[%d,%d], Free heap: %d bytes", im.rows, im.cols, start_heap);
    
    float omega = 0.95;

    Mat iaux;

    vector<Mat> im_ch(3);
    cv::split(im, im_ch);

    im_ch[0] = (im_ch[0] / (int)A.val[0]) * 255;
    im_ch[1] = (im_ch[1] / (int)A.val[1]) * 255;
    im_ch[2] = (im_ch[2] / (int)A.val[2]) * 255;

    cv::merge(im_ch, iaux);

    for(int i = 0; i < im_ch.size(); i++)
    {
        im_ch[i].release();
    }

    Mat _dark;
    DarkChannel2(iaux,sz,_dark);
    dst = 255 - omega*_dark;

    _dark.release();
    iaux.release();

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGW(TAG,"End TransmissionEstimate, Free heap: %d bytes", end_heap);
    ESP_LOGW(TAG,"Total memory used by TransmissionEstimate: %d bytes", (start_heap - end_heap));
}

void TransmissionRefine2(Mat &im, Mat &et)
{

    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGW(TAG,"++ Start TransmissionRefine im[%d,%d], Free heap: %d bytes", im.rows, im.cols, start_heap);
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes, Core: %d", uxTaskGetStackHighWaterMark(NULL), xPortGetCoreID());
    
    Mat gray;
    cvtColor(im, gray, cv::COLOR_BGR2GRAY);

    // downscale
    cv::pyrDown(et,et);
    cv::pyrDown(gray,gray);
    

    Guidedfilter2(gray, et, 60, 0.0001);


    // upscale
    cv::pyrUp(et,et);
    gray.release();

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGW(TAG,"++ End TransmissionRefine, Free heap: %d bytes", end_heap);
    ESP_LOGW(TAG,"++ Total memory used by TransmissionRefine: %d bytes", (start_heap - end_heap));
}

void Guidedfilter2(Mat &im_grey, Mat &transmission_map, int r, float eps)
{ 
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGW(TAG,"++++ Start Guidedfilter, Free heap: %d bytes", start_heap);
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes, Core: %d", uxTaskGetStackHighWaterMark(NULL), xPortGetCoreID());
    
    Mat mean_I = Mat(im_grey.rows, im_grey.cols, CV_32FC1);
    Mat mean_Ip = Mat(im_grey.rows, im_grey.cols, CV_32FC1);
    Mat mean_II = Mat(im_grey.rows, im_grey.cols, CV_32FC1);

    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes, Core: %d", uxTaskGetStackHighWaterMark(NULL), xPortGetCoreID());
 
    // Conver to float
    transmission_map.convertTo(transmission_map, CV_32FC1);
    transmission_map = transmission_map/255;

    ESP_LOGW(TAG,"after converting transmission_map to float, Free heap: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    im_grey.convertTo(im_grey, CV_32FC1);
    im_grey = im_grey/255;
    
    ESP_LOGW(TAG,"after converting im input to float, Free heap: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    
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

    // ESP_LOGW(TAG,"Before CopyTo, Free heap: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    Mat temp;

    // ESP_LOGW(TAG,"After CopyTo, Free heap: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    int end_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGW(TAG,"++++ End Guidedfilter, Free heap: %d bytes", end_heap);
    ESP_LOGW(TAG,"++++ Total memory used by Guidedfilter: %d bytes", (start_heap - end_heap));
}

void Recover2(Mat &im, Mat &t, Mat &dst, Scalar A, int tx)
{
    int start_heap = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGW(TAG,"++++ Start Recover im[%d,%d], Free heap: %d bytes", im.rows, im.cols, start_heap);
    
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
    ESP_LOGW(TAG,"++++ End Recover, Free heap: %d bytes", end_heap);
    ESP_LOGW(TAG,"++++ Total memory used by Recover: %d bytes", (start_heap - end_heap));
}
