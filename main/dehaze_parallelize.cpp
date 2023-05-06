
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

using namespace cv;
using namespace std;

static char TAG[] = "parallel_dehaze";

#define printf(...) ESP_LOGI(TAG, ##__VA_ARGS__)

extern xQMatMessage *message_tx;

void parallel_DarkChannel(Mat &src_T, Mat &src_B, int sz, Mat &dst_T, Mat &dst_B)
{
    message_tx->id      = 911;
    message_tx->opcode  = DARK_CHANNEL_OP;
    message_tx->src     = &src_B;
    message_tx->dst     = &dst_B;
    message_tx->done    = 0;

    long now = esp_timer_get_time();
    printf("Sending message, Core %d ts: %li micro-seconds", xPortGetCoreID(), now);
    printf("Pointer to send: %p", message_tx);
    printf("Pointer to send: & %p", &message_tx);
    xQueueSend( xDehazeToOffload_Queue, ( void * ) &message_tx, ( TickType_t ) 0 );
    // xTaskGenericNotify( offload_task_handle,        // TaskHandle_t xTaskToNotify, 
    //                     0x00,                       // UBaseType_t uxIndexToNotify, 
    //                     (uint32_t)message_tx,       // uint32_t ulValue, 
    //                     eSetValueWithoutOverwrite,  // eNotifyAction eAction, 
    //                     NULL                        // uint32_t *pulPreviousNotificationValue
    //                     );
   
    // Process half Mat
    start_darkc = esp_timer_get_time(); 
    DarkChannel(src_T, sz, dst_T);

    now = esp_timer_get_time();
    printf("Waiting Rendezvous, %li", now);   
    xEventGroupWaitBits(xMatEvents,       // Event group handler
                                 MAT_SPLIT_EVENT,  // Event to wait for
                                 pdTRUE,           // clear bits
                                 pdFALSE,          // do not wait for all, either event suffice
                                 portMAX_DELAY);   // wait forever
    stop_darkc = esp_timer_get_time();
    printf("Rendezvous received, %li", stop_darkc);   
}

Scalar parallel_AtmLight(Mat &src_T, Mat &src_B, Mat &dark_T, Mat &dark_B)
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
    // xTaskGenericNotify( offload_task_handle,        // TaskHandle_t xTaskToNotify, 
    //                     0x00,                       // UBaseType_t uxIndexToNotify, 
    //                     (uint32_t)message_tx,       // uint32_t ulValue, 
    //                     eSetValueWithoutOverwrite,  // eNotifyAction eAction, 
    //                     NULL                        // uint32_t *pulPreviousNotificationValue
    //                     );

    // Process half Mat
    start_atml = esp_timer_get_time(); 
    A_T = AtmLight(src_T, dark_T);

    printf("Waiting Rendezvous");   
    xEventGroupWaitBits(xMatEvents,       // Event group handler
                                 MAT_SPLIT_EVENT,  // Event to wait for
                                 pdTRUE,           // clear bits
                                 pdFALSE,          // do not wait for all, either event suffice
                                 portMAX_DELAY);   // wait forever
    stop_atml = esp_timer_get_time();
    printf("Rendezvous received, %li", stop_atml);  
    
    // consolidate atmospheric light
    A.val[0] = cv::max(A_T.val[0], A_B.val[0]);
    A.val[1] = cv::max(A_T.val[1], A_B.val[1]);
    A.val[2] = cv::max(A_T.val[2], A_B.val[2]);
    
    return A;
}

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
    // xTaskGenericNotify( offload_task_handle,        // TaskHandle_t xTaskToNotify, 
    //                     0x00,                       // UBaseType_t uxIndexToNotify, 
    //                     (uint32_t)message_tx,       // uint32_t ulValue, 
    //                     eSetValueWithoutOverwrite,  // eNotifyAction eAction, 
    //                     NULL                        // uint32_t *pulPreviousNotificationValue
    //                     );

    // Process half Mat
    start_tranEst = esp_timer_get_time();
    TransmissionEstimate(src_T, A, sz, dst_T);

    printf("Waiting Rendezvous");   
    xEventGroupWaitBits(xMatEvents,       // Event group handler
                                 MAT_SPLIT_EVENT,  // Event to wait for
                                 pdTRUE,           // clear bits
                                 pdFALSE,          // do not wait for all, either event suffice
                                 portMAX_DELAY);   // wait forever
    stop_tranEst = esp_timer_get_time();
    printf("Rendezvous received, %li", stop_tranEst);  
}

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
    // xTaskGenericNotify( offload_task_handle,        // TaskHandle_t xTaskToNotify, 
    //                     0x00,                       // UBaseType_t uxIndexToNotify, 
    //                     (uint32_t)message_tx,       // uint32_t ulValue, 
    //                     eSetValueWithoutOverwrite,  // eNotifyAction eAction, 
    //                     NULL                        // uint32_t *pulPreviousNotificationValue
    //                     );

    // Process half Mat
    start_tranRef = esp_timer_get_time();
    TransmissionRefine(src_T, dst_T);

    printf("Waiting Rendezvous");   
    xEventGroupWaitBits(xMatEvents,       // Event group handler
                                 MAT_SPLIT_EVENT,  // Event to wait for
                                 pdTRUE,           // clear bits
                                 pdFALSE,          // do not wait for all, either event suffice
                                 portMAX_DELAY);   // wait forever
    stop_tranRef = esp_timer_get_time();
    printf("Rendezvous received, %li", stop_tranRef); 

}

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
    // xTaskGenericNotify( offload_task_handle,        // TaskHandle_t xTaskToNotify, 
    //                     0x00,                       // UBaseType_t uxIndexToNotify, 
    //                     (uint32_t)message_tx,       // uint32_t ulValue, 
    //                     eSetValueWithoutOverwrite,  // eNotifyAction eAction, 
    //                     NULL                        // uint32_t *pulPreviousNotificationValue
    //                     );

    // Process half Mat
    start_recover = esp_timer_get_time();
    Recover(src_T, te_T, dst_T, A, tx);

    printf("Waiting Rendezvous");   
    xEventGroupWaitBits(xMatEvents,       // Event group handler
                                 MAT_SPLIT_EVENT,  // Event to wait for
                                 pdTRUE,           // clear bits
                                 pdFALSE,          // do not wait for all, either event suffice
                                 portMAX_DELAY);   // wait forever
    stop_recover = esp_timer_get_time();
    printf("Rendezvous received, %li", stop_recover); 

}
