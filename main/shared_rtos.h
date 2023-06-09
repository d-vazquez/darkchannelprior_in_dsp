
#ifndef SHARED_RTOS_H
#define SHARED_RTOS_H


#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"

#include "opencv_interface.h"
#include "dehaze.h"

typedef struct _queueMatMessage
{
    uint32_t            id;
    dehaze_op           opcode;
    int                 ksize;
    cv::Mat             *src;
    cv::Mat             *dst;
    cv::Mat             *aux;
    cv::Scalar          *atmlight;
} xQMatMessage;


#define MAT_SPLIT_EVENT (0x01)
#define PARALELLIZE

extern QueueHandle_t       xDehazeToOffload_Queue;
extern EventGroupHandle_t  xMatEvents;


#endif // SHARED_RTOS_H
