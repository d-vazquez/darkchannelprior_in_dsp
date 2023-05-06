
#include "opencv_interface.h"
#include <esp_log.h>
#include <string>
#include "sdkconfig.h"
#include <iostream>
#include <sys/unistd.h>
#include <sys/stat.h>
#include <esp_err.h>
#include <esp_spiffs.h>

#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include <esp_heap_caps.h>

#include "dehaze.h"
#include "offload_task.h"
#include "dehaze_task.h"
#include "filesystem.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "freertos/event_groups.h"

#include "shared_rtos.h"

using namespace cv;
using namespace std;

extern "C"
{
    void app_main(void);
}

#ifdef PARALLELIZE
    #define DEHAZE_TASK_PRIO (1)
    // #define DEHAZE_TASK_PRIO (configMAX_PRIORITIES - 2)
    #define DEHAZE_STACK_SIZE (1024 * 35)
    StackType_t xDehaze_stack[ DEHAZE_STACK_SIZE ];
    StaticTask_t xDehaze_TaskBuffer;

    #define OFFLOAD_TASK_PRIO (1)
    #define OFFLOAD_STACK_SIZE (1024 * 35)
    StackType_t xOffload_stack[ OFFLOAD_STACK_SIZE ];
    StaticTask_t xOffload_TaskBuffer;
#else
    #define DEHAZE_TASK_PRIO (1)
    #define DEHAZE_STACK_SIZE (1024 * 40)
    StackType_t xDehaze_stack[ DEHAZE_STACK_SIZE ];
    StaticTask_t xDehaze_TaskBuffer;
#endif

TaskHandle_t mat_split_task_handle = NULL;
TaskHandle_t offload_task_handle = NULL;

void app_main()
{
    static char TAG[] = "startup_task";
    ESP_LOGI(TAG, "Starting main");

    /* Create queues */
    // Create a queue capable of containing 10 pointers to AMessage structures.
    // These should be passed by pointer as they contain a lot of data.
    xDehazeToOffload_Queue = xQueueCreate( 10, sizeof(xQMatMessage *) );
    if( xDehazeToOffload_Queue == 0 )
    {
        // Failed to create the queue.
        ESP_LOGE(TAG, "Failed creating queue");
        return;
    }

    xMatEvents = xEventGroupCreate();
    if( xMatEvents == NULL )
    {
        ESP_LOGE(TAG, "Failed creating event Group");
        return;
    }

    /* Start the tasks */
    mat_split_task_handle = xTaskCreateStaticPinnedToCore(dehaze_task,              // Function Ptr
                                                        "dehaze Task",              // Name
                                                        DEHAZE_STACK_SIZE,                // Stack size
                                                        nullptr,                    // Parameter
                                                        DEHAZE_TASK_PRIO,   // Prio
                                                        xDehaze_stack,                    // Static stack array
                                                        &xDehaze_TaskBuffer,              // Static TCB
                                                        tskNO_AFFINITY);                         // Core 0
#ifdef PARALLELIZE
    offload_task_handle = xTaskCreateStaticPinnedToCore(dehaze_offload_task,        // Function Ptr
                                                        "Offload Task",             // Name
                                                        OFFLOAD_STACK_SIZE,                // Stack size
                                                        nullptr,                    // Parameter
                                                        OFFLOAD_TASK_PRIO,   // Prio
                                                        xOffload_stack,                    // Static stack array
                                                        &xOffload_TaskBuffer,              // Static TCB
                                                        tskNO_AFFINITY);            // Core 1
#endif
}


