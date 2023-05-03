
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


void mat_split(Mat &src, Mat &top, Mat &bot);

#define STACK_SIZE0 (1024 * 40)
#define STACK_SIZE1 (1024 * 40)
StackType_t xStack0[ STACK_SIZE0 ];
StackType_t xStack1[ STACK_SIZE1 ];
StaticTask_t xTaskBuffer0;
StaticTask_t xTaskBuffer1;


void app_main()
{
    static char TAG[] = "startup_task";
    ESP_LOGI(TAG, "Starting main");

    TaskHandle_t mat_split_task_handle = NULL;

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
                                                        STACK_SIZE0,                // Stack size
                                                        nullptr,                    // Parameter
                                                        configMAX_PRIORITIES - 2,   // Prio
                                                        xStack0,                    // Static stack array
                                                        &xTaskBuffer0,              // Static TCB
                                                        0);                         // Core 0
    
    offload_task_handle = xTaskCreateStaticPinnedToCore(dehaze_offload_task,        // Function Ptr
                                                        "Offload Task",             // Name
                                                        STACK_SIZE1,                // Stack size
                                                        nullptr,                    // Parameter
                                                        configMAX_PRIORITIES - 2,   // Prio
                                                        xStack1,                    // Static stack array
                                                        &xTaskBuffer1,              // Static TCB
                                                        1);                         // Core 1
}








