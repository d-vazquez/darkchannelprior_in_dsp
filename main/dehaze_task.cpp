

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "freertos/event_groups.h"


#include "filesystem.h"
#include "offload_task.h"
#include "shared_rtos.h"
#include <esp_log.h>

#include "opencv_interface.h"
#include "dehaze_task.h"
#include "dehaze.h"


#define block_size 503910


using namespace cv;

void dehaze_task(void *arg)
{
    static char TAG[] = "dehaze_task";

    ESP_LOGI(TAG, "Starting dehaze_task in Core %d", xPortGetCoreID());
    
    ESP_LOGI(TAG, "Free heap: %u bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes, Core: %d", uxTaskGetStackHighWaterMark(NULL), xPortGetCoreID());

    // -------- start ------------------------------------------
    esp_err_t ret;

    ESP_LOGI(TAG, "Starting main");

    ESP_LOGI(TAG, "Free heap: %u bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes, Core: %d", uxTaskGetStackHighWaterMark(NULL), xPortGetCoreID());

    ESP_LOGI(TAG, "Allocating memory");

    char *image_buffer = (char *)heap_caps_malloc(sizeof(char) * block_size, MALLOC_CAP_SPIRAM);

    if (NULL == image_buffer)
    {
        ESP_LOGE(TAG, "Error allocating memory");
        return;
    }
    ESP_LOGI(TAG, "memory value allocated: %x", image_buffer[0]);

    image_buffer[0] = 0xDE;

    ESP_LOGI(TAG, "memory value after write: %x", image_buffer[0]);

    ESP_LOGI(TAG, "memory wrote");

    ESP_LOGI(TAG, "Free heap: %u bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes, Core: %d", uxTaskGetStackHighWaterMark(NULL), xPortGetCoreID());

    ret = mount_sd_filesystem();

    // ------------ Opening file
    const char *file_src = MOUNT_POINT "/fuente.bin";
    unsigned char *img_buffer = NULL;
    size_t img_buffer_lenght = 0;


    ESP_LOGI(TAG, "img_buffer reference & %p", &img_buffer  );
    ESP_LOGI(TAG, "img_buffer ptr should be null %p", img_buffer  );


    read_from_file(file_src, &img_buffer, &img_buffer_lenght);

    ESP_LOGI(TAG, "img_buffer_lenght %d",img_buffer_lenght);
    ESP_LOGI(TAG, "img_buffer %p", img_buffer);

    ESP_LOGI(TAG, "Reading image to MAT");
    ESP_LOGI(TAG, "Free heap: %u bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes, Core: %d", uxTaskGetStackHighWaterMark(NULL), xPortGetCoreID());

    
    Mat I(480, 640, CV_8UC3, img_buffer);
    Mat J;

    dehaze(I, J);
    
    ESP_LOGI(TAG, "finished processing MAT");
    ESP_LOGI(TAG, "Free heap: %u bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes, Core: %d", uxTaskGetStackHighWaterMark(NULL), xPortGetCoreID());

    // Create output file
    const char *file_dst = MOUNT_POINT "/restored.bin";
    write_MAT_to_file(file_dst, J);

    ESP_LOGE(TAG, "Exit.....");

    while(1)
    {
        // Task should not return so infinite
    }

    return;
}

void write_MAT_to_file(const char *file_dst, cv::Mat &src)
{
    write_to_file(file_dst, src.data, src.rows * src.cols * src.channels());
}
