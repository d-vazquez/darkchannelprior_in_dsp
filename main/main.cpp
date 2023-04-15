
#undef EPS
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#define EPS 192

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

using namespace cv;
using namespace std;

static char TAG[] = "hello_opencv";

#define block_size 503910

#define MOUNT_POINT "/sdcard"

#define CONFIG_EXAMPLE_PIN_MOSI 15
#define CONFIG_EXAMPLE_PIN_MISO 2
#define CONFIG_EXAMPLE_PIN_CLK 14
#define CONFIG_EXAMPLE_PIN_CS 13

#define PIN_NUM_MISO CONFIG_EXAMPLE_PIN_MISO
#define PIN_NUM_MOSI CONFIG_EXAMPLE_PIN_MOSI
#define PIN_NUM_CLK CONFIG_EXAMPLE_PIN_CLK
#define PIN_NUM_CS CONFIG_EXAMPLE_PIN_CS

extern "C"
{
    void app_main(void);
}

void write_to_file(const char *file_dst, unsigned char *data, size_t lenght);
void read_from_file(const char *file_src, unsigned char **data, size_t *lenght);
void write_MAT_to_file(const char *file_dst, cv::Mat &src);

void app_main(void)
{
    ESP_LOGI(TAG, "Starting main");
    ESP_LOGI(TAG, "Convert to Grey");
    
    ESP_LOGI(TAG, "Free heap: %u bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes", uxTaskGetStackHighWaterMark(NULL));


    // -------- start ------------------------------------------
    esp_err_t ret;

    ESP_LOGI(TAG, "Starting main");

    ESP_LOGI(TAG, "Free heap: %u bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes", uxTaskGetStackHighWaterMark(NULL));

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
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes", uxTaskGetStackHighWaterMark(NULL));

    // ----------------- File System

    // Options for mounting the filesystem.
    // If format_if_mount_failed is set to true, SD card will be partitioned and
    // formatted in case when mounting fails.
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 32 * 1024};
    sdmmc_card_t *card;
    const char mount_point[] = MOUNT_POINT;
    ESP_LOGI(TAG, "Initializing SD card");

    // Use settings defined above to initialize SD card and mount FAT filesystem.
    // Note: esp_vfs_fat_sdmmc/sdspi_mount is all-in-one convenience functions.
    // Please check its source code and implement error recovery when developing
    // production applications.
    ESP_LOGI(TAG, "Using SPI peripheral");

    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    // host.flags |= BIT(1);
    host.max_freq_khz = 400;
    // host.max_freq_khz = 400;

    spi_bus_config_t bus_cfg = {
        .mosi_io_num = (gpio_num_t)PIN_NUM_MOSI,
        .miso_io_num = (gpio_num_t)PIN_NUM_MISO,
        .sclk_io_num = (gpio_num_t)PIN_NUM_CLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 0};
    ret = spi_bus_initialize((spi_host_device_t)host.slot, &bus_cfg, SDSPI_DEFAULT_DMA);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to initialize bus.");
        return;
    }

    // This initializes the slot without card detect (CD) and write protect (WP) signals.
    // Modify slot_config.gpio_cd and slot_config.gpio_wp if your board has these signals.
    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = (gpio_num_t)PIN_NUM_CS;
    slot_config.host_id = (spi_host_device_t)host.slot;

    ESP_LOGI(TAG, "Mounting filesystem");
    ret = esp_vfs_fat_sdspi_mount(mount_point, &host, &slot_config, &mount_config, &card);

    if (ret != ESP_OK)
    {
        if (ret == ESP_FAIL)
        {
            ESP_LOGE(TAG, "Failed to mount filesystem. "
                          "If you want the card to be formatted, set the CONFIG_EXAMPLE_FORMAT_IF_MOUNT_FAILED menuconfig option.");
        }
        else
        {
            ESP_LOGE(TAG, "Failed to initialize the card (%s). "
                          "Make sure SD card lines have pull-up resistors in place.",
                     esp_err_to_name(ret));
        }
        return;
    }
    ESP_LOGI(TAG, "Filesystem mounted");

    // Card has been initialized, print its properties
    sdmmc_card_print_info(stdout, card);

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
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes", uxTaskGetStackHighWaterMark(NULL));

    
    Mat I(480, 640, CV_8UC3, img_buffer);
    Mat J;

    long start = esp_timer_get_time();

    dehaze(I, J);

    long stop = esp_timer_get_time();

    long duration = stop - start;
    ESP_LOGI(TAG, "dehaze elapsed time: %li ticks", duration);
    ESP_LOGI(TAG, "dehaze elapsed time: %li ms", duration/1000 );

    ESP_LOGI(TAG, "finished processing MAT");
    ESP_LOGI(TAG, "Free heap: %u bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "uxTaskGetStackHighWaterMark: %u bytes", uxTaskGetStackHighWaterMark(NULL));

    // Create output file
    const char *file_dst = MOUNT_POINT "/fuente2.bin";
    write_MAT_to_file(file_dst, J);

    ESP_LOGE(TAG, "Exit.....");
    return;
}




void write_MAT_to_file(const char *file_dst, cv::Mat &src)
{
    write_to_file(file_dst, src.data, src.rows * src.cols * src.channels());
}

void read_from_file(const char *file_src, unsigned char **data, size_t *lenght)
{
    ESP_LOGI(TAG, "Opening file %s", file_src);
    FILE *f_src = fopen(file_src, "rb");
    if (f_src == NULL)
    {
        ESP_LOGE(TAG, "Failed to open file for writing");
        return;
    }

    // Get file size
    fseek(f_src, 0, SEEK_END);         // seek to end of file
    *lenght = ftell(f_src); // get current file pointer
    fseek(f_src, 0, SEEK_SET);         // seek back to beginning of file

    ESP_LOGI(TAG, "data before malloc %p", *data  );
    ESP_LOGI(TAG, "File size %d", (*lenght));
    *data = (unsigned char *)heap_caps_malloc(sizeof(unsigned char) * (*lenght), MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "data malloced %p", *data  );

    ESP_LOGI(TAG, "Free heap: %u bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    // Move image to heap
    ESP_LOGE(TAG, "Reading image.....");
    uint32_t bytesread = fread(*data, 1, sizeof(unsigned char) * (*lenght), f_src);
    ESP_LOGE(TAG, "Bytes read: %u", bytesread);
    ESP_LOGI(TAG, "data[0] = %d", *data[0]  );


    // Close source image
    ESP_LOGI(TAG, "closing file");
    fclose(f_src);
}

void write_to_file(const char *file_dst, unsigned char *data, size_t lenght)
{
    ESP_LOGI(TAG, "Opening file %s", file_dst);
    FILE *f_dst = fopen(file_dst, "wb");
    if (f_dst == NULL)
    {
        ESP_LOGE(TAG, "Failed to open file for writing");
        return;
    }
    uint32_t newfile_size = lenght;

    // Write buffer to image
    ESP_LOGE(TAG, "writing image.....");
    ESP_LOGE(TAG, "writing %u bytes.....", newfile_size);
    uint32_t byteswriten = fwrite(data, 1, sizeof(unsigned char) * newfile_size, f_dst);
    ESP_LOGE(TAG, "Bytes writen: %u", byteswriten);

    // Close source image
    ESP_LOGE(TAG, "image closed.....");
    fclose(f_dst);

}

