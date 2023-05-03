
#include <esp_log.h>
#include <esp_err.h>
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "filesystem.h"

static char TAG[] = "filesystem";


#define CONFIG_EXAMPLE_PIN_MOSI 15
#define CONFIG_EXAMPLE_PIN_MISO 2
#define CONFIG_EXAMPLE_PIN_CLK 14
#define CONFIG_EXAMPLE_PIN_CS 13

#define PIN_NUM_MISO CONFIG_EXAMPLE_PIN_MISO
#define PIN_NUM_MOSI CONFIG_EXAMPLE_PIN_MOSI
#define PIN_NUM_CLK CONFIG_EXAMPLE_PIN_CLK
#define PIN_NUM_CS CONFIG_EXAMPLE_PIN_CS

esp_err_t mount_sd_filesystem(void)
{
    esp_err_t ret;

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
        return ret;
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
        return ret;
    }
    ESP_LOGI(TAG, "Filesystem mounted");

    // Card has been initialized, print its properties
    sdmmc_card_print_info(stdout, card);

    return ret;
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
    ESP_LOGI(TAG, "Reading image.....");
    uint32_t bytesread = fread(*data, 1, sizeof(unsigned char) * (*lenght), f_src);
    ESP_LOGI(TAG, "Bytes read: %u", bytesread);
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
    ESP_LOGI(TAG, "writing image.....");
    ESP_LOGI(TAG, "writing %u bytes.....", newfile_size);
    uint32_t byteswriten = fwrite(data, 1, sizeof(unsigned char) * newfile_size, f_dst);
    ESP_LOGI(TAG, "Bytes writen: %u", byteswriten);

    // Close source image
    ESP_LOGI(TAG, "image closed.....");
    fclose(f_dst);
}
