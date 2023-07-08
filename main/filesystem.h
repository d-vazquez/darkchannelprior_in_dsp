
#ifndef FILESYSTEM_H
#define FILESYSTEM_H

/**
* @file filesystem.h
* @brief File operations on the SD Card, read and write of images in memory
* @author Dario Vazquez
*/

#define MOUNT_POINT "/sdcard"

esp_err_t mount_sd_filesystem(void);
void      write_to_file(const char *file_dst, unsigned char *data, size_t lenght);
void      read_from_file(const char *file_src, unsigned char **data, size_t *lenght);

#endif // FILESYSTEM_H
