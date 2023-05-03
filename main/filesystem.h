
#ifndef FILESYSTEM_H
#define FILESYSTEM_H


#define MOUNT_POINT "/sdcard"


esp_err_t mount_sd_filesystem(void);
void      write_to_file(const char *file_dst, unsigned char *data, size_t lenght);
void      read_from_file(const char *file_src, unsigned char **data, size_t *lenght);

#endif // FILESYSTEM_H
