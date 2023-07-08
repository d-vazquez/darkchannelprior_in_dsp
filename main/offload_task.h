
#ifndef OFFLOAD_TASK_H
#define OFFLOAD_TASK_H

/**
* @file offload_task.h
* @brief Core 1 offload task header
* @author Dario Vazquez
*/

#define OFFLOAD_EVENT_WAIT_MS  (portMAX_DELAY)   // wait forever

extern TaskHandle_t offload_task_handle;

void dehaze_offload_task(void *arg);

#endif // OFFLOAD_TASK_H
