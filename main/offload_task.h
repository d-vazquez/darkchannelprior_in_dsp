
#ifndef OFFLOAD_TASK_H
#define OFFLOAD_TASK_H


// #define OFFLOAD_EVENT_WAIT_MS  (portMAX_DELAY)   // wait forever
#define OFFLOAD_EVENT_WAIT_MS  (( TickType_t )10)   // ticks   

extern TaskHandle_t offload_task_handle;

void dehaze_offload_task(void *arg);


#endif // OFFLOAD_TASK_H
