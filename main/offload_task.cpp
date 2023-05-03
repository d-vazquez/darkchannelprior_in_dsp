

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "freertos/event_groups.h"


#include "offload_task.h"
#include "shared_rtos.h"
#include <esp_log.h>

#include "opencv_interface.h"
#include "dehaze.h"

// QueueHandle_t       xDehazeToOffload_Queue;
// QueueHandle_t       xOffloadToDehaze_Queue;
// EventGroupHandle_t  xMatEvents;
TaskHandle_t        offload_task_handle = NULL;


using namespace std;
using namespace cv;

void dehaze_offload_task(void *arg)
{
    // Task memory
    // static xQMatMessage _message;
    // static xQMatMessage *message_tx = &_message;
    static xQMatMessage *message_rx;
    static char TAG[] = "Mat_offload";
    ESP_LOGW(TAG, "Starting task in Core %d", xPortGetCoreID());
    

    while(1)
    {
        ESP_LOGW(TAG, "waiting for message...");
        if (xQueueReceive(xDehazeToOffload_Queue, (void *)&message_rx, OFFLOAD_EVENT_WAIT_MS) == pdTRUE) 
        {
            // offload message
            Mat         *src   = message_rx->src;
            Mat         *dst   = message_rx->dst;
            Mat         *aux   = message_rx->aux;
            Scalar      *AtmL  = message_rx->atmlight;
            int         size   = message_rx->ksize;
            dehaze_op   opcode = message_rx->opcode;

            ESP_LOGW(TAG, "id:        %d", message_rx->id);
            ESP_LOGW(TAG, "Operation: %d", opcode);
            ESP_LOGW(TAG, "size:      %d", size);

            switch(opcode)
            {
                case DARK_CHANNEL_OP:
                {
                    DarkChannel(*src, 15, *dst);
                    break;
                }
                case ATMLIGHT_OP:
                {
                    *AtmL = AtmLight(*src, *dst);
                    ESP_LOGW(TAG, "Atmlight = [%f %f %f]", (*AtmL).val[0], (*AtmL).val[1], (*AtmL).val[2]);
                    
                    break;
                }
                case TRANSMITION_ESTIMATE_OP:
                {
                    TransmissionEstimate(*src, *AtmL, size, *dst);
                    break;
                }
                case TRANSMITION_REFINE_OP:
                {
                    TransmissionRefine(*src, *dst);
                    break;
                }
                case RECOVER_OP:
                {
                    Recover(*src, *aux, *dst, *AtmL, size);
                    break;
                }
                default:
                {
                    ESP_LOGW(TAG, "Invalid opcode");
                    abort();
                    break;
                }
            }

            ESP_LOGW(TAG, "Sending event");
            xEventGroupSetBits(xMatEvents, MAT_SPLIT_EVENT);
            ESP_LOGW(TAG, "Event sent");
        }
        else
        {
            ESP_LOGW(TAG, "timeout waiting for message");
        }
    }
}
