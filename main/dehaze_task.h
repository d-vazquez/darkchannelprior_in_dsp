
#ifndef DEHAZE_TASK_H
#define DEHAZE_TASK_H

void dehaze_task(void *arg);
void write_MAT_to_file(const char *file_dst, cv::Mat &src);

#endif // DEHAZE_TASK_H
