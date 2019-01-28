#ifndef UTIL_H
#define UTIL_H

#include "mat.h"

#ifdef __ANDROID__
#define LOG_TAG "IMPUNET/Benchmark"
#include <android/log.h>
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(...) fprintf(stderr, __VA_ARGS__); fprintf(stderr,"\n");
#endif

void print_matrix(ncnn::Mat m);

#endif
