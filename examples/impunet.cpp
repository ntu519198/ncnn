#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "net.h"

static void multiply(ncnn::Mat& m, float multiplier) {
    for(int i=0; i<(int)m.total(); i++) {
        ((float*)m.data)[i] *= multiplier;
    }
}

static int detect_impunet(const char* param_path, const char* bin_path, const cv::Mat& bgr, cv::Mat& bgr_out)
{
    ncnn::Net impunet;
    impunet.load_param(param_path);
    impunet.load_model(bin_path);
    //impunet.load_param("impunet_v1.2.param");
    //impunet.load_model("impunet_v1.2.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 2048, 1024);
    multiply(in, 1.0/255);

    ncnn::Extractor ex = impunet.create_extractor();

    ex.input("input", in);

    ncnn::Mat out;
    ex.extract("output", out);
    multiply(out, 255);

    unsigned char* pixels = new unsigned char[out.total()];
    out.to_pixels(pixels, ncnn::Mat::PIXEL_BGR);

    bgr_out = cv::Mat(out.h, out.w, CV_8UC3, pixels);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s [parampath] [binpath] [imagepath] [outputpath]\n", argv[0]);
        return -1;
    }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const char* imagepath = argv[3];

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    cv::Mat m_out;
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    detect_impunet(param_path, bin_path, m, m_out);

    const char* outputpath = argv[4];
    int w = m_out.cols;
    int h = m_out.rows;
    int c = m_out.channels();
    cv::imwrite(outputpath, m_out);

    return 0;
}
