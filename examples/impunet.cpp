#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "net.h"

static void multiply(ncnn::Mat& m, float multiplier) {
    for(int i=0; i<(int)m.total(); i++) {
        ((float*)m.data)[i] *= multiplier;
    }
}

static int detect_impunet(const char* param_path, const char* bin_path,
                          const cv::Mat& m_in, cv::Mat& m_out,
                          const char* input_name, const char* output_name,
                          const bool isbin, const bool keepname)
{
    const int h = m_in.rows;
    const int w = m_in.cols;
    const int color_format = ncnn::Mat::PIXEL_RGB;

    ncnn::Net impunet;
    impunet.load_param(param_path, isbin, keepname);
    impunet.load_model(bin_path);

    // Convert BGR image to RGB
    cvtColor(m_in, m_in, CV_BGR2RGB);

    ncnn::Mat in = ncnn::Mat::from_pixels(m_in.data, color_format, w, h);
    multiply(in, 1.0/255);

    ncnn::Extractor ex = impunet.create_extractor();
    ncnn::Mat out;

    if(isbin)
    {
        ex.input(0, in);
        ex.extract(atoi(output_name), out);
    }
    else
    {
        ex.input(input_name, in);
        ex.extract(output_name, out);
    }

    multiply(out, 255);

    unsigned char* pixels = new unsigned char[out.total()];
    out.to_pixels(pixels, ncnn::Mat::PIXEL_RGB);

    m_out = cv::Mat(out.h, out.w, CV_8UC3, pixels);

    // Convert RGB image to BGR
    cvtColor(m_out, m_out, CV_RGB2BGR);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 7 && argc != 8 && argc !=9)
    {
        fprintf(stderr, "Usage: %s [parampath] [binpath] "
                        "[imagepath] [outputpath] "
                        "[input_name] [output_name] "
                        "[isbin] [keepname]\n", argv[0]);
        return -1;
    }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const char* imagepath = argv[3];
    const char* outputpath = argv[4];
    const char* input_name = argv[5];
    const char* output_name = argv[6];
    bool isbin = false;
    bool keepname = false;

    if (argc >= 8)
    {
        isbin = !strcmp(argv[7], "isbin");
        if (argc == 9)
        {
            keepname = !strcmp(argv[8], "keepname");
        }
    }

    const std::string outputpath_str = std::string(outputpath);
    const std::string extension =
        outputpath_str.substr(outputpath_str.length()-4);
    if (extension != ".png")
    {
        fprintf(stderr, "Output path must be png file");
        exit(1);
    }

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    cv::Mat m_out;
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    detect_impunet(param_path, bin_path, m, m_out,
                   input_name, output_name,
                   isbin, keepname);

    cv::imwrite(outputpath, m_out);
    return 0;
}
