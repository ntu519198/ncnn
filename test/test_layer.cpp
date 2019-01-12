#include <stdio.h>
#include <map>
#include <vector>
#include <string>
#include "net.h"
#include "util.h"

bool do_test(const std::string layer_name, const ncnn::Mat& in)
{
    ncnn::Net test_net;
    std::string root_dir = "/home/alan/ncnn/";
    std::string param_path = root_dir+layer_name+".param";
    std::string bin_path = root_dir+layer_name+".bin";

    test_net.load_param(param_path.c_str());
    test_net.load_model(bin_path.c_str());

    ncnn::Extractor ex = test_net.create_extractor();

    ex.input("input", in);

    ncnn::Mat out;
    ex.extract("output", out);

    printf("Input:\n");
    print_matrix(in);

    printf("Output:\n");
    print_matrix(out);

    return true;
}

int main(int argc, char** argv) {

    if(argc != 2) {
        fprintf(stderr, "Usage: %s [layer_name]\n", argv[0]);
        exit(1);
    }
    std::string layer_name = argv[1];

    std::vector<int> shape;
    int h = 2, w = 4, c = 3;
    shape.push_back(h);
    shape.push_back(w);
    shape.push_back(c);

    float* data = new float[24];
    for(int i=0; i<24; ++i)
    {
        data[i] = float(i);
    }
    ncnn::Mat m = ncnn::Mat(w, h, c, data);
    do_test(layer_name, m);

    return 0;
}
