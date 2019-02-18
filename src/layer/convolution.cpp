// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "convolution.h"

#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Convolution)

Convolution::Convolution()
{
    one_blob_only = true;
    support_inplace = false;

    quantize = 0;
    dequantize = 0;
}

Convolution::~Convolution()
{
    delete quantize;
    delete dequantize;
}

int Convolution::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_w = pd.get(4, 0);
    pad_h = pd.get(14, pad_w);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    int8_scale_term = pd.get(8, 0);

    use_int8_inference = pd.use_int8_inference;

    if (int8_scale_term == 0)
        use_int8_inference = false;

    return 0;
}

int Convolution::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    if (int8_scale_term)
    {
        weight_data_int8_scale = mb.load(1, 1)[0];
        bottom_blob_int8_scale = mb.load(1, 1)[0];
    }

    bool weight_data_is_int8 = (weight_data.elemsize == (size_t)1u);
    bool weight_data_is_float32 = (weight_data.elemsize == (size_t)4u);

    if (weight_data_is_int8 && !use_int8_inference)
    {
        fprintf(stderr, "quantized int8 weight loaded but use_int8_inference disabled\n");
        return -1;
    }

    if (weight_data_is_float32 && use_int8_inference)
    {
        // quantize weight to int8
        Layer* op = ncnn::create_layer(ncnn::LayerType::Quantize);

        ncnn::ParamDict pd;
        pd.set(0, weight_data_int8_scale);// scale

        op->load_param(pd);

        Mat int8_weight_data;
        op->forward(weight_data, int8_weight_data);

        delete op;

        if (int8_weight_data.empty())
            return -100;

        weight_data = int8_weight_data;
    }

    if (use_int8_inference)
    {
        quantize = ncnn::create_layer(ncnn::LayerType::Quantize);
        {
            ncnn::ParamDict pd;
            pd.set(0, bottom_blob_int8_scale);// scale

            quantize->load_param(pd);
        }

        dequantize = ncnn::create_layer(ncnn::LayerType::Dequantize);
        {
            float top_rescale = 1.f / (bottom_blob_int8_scale * weight_data_int8_scale);

            ncnn::ParamDict pd;
            pd.set(0, top_rescale);// scale
            pd.set(1, bias_term);// bias_term
            pd.set(2, num_output);// bias_data_size

            dequantize->load_param(pd);

            ncnn::Mat weights[1];
            weights[0] = bias_data;

            dequantize->load_model(ModelBinFromMatArray(weights));
        }
    }

    return 0;
}

int Convolution::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        int num_input = weight_data_size / num_output;
        if (bottom_blob.w == num_input)
        {
            // call InnerProduct
            ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::InnerProduct);

            // set param
            ncnn::ParamDict pd;
            pd.set(0, num_output);
            pd.set(1, bias_term);
            pd.set(2, weight_data_size);
            pd.set(8, int8_scale_term);

            pd.use_int8_inference = use_int8_inference;

            op->load_param(pd);

            // set weights
            ncnn::Mat weights[4];
            weights[0] = weight_data;
            weights[1] = bias_data;

            if (int8_scale_term)
            {
                weights[2] = Mat(1, (size_t)4u, (void*)&weight_data_int8_scale);
                weights[3] = Mat(1, (size_t)4u, (void*)&bottom_blob_int8_scale);
            }

            op->load_model(ModelBinFromMatArray(weights));

            // forward
            op->forward(bottom_blob, top_blob, opt);

            delete op;

            return 0;
        }
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

//     fprintf(stderr, "Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_unbordered = bottom_blob;
    if (use_int8_inference && elemsize != 1)
    {
        Mat bottom_blob_int8;
        bottom_blob_int8.create(w, h, channels, (size_t)1u, opt.workspace_allocator);
        if (bottom_blob_int8.empty())
            return -100;

        // quantize, scale and round to nearest
        {
            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = bottom_blob_int8.allocator;

            quantize->forward(bottom_blob, bottom_blob_int8, opt_g);
        }

        bottom_blob_unbordered = bottom_blob_int8;
    }

    Mat bottom_blob_bordered = bottom_blob_unbordered;
    if (pad_w > 0 || pad_h > 0)
    {
        copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_w == -233 && pad_h == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    if (use_int8_inference)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output; p++)
        {
            int* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    int sum = 0;

                    const signed char* kptr = (const signed char*)weight_data + maxk * channels * p;

                    // channels
                    for (int q=0; q<channels; q++)
                    {
                        const Mat m = bottom_blob_bordered.channel(q);
                        const signed char* sptr = m.row<signed char>(i*stride_h) + j*stride_w;

                        for (int k = 0; k < maxk; k++)
                        {
                            int val = sptr[ space_ofs[k] ];
                            int w = kptr[k];
                            sum += val * w;
                        }

                        kptr += maxk;
                    }

                    outptr[j] = sum;
                }

                outptr += outw;
            }
        }

        // dequantize, reverse scale inplace
        {
            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = top_blob.allocator;

            dequantize->forward_inplace(top_blob, opt_g);
        }

        return 0;
    }

//////
#if 1
    if (maxk == 15 && stride_h == 1 && stride_w == 1)
    {
        int cnt = 0;
        int kernel_stride_w = (kernel_w/stride_w)*stride_w;
        int kernel_max_w = (w-kernel_w)/stride_w*stride_w;
        int max_n_w = 5;
        int out_size = outh*outw;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output; p++)
        {
            float* outptr = top_blob.channel(p);
            const float* kptr = (const float*)weight_data + 15 * channels * p;
            for (int s=0; s<out_size; s++)
            {
                outptr[s] = 0.0;
            }

            for (int q=0; q<channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                for(int i=0; i<h; i++)
                {
                    int h_idx_out = i>=kernel_h? (i-kernel_h)/stride_h+1:0;
                    int h_idx_in = i-h_idx_out*stride_h;

                    for(; h_idx_out<outh && h_idx_in>=0; h_idx_out++, h_idx_in-=stride_h)
                    {
                        int j = 0;
                        int base_out = h_idx_out*outw;
                        int base_in = h_idx_in*kernel_w;
                        int base = i*w;

                        int n_w = 1;
                        //outptr[h_idx_out*outw+w_idx_out] += m[i*w+j]*kptr[h_idx_in*9+w_idx_in];
                        for(; j<kernel_stride_w; j+=stride_w)
                        {
                            for(int k_w=0; k_w<stride_w; k_w++)
                            {
                                int global_j = j+k_w;
                                int kernel_j = global_j;
                                float val = m[base+global_j];
                                for(int kk_w=0; kk_w<n_w; kk_w++)
                                {
                                    outptr[base_out+kk_w] += val*kptr[base_in+kernel_j];
                                    kernel_j -= stride_w;
                                    //cnt++;
                                }
                            }
                            n_w++;
                        }
                        for(; j<kernel_w; j++)
                        {
                            int kernel_j = j;
                            float val = m[base+j];
                            for(int kk_w=0; kk_w<max_n_w; kk_w++)
                            {
                                outptr[base_out+kk_w] += val*kptr[base_in+kernel_j];
                                kernel_j -= stride_w;
                                //cnt++;
                            }
                        }

                        //////////////////////////////////////////
                        int w_idx_out = 1;
                        int w_idx_in = j-stride_w;
                        for(; j<kernel_max_w; j+=stride_w)
                        {
                            float val0 = m[base+j];
                            int idx = base_in+w_idx_in;
                            int idx2 = base_out+w_idx_out;

                            outptr[idx2+4] += val0*kptr[idx-4];

                            outptr[idx2+3] += val0*kptr[idx-3];

                            outptr[idx2+2] += val0*kptr[idx-2];

                            outptr[idx2+1] += val0*kptr[idx-1];

                            outptr[idx2] += val0*kptr[idx];

                            w_idx_out++;
                            //cnt+=5;
                        }

                        int ww = j+kernel_w;
                        n_w--;
                        ////////////////////////////////////////////
                        int idx_in = base_in+w_idx_in;
                        int idx_out = base_out+w_idx_out;
                        int round = 0;

                        for(; j<ww; j+=stride_w)
                        {
                            for(int k_w=0; k_w<stride_w; k_w++)
                            {
                                int global_j = j+k_w;
                                int kernel_j = 0;
                                float val = m[base+global_j];
                                for(int kk_w=0; kk_w<n_w; kk_w++)
                                {
                                    outptr[idx_out+round+kk_w] += val*kptr[idx_in+kernel_j+k_w];
                                    kernel_j -= stride_w;
                                    //cnt++;
                                }
                            }
                            round++;
                            n_w--;
                        }
                    }
                }
            }
        }
        return 0;
    }
#endif

//////
#if 1
    if (maxk == 15 && stride_h == 2 && stride_w == 2)
    {
        int cnt = 0;
        int kernel_stride_w = (kernel_w/stride_w)*stride_w;
        int kernel_max_w = (w-kernel_w)/stride_w*stride_w;
        int num_n_w = 1;
        int max_n_w = 3;
        int out_size = outh*outw;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output; p++)
        {
            float* outptr = top_blob.channel(p);
            const float* kptr = (const float*)weight_data + 15 * channels * p;

            for (int s=0; s<out_size; s++)
            {
                outptr[s] = 0.0;
            }

            for (int q=0; q<channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                for(int i=0; i<h; i++)
                {
                    int h_idx_out = i>=kernel_h? (i-kernel_h)/stride_h+1:0;
                    int h_idx_in = i-h_idx_out*stride_h;

                    for(; h_idx_out<outh && h_idx_in>=0; h_idx_out++, h_idx_in-=stride_h)
                    {
                        int j = 0;
                        int base_out = h_idx_out*outw;
                        int base_in = h_idx_in*kernel_w;
                        int base = i*w;

                        int n_w = 1;
                        //outptr[h_idx_out*outw+w_idx_out] += m[i*w+j]*kptr[h_idx_in*9+w_idx_in];
                        for(; j<kernel_stride_w; j+=stride_w)
                        {
                            for(int k_w=0; k_w<stride_w; k_w++)
                            {
                                int global_j = j+k_w;
                                int kernel_j = global_j;
                                float val = m[base+global_j];
                                for(int kk_w=0; kk_w<n_w; kk_w++)
                                {
                                    outptr[base_out+kk_w] += val*kptr[base_in+kernel_j];
                                    kernel_j -= stride_w;
                                    //cnt++;
                                }
                            }
                            n_w++;
                        }
                        for(; j<kernel_w; j++)
                        {
                            int kernel_j = j;
                            float val = m[base+j];
                            for(int kk_w=0; kk_w<max_n_w; kk_w++)
                            {
                                outptr[base_out+kk_w] += val*kptr[base_in+kernel_j];
                                kernel_j -= stride_w;
                                //cnt++;
                            }
                        }

                        //////////////////////////////////////////
                        int w_idx_out = 1;
                        int w_idx_in = j-stride_w;
                        for(; j<kernel_max_w; j+=stride_w)
                        {
                            float val0 = m[base+j];
                            float val1 = m[base+j+1];
                            int idx = base_in+w_idx_in;
                            int idx2 = base_out+w_idx_out;

                            outptr[idx2+2] += val1*kptr[idx-3];

                            outptr[idx2+1] += val0*kptr[idx-2];
                            outptr[idx2+1] += val1*kptr[idx-1];

                            outptr[idx2] += val0*kptr[idx];
                            outptr[idx2] += val1*kptr[idx+1];

                            w_idx_out++;
                            //cnt+=5;
                        }

                        int ww = j+kernel_w;
                        n_w--;
                        ////////////////////////////////////////////
                        int idx_in = base_in+w_idx_in;
                        int idx_out = base_out+w_idx_out;
                        int round = 0;

                        for(; j<ww; j+=stride_w)
                        {
                            for(int k_w=0; k_w<stride_w; k_w++)
                            {
                                int global_j = j+k_w;
                                int kernel_j = 0;
                                float val = m[base+global_j];
                                for(int kk_w=0; kk_w<n_w; kk_w++)
                                {
                                    outptr[idx_out+round+kk_w] += val*kptr[idx_in+kernel_j+k_w];
                                    kernel_j -= stride_w;
                                    //cnt++;
                                }
                            }
                            round++;
                            n_w--;
                        }
                    }
                }
            }
        }
        return 0;
    }
#endif

//////
#if 1
    if (maxk == 45 && stride_h == 4 && stride_w == 4)
    {
        int cnt = 0;
        //int kernel_stride_h = (kernel_h/stride_h)*stride_h;
        //int kernel_max_h = (h-kernel_h)/stride_h*stride_h;
        //int num_n_h = 3;
        //int max_n_h = 2;

        int kernel_stride_w = (kernel_w/stride_w)*stride_w;
        int kernel_max_w = (w-kernel_w)/stride_w*stride_w;
        int num_n_w = 3;
        int max_n_w = 3;
        int out_size = outh*outw;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output; p++)
        {
            float* outptr = top_blob.channel(p);
            const float* kptr = (const float*)weight_data + 45 * channels * p;

            for (int s=0; s<out_size; s++)
            {
                outptr[s] = 0.0;
            }

            for (int q=0; q<channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                for(int i=0; i<h; i++)
                {
                    int h_idx_out = i>=kernel_h? (i-kernel_h)/stride_h+1:0;
                    int h_idx_in = i-h_idx_out*stride_h;

                    for(; h_idx_out<outh && h_idx_in>=0; h_idx_out++, h_idx_in-=stride_h)
                    {
                        int j = 0;
                        int base_out = h_idx_out*outw;
                        int base_in = h_idx_in*kernel_w;
                        int base = i*w;

                        int n_w = 1;
                        //outptr[h_idx_out*outw+w_idx_out] += m[i*w+j]*kptr[h_idx_in*9+w_idx_in];
                        for(; j<kernel_stride_w; j+=stride_w)
                        {
                            for(int k_w=0; k_w<stride_w; k_w++)
                            {
                                int global_j = j+k_w;
                                int kernel_j = global_j;
                                float val = m[base+global_j];
                                for(int kk_w=0; kk_w<n_w; kk_w++)
                                {
                                    outptr[base_out+kk_w] += val*kptr[base_in+kernel_j];
                                    kernel_j -= stride_w;
                                    //cnt++;
                                }
                            }
                            n_w++;
                        }
                        for(; j<kernel_w; j++)
                        {
                            int kernel_j = j;
                            float val = m[base+j];
                            for(int kk_w=0; kk_w<max_n_w; kk_w++)
                            {
                                outptr[base_out+kk_w] += val*kptr[base_in+kernel_j];
                                kernel_j -= stride_w;
                                //cnt++;
                            }
                        }

                        //////////////////////////////////////////
                        int w_idx_out = 1;
                        int w_idx_in = j-stride_w;
                        for(; j<kernel_max_w; j+=stride_w)
                        {
                            float val0 = m[base+j];
                            float val1 = m[base+j+1];
                            float val2 = m[base+j+2];
                            float val3 = m[base+j+3];
                            int idx = base_in+w_idx_in;
                            int idx2 = base_out+w_idx_out;

                            outptr[idx2+2] += val3*kptr[idx-5];

                            outptr[idx2+1] += val0*kptr[idx-4];
                            outptr[idx2+1] += val1*kptr[idx-3];
                            outptr[idx2+1] += val2*kptr[idx-2];
                            outptr[idx2+1] += val3*kptr[idx-1];

                            outptr[idx2] += val0*kptr[idx];
                            outptr[idx2] += val1*kptr[idx+1];
                            outptr[idx2] += val2*kptr[idx+2];
                            outptr[idx2] += val3*kptr[idx+3];

                            w_idx_out++;
                            //cnt+=9;
                            /*
                            //float val[stride_w];
                            int idx_in = base_in+w_idx_in;
                            int idx_out = base_out+w_idx_out;
                            int k_j = -(max_n_w-2)*stride_w-(stride_w-num_n_w);
                            int out_j=max_n_w-1;
                            for(int k_w=0; k_w<stride_w; k_w++)
                            {
                                val[k_w] = m[base+j+k_w];
                            }
                            for(int m_j=num_n_w; m_j<stride_w; m_j++)
                            {
                                outptr[idx_out+out_j] += val[m_j]*kptr[idx_in+k_j];
                                k_j++;
                                //cnt++;
                            }

                            out_j--;
                            for(; out_j>=0; out_j--)
                            {
                                for(int m_j=0; m_j<stride_w; m_j++)
                                {
                                    outptr[idx_out+out_j] += val[m_j]*kptr[idx_in+k_j];
                                    k_j++;
                                    //cnt++;
                                }
                            }
                            w_idx_out++;
                            */
                        }

                        int ww = j+kernel_w;
                        n_w--;
                        ////////////////////////////////////////////
                        int idx_in = base_in+w_idx_in;
                        int idx_out = base_out+w_idx_out;
                        int round = 0;

                        for(; j<ww; j+=stride_w)
                        {
                            for(int k_w=0; k_w<stride_w; k_w++)
                            {
                                int global_j = j+k_w;
                                int kernel_j = 0;
                                float val = m[base+global_j];
                                for(int kk_w=0; kk_w<n_w; kk_w++)
                                {
                                    outptr[idx_out+round+kk_w] += val*kptr[idx_in+kernel_j+k_w];
                                    kernel_j -= stride_w;
                                    //cnt++;
                                }
                            }
                            round++;
                            n_w--;
                        }
                    }
                }
            }
        }
        return 0;
    }
#endif

#if 0
    if (maxk == 45 && stride_h == 4 && stride_w == 4)
    {
        int cnt = 0;

        int kernel_stride_h = (kernel_h/stride_h)*stride_h;
        int kernel_max_h = (h-kernel_h)/stride_h*stride_h;
        int num_n_h = 3;
        int max_n_h = 2;

        int kernel_stride_w = (kernel_w/stride_w)*stride_w;
        int kernel_max_w = (w-kernel_w)/stride_w*stride_w;
        int num_n_w = 3;
        int max_n_w = 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output; p++)
        {
            float* outptr = top_blob.channel(p);
            const float* kptr = (const float*)weight_data + 45 * channels * p;

            for (int q=0; q<channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                int i = 0;
                int n_h = 1;

                for(; i<kernel_stride_h; i+=stride_h)
                {
                    for(int k_h=0; k_h<stride_h; k_h++)
                    {
                        for(int kk_h=0; kk_h<n_h; kk_h++)
                        {
                            //here
                            int j = 0;
                            int n_w = 1;

                            for(; j<kernel_stride_w; j+=stride_w)
                            {
                                for(int k_w=0; k_w<stride_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<n_w; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++;
                                    }
                                }
                                n_w++;
                            }
                            for(; j<kernel_w; j++)
                            {
                                for(int kk_w=0; kk_w<max_n_w; kk_w++)
                                {
                                    outptr[0]++;
                                    cnt++;
                                }
                            }

                            //////////////////////////////////////////
                            for(; j<kernel_max_w; j+=stride_w)
                            {
                                int k_w = 0;
                                for(; k_w<num_n_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<max_n_w-1; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++;
                                    }
                                }
                                for(; k_w<stride_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<max_n_w; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++;
                                    }
                                }
                            }

                            int ww = j+kernel_w;
                            n_w--;
                            ////////////////////////////////////////////
                            for(; j<ww; j+=stride_w)
                            {
                                for(int k_w=0; k_w<stride_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<n_w; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++;
                                    }
                                }
                                n_w--; 
                            }
                        }
                    }
                    n_h++;
                }

                for(; i<kernel_h; i++)
                {
                    for(int kk_h=0; kk_h<max_n_h; kk_h++)
                    {
                        //here
                        int j = 0;
                        int n_w = 1;

                        for(; j<kernel_stride_w; j+=stride_w)
                        {
                            for(int k_w=0; k_w<stride_w; k_w++)
                            {
                                for(int kk_w=0; kk_w<n_w; kk_w++)
                                {
                                    outptr[0]++;
                                    cnt++;
                                }
                            }
                            n_w++;
                        }
                        for(; j<kernel_w; j++)
                        {
                            for(int kk_w=0; kk_w<max_n_w; kk_w++)
                            {
                                outptr[0]++;
                                cnt++;
                            }
                        }

                        //////////////////////////////////////////
                        for(; j<kernel_max_w; j+=stride_w)
                        {
                            int k_w = 0;
                            for(; k_w<num_n_w; k_w++)
                            {
                                for(int kk_w=0; kk_w<max_n_w-1; kk_w++)
                                {
                                    outptr[0]++;
                                    cnt++;
                                }
                            }
                            for(; k_w<stride_w; k_w++)
                            {
                                for(int kk_w=0; kk_w<max_n_w; kk_w++)
                                {
                                    outptr[0]++;
                                    cnt++;
                                }
                            }
                        }

                        int ww = j+kernel_w;
                        n_w--;
                        ////////////////////////////////////////////
                        for(; j<ww; j+=stride_w)
                        {
                            for(int k_w=0; k_w<stride_w; k_w++)
                            {
                                for(int kk_w=0; kk_w<n_w; kk_w++)
                                {
                                    outptr[0]++;
                                    cnt++;
                                }
                            }
                            n_w--; 
                        }
                    }
                }

                //########################################
                for(; i<kernel_max_h; i+=stride_h)
                {
                    int k_h = 0; 
                    for(; k_h<num_n_h; k_h++)
                    {
                        for(int kk_h=0; kk_h<max_n_h-1; kk_h++)
                        {
                            //here
                            int j = 0;
                            int n_w = 1;

                            for(; j<kernel_stride_w; j+=stride_w)
                            {
                                for(int k_w=0; k_w<stride_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<n_w; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++; 
                                    }
                                }
                                n_w++;
                            }
                            for(; j<kernel_w; j++)
                            {
                                for(int kk_w=0; kk_w<max_n_w; kk_w++)
                                {
                                    outptr[0]++;
                                    cnt++; 
                                }
                            }

                            //////////////////////////////////////////
                            for(; j<kernel_max_w; j+=stride_w)
                            {
                                int k_w = 0;
                                for(; k_w<num_n_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<max_n_w-1; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++;
                                    }
                                }
                                for(; k_w<stride_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<max_n_w; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++;
                                    }
                                }
                            }

                            int ww = j+kernel_w;
                            n_w--;
                            ////////////////////////////////////////////
                            for(; j<ww; j+=stride_w)
                            {
                                for(int k_w=0; k_w<stride_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<n_w; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++;
                                    }
                                }
                                n_w--; 
                            }
                        }
                    }
                    for(; k_h<stride_h; k_h++)
                    {
                        for(int kk_h=0; kk_h<max_n_h; kk_h++)
                        {
                            //here
                            int j = 0;
                            int n_w = 1;

                            for(; j<kernel_stride_w; j+=stride_w)
                            {
                                for(int k_w=0; k_w<stride_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<n_w; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++; 
                                    }
                                }
                                n_w++;
                            }
                            for(; j<kernel_w; j++)
                            {
                                for(int kk_w=0; kk_w<max_n_w; kk_w++)
                                {
                                    outptr[0]++;
                                    cnt++; 
                                }
                            }

                            //////////////////////////////////////////
                            for(; j<kernel_max_w; j+=stride_w)
                            {
                                int k_w = 0;
                                for(; k_w<num_n_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<max_n_w-1; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++;
                                    }
                                }
                                for(; k_w<stride_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<max_n_w; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++;
                                    }
                                }
                            }

                            int ww = j+kernel_w;
                            n_w--;
                            ////////////////////////////////////////////
                            for(; j<ww; j+=stride_w)
                            {
                                for(int k_w=0; k_w<stride_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<n_w; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++;
                                    }
                                }
                                n_w--; 
                            }
                        }
                    }
                }

                int hh = i+kernel_h;
                n_h--;

                //########################################
                for(; i<hh; i+=stride_h)
                {
                    for(int k_h=0; k_h<stride_h; k_h++)
                    {
                        for(int kk_h=0; kk_h<n_h; kk_h++)
                        {
                            //here
                            int j = 0;
                            int n_w = 1;

                            for(; j<kernel_stride_w; j+=stride_w)
                            {
                                for(int k_w=0; k_w<stride_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<n_w; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++; 
                                    }
                                }
                                n_w++;
                            }
                            for(; j<kernel_w; j++)
                            {
                                for(int kk_w=0; kk_w<max_n_w; kk_w++)
                                {
                                    outptr[0]++;
                                    cnt++; 
                                }
                            }

                            //////////////////////////////////////////
                            for(; j<kernel_max_w; j+=stride_w)
                            {
                                int k_w = 0;
                                for(; k_w<num_n_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<max_n_w-1; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++;
                                    }
                                }
                                for(; k_w<stride_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<max_n_w; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++;
                                    }
                                }
                            }

                            int ww = j+kernel_w;
                            n_w--;
                            ////////////////////////////////////////////
                            for(; j<ww; j+=stride_w)
                            {
                                for(int k_w=0; k_w<stride_w; k_w++)
                                {
                                    for(int kk_w=0; kk_w<n_w; kk_w++)
                                    {
                                        outptr[0]++;
                                        cnt++;
                                    }
                                }
                                n_w--; 
                            }
                        }
                    }
                    n_h--;
                }
            }
        }
        return 0;
    }
#endif

#if 0
    if (maxk == 45 && stride_h == 4 && stride_w == 4)
    {
        int cnt = 0;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output; p++)
        {
            float* outptr = top_blob.channel(p);
            const float* kptr = (const float*)weight_data + 45 * channels * p;

            for (int q=0; q<channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                for(int i=0; i<h; i++)
                {
                    int h_idx_out = i>=5? (i-5)/stride_h+1:0;
                    int h_idx_in = i-h_idx_out*stride_h;

                    for(; h_idx_out<outh && h_idx_in>=0; h_idx_out++, h_idx_in-=stride_h)
                    {
                        int j = 0;
                        int base_out = h_idx_out*outw;
                        int base_in = h_idx_in*9;
                        int base = i*w;
                        //////////////////////////////////

                        //outptr[h_idx_out*outw+w_idx_out] += m[i*w+j]*kptr[h_idx_in*9+w_idx_in];
                        for(; j<4; j++)
                        {
                            outptr[base_out] += m[base+j]*kptr[base_in+j];
                            //cnt++;
                        }
                        for(; j<8; j++)
                        {
                            float val = m[base+j];
                            outptr[base_out] += val*kptr[base_in+j];
                            outptr[base_out+1] += val*kptr[base_in+j-stride_w];
                            //cnt += 2;
                        }
                        float val = m[base+j];
                        outptr[base_out] += val*kptr[base_in+8];
                        outptr[base_out+1] += val*kptr[base_in+4];
                        outptr[base_out+2] += val*kptr[base_in];
                        //cnt += 3;
                        j++;

                        //////////////////////////////////
                        int ww = (outw-1)*stride_w;
                        int w_idx_out = 1;
                        int w_idx_in = j-stride_w;
                        for(; j<ww; j+=4)
                        {
                            float val0 = m[base+j];
                            float val1 = m[base+j+1];
                            float val2 = m[base+j+2];
                            float val3 = m[base+j+3];
                            int idx = base_in+w_idx_in;
                            int idx2 = base_out+w_idx_out;

                            outptr[idx2+2] += val3*kptr[idx-5];

                            outptr[idx2+1] += val0*kptr[idx-4];
                            outptr[idx2+1] += val1*kptr[idx-3];
                            outptr[idx2+1] += val2*kptr[idx-2];
                            outptr[idx2+1] += val3*kptr[idx-1];

                            outptr[idx2] += val0*kptr[idx];
                            outptr[idx2] += val1*kptr[idx+1];
                            outptr[idx2] += val2*kptr[idx+2];
                            outptr[idx2] += val3*kptr[idx+3];

                            //cnt+=9;
                            w_idx_out++;
                            /*
                            val = m[base+j];
                            outptr[base_out+w_idx_out] += val*kptr[base_in+w_idx_in];
                            outptr[base_out+w_idx_out+1] += val*kptr[base_in+w_idx_in-stride_w];

                            val = m[base+j+1];
                            outptr[base_out+w_idx_out] += val*kptr[base_in+w_idx_in+1];
                            outptr[base_out+w_idx_out+1] += val*kptr[base_in+w_idx_in-stride_w+1];

                            val = m[base+j+2];
                            outptr[base_out+w_idx_out] += val*kptr[base_in+w_idx_in+2];
                            outptr[base_out+w_idx_out+1] += val*kptr[base_in+w_idx_in-stride_w+2];

                            val = m[base+j+3];
                            outptr[base_out+w_idx_out] += val*kptr[base_in+w_idx_in+3];
                            outptr[base_out+w_idx_out+1] += val*kptr[base_in+w_idx_in-stride_w+3];
                            outptr[base_out+w_idx_out+2] += val*kptr[base_in+w_idx_in-stride_w*2+3];
                            //cnt+=9;
                            w_idx_out++;
                            */
                        }
                        ////////////////////////////////////////

                        //outptr[h_idx_out*outw+w_idx_out] += m[i*w+j]*kptr[h_idx_in*9+w_idx_in];
                        float val0 = m[base+j];
                        float val1 = m[base+j+1];
                        float val2 = m[base+j+2];
                        float val3 = m[base+j+3];
                        float val4 = m[base+j+4];
                        float val5 = m[base+j+5];
                        float val6 = m[base+j+6];
                        float val7 = m[base+j+7];
                        int idx = base_in+w_idx_in;
                        int idx2 = base_out+w_idx_out;

                        outptr[idx2+1] += val0*kptr[idx-4];
                        outptr[idx2+1] += val1*kptr[idx-3];
                        outptr[idx2+1] += val2*kptr[idx-2];
                        outptr[idx2+1] += val3*kptr[idx-1];
                        outptr[idx2+1] += val4*kptr[idx];
                        outptr[idx2+1] += val5*kptr[idx+1];
                        outptr[idx2+1] += val6*kptr[idx+2];
                        outptr[idx2+1] += val7*kptr[idx+3];

                        outptr[idx2] += val0*kptr[idx];
                        outptr[idx2] += val1*kptr[idx+1];
                        outptr[idx2] += val2*kptr[idx+2];
                        outptr[idx2] += val3*kptr[idx+3];

                        //cnt += 12;
                        //j += 4;
                        //w_idx_out++;
                        //cnt+=8;

                        //val0 = m[base+j];
                        //val1 = m[base+j+1];
                        //val2 = m[base+j+2];
                        //val3 = m[base+j+3];

                        //outptr[base_out+w_idx_out] += val0*kptr[base_in+w_idx_in];
                        //outptr[base_out+w_idx_out] += val1*kptr[base_in+w_idx_in+1];
                        //outptr[base_out+w_idx_out] += val2*kptr[base_in+w_idx_in+2];
                        //outptr[base_out+w_idx_out] += val3*kptr[base_in+w_idx_in+3];
                        //cnt += 4
                    }
                }
            }
        }
        //printf("%d\n", cnt);
        return 0;
    }
#endif
    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<num_output; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_term)
                    sum = bias_data[p];

                const float* kptr = (const float*)weight_data + maxk * channels * p;

                // channels
                for (int q=0; q<channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float val = sptr[ space_ofs[k] ]; // 20.72
                        float w = kptr[k];
                        sum += val * w; // 41.45
                    }
                    kptr += maxk;
                }

                outptr[j] = sum;
            }
            outptr += outw;
        }
    }

    return 0;
}

} // namespace ncnn
