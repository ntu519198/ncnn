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

#include "interp.h"
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(Interp);

Interp::Interp()
{
    one_blob_only = true;
}

int Interp::load_param(const ParamDict& pd)
{
    resize_type = pd.get(0, 0);
    height_scale = pd.get(1, 1.f);
    width_scale = pd.get(2, 1.f);
    output_height = pd.get(3, 0);
    output_width = pd.get(4, 0);

    return 0;
}

int Interp::forward(const Mat &bottom_blob, Mat &top_blob, const Option& opt) const
{
    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int c = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int oh = output_height;
    int ow = output_width;
    float hscale = height_scale;
    float wscale = width_scale;
    if (bottom_blob.dims == 1)
    {
        h = 1;
        w = 1;
        c = bottom_blob.w;
    }
    if (oh == 0 || ow == 0)
    {
        //oh = h * height_scale;
        //ow = w * width_scale;
        oh = h * hscale;
        ow = w * wscale;
    }
    else
    {
        hscale = int(oh/h);
        wscale = int(ow/w);
    }
    if (oh == h && ow == w)
    {
        top_blob = bottom_blob;
        return 0;
    }
    top_blob.create(ow, oh, c, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (bottom_blob.dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            Mat top_blob_c = top_blob.channel(q);
            const float *ptr = ((const float*)bottom_blob.data + q);
            top_blob_c.fill(*ptr);
        }
        return 0;
    }

    if (resize_type == 1)//nearest
    {
        if(hscale == 4 && wscale == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < c; ++q)
            {
                const float *ptr = bottom_blob.channel(q);
                float *output_ptr = top_blob.channel(q);
                int in_y = 0;
                for (int y = 0; y < oh; y += 4)
                {
                    for (int i=0; i < 4; ++i)
                    {
                        const float* tmp_ptr = ptr+in_y;
                        for (int x = 0; x < ow; x += 4)
                        {
                            float tmp = *tmp_ptr;
                            output_ptr[x] = tmp;
                            output_ptr[x+1] = tmp;
                            output_ptr[x+2] = tmp;
                            output_ptr[x+3] = tmp;
                            ++tmp_ptr;
                        }
                        output_ptr += ow;
                    }
                    in_y += w;
                }
            }
            return 0;
        }

        if(hscale == 2 && wscale == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < c; ++q)
            {
                const float *ptr = bottom_blob.channel(q);
                float *output_ptr = top_blob.channel(q);
                int in_y = 0;
                for (int y = 0; y < oh; y += 2)
                {
                    for (int i=0; i < 2; ++i)
                    {
                        const float* tmp_ptr = ptr+in_y;
                        for (int x = 0; x < ow; x += 2)
                        {
                            float tmp = *tmp_ptr;
                            output_ptr[x] = tmp;
                            output_ptr[x+1] = tmp;
                            ++tmp_ptr;
                        }
                        output_ptr += ow;
                    }
                    in_y += w;
                }
            }
            return 0;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            const float *ptr = bottom_blob.channel(q);
            float *output_ptr = top_blob.channel(q);
            for (int y = 0; y < oh; ++y)
            {
                const int in_y = std::min((int) (y / hscale), (h - 1));
                for (int x = 0; x < ow; x += wscale)
                {
                    const int in_x = std::min((int) (x / wscale), (w - 1));
                    const float tmp = ptr[in_y * w + in_x];
                    for(int i=0; i<wscale; i++)
                    {
                        output_ptr[x+i] = tmp;
                    }
                }
                output_ptr += ow;
            }
        }

        return 0;

    }
    else if (resize_type == 2)// bilinear
    {
        resize_bilinear(bottom_blob, top_blob, ow, oh);
        return 0;

    }
    else
    {
        fprintf(stderr, "unsupported resize type %d %d %d\n", resize_type, oh, ow);
        return -233;
    }
}


} // namespace ncnn
