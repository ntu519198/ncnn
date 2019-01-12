#include <stdio.h>
#include <string>
#include "util.h"

void print_matrix(ncnn::Mat m) {
    int h = m.h;
    int w = m.w;
    int c = m.c;
    printf("c: %d h: %d w: %d\n", c, h, w);
    for (int q=0; q<c; ++q)
    {
        const float* ptr = m.channel(q);
        if (q == 0)
            printf("[\n");

        for(int i=0; i<h; ++i)
        {
            if (i == 0)
                printf("\t[\n");
            for(int j=0; j<w; ++j)
            {
                if (j == 0)
                    printf("\t\t[");
                printf("%f ", ptr[j]);
                if (j == w-1)
                    printf("]\n");
            }
            
            ptr += w;
            if (i == h-1)
                printf("\t]\n");
        }

        if (q == c-1)
            printf("]\n");
    }
}
