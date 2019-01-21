#!/bin/bash

function exe_print_cmd
{
    cmd_string=$1
    echo ${cmd_string}
    eval ${cmd_string}
}

if [ $# -ne 2 ];
then
    echo "Usage:  ${0} [pb_path] [output_name]"
    exit
fi

pb_path=$1
name=$2
tf2ncnn_exe="./build/tools/tensorflow/tensorflow2ncnn"
ncnn2mem_exe="./build/tools/ncnn2mem"
param_path="./model/${name}.param"
bin_path="./model/${name}.bin"
id_path="./mem_model/${name}.id.h"
mem_path="./mem_model/${name}.mem.h"
param_bin_path="./model/${name}.param_keepname.bin"
new_param_bin_path="./mem_model/${name}.param_keepname.bin"

tf2ncnn="${tf2ncnn_exe} ${pb_path} ${param_path} ${bin_path}"
ncnn2mem="${ncnn2mem_exe} ${param_path} ${bin_path} ${id_path} ${mem_path} keepname"
copy_param_bin="cp ${param_bin_path} ${new_param_bin_path}"

exe_print_cmd "${tf2ncnn}"
exe_print_cmd "${ncnn2mem}"
exe_print_cmd "${copy_param_bin}"
