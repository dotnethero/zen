#pragma once

#include "helpers.h"
#include "kernels.h"

using Element = float;
using Layout  = cutlass::layout::TensorNHWC;
using Conv2d  = zen::device::Conv2d<
    Element, Layout,
    Element, Layout,
    Element, Layout>;

using Conv2dHandle = zen::Handle<Conv2d>;

struct Conv2dParams
{
    int input_n;
    int input_h;
    int input_w;
    int input_c;
    int filter_c; 
    int filter_h; 
    int filter_w;
    int padding_h;
    int padding_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;
    int output_h;
    int output_w;
};

extern "C"
{
    DLL_EXPORT void zenCreateConv2d(
        Conv2dHandle** handle,
        Conv2dParams* params,
        Element* input,
        Element* filter,
        Element* bias,
        Element* output)
    {
        const cutlass::Tensor4DCoord input_size(
            params->input_n,
            params->input_h,
            params->input_w,
            params->input_c);

        const cutlass::Tensor4DCoord filter_size(
            params->filter_c,
            params->filter_h,
            params->filter_w,
            params->input_c);

        const cutlass::Tensor4DCoord padding_size(
            params->padding_h,
            params->padding_h,
            params->padding_w,
            params->padding_w);

        const cutlass::MatrixCoord stride(
            params->stride_h,
            params->stride_w);
        
        const cutlass::MatrixCoord dilation(
            params->dilation_h,
            params->dilation_w);

        const cutlass::Tensor4DCoord output_size(
            params->input_n,
            params->output_h,
            params->output_w,
            params->filter_c);

        const auto split_k_slices = 1;
        const auto mode = cutlass::conv::Mode::kCrossCorrelation;

        auto problem = new cutlass::conv::Conv2dProblemSize
        {      
            input_size,
            filter_size,
            padding_size,
            stride,
            dilation,
            output_size,
            mode,
            split_k_slices
        };

        auto input_layout  = Conv2d::LayoutA::packed(problem->activation_extent());
        auto filter_layout = Conv2d::LayoutB::packed(problem->filter_extent());
        auto bias_layout   = Conv2d::LayoutC::packed(problem->output_extent());
        auto output_layout = Conv2d::LayoutC::packed(problem->output_extent());

        Conv2d* gemm = new Conv2d();
        Conv2d::Arguments args{
            *problem,
            {input,  input_layout},
            {filter, filter_layout},
            {bias,   bias_layout},
            {output, output_layout},
            {1.0f, 0.0f}};
        
        auto status = gemm->initialize(args);
        
        if (status != cutlass::Status::kSuccess) {
            delete gemm;
            return;
        }
        
        *handle = new Conv2dHandle(gemm);
    }
    
    DLL_EXPORT void zenExecuteConv2d(Conv2dHandle* handle, cudaStream_t stream) {
        handle->run(stream);
    }

    DLL_EXPORT void zenDestroyConv2d(Conv2dHandle* handle) {
        delete handle->gemm;
        delete handle;
    }
}
