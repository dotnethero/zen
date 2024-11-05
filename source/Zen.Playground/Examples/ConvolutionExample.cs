using Zen.CUDA;
using Zen.CUDA.Interop;
using Zen.CUDA.Wrappers;
using static Zen.CUDA.Interop.CudaImports;

namespace Zen.Playground.Examples;

public static unsafe class ConvolutionExample
{
    public static void Run()
    {
        nuint size;
        zenConv2dHandle* plan;
        zenConv2dParams parameters = new zenConv2dParams
        {
            input_n = 1,
            input_h = 8,
            input_w = 8,
            input_c = 1,
            filter_c = 1,
            filter_h = 3,
            filter_w = 3,
            padding_h = 1,
            padding_w = 1,
            stride_h = 1,
            stride_w = 1,
            dilation_h = 1,
            dilation_w = 1,
            output_h = 8,
            output_w = 8,
        };

        Shape input_shape = [
            parameters.input_n,
            parameters.input_h,
            parameters.input_w,
            parameters.input_c];
        
        Shape filter_shape = [
            parameters.filter_c,
            parameters.filter_h,
            parameters.filter_w,
            parameters.input_c];
        
        Shape output_shape = [
            parameters.input_n,
            parameters.output_h,
            parameters.output_w,
            parameters.filter_c];

        using var host_input  = HostTensor.Allocate<float>(input_shape);
        using var host_filter = HostTensor.Allocate<float>(filter_shape);
        using var host_output = HostTensor.Allocate<float>(output_shape);
        
        using var input  = DeviceTensor.Allocate<float>(input_shape);
        using var filter = DeviceTensor.Allocate<float>(filter_shape);
        using var output = DeviceTensor.Allocate<float>(output_shape);
        
        for (var i = 0; i < host_input.Array.Size; ++i)
        {
            host_input.Array[i] = 1;
        }

        for (var i = 0; i < host_filter.Array.Size; ++i)
        {
            host_filter.Array[i] = 1f / filter_shape[1..].Size; // avg
            host_filter.Array[i] = 1f; // sum
        }
        
        for (var i = 0; i < host_output.Array.Size; ++i)
        {
            host_output.Array[i] = float.NaN;
        }

        using var stream = new CudaStream();

        host_input.CopyTo(input, stream);
        host_filter.CopyTo(filter, stream);
        stream.Synchronize();

        zenCreateConv2d(
            &plan,
            &parameters,
            input.Array.Pointer,
            filter.Array.Pointer,
            output.Array.Pointer,
            output.Array.Pointer);
        
        stream.BeginCapture();
        
        zenExecuteConv2d(
            plan,
            stream.Pointer);

        using var graph = stream.EndCapture();
        using var graphInstance = graph.CreateInstance();

        for (var i = 0; i < 5; i++)
        {
            graphInstance.Launch(stream);
        }
        
        stream.Synchronize();
        output.CopyTo(host_output);
        
        Utils.WriteLine(host_input.Slice(0, .., .., 0));
        Utils.WriteLine(host_filter.Slice(0, .., .., 0));
        Utils.WriteLine(host_output.Slice(0, .., .., 0));
    }
}