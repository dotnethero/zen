using Zen.CUDA;
using Zen.CUDA.Interop;
using Zen.CUDA.Wrappers;
using Zen.Playground.Examples;

namespace Zen.Playground;

internal static unsafe class Program
{
    public static void Main()
    {
        // CopyExample.Run();
        // GraphExample.Run();
        // EventExample.Run();
        // SliceExample.Run();
        // ConvolutionExample.Conv1D();
        // ConvolutionExample.Conv2D();

        Run();
    }

    private static void Run()
    {
        nuint size;
        zenConv2dPlan* plan;
        zenConv2dParams parameters = new zenConv2dParams
        {
            input_n = 1,
            input_h = 8,
            input_w = 8,
            input_c = 4,
            filter_c = 4,
            filter_h = 3,
            filter_w = 3,
            padding_h = 0,
            padding_w = 0,
            stride_h = 1,
            stride_w = 1,
            dilation_h = 1,
            dilation_w = 1,
            output_h = 6,
            output_w = 6,
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

        using var host_input = HostArray.Allocate<float>(input_shape.Size);
        using var host_filter = HostArray.Allocate<float>(filter_shape.Size);
        using var host_output = HostArray.Allocate<float>(output_shape.Size);
        
        for (var i = 0; i < host_input.Size; ++i)
        {
            host_input[i] = 1;
        }

        for (var i = 0; i < host_filter.Size; ++i)
        {
            host_filter[i] = 1f / filter_shape[1..].Size; // avg
            host_filter[i] = 1f; // sum
        }
        
        for (var i = 0; i < host_output.Size; ++i)
        {
            host_output[i] = float.NaN;
        }

        using var input_array  = DeviceArray.Allocate<float>(input_shape.Size);
        using var output_array = DeviceArray.Allocate<float>(filter_shape.Size);
        using var filter_array = DeviceArray.Allocate<float>(output_shape.Size);
        
        host_input.CopyTo(input_array, CudaStream.Default);
        host_filter.CopyTo(filter_array, CudaStream.Default);

        LibZen.zenCreateConv2dPlan(&plan, &parameters);
        LibZen.zenGetConv2dWorkspaceSize(plan, &size);

        var ws = DeviceArray.Allocate<byte>((int)size);
        
        LibZen.zenExecuteConv2d(
            plan,
            input_array.Pointer,
            filter_array.Pointer,
            output_array.Pointer,
            1.0f,
            0.0f,
            output_array.Pointer,
            ws.Pointer,
            null);

        CudaStream.Default.Synchronize();
        output_array.CopyTo(host_output, CudaStream.Default);
        
        var input_tensor = Tensor.Create(input_shape, host_input);
        var filter_tensor = Tensor.Create(filter_shape, host_filter);
        var output_tensor = Tensor.Create(output_shape, host_output);
        
        Utils.WriteLine(input_tensor.Slice(0, .., .., 0));
        Utils.WriteLine(filter_tensor.Slice(0, .., .., 0));
        Utils.WriteLine(output_tensor);
    }
}