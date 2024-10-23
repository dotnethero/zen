using Zen.CUDA;

namespace Zen.Playground.Examples;

public static class ConvolutionExample
{
    public static void Conv1D()
    {
        using var array = HostArray.Allocate<float>(16);

        for (var i = 0; i < array.Size; ++i)
        {
            array[i] = i + 1;
        }
        
        var original = new Tensor<float>(shape: [16], array);

        var kernel = Shape.Create([3], original.Shape.Strides); // original strides = no dilation
        var shape  = Shape.Create(
            extents: [..original.Shape.Extents, ..kernel.Extents],
            strides: [..original.Shape.Strides, ..kernel.Strides]);

        var conv = new Tensor<float>(shape, array);
        
        Utils.WriteLine(original);
        Utils.WriteLine(conv);
    }

    public static void Conv2D()
    {
        using var array = HostArray.Allocate<float>(16);
        
        for (var i = 0; i < array.Size; ++i)
        {
            array[i] = i + 1;
        }

        var original = new Tensor<float>(shape: [4, 4], array);
        
        var kernel = Shape.Create([2, 2], original.Shape.Strides); // original strides = no dilation
        var shape  = Shape.Create(
            extents: [..original.Shape.Extents, ..kernel.Extents],
            strides: [..original.Shape.Strides, ..kernel.Strides]);
        
        var conv = new Tensor<float>(shape, array);
        
        Utils.WriteLine(original);
        Utils.WriteLine(conv);
    }
}