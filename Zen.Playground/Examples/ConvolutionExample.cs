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
        
        var original = Tensor.Create(shape: [16], array);

        var kernel = Shape.Create([3], original.Shape.Strides); // original strides = no dilation
        var shape  = Shape.Create(
            extents: [..original.Shape.Extents, ..kernel.Extents],
            strides: [..original.Shape.Strides, ..kernel.Strides]);

        var conv = Tensor.Create(shape, array);
        
        Utils.WriteLine(original);
        
        // original(16) =
        // [  1.00   2.00   3.00   4.00   5.00   6.00   7.00   8.00   9.00  10.00  11.00  12.00  13.00  14.00  15.00  16.00 ]
        
        Utils.WriteLine(conv);

        // conv(16,3) =
        // [[  1.00   2.00   3.00 ]
        //  [  2.00   3.00   4.00 ]
        //  [  3.00   4.00   5.00 ]
        //  [  4.00   5.00   6.00 ]
        //  [  5.00   6.00   7.00 ]
        //  [  6.00   7.00   8.00 ]
        //  [  7.00   8.00   9.00 ]
        //  [  8.00   9.00  10.00 ]
        //  [  9.00  10.00  11.00 ]
        //  [ 10.00  11.00  12.00 ]
        //  [ 11.00  12.00  13.00 ]
        //  [ 12.00  13.00  14.00 ]
        //  [ 13.00  14.00  15.00 ]
        //  [ 14.00  15.00  16.00 ]
        //  [ 15.00  16.00   0.00 ]
        //  [ 16.00   0.00   0.00 ]]
    }

    public static void Conv2D()
    {
        using var array = HostArray.Allocate<float>(16);
        
        for (var i = 0; i < array.Size; ++i)
        {
            array[i] = i + 1;
        }

        var original = Tensor.Create(shape: [4, 4], array);
        
        var kernel = Shape.Create([2, 2], original.Shape.Strides); // original strides = no dilation
        var shape  = Shape.Create(
            extents: [..original.Shape.Extents, ..kernel.Extents],
            strides: [..original.Shape.Strides, ..kernel.Strides]);
        
        var conv = Tensor.Create(shape, array);
        
        Utils.WriteLine(original);
        
        // original(4,4) =
        // [[  1.00   2.00   3.00   4.00 ]
        //  [  5.00   6.00   7.00   8.00 ]
        //  [  9.00  10.00  11.00  12.00 ]
        //  [ 13.00  14.00  15.00  16.00 ]]
        
        Utils.WriteLine(conv);
        
        // conv(4,4,2,2) =
        // [[[[  1.00   2.00 ]
        //    [  5.00   6.00 ]]
        //   [[  2.00   3.00 ]
        //    [  6.00   7.00 ]]
        //   [[  3.00   4.00 ]
        //    [  7.00   8.00 ]]
        //   [[  4.00   5.00 ]
        //    [  8.00   9.00 ]]]
        //  [[[  5.00   6.00 ]
        //    [  9.00  10.00 ]]
        //   [[  6.00   7.00 ]
        //    [ 10.00  11.00 ]]
        //   [[  7.00   8.00 ]
        //    [ 11.00  12.00 ]]
        //   [[  8.00   9.00 ]
        //    [ 12.00  13.00 ]]]
        // ....
    }
}