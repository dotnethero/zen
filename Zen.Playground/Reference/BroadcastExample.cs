using System.Numerics;
using Zen.CUDA;

namespace Zen.Playground.Reference;

public static class BroadcastExample
{
    public static void Run()
    {
        using var a = HostTensor.Allocate<float>([4, 4]);
        using var b = HostTensor.Allocate<float>([4]);
        using var c = HostTensor.Allocate<float>([4, 4]);

        for (var i = 0; i < a.Array.Size; ++i)
        {
            a.Array[i] = i + 1;
        }
        
        for (var i = 0; i < b.Array.Size; ++i)
        {
            b.Array[i] = i + 1;
        }

        var col = b.AppendDimension();
        var row = b.PrependDimension();
        
        Sum2D(a, row, c);
        Utils.WriteLine(a);
        Utils.WriteLine(row);
        Utils.WriteLine(c);
        
        Console.WriteLine(new string('=', 40));
        
        Sum2D(a, col, c);
        Utils.WriteLine(a);
        Utils.WriteLine(col);
        Utils.WriteLine(c);
    }

    private static void Sum2D<T>(Tensor<T> a, Tensor<T> b, Tensor<T> c) where T : unmanaged, IFloatingPoint<T>
    {
        for (var i = 0; i < c.Shape.Extents[0]; ++i)
        {
            for (var j = 0; j < c.Shape.Extents[1]; ++j)
            {
                c[i, j] = a[i, j] + b[i, j];
            }
        }
    }
}