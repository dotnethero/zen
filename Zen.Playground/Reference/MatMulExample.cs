using System.Numerics;
using Zen.CUDA;

namespace Zen.Playground.Reference;

public static class MatMulExample
{
    public static void Run()
    {
        using var a = HostTensor.Allocate<float>([3, 4]);
        using var b = HostTensor.Allocate<float>([5, 4]); // b is allocated transposed
        using var c = HostTensor.Allocate<float>([3, 5]);

        for (var i = 0; i < a.Array.Size; ++i)
        {
            a.Array[i] = i + 1;
        }
        
        for (var i = 0; i < b.Array.Size; ++i)
        {
            b.Array[i] = i % 3;
        }
        
        MatMul(a, b.Transpose(), c); // permutations are non-allocative

        Utils.WriteLine(a);
        Utils.WriteLine(b);
        Utils.WriteLine(c);
        
        // a(3,4):(4,1) =
        // [[  1.00   2.00   3.00   4.00 ]
        //  [  5.00   6.00   7.00   8.00 ]
        //  [  9.00  10.00  11.00  12.00 ]]
        
        // b(5,4):(4,1) =
        // [[  0.00   1.00   2.00   0.00 ]
        //  [  1.00   2.00   0.00   1.00 ]
        //  [  2.00   0.00   1.00   2.00 ]
        //  [  0.00   1.00   2.00   0.00 ]
        //  [  1.00   2.00   0.00   1.00 ]]
        
        // c(3,5):(5,1) =
        // [[  8.00   9.00  13.00   8.00   9.00 ]
        //  [ 20.00  25.00  33.00  20.00  25.00 ]
        //  [ 32.00  41.00  53.00  32.00  41.00 ]]
    }

    private static void MatMul<T>(Tensor<T> a, Tensor<T> b, Tensor<T> c) where T : unmanaged, IFloatingPoint<T>
    {
        if (a.Shape.Extents[1] != b.Shape.Extents[0])
            throw new InvalidOperationException("Contraction dimensions are different");
        
        for (var i = 0; i < c.Shape.Extents[0]; ++i)
        {
            for (var j = 0; j < c.Shape.Extents[1]; ++j)
            {
                for (var k = 0; k < a.Shape.Extents[1]; ++k) // contraction dimension
                {
                    c[i, j] += a[i, k] * b[k, j];
                }
            }
        }
    }
}