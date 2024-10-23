using Zen.CUDA;

namespace Zen.Playground.Examples;

public static class SliceExample
{
    public static void Run()
    {
        using var array = HostArray.Allocate<float>(24);

        for (var i = 0; i < array.Size; ++i)
        {
            array[i] = i;
        }
        
        var a = new Tensor<float>([4, 3, 2], array);
        var b = a.Slice(1.., ..^1); 
        var c = a.Permute(0, ^1, ^2);
        var d = a.Slice(1.., .., 1);

        Utils.WriteLine(a);
        
        // a(4,3,2) =
        // [[[  0.00   1.00 ]
        //   [  2.00   3.00 ]
        //   [  4.00   5.00 ]]
        //  [[  6.00   7.00 ]
        //   [  8.00   9.00 ]
        //   [ 10.00  11.00 ]]
        //  [[ 12.00  13.00 ]
        //   [ 14.00  15.00 ]
        //   [ 16.00  17.00 ]]
        //  [[ 18.00  19.00 ]
        //   [ 20.00  21.00 ]
        //   [ 22.00  23.00 ]]]

        Utils.WriteLine(b);
        
        // b(3,2,2) =
        // [[[  6.00   7.00 ]
        //   [  8.00   9.00 ]]
        //  [[ 12.00  13.00 ]
        //   [ 14.00  15.00 ]]
        //  [[ 18.00  19.00 ]
        //   [ 20.00  21.00 ]]]
        
        Utils.WriteLine(c);
        
        // c(4,2,3) =
        // [[[  0.00   2.00   4.00 ]
        //   [  1.00   3.00   5.00 ]]
        //  [[  6.00   8.00  10.00 ]
        //   [  7.00   9.00  11.00 ]]
        //  [[ 12.00  14.00  16.00 ]
        //   [ 13.00  15.00  17.00 ]]
        //  [[ 18.00  20.00  22.00 ]
        //   [ 19.00  21.00  23.00 ]]]
        
        Utils.WriteLine(d);
        
        // d(3,3) =
        // [[  7.00   9.00  11.00 ]
        //  [ 13.00  15.00  17.00 ]
        //  [ 19.00  21.00  23.00 ]]
    }
}