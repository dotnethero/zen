using Zen.CUDA;
using Zen.Playground.Examples;

namespace Zen.Playground;

internal static class Program
{
    public static void Main()
    {
        CopyExample.Run();
        GraphExample.Run();
        EventExample.Run();

        using var array = HostArray.Allocate<float>(24);

        for (var i = 0; i < array.Size; ++i)
        {
            array[i] = i;
        }
        
        var a = new Tensor<float>([4, 3, 2], array);
        var b = a.Slice(1.., ..^1); 
        var c = a.Permute(0, ^1, ^2);
        var d = b.Permute(0, ^1, ^2);

        Utils.WriteLine(a);
        Utils.WriteLine(b);
        Utils.WriteLine(c);
        Utils.WriteLine(d);
    }
}