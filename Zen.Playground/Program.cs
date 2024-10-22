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

        using var a = new HostTensor<float>([4, 3, 2]);

        for (var i = 0; i < a.Array.Size; ++i)
        {
            a.Array[i] = i;
        }
        
        var b = a.Slice(1.., ..^1); 
        var c = a.Permute(0, ^1, ^2);

        Utils.WriteLine(a);
        Utils.WriteLine(b);
        Utils.WriteLine(c);
    }
}