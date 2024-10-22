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
        using var b = a.Permute([0, 2, 1]);
        
        Utils.WriteLine(a);
        Utils.WriteLine(b);
    }
}