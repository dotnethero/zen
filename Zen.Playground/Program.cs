using Zen.CUDA;

namespace Zen.Playground;

internal static class Program
{
    public static unsafe void Main(string[] args)
    {
        using var host = HostArray.Allocate<float>(4096);
        using var dev = DeviceArray.Allocate<float>(4096);

        for (var i = 0; i < 4096; ++i)
        {
            host[i] = i;
        }
        
        Console.WriteLine(host[128]);
    }
}