using Zen.CUDA;

namespace Zen.Playground;

internal static class Program
{
    public static unsafe void Main(string[] args)
    {
        using var host1 = HostArray.Allocate<float>(4096);
        using var host2 = HostArray.Allocate<float>(4096);
        using var dev = DeviceArray.Allocate<float>(4096);

        for (var i = 0; i < 4096; ++i)
        {
            host1[i] = i;
        }
        
        host1.CopyTo(dev);
        dev.CopyTo(host2);
        host2.Sync();
        
        Console.WriteLine(host1[128]);
        Console.WriteLine(host2[128]);
    }
}