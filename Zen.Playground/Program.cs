using System.Diagnostics;
using Zen.CUDA;
using Zen.CUDA.Wrappers;

namespace Zen.Playground;

internal static class Program
{
    public static void Main()
    {
        var sw = Stopwatch.StartNew();
        
        using var host1 = HostArray.Allocate<float>(4096);
        using var host2 = HostArray.Allocate<float>(4096);
        using var dev = DeviceArray.Allocate<float>(4096);

        for (var i = 0; i < 4096; ++i)
        {
            host1[i] = i;
        }

        using var stream = new CudaStream();
        stream.BeginCapture();
        host1.CopyTo(dev, stream);
        dev.CopyTo(host2, stream);
        
        using var graph = stream.EndCapture();
        using var instance = graph.CreateInstance();
        instance.Launch(stream);
        stream.Synchronize();
        
        Console.WriteLine(host1[128]);
        Console.WriteLine(host2[128]);
        
        // repeat graph execution
        
        for (var i = 0; i < 4096; ++i)
        {
            host1[i] = i + 64;
        }
        
        instance.Launch(stream);
        stream.Synchronize();
        
        Console.WriteLine(host1[128]);
        Console.WriteLine(host2[128]);
        
        Console.WriteLine(sw.Elapsed);
    }
}