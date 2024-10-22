using System.Diagnostics;
using Zen.CUDA;
using Zen.CUDA.Wrappers;

namespace Zen.Playground;

internal static class Program
{
    public static void Main()
    {
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

        using var start = new CudaEvent();
        using var stop = new CudaEvent();
        
        var sw = Stopwatch.StartNew();

        start.Record(stream);
        instance.Launch(stream);
        stop.Record(stream);
        stop.Synchronize();
        
        var elapsedHost = sw.Elapsed;
        var elapsedGraph = CudaEvent.Elapsed(start, stop);
        
        Console.WriteLine(host1[128]);
        Console.WriteLine(host2[128]);
        
        Console.WriteLine($"Host execution: {elapsedHost}");
        Console.WriteLine($"Graph execution: {elapsedGraph}");
    }
}