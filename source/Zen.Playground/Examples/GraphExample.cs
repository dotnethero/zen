using Zen.CUDA;
using Zen.CUDA.Wrappers;

namespace Zen.Playground.Examples;

public static class GraphExample
{
    public static void Run()
    {
        using var host1 = HostArray.Allocate<float>(4096);
        using var host2 = HostArray.Allocate<float>(4096);
        using var dev = DeviceArray.Allocate<float>(4096);

        for (var i = 0; i < 4096; ++i)
        {
            host1[i] = i;
        }

        using var stream = new CudaStream();
        
        stream.BeginCapture(); // capture operations into graph
        host1.CopyTo(dev, stream);
        dev.CopyTo(host2, stream);
        
        using var graph = stream.EndCapture();
        using var graphInstance = graph.CreateInstance();
        
        // execute graph on new data
        
        for (var i = 0; i < 4096; ++i)
        {
            host1[i] = i + 10;
        }

        graphInstance.Launch(stream);
        stream.Synchronize();
        
        Console.WriteLine($"Host 1: {host1[64]}"); // 74
        Console.WriteLine($"Host 2: {host2[64]}"); // 74
    }
}