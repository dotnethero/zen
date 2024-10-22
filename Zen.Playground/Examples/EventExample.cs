using System.Diagnostics;
using Zen.CUDA;
using Zen.CUDA.Wrappers;

namespace Zen.Playground.Examples;

public static class EventExample
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
        using var start = new CudaEvent();
        using var stop = new CudaEvent();

        var sw = Stopwatch.StartNew();

        start.Record(stream); // record event
        host1.CopyTo(dev, stream);
        dev.CopyTo(host2, stream);
        stop.Record(stream); // record event
        stop.Synchronize(); // wait for event

        var elapsedHost = sw.Elapsed;
        var elapsedDevice = CudaEvent.Elapsed(start, stop);

        Console.WriteLine($"Host execution: {elapsedHost}");
        Console.WriteLine($"Device execution: {elapsedDevice}");
    }
}