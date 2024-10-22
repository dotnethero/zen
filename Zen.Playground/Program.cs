﻿using Zen.CUDA;
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
        host1.CopyTo(dev, stream);
        dev.CopyTo(host2, stream);
        stream.Synchronize();
        
        Console.WriteLine(host1[128]);
        Console.WriteLine(host2[128]);
    }
}