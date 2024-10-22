using Zen.CUDA;
using Zen.CUDA.Wrappers;

namespace Zen.Playground.Examples;

public static class CopyExample
{
    public static void Run()
    {
        using var host1 = HostArray.Allocate<float>(4096);
        using var host2 = HostArray.Allocate<float>(4096);
        using var dev1 = DeviceArray.Allocate<float>(4096);
        using var dev2 = DeviceArray.Allocate<float>(4096);

        for (var i = 0; i < 4096; ++i)
        {
            host1[i] = i; // initialize host array
        }

        using (var stream = new CudaStream())
        {
            host1.CopyTo(dev1, stream); // host to device
            dev1.CopyTo(dev2, stream);  // device to device
            dev2.CopyTo(host2, stream); // device to host
            stream.Synchronize();
        }

        var result = string.Join(" ", host2[128..144].ToArray());
        
        // [128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143]
        
        Console.WriteLine($"Host 2: [{result}]");
    }
}