using System.Diagnostics;
using Zen.CUDA;

namespace Zen.Playground;

internal static class Program
{
    public static unsafe void Main(string[] args)
    {
        using var a1 = DeviceArray.Allocate<float>(4096);
        var a = new Tensor<float>();
    }
}