using Zen.CUDA;
using Zen.CUDA.Interop;
using static Zen.CUDA.Interop.CudaRuntime;
using static Zen.CUDA.Interop.CudaImports;

namespace Zen.Playground.Examples;

public static class GemmExample
{
    public static unsafe void Run()
    {
        using var a = ReferenceTensor.Allocate<float>([3, 4]);
        using var b = ReferenceTensor.Allocate<float>([5, 4]);
        using var c = ReferenceTensor.Allocate<float>([3, 5]);
        
        for (var i = 0; i < a.Cosize; ++i)
        {
            a.Host.Array[i] = i + 1;
        }
        
        for (var i = 0; i < b.Cosize; ++i)
        {
            b.Host.Array[i] = i % 3;
        }
        
        a.SyncToDevice();
        b.SyncToDevice();

        zenGemmHandle* handle;
        zenCreateGemm(
            &handle,
            m: a.Shape.Extents[0],
            n: b.Shape.Extents[0],
            k: a.Shape.Extents[1],
            a.Device.Pointer, lda: a.Shape.Strides[0],
            b.Device.Pointer, ldb: b.Shape.Strides[0],
            c.Device.Pointer, ldc: c.Shape.Strides[0]);

        if (handle == null)
            throw new InvalidOperationException();
        
        zenExecuteGemm(handle);
        cudaDeviceSynchronize();
        
        c.SyncToHost();
        
        zenDestroyGemm(handle);
        
        Utils.WriteLine(a.Host);
        Utils.WriteLine(b.Host);
        Utils.WriteLine(c.Host);
    }
}