using Zen.CUDA.Interop;

namespace Zen.CUDA.Wrappers;

public sealed unsafe class CudaEvent : IDisposable
{
    public readonly cudaEvent* Pointer;

    public CudaEvent()
    {
        cudaEvent* pointer = null;
        var status = cudaEventCreate(&pointer);
        Status.EnsureIsSuccess(status);
        Pointer = pointer;
    }

    public void Record(CudaStream stream)
    {
        cudaEventRecord(Pointer, stream.Pointer);
    }

    public void Synchronize()
    {
        cudaEventSynchronize(Pointer);
    }

    public static TimeSpan Elapsed(CudaEvent start, CudaEvent end)
    {
        float ms;
        cudaEventElapsedTime(&ms, start.Pointer, end.Pointer);
        return TimeSpan.FromMilliseconds(ms);
    }
    
    public void Dispose()
    {
        cudaEventDestroy(Pointer);
    }
}