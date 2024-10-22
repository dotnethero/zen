using Zen.CUDA.Interop;

namespace Zen.CUDA.Wrappers;

public sealed unsafe class CudaStream : IDisposable
{
    internal readonly cudaStream* Pointer;

    public CudaStream()
    {
        cudaStream* stream = null;
        var status = cudaStreamCreate(&stream);
        Status.EnsureIsSuccess(status);
        Pointer = stream;
    }
    
    public void Synchronize()
    {
        var status = cudaStreamSynchronize(Pointer);
        Status.EnsureIsSuccess(status);
    }

    public void BeginCapture()
    {
        var status = cudaStreamBeginCapture(Pointer, cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal);
        Status.EnsureIsSuccess(status);
    }

    public CudaGraph EndCapture()
    {
        cudaGraph* graph = null;
        var status = cudaStreamEndCapture(Pointer, &graph);
        Status.EnsureIsSuccess(status);
        return new(graph);
    }

    public void Dispose()
    {
        cudaStreamDestroy(Pointer);
    }
}