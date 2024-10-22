using Zen.CUDA.Interop;

namespace Zen.CUDA.Wrappers;

public sealed unsafe class CudaGraphInstance : IDisposable
{
    internal readonly cudaGraphInstance* Pointer;
    
    public CudaGraphInstance(cudaGraphInstance* graphInstance)
    {
        Pointer = graphInstance;
    }

    public void Launch(CudaStream stream)
    {
        cudaGraphLaunch(Pointer, stream.Pointer);
    }

    public void Dispose()
    {
        cudaGraphExecDestroy(Pointer);
    }
}