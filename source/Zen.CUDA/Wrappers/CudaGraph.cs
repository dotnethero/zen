using Zen.CUDA.Interop;

namespace Zen.CUDA.Wrappers;

public sealed unsafe class CudaGraph : IDisposable
{
    public readonly cudaGraph* Pointer;

    internal CudaGraph(cudaGraph* graph)
    {
        Pointer = graph;
    }

    public CudaGraphInstance CreateInstance()
    {
        cudaGraphInstance* graphInstance;
        var status = cudaGraphInstantiate(&graphInstance, Pointer, 1);
        Status.EnsureIsSuccess(status);
        return new(graphInstance);
    }

    public void Dispose()
    {
        cudaGraphDestroy(Pointer);
    }
}