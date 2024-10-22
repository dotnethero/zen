using Zen.CUDA.Interop;

namespace Zen.CUDA.Wrappers;

public sealed unsafe class CudaGraph : IDisposable
{
    internal readonly cudaGraph* Pointer;

    public CudaGraph()
    {
        cudaGraph* graph = null;
        
        var status = cudaGraphCreate(&graph, 1); // TODO: Add flags enum
        Status.EnsureIsSuccess(status);
    }
    
    public CudaGraph(cudaGraph* graph)
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