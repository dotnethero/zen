using Zen.CUDA.Wrappers;

namespace Zen.Inference;

public static class ModelExtensions
{
    public static CudaGraphInstance Compile(this IModel model)
    {
        using var stream = new CudaStream();
        
        stream.BeginCapture();
        model.Execute(stream);

        var graph = stream.EndCapture();
        var graphInstance = graph.CreateInstance();
        
        return graphInstance;
    }
}