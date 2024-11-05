using System.Collections;
using System.Runtime.CompilerServices;
using Zen.CUDA;
using Zen.CUDA.Wrappers;

namespace Zen.Inference;

[CollectionBuilder(typeof(Sequential), nameof(Create))]
public sealed class Sequential : IModel, IEnumerable<IModel>
{
    public IEnumerable<IModel> Layers { get; }

    public static Sequential Create(ReadOnlySpan<IModel> layers) => new(layers);

    public Sequential(ReadOnlySpan<IModel> layers)
    {
        Layers = layers.ToArray();
    }

    public DeviceTensor<float> Compose(DeviceTensor<float> inputs)
    {
        var activation = inputs;
        foreach (var layer in Layers)
        {
            activation = layer.Compose(activation);
        }
        return activation;
    }

    public void Execute(CudaStream stream)
    {
        foreach (var layer in Layers)
        {
            layer.Execute(stream);
        }
    }

    public IEnumerator<IModel> GetEnumerator() => Layers.GetEnumerator();

    public void Dispose()
    {
        foreach (var layer in Layers)
        {
            layer.Dispose();
        }
    }
    
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
