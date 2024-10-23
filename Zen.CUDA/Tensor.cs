namespace Zen.CUDA;

public sealed class Tensor<T> where T : unmanaged
{
    public readonly TensorRef<T> Reference;
    public readonly Shape Shape;

    public Tensor(Shape shape, TensorRef<T> reference)
    {
        Reference = reference;
        Shape = shape;
    }
    
    public Tensor<T> Slice(params Coord[] coords)
    {
        var shape = Shape.Slice(coords, out var offset);
        return new(shape, Reference + offset);
    }

    public Tensor<T> Permute(params Axis[] axis)
    {
        return new(Shape.Permute(axis), Reference);
    }
}