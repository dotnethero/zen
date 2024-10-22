namespace Zen.CUDA;

public sealed class Tensor<T> where T : unmanaged
{
    public readonly TensorRef<T> Reference;
    public readonly Shape Shape;
    public readonly bool Owned;

    public Tensor(Shape shape, TensorRef<T> reference)
    {
        Reference = reference;
        Shape = shape;
        Owned = false;
    }
    
    public Tensor<T> View(Index coord)
    {
        var shape = Shape[1..];
        var offset = Shape.Strides[0] * coord.GetOffset(Shape.Rank);
        return new(shape, Reference + offset);
    }
    
    public Tensor<T> Slice(params Range[] ranges)
    {
        var shape = Shape.Slice(ranges);
        var offset = Shape.GetOffset(ranges);
        return new(shape, Reference + offset);
    }

    public Tensor<T> Permute(params Index[] axis)
    {
        return new(Shape.Permute(axis), Reference);
    }
}