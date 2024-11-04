namespace Zen;

public static class Tensor
{
    public static Tensor<T> Create<T>(Shape shape, TensorRef<T> reference) 
        where T : unmanaged => 
        new(shape, reference);
}

public class Tensor<T> where T : unmanaged
{
    public readonly TensorRef<T> Reference;
    public readonly Shape Shape;

    public Tensor(Shape shape, TensorRef<T> reference)
    {
        Reference = reference;
        Shape = shape;
    }
    
    public ref T this[params ReadOnlySpan<LogicalCoord> coords]
    {
        get
        {
            var offset = Shape.GetOffset(coords);
            return ref Reference[offset];
        }
    }

    public Tensor<T> PrependDimension() => new(Shape.Prepend(1, 0), Reference);
    
    public Tensor<T> AppendDimension() => new(Shape.Append(1, 0), Reference);

    public Tensor<T> Slice(params ReadOnlySpan<LogicalRange> slice)
    {
        var shape = Shape.Slice(slice, out var offset);
        return new(shape, Reference + offset);
    }

    public Tensor<T> Permute(params ReadOnlySpan<Axis> axis)
    {
        return new(Shape.Permute(axis), Reference);
    }
}