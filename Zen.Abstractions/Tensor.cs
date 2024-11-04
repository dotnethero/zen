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
    
    public T this[params ReadOnlySpan<Index> coord]
    {
        get
        {
            EnsureRanksAreEqual(coord);

            var offset = Shape.GetOffset(coord);
            return Reference[offset];
        }
        set
        {
            EnsureRanksAreEqual(coord);
            var offset = Shape.GetOffset(coord);
            Reference[offset] = value;
        }
    }

    public Tensor<T> PrependDimension() => new(Shape.Prepend(1, 0), Reference);
    
    public Tensor<T> AppendDimension() => new(Shape.Append(1, 0), Reference);

    public Tensor<T> Slice(params ReadOnlySpan<RangeOrIndex> coords)
    {
        var shape = Shape.Slice(coords, out var offset);
        return new(shape, Reference + offset);
    }

    public Tensor<T> Permute(params ReadOnlySpan<Axis> axis)
    {
        return new(Shape.Permute(axis), Reference);
    }
    
    // TODO: Extract common checks
    private void EnsureRanksAreEqual(ReadOnlySpan<Index> coord)
    {
        if (coord.Length != Shape.Rank)
            throw new InvalidOperationException($"Can not apply {coord.Length} rank coordinate to {Shape.Rank} tensor");
    }
}