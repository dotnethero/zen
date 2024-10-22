namespace Zen.CUDA;

public sealed class HostTensor<T> : IDisposable where T : unmanaged
{
    public readonly HostArray<T> Array;
    public readonly Shape Shape;
    public readonly bool Owned;

    public HostTensor(Shape shape)
    {
        Array = HostArray.Allocate<T>(shape.Size);
        Shape = shape;
        Owned = true;
    }
    
    public HostTensor(Shape shape, HostArray<T> array)
    {
        Array = array;
        Shape = shape;
        Owned = false;
    }
    
    public HostTensor<T> View(Index coord)
    {
        var shape = Shape[1..];
        var offset = Shape.Strides[0] * coord.GetOffset(Shape.Rank);
        return new(shape, Array + offset);
    }
    
    public HostTensor<T> Slice(params Range[] ranges)
    {
        var shape = Shape.Slice(ranges);
        var offset = Shape.GetOffset(ranges);
        return new(shape, Array + offset);
    }

    public HostTensor<T> Permute(params Index[] axis)
    {
        return new(Shape.Permute(axis), Array);
    }
    
    public void Dispose()
    {
        if (Owned)
            Array.Dispose();
    }
}