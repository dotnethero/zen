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
    
    public HostTensor<T> Permute(Index[] axis)
    {
        return new(Shape.Permute(axis), Array);
    }
    
    public void Dispose()
    {
        if (Owned)
            Array.Dispose();
    }
}