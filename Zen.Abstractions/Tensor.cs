namespace Zen;

public readonly unsafe struct Tensor<T> where T : unmanaged
{
    public readonly Shape Shape;
    public readonly T* Pointer;

    public Tensor(T* pointer, Shape shape)
    {
        Pointer = pointer;
        Shape = shape;
    }
}