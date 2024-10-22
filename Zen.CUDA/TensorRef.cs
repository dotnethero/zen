namespace Zen.CUDA;

public unsafe class TensorRef<T> where T : unmanaged
{
    internal readonly T* Pointer;
    internal readonly int ElementSize;
    
    internal TensorRef(T* pointer)
    {
        ElementSize = sizeof(T);
        Pointer = pointer;
    }
    
    public T this[int offset]
    {
        get => Pointer[offset];
        set => Pointer[offset] = value;
    }
    
    public static TensorRef<T> operator +(TensorRef<T> array, int offset) => new(array.Pointer + offset);
    public static TensorRef<T> operator -(TensorRef<T> array, int offset) => new(array.Pointer - offset);
}