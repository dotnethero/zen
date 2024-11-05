using Zen.CUDA.Wrappers;

namespace Zen.CUDA;

public static class HostTensor
{
    public static HostTensor<T> Allocate<T>(Shape shape) where T : unmanaged
    {
        var array = HostArray.Allocate<T>(shape.Size);
        var tensor = new HostTensor<T>(shape, array);
        return tensor;
    }
}

public sealed class HostTensor<T> : Tensor<T>, IDisposable where T : unmanaged
{
    public readonly HostArray<T> Array;

    public HostTensor(Shape shape, HostArray<T> array) : base(shape, array)
    {
        Array = array;
    }

    public void CopyTo(HostTensor<T> tensor, CudaStream? stream = null) // TODO: check shapes
    {
        Array.CopyTo(tensor.Array, stream ?? CudaStream.Default);
    }

    public void CopyTo(DeviceTensor<T> tensor, CudaStream? stream = null)
    {
        Array.CopyTo(tensor.Array, stream ?? CudaStream.Default);
    }
    
    public void Dispose()
    {
        Array.Dispose();
    }
}
