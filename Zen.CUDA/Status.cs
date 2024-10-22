using Zen.CUDA.Interop;
using Zen.CUDA.Wrappers;

namespace Zen.CUDA;

internal static class Status
{
    public static void EnsureIsSuccess(cudaError error)
    {
        if (error != cudaError.cudaSuccess)
            throw new CudaException(error);
    }
}