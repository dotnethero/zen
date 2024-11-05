using Zen.CUDA.Interop;
using Zen.CUDA.Wrappers;

namespace Zen.CUDA;

internal static class Status
{
    public static void EnsureIsSuccess(cudaStatus status)
    {
        if (status != cudaStatus.cudaSuccess)
            throw new CudaException(status);
    }
}