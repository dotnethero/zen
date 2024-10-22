using Zen.CUDA.Interop;

namespace Zen.CUDA;

internal sealed class CudaException(cudaError status) : Exception($"Operation is not completed: {status}");