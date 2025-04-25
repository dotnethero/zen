# Zen

Zen is a high-performance library providing convenient abstractions over CUDA primitives, allowing for efficient GPU programming with a clean .NET API.

## Overview

Zen offers low-level but ergonomic wrappers around CUDA functionality, making GPU programming more accessible while maintaining performance. The library supports tensor operations, CUDA graphs, memory management, and mathematical operations like GEMM.

## Features

- **Tensor Operations**: Convenient tensor allocation, manipulation, and computation
- **CUDA Graph Support**: Capture and replay operation sequences for optimal performance
- **Memory Management**: Simple host and device memory allocation with automatic resource disposal
- **Shape Manipulation**: Rich API for tensor shape operations (reduction, transposition, slicing)
- **Neural Network Building Blocks**: Linear layers and sequential model composition
- **Stream Management**: Asynchronous operation execution

## Examples

### GEMM (General Matrix Multiplication)

```csharp
public static unsafe void RunGemm()
{
    using var a = ReferenceTensor.Allocate<float>([3, 4]);
    using var b = ReferenceTensor.Allocate<float>([5, 4]);
    using var c = ReferenceTensor.Allocate<float>([3, 5]);
    
    // Initialize tensors
    for (var i = 0; i < a.Cosize; ++i)
        a.Host.Array[i] = i + 1;
    
    for (var i = 0; i < b.Cosize; ++i)
        b.Host.Array[i] = i % 3;
    
    a.SyncToDevice();
    b.SyncToDevice();

    // Create and execute GEMM operation
    zenGemmHandle* handle;
    zenCreateGemm(
        &handle,
        m: a.Shape.Extents[0],
        n: b.Shape.Extents[0],
        k: a.Shape.Extents[1],
        a.Device.Pointer, lda: a.Shape.Strides[0],
        b.Device.Pointer, ldb: b.Shape.Strides[0],
        c.Device.Pointer, ldc: c.Shape.Strides[0]);
    
    zenExecuteGemm(handle);
    cudaDeviceSynchronize();
    
    c.SyncToHost();
    zenDestroyGemm(handle);
}
```

### CUDA Graph Capture and Replay

```csharp
public static void RunGraph()
{
    using var host1 = HostArray.Allocate<float>(4096);
    using var host2 = HostArray.Allocate<float>(4096);
    using var dev = DeviceArray.Allocate<float>(4096);

    // Initialize data
    for (var i = 0; i < 4096; ++i)
        host1[i] = i;

    // Capture operations into graph
    using var stream = new CudaStream();
    stream.BeginCapture();
    host1.CopyTo(dev, stream);
    dev.CopyTo(host2, stream);
    
    using var graph = stream.EndCapture();
    using var graphInstance = graph.CreateInstance();
    
    // Execute graph on new data
    for (var i = 0; i < 4096; ++i)
        host1[i] = i + 10;

    graphInstance.Launch(stream);
    stream.Synchronize();
}
```

### Shape Manipulation

```csharp
Shape shape = [3, 4, 5];
var reduced = shape.Reduce(axis: 1);
var appended = shape.Append(extent: 1, stride: 0);
var prepended = shape.Prepend(extent: 1, stride: 0);
var transposed = shape.Transpose(^1, ^2);

var rowMajor = Shape.Create([3, 5], Layout.Right);
var colMajor = Shape.Create([3, 5], Layout.Left);
```

### Neural Network Example

```csharp
public static void RunToyModel()
{
    const int batchSize = 2048;
    const int inputSize = 28 * 28;
    const int hiddenSize = 512;
    const int outputSize = 10;

    using Sequential model =
    [
        new LinearLayer(inputSize, hiddenSize),
        new LinearLayer(hiddenSize, outputSize),
    ];
    
    // Allocate outputs and prepare kernels
    var inputs = DeviceTensor.Allocate<float>([batchSize, inputSize]);
    var outputs = model.Compose(inputs);
    var graph = model.Compile();

    // Execute graph
    using var stream = new CudaStream();
    using var start = new CudaEvent();
    using var stop = new CudaEvent();
    
    start.Record(stream);
    
    for (var epoch = 0; epoch < 10000; epoch++)
        graph.Launch(stream);
    
    stop.Record(stream);
    stop.Synchronize();
    
    // Get results
    var elapsed = CudaEvent.Elapsed(start, stop);
    var results = HostTensor.Allocate<float>(outputs.Shape);
    outputs.CopyTo(results, stream);
}
```

## Requirements

- .NET 8.0
- NVIDIA GPU

**Note:** CUDA Libraries are not required to run .NET part<br/>
(only uses small C++ library with statically linked CUDA runtime functions)

## License

MIT
