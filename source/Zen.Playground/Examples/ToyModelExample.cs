using Zen.CUDA;
using Zen.CUDA.Wrappers;
using Zen.Inference;

namespace Zen.Playground.Examples;

public static class ToyModelExample
{
    public static void Run()
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
        
        // allocate outputs and prepare kernels

        var inputs = DeviceTensor.Allocate<float>([batchSize, inputSize]);
        var outputs = model.Compose(inputs);
        var graph = model.Compile();

        // execute graph for desired iterations count

        using var stream = new CudaStream();
        using var start = new CudaEvent();
        using var stop = new CudaEvent();
        
        start.Record(stream);
        
        for (var epoch = 0; epoch < 10000; epoch++)
        {
            graph.Launch(stream);
        }
        
        stop.Record(stream);
        stop.Synchronize();
        
        // record elapsed time and copy results
        
        var elapsed = CudaEvent.Elapsed(start, stop);
        var results = HostTensor.Allocate<float>(outputs.Shape);
        
        outputs.CopyTo(results, stream);
        stream.Synchronize();
        
        Utils.WriteLine(results);
        Console.WriteLine($"Device execution: {elapsed}");
        
        // Device execution: 00:00:01.1092039 for 10'000 iterations
    }
}