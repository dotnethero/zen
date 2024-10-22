using System.Runtime.CompilerServices;
using System.Text;

namespace Zen.CUDA;

public static class Utils
{
    public static void WriteLine<T>(Tensor<T> tensor, [CallerArgumentExpression(nameof(tensor))] string tensorName = "") where T : unmanaged
    {
        var text = GetString(tensor);
        Console.WriteLine($"{tensorName}{tensor.Shape} =\n{text}");
    }
    
    private static string GetString<T>(Tensor<T> tensor, int depth = 0) where T : unmanaged
    {
        var sb = new StringBuilder("[");

        if (tensor.Shape.Rank == 1)
        {
            for (var i = 0; i < tensor.Shape.Extents[0]; ++i)
            {
                var offset = tensor.Shape.Strides[0] * i;
                var value = tensor.Reference[offset];
                sb.Append($"{value,6:0.00} ");
            }
        }
        else
        {
            for (var i = 0; i < tensor.Shape.Extents[0]; ++i)
            {
                var view = tensor.View(i);
                var text = GetString(view, depth + 1);
                if (i != 0)
                {
                    sb.Append(' ', depth + 1);                    
                }
                sb.Append($"{text}");
                if (i != tensor.Shape.Extents[0] - 1)
                {
                    sb.AppendLine();
                }
            }
        }

        sb.Append("]");
        return sb.ToString();
    }
}