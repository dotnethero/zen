using System.Runtime.CompilerServices;
using System.Text;

namespace Zen.Playground.Examples;

public static class ShapeExample
{
    public static void Run()
    {
        Shape shape = [3, 4, 5];
        
        var reduced = shape.Reduce(axis: 1);
        var appended = shape.Append(extent: 1, stride: 0);
        var prepended = shape.Prepend(extent: 1, stride: 0);
        var transposed = shape.Transpose(^1, ^2);
        
        Console.WriteLine(shape);
        Console.WriteLine(reduced);
        Console.WriteLine(appended);
        Console.WriteLine(prepended);
        Console.WriteLine(transposed);

        var rowMajor = Shape.Create([3, 5], Layout.Right);
        var colMajor = Shape.Create([3, 5], Layout.Left);

        PrintLayout(rowMajor);
        PrintLayout(rowMajor.Slice([1.., 1..^1], out var offset), offset);
        PrintLayout(colMajor);
    }

    private static void PrintLayout(Shape shape, int offset = 0, [CallerArgumentExpression(nameof(shape))] string name = "")
    {
        var layout = new StringBuilder($"{name} has {shape} layout:\n");
        
        for (var i = 0; i < shape.Extents[0]; ++i)
        {
            if (i != 0)
            {
                layout.AppendLine();
            }
            for (var j = 0; j < shape.Extents[1]; ++j)
            {
                layout.Append($"{shape.GetOffset(i, j) + offset,4}");
            }
        }

        Console.WriteLine(layout.ToString());
    }
}