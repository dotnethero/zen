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
        
        Console.WriteLine(shape.ToLayoutString());
        Console.WriteLine(reduced.ToLayoutString());
        Console.WriteLine(appended.ToLayoutString());
        Console.WriteLine(prepended.ToLayoutString());
        Console.WriteLine(transposed.ToLayoutString());

        var rowMajor = Shape.Create([3, 5], Layout.Right);
        var colMajor = Shape.Create([3, 5], Layout.Left);

        var rowMajorLayout = new StringBuilder("Row-major layout:\n");
        var colMajorLayout = new StringBuilder("Column-major layout:\n");
        
        for (var i = 0; i < rowMajor.Extents[0]; ++i)
        {
            if (i != 0)
            {
                rowMajorLayout.AppendLine();
                colMajorLayout.AppendLine();
            }
            for (var j = 0; j < rowMajor.Extents[1]; ++j)
            {
                rowMajorLayout.Append($"{rowMajor.GetOffset([i, j]),4}");
                colMajorLayout.Append($"{colMajor.GetOffset([i, j]),4}");
            }
        }
        
        Console.WriteLine(rowMajorLayout.ToString());
        Console.WriteLine(colMajorLayout.ToString());
    }
}