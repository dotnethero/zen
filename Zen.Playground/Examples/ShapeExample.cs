namespace Zen.Playground.Examples;

public static class ShapeExample
{
    public static void Run()
    {
        Shape shape = [3, 4, 5];
        Shape reduced = shape.Reduce(axis: 1);
        Shape appended = shape.Append(extent: 1, stride: 0);
        Shape prepended = shape.Prepend(extent: 1, stride: 0);
        Shape transposed = shape.Transpose(^1, ^2);
        
        Console.WriteLine(shape.ToLayoutString());
        Console.WriteLine(reduced.ToLayoutString());
        Console.WriteLine(appended.ToLayoutString());
        Console.WriteLine(prepended.ToLayoutString());
        Console.WriteLine(transposed.ToLayoutString());
    }
}