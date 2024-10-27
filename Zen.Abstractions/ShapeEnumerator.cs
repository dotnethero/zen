using System.Collections;

namespace Zen;

public struct ShapeEnumerator : IEnumerator<Shape>
{
    private int _index = 0;
        
    public Shape Current { get; private set; }
    public Shape Shape { get; }

    object IEnumerator.Current => Current;
        
    public ShapeEnumerator(Shape shape)
    {
        Shape = shape;
        Current = default;
    }

    public bool MoveNext()
    {
        if (_index == Shape.Rank) 
            return false;
            
        Current = Shape[_index++];
        return true;
    }

    public void Reset()
    {
        Current = default;
        _index = 0;
    }

    public void Dispose()
    {
    }
}