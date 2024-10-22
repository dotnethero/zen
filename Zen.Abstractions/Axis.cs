namespace Zen;

public readonly struct Axis
{
    public readonly Index Index;

    public Axis(Index index)
    {
        Index = index;
    }

    public static implicit operator Axis(int index) => new(index);
    public static implicit operator Axis(Index index) => new(index);
}