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
    public static implicit operator int(Axis axis) => axis.Index.Value;
    public static implicit operator Index(Axis axis) => axis.Index;
}