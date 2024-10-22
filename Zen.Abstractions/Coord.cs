namespace Zen;

public readonly struct Coord
{
    public readonly Index Start;
    public readonly Index End;
    public readonly bool IsIndex;

    public Coord(Index index)
    {
        Start = index;
        End = index;
        IsIndex = true;
    }

    public Coord(Range range)
    {
        Start = range.Start;
        End = range.End;
        IsIndex = false;
    }

    public static implicit operator Coord(int index) => new(index);
    public static implicit operator Coord(Index index) => new(index);
    public static implicit operator Coord(Range index) => new(index);
}