namespace Zen;

public readonly struct RangeOrIndex
{
    public readonly Index Start;
    public readonly Index End;
    public readonly bool IsIndex;

    public RangeOrIndex(Index index)
    {
        Start = index;
        End = index;
        IsIndex = true;
    }

    public RangeOrIndex(Range range)
    {
        Start = range.Start;
        End = range.End;
        IsIndex = false;
    }

    public static implicit operator RangeOrIndex(int index) => new(index);
    public static implicit operator RangeOrIndex(Index index) => new(index);
    public static implicit operator RangeOrIndex(Range index) => new(index);
}