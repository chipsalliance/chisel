# Intervals
**Intervals** are a new experimental numeric type that comprises UInt, SInt and FixedPoint numbers.
It augments these types with range information, i.e. upper and lower numeric bounds.
This information can be used to exercise tighter programmatic control over the ultimate widths of
signals in the final circuit.  The **Firrtl** compiler can infer this range information based on 
operations and earlier values in the circuit. Intervals support all the ordinary bit and arithmetic operations
associated with UInt, SInt, and FixedPoint and adds the following methods for manipulating the range of
a **source** Interval with the IntervalRange of **target** Interval 

### Clip -- Fit the value **source** into the IntervalRange of **target**, saturate if out of bounds
The clip method applied to an interval creates a new interval based on the argument to clip,
and constructs the necessary hardware so that the source Interval's value will be mapped into the new Interval.
Values that are outside the result range will be pegged to either maximum or minimum of result range as appropriate.

> Generates necessary hardware to clip values, values greater than range are set to range.high, values lower than range are set to range min.

### Wrap -- Fit the value **source** into the IntervalRange of **target**, wrapping around if out of bounds
The wrap method applied to an interval creates a new interval based on the argument to wrap,
and constructs the necessary
hardware so that the source Interval's value will be mapped into the new Interval.
Values that are outside the result range will be wrapped until they fall within the result range.

> Generates necessary hardware to wrap values, values greater than range are set to range.high, values lower than range are set to range min.

> Does not handle out of range values that are less than half the minimum or greater than twice maximum

### Squeeze -- Fit the value **source** into the smallest IntervalRange based on source and target.
The squeeze method applied to an interval creates a new interval based on the argument to clip, the two ranges must overlap
behavior of squeeze with inputs outside of the produced range is undefined.

> Generates no hardware, strictly a sizing operation

#### Range combinations

| Condition | A.clip(B) | A.wrap(B) | A.squeeze(B) |
| --------- | --------------- | --------------- | --------------- |
| A === B   | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| A contains B   | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| B contains A   | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| A min < B min, A max in B  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| A min in B, A max > B max  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| A strictly less than B   | error               | error               | error               |
| A strictly greater than B   | error               | error               | error               |


### Applying binary point operators to an Interval

Consider a Interval with a binary point of 3: aaa.bbb 

| operation | after operation | binary point | lower | upper | meaning | 
| --------- | --------------- | ------------ | ----- | ----- | ------- |
| setBinaryPoint(2) | aaa.bb |  2 | X | X  | set the precision |
| shiftLeftBinaryPoint(2) | a.aabbb |  5 | X | X  | increase the precision |
| shiftRighBinaryPoint(2) | aaaa.b |  1 | X | X  | reduce the precision |

