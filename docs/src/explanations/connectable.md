---

layout: docs

title:  "Connectable Operators"

section: "chisel3"

---

The `Connectable` operators are the standard way to connect Chisel hardware components to one another.

Note: For descriptions of the semantics for the previous operators, see `connection-operators.md`


## Overview

## Questions to consider

Do vector sizes match?
Do all bundle field names match?
Do all bundle field name have the same order?
Do all bundle fields have the same orientation?
Do all types match?
Do all widths match?

### (c :<= p) Dangling producer field is an ERROR

```
foo: {x, z}
bar: {x, y, z}
foo :<= bar (ERROR)
```

"Should be an error" reasoning:
 - Lean towards being more strict
 - The extra field in bar could have be added later without knowledge of this connect, which would be a silent failure unless there was a strict check

"Should be legal" reasoning:
 - We use c to determine things, why do we care if there is an extra dangling field?
 - Wes's operators don't error in this case (double-check this)

### (c :<= p) Unassigned consumer field is an ERROR

```
foo: {x, y, z}
bar: {x, z}
foo :<= bar (ERROR)
```

"Should be an error" reasoning:
 - Lean towards being more strict
 - The extra field in bar could have be added later without knowledge of this connect, which would be a silent failure unless there was a strict check

### (c :<= p) Extra consumer backpressure field is LEGAL

```
c: {x, flip y, z}
p: {x, z}
c :<= p
```

"Should be legal" reasoning:
 - This operator ignores backpressure fields intentionally
 - Uses c to determine directionality, so it should ignore this field

### (c :<= p) Extra producer backpressure field is LEGAL

```
c: {x, z}
p: {x, flip y, z}
c :<= p
```

"Should be legal" reasoning:
 - This operator ignores backpressure fields intentionally
 - Uses c to determine directionality, so it should ignore this field