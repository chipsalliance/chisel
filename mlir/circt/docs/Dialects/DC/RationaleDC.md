# DC Dialect Rationale

[TOC]

## Introduction

DC (**D**ynamic **C**ontrol) IR describes independent, unsynchronized processes
communicating data through First-in First-out (FIFO) communication channels. 
This can be implemented in many ways, such as using synchronous logic, or with
processors. 

The intention of DC is to model all operations required to represent such a
control flow language. DC aims to be strictly a control
flow dialect - as opposed to the Handshake dialect, which assigns control
semantics to _all_ SSA values. As such, data values are
only present in DC where they are required to model control flow.

By having such a control language, the dialect aims to facilitate the
construction of dataflow programs where control and data
is explicitly separate. This enables optimization of the control and data side
of the program independently, as well as mapping
of the data side into functional units, pipelines, etc.. Furthermore, separating
data and control will make it easier to reason about possible critical paths of
either circuit, which may inform buffer placement.

The DC dialect has been heavily influenced by the Handshake dialect, and can
either be seen as a successor to it, or a lower level
abstraction. As of writing, DC is _fully deterministic_. This means that
non-deterministic operators such as the ones found in Handshake -
`handshake.merge, handshake.control_merge` - do **not** have a lowering to DC.
Handshake programs must therefore be converted or by construction not contain
any of these non-deterministic operators. Apart from that, **all** handshake
operations can be lowered to a combination of DC
and e.g. `arith` operations (to represent the data side semantics of any given
operation).

In DC IR, all values have implicit fork and sink semantics. That is, a DC-typed
value may be referenced multiple times, as well as it being legal that said
value is not referenced at all. This has been chosen to facilitate
canonicalization, thus removing the need for all canonicalization patterns to
view forks as opaque/a special case. If a given DC lowering requires explcit
fork/sink semantics, forks and sinks can be _materialized_ throuh use of the
`--dc-materialize-forks-sinks` pass. Conversely, if one wishes to optimize DC
IR which already contains fork and sink operations, one may use the
`--dc-dematerialize-forks-sinks` pass, run canonicalization, and then re-apply
the `--dc-materialize-forks-sinks` pass.


## Value (channel) semantics

1. **Latency insensitive**:
    * Any DC-typed value (`dc.token/dc.value<T...>`) has latency insensitive
semantics.
    * DC does **not** specify the implementation of this latency
insensitivity, given that it strictly pertains to the **control** of latency
insensitive values. This should reinforce the mental model that DC isn't
strictly a hardware construct - that is, DC values could be implemented in
hardware by e.g ready/valid semantics or by FIFO interfaces (read/write, full, empty, ...)
or in software by e.g. message queues, RPC, or other streaming protocols.
    * In the current state of the world (CIRCT), DC uses ESI to implement its
latency insensitive hardware protocol. By doing so, we let DC do what DC does
best (control language) and likewise with ESI (silicon interconnect).
2. **Values are channels**:
    * Given the above latency insensitivity, it is useful to think of DC values
as channels, wherein a channel can be arbitrarily buffered without changing the
semantics of the program.
2. **FIFO semantics**:
    * DC-typed values have FIFO semantics, meaning that the order of values in
the 'channel' is preserved (i.e. the order of values written to the channel is
the same as the order of values read from the channel).

## Canonicalization
By explicitly separating data and control parts of a program, we allow for
control-only canonicalization to take place.
Here are some examples of non-trivial canonicalization patterns:
* **Transitive join closure**:
  * Taking e.g. the Handshake dialect as the source abstraction, all operations
  - unless some specific Handshake operations - will be considered as
  *unit rate actors* and have join semantics. When lowering Handshake to DC,
  and by separating the data and control paths, we can easily identify `join`
  operations which are staggered, and can be merged through a transitive closure
  of the control graph.
* **Branch to select**: Canonicalizes away a select where its inputs originate
from a branch, and both have the same select signal.
* **Identical join**: Canonicalizes away joins where all inputs are the same
(i.e. a single join can be used).
