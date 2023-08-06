# The Elastic Silicon Interconnect dialect

Long ago, software function calling conventions were ad-hoc. This led to
issues, particularly with register clobbering and stack corruption. This is --
in large part -- the state of FPGA/ASIC design today: wire signaling protocols
are often ad-hoc, which also leads to major issues. Though there are efforts
to standardize the signaling protocols there are many minor and major
variants, both of which lead to confusion which can cause real problems when
one is listening to and twiddling the wires manually. ESI solves this by
providing a simple, intuitive, common, and standardized interface to developers
and then figures out the signaling details and conversions between them.

While the ABI/signaling problem is slowly being partially solved, it does not
speak to the types of data on the wires – the software analogy being memory
and registers. In the software world, data types were added. More and more
complex type systems began to evolve – to great successes in some cases as
strong typing can help developers avoid bugs and assist in debugging. In the
FPGA/ASIC world, RTL-level languages are starting to get basic types but
across interconnects it is still common for the data types to be informally
specified in a data sheet. This indicates a total failure of the basic type
system which RTL supports.

The Elastic Silicon Interconnect (ESI) project raises the bar on both fronts. On
the data type front, it (will) define a rich, hardware-centric type system to
allow more formal data type definitions and strong static type safety. On the
ABI/signaling front, it can build simple, latency-insensitive interfaces and
abstract away the signaling protocol. Essentially, the intent is to cleanly
separate/abstract the physical signaling layer from the message layer. This
enables many tasks to be automated including – but not limited to – the
following:

1) Inter-language communication
2) Type checking to reduce bugs at interface boundaries
3) Correct-by-construction building of communication fabric (including clock
domain crossings)
4) Decision making about the physical signaling between modules
5) Software API generation which bridges over PCIe, network, or simulation
6) Pipelining based on floor planning between modules to reduce timing closure
pressure
7) Compatibility between modules with different bandwidths (automatic
gearboxing)
8) Type and signal aware debuggers/monitors in communication fabric
9) Common interface for board support packages
10) Extensible services to support global resources (e.g. telemetry)

## Status

The ESI project is in its infancy -- it is not complete by any means. We are
always looking for people to experiment with it and contribute!

## Publications

"Elastic Silicon Interconnects: Abstracting Communication in Accelerator
Design", John Demme (Microsoft).
[paper](https://capra.cs.cornell.edu/latte21/paper/8.pdf),
[talk](https://www.youtube.com/watch?v=gjOkGX2E7EY).
