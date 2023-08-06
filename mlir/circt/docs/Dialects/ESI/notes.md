# Miscellaneous Notes

## ABI

The ABI specifies how an ESI "API" is translated to hardware,
specifically RTL. This involves both the wire-level signaling between
modules and how data is arranged on those wires.

This section is purposely underspecified in this proposal as it should be an
implementation detail which only advanced users need know. The main issues
discussed here are how lists and data windows are lowered and presented to
RTL modules. Lowering of fixed-size, default-presentation semantics ports is
mostly straight forward, so is not discussed here.

### Wire-level signaling

Since ESI connections are elastic (latency-insensitive), the signaling
scheme must include some notion of validity and some notion of
backpressure. This rules out interfaces which inherently cannot be
back-pressured, though future ESI specs may support non-backpressure
(feed-forward) interfaces. For now, a buffering gasket which
converts between ESI and the non-backpressure-able interface is
necessary, perhaps implementing backpressure in a higher-level protocol
(e.g. not initiating DMA transfers until space exists to buffer them).
These typically exist somewhere in many if not most designs.

It is up to the compiler to determine the appropriate wire signaling
scheme to use for message transfer between ESI interfaces. (The designer
can optionally specify it manually.) If buffering is required, the
module should state that it is required and then the compiler should
build it automatically. The interface to an RTL module, however, is up
to that particular module.

There are several existing standards which we should consider
implementing:

- AXI Stream/AMBA
- Avalon-MM/Avalon-ST
- Simple valid/ready-ack semantics (this should be the default) for
  streaming
- Simple RW RAM-style interface (this should be the default) for MMIO

### Lists and Data Windows

The basic idea behind presenting lists and data windows to modules (at
least with simple valid/ready-ack semantics) is that ESI can
automatically construct the proper state machines automatically and to
essentially present a discriminated union which informs the RTL which
part of the message is currently being presented -- the particular data
window part and/or position in a list.

## A note on language implementation

It is not always clear how a particular hardware design language (HDL)
would be made to support all of the ESI constructs. That's fine -- not
all HDLs are suited to all the ESI constructs. For instance, some
languages are only designed for data stream processing so only the data
channel (streaming) parts of ESI make sense. Maybe a compiler for said
language would also generate some MMIO regions/clients (to access DRAM, PCIe,
network, et cetera).

An HDL compiler which supports ESI is not required to implement the
entire ESI spec. Rather, it is encouraged to only implement the parts of
ESI that make sense; however, it is also encouraged that the compiler
authors think long and hard about which constructs make sense. For
example, for an RTL compiler it may not be immediately obvious how to
support **`lists`**, the variable length data type. After further
consideration, however, one may realize that RTL can accept variable
length data over multiple cycles. In fact, this is the intention as
variable length data in hardware is generally reasoned about and
implemented in the temporal dimension rather than the spatial one.
"General purpose" HDLs (i.e. SystemVerilog) compilers are encouraged to
support all of the ESI specification.
