# ESI Software APIs

*More on this to be written.*

Status: **unimplemented**

Thanks to ESI's strong static typing, typed, design-dependent software APIs can
be automatically generated. Said APIs would be mostly independent of the
transport mechanism (PCIe, network, etc.) used to communicate with the silicon.
The same API could even drive a simulation of the ESI system.

Said APIs would need to ensure that the software is talking to the correct
hardware. There are several possible approaches:

- Generate a hash which is contained in the API and the hardware then compare
at runtime. This has the downside of software not being able to communicate
with logically compatible hardware. (E.g. mismatched interfaces which aren't
used.)
- Communicate via some self-describing message format (e.g. protobuf or
bond). This has the downside of encoding/decoding overhead.
- Auto-generate self-describing hardware for dynamic discovery of
functionality. This would allow software to be built without a static API
(e.g. a generic poke/prod script). The downside is larger hardware area, though
maybe not by much.
