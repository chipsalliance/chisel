# 'esi' Dialect

The Elastic Silicon Interconnect dialect aims to aid in accelerator system construction.

[TOC]

## Application channels

The main component of ESI are point-to-point, typed channels that allow
designers to connect modules to each other and software, then communicate by
sending messages. Channels largely abstract away the details of message
communication from the designer, though the designer can declaratively specify
how to implement the channel.

Messages have types: ints, structs, arrays, unions, and variable-length lists.
The width of a channel is not necessarily the same width as the message. ESI
“windows” can be used to break up a message into a series of “frames”. IP blocks
can emit / absorb “windowed” messages or full-sized messages, which can be
automatically broken up to save wire area at the cost of bandwidth.

Any channel which is exposed to the host will have a platform-agnostic software
API constructed for it based on the type of the channel. The software
application merely has to connect to the accelerator then invoke a method to
send or receive messages from the accelerator system.

[include "Dialects/ESIChannelTypes.md"]
[include "Dialects/ESITypes.md"]
[include "Dialects/ESIChannels.md"]

## Services

ESI "services" provide device-wide connectivity and arbitration for shared
resources, which can be requested from any IP block (service "client"). Standard
services will include DRAM, clock/reset, statistical counter reporting, and
debug.

[include "Dialects/ESIServices.md"]
[include "Dialects/ESIStdServices.md"]

## Structural

ESI has a special module which doesn't expose ports. All external interactions
are expected to be done through services.

[include "Dialects/ESIStructure.md"]

## Interfaces

Misc CIRCT interfaces.

[include "Dialects/ESIInterfaces.md"]
