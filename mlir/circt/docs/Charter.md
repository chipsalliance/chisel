# CIRCT Charter

## Abstract

Recent trends in computer architecture have resulted in two core problems.
Firstly, how do we design complex, heterogeneous systems-on-chip mixing
general purpose and specialized components?  Secondly, how do we
program them?  We believe that design tools that represent and
manipulate a wide variety of abstractions are central to solving these
problems.  This projects is focused on using LLVM/MLIR to express
these abstractions and to build useable open-source flows based on
those abstractions to solve the design problems of the next decade.

## Introduction

With the slowing rate of scaling in semiconductor process technology,
there has become a widespread shift to develop more specialized
architectures. Circuits implementing these specialized architectures are
generally called *accelerators*, although they are developed as a
tradeoff between performance, power usage, design cost and manufacturing
cost. Accelerators typically implement compute structures, interconnect,
and memory hierarchies optimized for particular workloads. Optimizing
memory and communication is often more important for overall design
goals than optimizing the computing structures.
Accelerators are almost always programmable to some extent, in order to
allow them to operate on a set of similar workloads, although they are
typically much less programmable than general purpose architectures.

Accelerators are often so valuable because of their specialization that
they may be included in devices even when they are not constantly
used.

The need for specialization has led to a wide variety of accelerators in
some areas, such as machine learning. Programming a wide
variety of accelerators from a wide scope of design languages is a
significant challenge. MLIR addresses this by using many abstractions
called *dialects*. Individual design languages are typically associated
with their own dialects, which are generally lowered into common
dialects and optimized. Implementing optimizations on common dialects
allows the effort of developing and maintaining optimizations to be
shared between design languages and target architectures. Eventually,
dialects are lowered (perhaps in a target-specific fashion) to low-level
dialects and eventually to target-specific dialects, facilitating code
generation for particular accelerator targets.

At the same time, developing accelerators themselves is becoming more
challenging. Accelerators are often complex and may
have multiple design constraints which are difficult to meet. Even
building one accelerator can have a significant cost. At the same time,
accelerators are also numerous, implying that there is less design
effort available for each accelerator and yet the cost of building an
accelerator must be incurred over and over again. We argue that the
fundamental concepts of MLIR can address this hardware design complexity
are as applicable to the process of *building* accelerators as they are
to the process of *programming* them. Furthermore, unifying the
abstractions for accelerator design and accelerator programming is a key
step towards enabling programming for arbitrary accelerators.

## Dialects for Hardware Design

In existing systems, we see the use of a wide variety of abstractions
commonly in use. These abstractions tend to exist in a hierarchy of
abstractions used at different parts of the design process, where
higher-level abstractions are used to generate lower-level abstractions.
Often this hierarchy is not explicit in tools, but is only apparent in
the informal processes applied by human engineers. For instance,
engineers might discuss the flow of data through a sequence of
accelerators using a block diagram on a whiteboard, capture that
architecture in informal documentation and then proceed with
implementing and integrating these components in a hardware design
language. We believe that using MLIR to define hardware design
abstractions using dialects will make these specifications more precise
and enable more automation of transformations between these
abstractions.

Note that these abstractions may be used in different ways at different
points in the design process. These differences in usage often come down
to a difference in component granularity. For instance early in the
design process an abstraction may be used with large, coarse-grained
components. After lowering, perhaps through other abstractions, the same
abstraction may again appear with smaller, fine-grained
components.

Below, we summarize the most important abstractions used in existing
processes. This list is unlikely to be comprehensive and it is likely
that the process of capturing these abstractions as MLIR dialects will
identify additional abstractions or constraints that help facilitate the
lowering between dialects. Many existing systems span multiple
abstractions, but we've tried here to focus on the most significant
internal representations used in each tool.

### Structural Circuit Netlists

Netlists are a longstanding abstraction used for circuit design,
expressing the instantiation and interconnection of primitive
components. Components are typically fine-grained, representing logic
gates in a standard-cell integrated circuit, or architectural primitives
in programmable logic. The EDIF format has been almost ubiquitous
since the early days of Electronic Design Automation, although the
Verilog language is also commonly used to store netlists in
modern tools.

Netlist descriptions often blur the distinctions between the inputs and
outputs of components. Connections between components may represent
physical connections where multiple components may (hopefully at
different times!) drive a signal onto the same wire. Alternatively, a
single component may sometimes use a wire as an input and at other times
use the same wire for output.

### Register-transfer level (RTL)

The Register-transfer level is has been commonly used to describe logic
circuits. The process of logic synthesis is commonly used to lower RTL
designs into structural netlists. Verilog, SystemVerilog, and VHDL are
commonly used in the industry. The LLHD framework
explicitly takes a multi-level approach to capturing RTL designs and
lowering them into more fundamental constructs which can be efficiently
simulated or synthesized. A key aspect of this systems is the
observation that useful input languages (such as SystemVerilog) provide
a much wider set of constructs than can be easily synthesized, and
managing this complexity in a structured way is key to building real
systems. Another recent system focused on improving RTL-level design is
Chisel, along with associated
FIRRTL intermediate
representation.

At the register-transfer level, the distinction between components which
are clocked (i.e. registers) and those which are not clocked (i.e.
combinational logic) is explicit. Many circuit synthesis techniques have
historically leveraged this distinction to focus only on the
combinational aspects. Modern circuit analysis techniques often consider
the behavior across synchronous boundaries. Retiming is a common
transformation performed on RTL descriptions.

### Finite-State Machine + Datapath (FSMD)

High-Level synthesis (HLS) of RTL designs from C code has become a common
paradigm for accelerator design.
Typically generating a high-performance implementation requires analysis
and scheduling of the FSMD model in order to understand the performance
implications of particular choices.

A common intermediate abstraction in HLS is a combination of
finite-state machines and datapath abstractions. Conceptually, in each
state of the state machine, certain operations in a datapath are
enabled. This abstraction facilitates reasoning about resource sharing
and pipelined behavior, while leaving the specifics of how those
mechanisms are implemented implicit. Lowering into an RTL-level
description typically requires explicitly representing these mechanisms,
converting both the control finite-state machine and the datapath into
registers and combinational logic. This lowering also requires
explicitly representing clocks and clock enables in the circuit in an
efficient way. Note that finite-state machines may also be represented
in other ways (for instance, using microcode) rather than being lowered
through logic synthesis into fixed logic.

Several projects have described the internal representation of HLS
compilers. LegUp leverages the Clang/LLVM framework for
representing designs, while Bambu builds an independent
internal representation. 

[Calyx](https://capra.cs.cornell.edu/calyx) is a new intermediate language
that combines a software-like control language with a hardware-like structural
language. This intermediate level of representation allows Calyx to optimize
structural programs using control flow information.
Calyx currently serves as the compilation target for [Dahlia](https://capra.cs.cornell.edu/dahlia)
and several other domain-specific architectures (systolic array, number 
theoretic transform unit).

### [Dataflow/handshake](Dialects/Handshake/RationaleHandshake.md)

Dataflow models have long been used to represent parallel computation.
Typically these models include components representing independent
processes communicating through streams of data. These models provide a
notion of concurrency which is intrinsically decoupled, unlike
synchronous models like RTL. This makes it easy to describe distributed
systems with *elastic* or *latency-insensitive* properties, where the
latency to communicate data from one process to another is guaranteed to
not change the streams of data being computed.

It is possible to extract dataflow models from fine-grained CDFG
representations of sequential code and the resulting
dataflow models can be implemented directly in a circuit, although the
circuits used to implement fine-grained handshake logic are often more
complex than an equivalent FSMD model (for designs which can be
statically scheduled). More commonly, dataflow models are used with
coarse-grained synchronous components to implement a globally
asynchronous-locally synchronous(GALS) architecture. This architecture
is often appropriate for large accelerator designs where clock
distribution and global timing closure can be challenging when
implementing large componentized RTL designs.

Another common use of dataflow models is as a simulation tool, as in
FireSim.  FireSim starts with a cycle-accurate RTL model, and extracts
a dataflow model where time is represented explicitly in the model.
This allows distributed simulation or for models of parts of a
hardware design to be inserted where RTL does not yet exist.  By
mapping the dataflow model back to hardware in a non-cycle accurate
way, very fast emulation can be built quickly.

### IP composition

IP composition has become a key methodology for most FPGA and ASIC
designs, enabling large designs to be constructed out of small
components. The key to making IP composition robust has been to focus on
interface-based composition using protocols like ARM AXI, rather than
signal-level composition which is commonly done in RTL design. Standard
interface protocols have enabled designers to independently solve the
problems of how components behave internally with how they are
integrated. At the lowest levels of the design flow, this has allowed
larger designs to meet timing during place and route, by enabling
hierarchical floorplanning and late-stage interconnect synthesis. It can
also enable the impact of late-stage changes, called Engineering Change
Orders (ECOs), to be encapsulated, avoiding unnecessary redundant
verification. IP composition also enables portable operating system
software and runtimes based on declarative descriptions of device
architectures, for instance using Device Trees.

IPXact is an industry standard file format, although a lack of
industry-standard IP definitions and heavy reliance in practice on
vendor extensions has meant that tools using IPXact are largely not
inter-operable. In addition, being based on XML, IPXact is somewhat
verbose and difficult for humans to interact with. DUH is a recent
open-source effort based on JSON formats also focusing on IP
composition.

## Tool Flows

Although it is important to have appropriate abstractions to build with,
those abstractions are simply a means to support transformations of
designs. Although many flows are potentially interested, we highlight
some of the most interesting flows.

### High-Level Synthesis (HLS)

A common toolflow is to synthesize circuits from sequential algorithmic
code. Many tools have implemented this type of tool flow, usually with a
focus on signal processing applications. In such a flow, building deeply
pipelined and parallelized implementations of loops is a key capability.
A typical high-level synthesis flow starts from an optimized
control-dataflow graph (CDFG). Additional hardware-oriented
optimizations, such as bitwidth minimization, ROM inference,
shift-register identification, and algebraic optimizations with hardware
cost models taken into account are typically performed to reduce
implementation cost. Memory analysis and optimization is also critical
for most code. Arrays are often implemented in local memories wherever
possible, and memory accesses optimized to eliminate cache hierarchy
wherever possible. Memories are often restructured through memory
partitioning to increase bandwidth. Accurate points-to analysis and
affine index analysis are necessary to statically identify limits on
pipelining. Operation scheduling is performed to identify initial
clock-cycle boundaries.

A scheduled design can be expressed as an FSMD model. In this form, it
becomes possible to identify micro-architectural optimizations.
Operations can be shared on the same hardware and the multiplexing cost
from this sharing can be accurately estimated. Control logic cost can
also be estimated and minimized. Verification is also important in this
stage: simulation can be used to verify a design with the original
sequential testbench and partial formal analysis, e.g. reachability
analysis on generated control logic, are possible. Interface synthesis
is also applicable at this level of abstraction, converting algorithmic
interfaces (like sequential accesses to memory) with implementation
interfaces (like FIFO interfaces).

After optimization, the FSMD model can be realized as a
Register-Transfer Level (RTL) design. This transformation makes the
control FSM explicit and blurs the distinction between the control logic
and data path. Logic optimization is possible between the control logic
and the data path and these optimizations can leverage Clock Enable
inputs on registers. RTL models are typically realized as Verilog or
VHDL, enabling implementation using vendor tools. Alternatively, these
models could be further lowered to more hardware-specific dialects in
MLIR.

### Dataflow-based Multicore Programming

Multicore programming is commonly done using Single-Program
Multiple-Data (SPMD) programming, such as CUDA for GPUs. Typically, each
processor reads a portion of a large data set from external memory into
local memory, processing it locally before writing results back to
external memory. As a result, communication between processors largely
happens through external memory, resulting in significant power
consumption and a communication bottleneck. While communication between
cores using on-chip memory is also possible, this must be managed
explicitly and separately from global memory. Explicit dataflow-based
programming represents inter-process communication explicitly as streams
of data, allowing communication intent to be explicitly separated from
stateful storage. Additionally, since dataflow processes do not share
state, dataflow-based programming describes memory locality explicitly
in a program.

Many optimizations are possible on dataflow models. Most critically,
fusion optimizations manipulate the dataflow graph to generate toplevel
dataflow components where each component represents execution on a
single processor. In order to generate a balanced
system, this is typically done with several goals. Primarily, this needs
to be done considering locality of access, often merging dataflow
components which share inputs or process data in a pipeline.
Secondarily, it is also important to balance the amount of processing
done on each physical processor. This minimizes the amount of time where
one processor is stalled waiting for data from another processor. In
more complicated systems with feedback, unbalanced workloads may be
preferable to ensure that feedback paths complete quickly. *Fission
transformations may also be important, where large granularity
components exist in a dataflow program. Obviously data-parallel
computations can be partitioned using Split/Join pairs, while more
complex components consisting of nested loops with recurrences require
more complex analysis.

Stream computation can be implemented using different mechanisms
depending on the architecture. Some architectures provide stream
operations in the ISA, enabling data to be communicated on physical
streams. However, in most cases, stream accesses are turned into memory
operations. For instance, streams on general purpose CPUs are typically
implemented through memory allocation, pointer manipulation and
synchronization between threads or perhaps using OS-level pipes between
processes. Stream communication may also occur between general purpose
CPUs and an accelerator with stream communication being implemented by a
combination of DMA hardware, shared memory access, and runtime API
calls. In distributed systems, stream communication can also be
implemented over a network, for instance using TCP/IP. In such cases,
understanding the bottleneck of a shared medium can be critical. In each
of these cases, there are lowerings from dataflow concepts to more
generic process/thread/fibre concepts, where there are significant
architecture-specific tradeoffs which must be managed. We argue that
modeling this lowering explicitly is almost always more convenient for a
programmer than exposing low-level concepts (e.g. threads and locks) to
the programmer.
