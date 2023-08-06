# FSM Dialect Rationale

This document describes various design points of the FSM dialect, why they are
the way they are, and current status.  This follows in the spirit of other
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

## Introduction

[Finite-state machine (FSM)](https://en.wikipedia.org/wiki/Finite-state_machine)
is an abstract machine that can be in exactly one of a finite number of states
at any given time.  The FSM can change from one state to another in response to
some inputs; the change from one state to another is called a transition.
Verification, Hardware IP block control, system state control, hardware design,
and software design have aspects that are succinctly described as some form of
FSM.  For integrated development purposes, late design-time choice, and
per-project choice, we want to encode system descriptions in an FSM form.  We
want a compiler to be able to manipulate, query, and generate code in multiple
domains, such as SW drivers, firmware, RTL hardware, and verification.

The FSM dialect in CIRCT is designed to provide a set of abstractions for FSM
with the following features:

1. Provide explicit and structural representations of states, transitions, and
internal variables of an FSM, allowing convenient analysis and transformation.
2. Provide a target-agnostic representation of FSM, allowing the state machine
to be instantiated and attached to other dialects from different domains.
3. By cooperating with two conversion passes, FSMToSV and FSMToStandard, allow
to lower the FSM abstraction into HW+Comb+SV (Hardware) and Standard+SCF+MemRef
(Software) dialects for the purposes of simulation, code generation, etc.

## Operations

### Two ways of instantiation

A state machine is defined by an `fsm.machine` operation, which contains all
the states and transitions of the state machine. `fsm.machine` has a list of
inputs and outputs and explicit state type:

```mlir
fsm.machine @foo(%arg0: i1) -> i1 attributes {stateType = i1} {
  ...
}
```

FSM dialect provides two ways to instantiate a state machine: `fsm.hw_instance`
is intended for use in HW context (usually described by graph regions) and
`fsm.instance`+`fsm.trigger` is intended for use in a SW context (usually
described by CFG regions).  In HW IRs (such as HW+Comb+SV and FIRRTL), although
an MLIR value is only defined once in the IR, it is actually "driven" by its
predecessors continuously during the runtime and can "hold" different values at
different moments.  However, in the world of SW IRs (such as Standard+SCF), we
don't have such a semantics -- SW IRs “run” sequentially.

Here we define that each *trigger* causes the possibility of a transition from
one state to another state through exactly one transition.  In a SW context,
`fsm.instance` generates an `InstanceType` value to represent a state machine
instance.  Each `fsm.trigger` targets a machine instance and explicitly causes a
*trigger*.  Therefore, `fsm.trigger` may change the state of the machine
instance thus is a side-effecting operation.  The following MLIR code shows an
example of instantiating and triggering the state machine defined above:

```mlir
func @bar() {
  %foo_inst = fsm.instance "foo_inst" @foo
  %in0 = ...
  %out0 = fsm.trigger %foo_inst(%in0) : (i1) -> i1
  ...
  %in1 = ...
  %out1 = fsm.trigger %foo_inst(%in1) : (i1) -> i1
  return
}
```

In the contrast, to comply with the HW semantics, `fsm.hw_instance` directly
consumes inputs and generates results.  The operand and result types must align
with the type of the referenced `fsm.machine`.  In a HW context, *trigger*s are
implicitly initiated by the processors of `fsm.hw_instance`.  The following
MLIR code shows an example of instantiating the same state machine in HW IRs:

```mlir
hw.module @qux() {
  %in = ...
  %out = fsm.hw_instance "foo_inst" @foo(%in) : (i1) -> i1
}
```

### Explicit state and transition representation

Each state of an FSM is represented explicitly with an `fsm.state` operation.
`fsm.state` must have an `output` CFG region representing the combinational
logic of generating the state machine's outputs.  The output region must have an
`fsm.output` operation as terminator and the operand types of the `fsm.output`
must align with the result types of the state machine.  `fsm.state` also
contains a list of `fsm.transition` operations representing the outgoing
transitions that can be triggered from the current state.  `fsm.state` also has
a symbol name that can be referred to by `fsm.transition`s as the next state.
The following MLIR code shows a running example:

```mlir
fsm.machine @foo(%arg0: i1) -> i1 attributes {stateType = i1} {
  fsm.state "IDLE" output  {
    %true = constant true
    fsm.output %true : i1
  } transitions  {
    fsm.transition @BUSY ...
  }

  fsm.state "BUSY" output  {
    %false = constant false
    fsm.output %false : i1
  } transitions  {
    fsm.transition @BUSY ...
    fsm.transition @IDLE ...
  }
}
```

### Guard region of transitions

`fsm.transition` has an optional `guard` CFG region, which must be terminated
with an `fsm.return` operation returning a Boolean value to indicate whether
the transition is taken:

```mlir
fsm.machine @foo(%arg0: i1) -> i1 attributes {stateType = i1} {
  fsm.state "IDLE" output  {
    %true = constant true
    fsm.output %true : i1
  } transitions  {
    fsm.transition @BUSY guard  {
      fsm.return %arg0 : i1
    }
  }
  ...
}
```

If a state has more than one transition, multiple transitions are prioritized
in the order that they appear in the transitions region.  Guards must also not
contain any operations with side effects, enabling the evaluation of guards to
be parallelized.  Note that an empty guard region is evaluated as true, which
means the corresponding transition is always taken.

### Action region of transitions and internal variables

To avoid *state explosion*, we introduce `fsm.variable` operation (similar to
the [extended state](https://en.wikipedia.org/wiki/UML_state_machine) in UML
state machine) to represent a variable associated with an FSM instance and can
hold a value of any type, which can be updated through `fsm.update` operations.

`fsm.transition` has an optional `action` CFG region representing the actions
associated with the current transition.  The action region can contain
side-effecting operations.  `fsm.update` must be contained by the action region
of a transition.  The following MLIR code shows a running example:

```mlir
fsm.machine @foo(%arg0: i1) -> i1 attributes {stateType = i1} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16
  fsm.state "IDLE" output  {
    %true = constant true
    fsm.output %true : i1
  } transitions  {
    fsm.transition @BUSY guard  {
      fsm.return %arg0 : i1
    } action  {
      %c256_i16 = constant 256 : i16
      fsm.update %cnt, %c256_i16 : i16
    }
  }
  ...
}
```
