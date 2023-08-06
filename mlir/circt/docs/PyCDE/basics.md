# PyCDE Basics

You know what's more difficult than forcing yourself to write documentation?
Maintaining it! We apologize for the inevitable inaccuracies.

## Modules, Generators, and Systems

```python
from pycde import Input, Output, Module, System
from pycde import generator
from pycde.types import Bits

class OrInts(Module):
    a = Input(Bits(32))
    b = Input(Bits(32))
    c = Output(Bits(32))

    @generator
    def construct(self):
        self.c = self.a | self.b


system = System([OrInts], name="ExampleSystem", output_directory="exsys")
system.compile()
```

Hardware modules extend `pycde.Module`. They define any number of typed inputs
and outputs by setting class members.

The `pycde.generator` decorator is used to denote the module construction code.
It is called once per module at `compile` time. Unlike standard python, the body
of the generator does not have access to instance members through `self` -- just
ports. All of the output ports **must** be set.

In order to compile a hardware system, use `System` constructing it with the
class or list of classes which are the top modules. Other modules instantiated
in the generators get generated and emitted automatically and recursively (i.e.
only the root of the hierarchy needs to be given). `name` defaults to the top
module name. `output_directory` defaults to `name`. The `compile` method outputs
a "build package" containing the generated system into `output_directory`.

## Instantiating Modules

Modules can be instantiated in other modules. The following example defines a module that instantiates the `AddInts` module defined above.

```python
class Top(Module):
    a = Input(Bits(32))
    b = Input(Bits(32))
    c = Output(Bits(32))

    @generator
    def construct(self):
        or_ints = OrInts(a=self.a, b=self.b)
        self.c = or_ints.c


system = System([Top], name="ExampleSystem")
system.compile()
```

The constructor of a `Module` expects named keyword arguments for each input
port. These keyword arguments can be any PyCDE Signal. The above example uses
module inputs. Instances of `Module`s support named access to output port values
like `add_insts.c`. These output port values are PyCDE Signals, and can be use
to further connect modules or instantiate CIRCT dialect operations.

## Types & Signals

Since CIRCT primarily targets hardware not software, it defines its own types.
PyCDE exposes them through the `pycde.types.Type` class hierarchy.  PyCDE
signals represent values on the target device. Signals have a particular `Type`
(which is distinct from the signal objects' Python `type`) stored in their
`type` instance member. `Type`s are heretofor referred to interchangably as
"PyCDE type", "CIRCT type", or "Type".

All signals extend the `pycde.signals.Signal` class, specialized by their Type.
The various specializations often include operator overrides to perform common
operations, as demonstrated in the hello world example's `|` bitwise or.

Note that the CIRCT type conveys information not just about the data type but
can also specify the signaling mechanism. In other words, a signal does not
necessarily imply standard wires (though it usually does).

** For CIRCT/MLIR developers: "signals" map 1:1 to MLIR Values.

### Constants and Python object conversion

Some Python objects (e.g. int, dict, list) can be converted to constants in
hardware. PyCDE tries its best to make this automatic, but it sometimes needs to
know the Type. For instance, we don't know the desired bitwidth of an int. In
some cases we default to the required number of bits to represent a number, but
sometimes that fails.

In those cases, you must manually specify the Type. So `Bits(16)(i)` would
create a 16-bit constant of `i`.

### Scalars

`Bits(width)` models a bitvector. Allows indexing, slicing, bitwise operations, etc. No math operations.

`UInt(width)`, `SInt(width)` math.

### Arrays

`Bits(32) * 10` creates an array of 32-bits of length 10.

`Bits(32) * 10 * 12` creates an array of arrays.

### Structs

```python
from pycde import Input, Output, generator, System, Module
from pycde.types import Bits
from pycde.signals import Struct, BitsSignal

class ExStruct(Struct):
  a: Bits(4)
  b: Bits(32)

  def get_b_xor(self, x: int) -> BitsSignal:
    return self.b ^ Bits(32)(x)


class StructExample(Module):
  inp1 = Input(ExStruct)
  out1 = Output(Bits(32))
  out2 = Output(Bits(4))
  out3 = Output(ExStruct)

  @generator
  def build(self):
    self.out1 = self.inp1.get_b_xor(5432)
    self.out2 = self.inp1.a
    self.out3 = ExStruct(a=self.inp1.a, b=42)
```


### NumPy features

PyCDE supports a subset of numpy array transformations (see `pycde/ndarray.py`)
that can be used to do complex reshaping and transformation of multidimensional
arrays.

The numpy functionality is provided by the `NDArray` class, which creates a view
on top of existing SSA values. Users may choose to perform transformations
directly on `ListSignal`s:

```python
class M1(Module):
  in1 = Input(dim(Bits(32), 4, 8))
  out = Output(dim(Bits(32), 2, 16))

  @generator
  def build(self):
    self.out = self.in1.transpose((1, 0)).reshape((16, 2))
    # Under the hood, this resolves to
    # Matrix(from_value=
    #    Matrix(from_value=ports.in1).transpose((1,0)).to_circt())
    #  .reshape(16, 2).to_circt()
```

or manually manage a `NDArray` object.

```python
class M1(Module):
  in1 = Input(dim(Bits(32), 4, 8))
  out = Output(dim(Bits(32), 2, 16))

  @generator
  def build(self):
    m = NDArray(from_value=self.in1).transpose((1, 0)).reshape((16, 2))
    self.out = m.to_circt()
```

Manually managing the NDArray object allows for postponing materialization
(`to_circt()`) until all transformations have been applied.  In short, this
allows us to do as many transformations as possible in software, before emitting
IR. Note however, that this might reduce debugability of the generated hardware
due to the lack of `sv.wire`s in between each matrix transformation.

For further usage examples, see `PyCDE/test/test_ndarray.py`, and inspect
`ListSignal` in `pycde/signals.py` for the full list of implemented numpy
functions.

## External Modules

External modules are how PyCDE and CIRCT support interacting with existing
System Verilog or Verilog modules. They must be declared and the ports must
match the externally defined implementation in SystemVerilog or other language.
We have no way of checking that they do indeed match so it'll be up to the EDA
synthesizer (and they generally do a poor job reporting mismatches).

In PyCDE, an external module is any module without a generator.

```python
class MulInts(Module):
    module_name = "MyMultiplier"
    a = Input(Bits(32))
    b = Input(Bits(32))
    c = Output(Bits(32))
```

The `MyMultiplier` module is declared in the default output file, `ExampleSystem/ExampleSystem.sv`.


## Parameterized modules

```python
from pycde import modparams

@modparams
def AddInts(width: int):

  class AddInts(Module):
    a = Input(UInt(width))
    b = Input(UInt(width))
    c = Output(UInt(width + 1))

    @generator
    def build(self):
      self.c = self.a + self.b

  return AddInts


class Top(Module):
  a = Input(UInt(32))
  b = Input(UInt(32))
  c = Output(UInt(33))

  @generator
  def construct(self):
    add_ints_m = AddInts(32)
    add_ints = add_ints_m(a=self.a, b=self.b)
    self.c = add_ints.c
```

In order to "parameterize" a module, simply return one from a function. Said
function must be decorated with `modparams` to inform PyCDE that the returned
module is a parameterized one. The `modparams` decorator does several things
including: (1) memoizing the parameterization function, and (2) automatically
derive a module name which includes the parameter values (for module name
uniqueness).

PyCDE does not produce parameterized SystemVerilog modules! The specialization
happens with Python code, which is far more powerful than SystemVerilog
parameterization constructs.

## External parameterized modules

Just like internally defined parameterized modules, leave off the generator and
PyCDE will output SystemVerilog instantations with the module parameters. The
parameter types are best effort based on the first instantiation encountered.

```python
from pycde import modparams, Module

@modparams
def AddInts(width: int):

  class AddInts(Module):
    a = Input(UInt(width))
    b = Input(UInt(width))
    c = Output(UInt(width + 1))

  return AddInts


class Top(Module):
  a = Input(UInt(32))
  b = Input(UInt(32))
  c = Output(UInt(33))

  @generator
  def construct(self):
    add_ints_m = AddInts(32)
    add_ints = add_ints_m(a=self.a, b=self.b)
    self.c = add_ints.c
```

For the instantiation produces:

```verilog
  AddInts #(
    .width(64'd32)
  ) AddInts (
    .a (a),
    .b (b),
    .c (c)
  );
```

## Using CIRCT dialects directly (instead of with PyCDE syntactic sugar)

Generally speaking, don't.

One can directly instantiate CIRCT operations through
`pycde.dialects.<dialect_name>`. The CIRCT operations contained therein provide
thin wrappers around the CIRCT operations to adapt them to [PyCDE
Signals](#signals) by overriding each operation's constructor. This
auto-wrapper, however, does not always "just work" depending on the complexity
of the operation it is attemption to wrap. So don't use it unless you know what
you're doing. User beware. Warranty voided. Caveat emptor. etc.
