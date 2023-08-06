# Python CIRCT Design Entry (PyCDE)

PyCDE is a python API for hardware related activities. It was intended to make
CIRCT functionality easy to expose to Python developers. PyCDE, therefore,
mostly maps down to CIRCT operations through "a bit" of syntactic sugar. The
vast majority of the work is done by CIRCT.

## Installation

Because PyCDE is rapidly evolving, we recommend always using the latest
pre-release. New packages are posted nightly if there have been updates (and so
long as the build and CI are working).

```
pip install pycde --pre
```

or [compile it yourself](compiling.md) (not recommended).

## Hello world!

The following example demonstrates a simple module that ors two integers:

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
