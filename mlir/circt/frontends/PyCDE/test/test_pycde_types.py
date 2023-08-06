# RUN: %PYTHON% %s | FileCheck %s

from pycde import dim, types, Input, Output, generator, System, Module
from pycde.types import bit, Bits, List, StructType, TypeAlias, UInt
from pycde.testing import unittestmodule
from pycde.signals import Struct, UIntSignal

# CHECK: [('foo', Bits<1>), ('bar', Bits<13>)]
st1 = StructType({"foo": types.i1, "bar": types.i13})
print(st1.fields)
# CHECK: Bits<1>
print(st1.foo)

array1 = dim(types.ui6)
# CHECK: UInt<6>
print(array1)

array2 = Bits(6) * 10 * 12
# CHECK: Bits<6>[10][12]
print(array2)

int_alias = TypeAlias(Bits(8), "myname1")
# CHECK: myname1
print(int_alias)
assert int_alias == types.int(8, "myname1")

# CHECK: struct { a: Bits<1>, b: SInt<1>}
struct = types.struct({"a": types.i1, "b": types.si1})
print(struct)

dim_alias = dim(1, 8, name="myname5")

# CHECK: List<Bits<5>>
i5list = List(Bits(5))
print(i5list)


class Dummy(Module):
  pass


# CHECK: hw.type_scope @pycde
# CHECK: hw.typedecl @myname1 : i8
# CHECK: hw.typedecl @myname5 : !hw.array<8xi1>
# CHECK-NOT: hw.typedecl @myname1
# CHECK-NOT: hw.typedecl @myname5
m = System(Dummy)
TypeAlias.declare_aliases(m)
TypeAlias.declare_aliases(m)
m.print()

assert bit == Bits(1)


class ExStruct(Struct):
  a: Bits(4)
  b: UInt(32)

  def get_b_plus1(self) -> UIntSignal:
    return self.b + 1


print(ExStruct)


# CHECK-LABEL:  msft.module @TestStruct {} (%inp1: !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>) -> (out1: ui33, out2: !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>)
# CHECK-NEXT:     %b = hw.struct_extract %inp1["b"] {sv.namehint = "inp1__b"} : !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>
# CHECK-NEXT:     [[r0:%.+]] = hwarith.constant 1 : ui1
# CHECK-NEXT:     [[r1:%.+]] = hwarith.add %b, [[r0]] : (ui32, ui1) -> ui33
# CHECK-NEXT:     %a = hw.struct_extract %inp1["a"] {sv.namehint = "inp1__a"} : !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>
# CHECK-NEXT:     %b_0 = hw.struct_extract %inp1["b"] {sv.namehint = "inp1__b"} : !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>
# CHECK-NEXT:     [[r2:%.+]] = hwarith.constant 1 : ui1
# CHECK-NEXT:     [[r3:%.+]] = hwarith.add %b_0, [[r2]] : (ui32, ui1) -> ui33
# CHECK-NEXT:     [[r4:%.+]] = hwarith.cast [[r3]] : (ui33) -> ui32
# CHECK-NEXT:     [[r5:%.+]] = hw.struct_create (%a, [[r4]]) : !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>
# CHECK-NEXT:     msft.output [[r1]], [[r5]] : ui33, !hw.typealias<@pycde::@ExStruct, !hw.struct<a: i4, b: ui32>>
@unittestmodule()
class TestStruct(Module):
  inp1 = Input(ExStruct)
  out1 = Output(UInt(33))
  out2 = Output(ExStruct)

  @generator
  def build(self):
    self.out1 = self.inp1.get_b_plus1()
    s = ExStruct(a=self.inp1.a, b=self.inp1.get_b_plus1().as_uint(32))
    assert type(s) is ExStruct._get_value_class()
    self.out2 = s
