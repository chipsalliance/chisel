# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1 | FileCheck %s
# RUN: FileCheck %s --input-file %t/hw/Test.tcl --check-prefix=OUTPUT

import pycde
import pycde.dialects.hw

from pycde.devicedb import (PhysLocation, PrimitiveDB, PrimitiveType)

import sys

from pycde.instance import InstanceDoesNotExistError, Instance, RegInstance
from pycde.module import AppID, Module


class Nothing(Module):
  pass


class Delay(Module):
  clk = pycde.Clock()
  x = pycde.Input(pycde.types.i1)
  y = pycde.Output(pycde.types.i1)

  @pycde.generator
  def construct(mod):
    r = mod.x.reg(mod.clk, appid=AppID("reg", 0))
    mod.y = r
    # CHECK: r appid: reg[0]
    print(f"r appid: {r.appid}")
    r.appid = AppID("reg", 4)
    # CHECK: r appid: reg[4]
    print(f"r appid: {r.appid}")


class UnParameterized(Module):
  clk = pycde.Clock()
  x = pycde.Input(pycde.types.i1)
  y = pycde.Output(pycde.types.i1)

  @pycde.generator
  def construct(mod):
    Nothing().name = "nothing_inst"
    mod.y = Delay(clk=mod.clk, x=mod.x).y


class Test(Module):
  clk = pycde.Clock()

  @pycde.generator
  def build(ports):
    c1 = pycde.dialects.hw.ConstantOp(pycde.types.i1, 1)
    UnParameterized(clk=ports.clk, x=c1, appid=AppID("unparam",
                                                     0)).name = "unparam"
    UnParameterized(clk=ports.clk, x=c1, appid=AppID("unparam",
                                                     1)).name = "unparam"


t = pycde.System([Test], name="Test", output_directory=sys.argv[1])
t.generate(["construct"])
t.print()
# CHECK: <pycde.Module: Test inputs: [('clk', Bits<1>)] outputs: []>
Test.print()
# CHECK: <pycde.Module: UnParameterized inputs: [('clk', Bits<1>), ('x', Bits<1>)] outputs: [('y', Bits<1>)]>
UnParameterized.print()

print(PhysLocation(PrimitiveType.DSP, 39, 25))
# CHECK: PhysLocation<PrimitiveType.DSP, x:39, y:25, num:0>

# CHECK-LABEL: === Hierarchy
print("=== Hierarchy")
# CHECK-NEXT: <instance: []>
# CHECK-NEXT: <instance: [UnParameterized]>
# CHECK-NEXT: <instance: [UnParameterized, Nothing]>
# CHECK-NEXT: <instance: [UnParameterized, Delay]>
# CHECK-NEXT: <instance: [UnParameterized, Delay, x__reg1]>
# CHECK-NEXT: <instance: [UnParameterized_1]>
# CHECK-NEXT: <instance: [UnParameterized_1, Nothing]>
# CHECK-NEXT: <instance: [UnParameterized_1, Delay]>
# CHECK-NEXT: <instance: [UnParameterized_1, Delay, x__reg1]>
test_inst = t.get_instance(Test)
test_inst.walk(lambda inst: print(inst))

reg = test_inst.unparam[0].reg[4]
print(f"unparam[0].reg[4]: {reg}")
# CHECK-NEXT: unparam[0].reg[4]: <instance: [UnParameterized, Delay, x__reg1]>
print(f"unparam[0].reg[4] appid: {reg.appid}")
# CHECK-NEXT: unparam[0].reg[4] appid: reg[4]

for u in test_inst.unparam:
  print(f"unparam list item: {u}")
  print(f"unparam list item appid: {u.appid}")
# CHECK-NEXT: unparam list item: <instance: [UnParameterized]>
# CHECK-NEXT: unparam list item appid: unparam[0]
# CHECK-NEXT: unparam list item: <instance: [UnParameterized_1]>
# CHECK-NEXT: unparam list item appid: unparam[1]

# Set up the primitive locations. Errors out if location is placed but doesn't
# exist.
primdb = PrimitiveDB()
primdb.add_coords("M20K", 39, 25)
primdb.add_coords(PrimitiveType.M20K, 15, 25)
primdb.add_coords(PrimitiveType.M20K, 40, 40)
primdb.add_coords("DSP", 0, 10)
primdb.add_coords(PrimitiveType.DSP, 1, 12)
primdb.add_coords(PrimitiveType.DSP, 39, 90)
primdb.add(PhysLocation(PrimitiveType.DSP, 39, 25))
t.createdb(primdb)

# CHECK-LABEL: === Placements
print("=== Placements")


def place_inst(inst: Instance):
  if inst.name == "UnParameterized_1":
    inst.place(PrimitiveType.M20K, 39, 25, 0, "memory|bank")
  if inst.path_names == ["UnParameterized", "Nothing"]:
    inst.add_named_attribute("FOO", "TRUE")


t.get_instance(Test).walk(place_inst)

# TODO: Add back physical region support

# region1 = t.create_physical_region("region_0").add_bounds((0, 10), (0, 10))
# region1.add_bounds((10, 20), (10, 20))
# ref = region1.get_ref()
# instance_attrs.lookup(pycde.AppID("UnParameterized",
#                                   "Nothing")).add_attribute(ref)

# region_anon = t.create_physical_region()
# assert region_anon._physical_region.sym_name.value == "region_1"

# region_explicit = t.create_physical_region("region_1")
# assert region_explicit._physical_region.sym_name.value == "region_1_1"

test_inst = t.get_instance(Test)
t.createdb()

test_inst["UnParameterized"].place(PrimitiveType.M20K, 15, 25, 0,
                                   ["memory", "bank"])
test_inst["UnParameterized"].add_named_attribute("FOO", "OFF",
                                                 ["memory", "bank"])
test_inst["UnParameterized"]["Nothing"].place(PrimitiveType.DSP, 39, 25, 0)

test_inst.walk(lambda inst: print(
    inst, inst.locations if not isinstance(inst, RegInstance) else "None"))
# CHECK-DAG: <instance: []> []
# CHECK-DAG: <instance: [UnParameterized]> [(PhysLocation<PrimitiveType.M20K, x:15, y:25, num:0>, '|memory|bank')]
# CHECK-DAG: <instance: [UnParameterized, Nothing]> [(PhysLocation<PrimitiveType.DSP, x:39, y:25, num:0>, None)]
# CHECK-DAG: <instance: [UnParameterized_1]> [(PhysLocation<PrimitiveType.M20K, x:39, y:25, num:0>, '|memory|bank')]
# CHECK-DAG: <instance: [UnParameterized_1, Nothing]> []
# CHECK-DAG: <instance: [UnParameterized_1, Delay, x__reg1]>

# TODO: add back anonymous reservations

# reserved_loc = PhysLocation(PrimitiveType.M20K, 40, 40, 0)
# entity_extern = t.create_entity_extern("tag")
# test_inst.placedb.reserve_location(reserved_loc, entity_extern)

# CHECK: PhysLocation<PrimitiveType.DSP, x:39, y:25, num:0> has ['UnParameterized', 'Nothing']
loc = PhysLocation(PrimitiveType.DSP, 39, 25, 0)
print(f"{loc} has {t.placedb.get_instance_at(loc)[0].path_names}")

assert t.placedb.get_instance_at(PhysLocation(PrimitiveType.M20K, 0, 0,
                                              0)) is None
# assert test_inst.placedb.get_instance_at(reserved_loc) is not None

# CHECK-LABEL: === Force-clean all the caches and test rebuilds
print("=== Force-clean all the caches and test rebuilds")
t._op_cache.release_ops()

test_inst.walk(lambda inst: print(
    inst, inst.locations if not isinstance(inst, RegInstance) else "None"))
# CHECK: <instance: []> []
# CHECK: <instance: [UnParameterized]> [(PhysLocation<PrimitiveType.M20K, x:15, y:25, num:0>, '|memory|bank')]
# CHECK: <instance: [UnParameterized, Nothing]> [(PhysLocation<PrimitiveType.DSP, x:39, y:25, num:0>, None)]
# CHECK: <instance: [UnParameterized_1]> [(PhysLocation<PrimitiveType.M20K, x:39, y:25, num:0>, '|memory|bank')]
# CHECK: <instance: [UnParameterized_1, Nothing]> []

# CHECK: PhysLocation<PrimitiveType.DSP, x:39, y:25, num:0> has (<instance: [UnParameterized, Nothing]>, None)
print(f"{loc} has {t.placedb.get_instance_at(loc)}")

foo_inst = t.get_instance(Test, "foo_inst")
foo_inst["UnParameterized"]["Nothing"].place(PrimitiveType.DSP, 39, 90, 0)

print("=== Pre-pass mlir dump")
t.print()

print("=== Running passes")
t.run_passes()

try:
  test_inst._dyn_inst()
  assert False
except InstanceDoesNotExistError:
  pass

print("=== Final mlir dump")
t.print()

# OUTPUT-LABEL: proc Test_config { parent }
# OUTPUT-NOT:  set_location_assignment M20K_X40_Y40
# OUTPUT-DAG:  set_location_assignment M20K_X39_Y25_N0 -to $parent|UnParameterized_1|memory|bank
# OUTPUT-DAG:  set_location_assignment M20K_X15_Y25_N0 -to $parent|UnParameterized|memory|bank
# OUTPUT-DAG:  set_instance_assignment -name FOO OFF -to $parent|UnParameterized|memory|bank
# OUTPUT-DAG:  set_location_assignment MPDSP_X39_Y25_N0 -to $parent|UnParameterized|Nothing
# OUTPUT-DAG:  set_instance_assignment -name FOO TRUE -to $parent|UnParameterized|Nothing
# OUTPUT-NOT:  set_location_assignment
# OUTPUT-NEXT: }
# OUTPUT-LABEL: proc Test_foo_inst_config { parent } {
# OUTPUT:         set_location_assignment MPDSP_X39_Y90_N0 -to $parent|UnParameterized|Nothing
# OUTPUT:       }
t.compile()
