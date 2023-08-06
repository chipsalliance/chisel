# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import circt
from circt.dialects import hw

from circt.ir import (Context, Location, InsertionPoint, IntegerType,
                      IntegerAttr, Module, StringAttr, TypeAttr)

with Context() as ctx, Location.unknown():
  circt.register_dialects(ctx)

  i1 = IntegerType.get_signless(1)
  i2 = IntegerType.get_signless(2)
  i32 = IntegerType.get_signless(32)

  m = Module.create()
  with InsertionPoint(m.body):

    def build(module):
      constI32 = hw.ConstantOp(IntegerAttr.get(i32, 1))
      constI2 = hw.ConstantOp(IntegerAttr.get(i2, 1))
      constI1 = hw.ConstantOp.create(i1, 1)

      # CHECK: Argument 1 has a different element type (i32) than the element type of the array (i1)
      try:
        hw.ArrayCreateOp.create([constI1, constI32])
      except TypeError as e:
        print(e)

      # CHECK: Cannot 'create' an array of length zero
      try:
        hw.ArrayCreateOp.create([])
      except ValueError as e:
        print(e)

      # CHECK: %[[CONST:.+]] = hw.constant 1 : i32

      # CHECK: [[ARRAY1:%.+]] = hw.array_create %[[CONST]], %[[CONST]], %[[CONST]] : i32
      array1 = hw.ArrayCreateOp.create([constI32, constI32, constI32])

      # CHECK: hw.array_get [[ARRAY1]][%c1_i2] : !hw.array<3xi32>
      hw.ArrayGetOp.create(array1, constI2)
      # CHECK: %c-2_i2 = hw.constant -2 : i2
      # CHECK: hw.array_get [[ARRAY1]][%c-2_i2] : !hw.array<3xi32>
      hw.ArrayGetOp.create(array1, 2)

      # CHECK: [[ARRAY2:%.+]] = hw.array_create %[[CONST]] : i32
      array2 = hw.ArrayCreateOp.create([constI32])

      # CHECK: %c0_i0 = hw.constant 0 : i0
      # CHECK: hw.array_get [[ARRAY2]][%c0_i0] : !hw.array<1xi32>
      hw.ArrayGetOp.create(array2, 0)

      # CHECK: [[STRUCT1:%.+]] = hw.struct_create (%c1_i32, %true) : !hw.struct<a: i32, b: i1>
      struct1 = hw.StructCreateOp.create([('a', constI32), ('b', constI1)])

      # CHECK: hw.struct_extract [[STRUCT1]]["a"] : !hw.struct<a: i32, b: i1>
      hw.StructExtractOp.create(struct1, 'a')

    hw.HWModuleOp(name="test", body_builder=build)

  print(m)

  # CHECK: !hw.typealias<@myscope::@myname, i1>
  # CHECK: i1
  # CHECK: i1
  # CHECK: myscope
  # CHECK: myname
  typeAlias = hw.TypeAliasType.get("myscope", "myname", i1)
  print(typeAlias)
  print(typeAlias.canonical_type)
  print(typeAlias.inner_type)
  print(typeAlias.scope)
  print(typeAlias.name)

  pdecl = hw.ParamDeclAttr.get("param1", i32, IntegerAttr.get(i32, 13))
  # CHECK: #hw.param.decl<"param1": i32 = 13>
  print(pdecl)

  pdecl = hw.ParamDeclAttr.get_nodefault("param2", i32)
  # CHECK: #hw.param.decl<"param2": i32>
  print(pdecl)

  # CHECK: #hw.param.decl.ref<"param2"> : i32
  pdeclref = hw.ParamDeclRefAttr.get(ctx, "param2")
  print(pdeclref)

  # CHECK: !hw.int<#hw.param.decl.ref<"param2">>
  pinttype = hw.ParamIntType.get_from_param(ctx, pdeclref)
  print(pinttype)

  pverbatim = hw.ParamVerbatimAttr.get(StringAttr.get("this is verbatim"))
  # CHECK: #hw.param.verbatim<"this is verbatim">
  print(pverbatim)

  inner_sym = hw.InnerSymAttr.get(StringAttr.get("some_sym"))
  # CHECK: #hw<innerSym@some_sym>
  print(inner_sym)
  # CHECK: "some_sym"
  print(inner_sym.symName)

  inner_ref = hw.InnerRefAttr.get(StringAttr.get("some_module"),
                                  StringAttr.get("some_instance"))
  # CHECK: #hw.innerNameRef<@some_module::@some_instance>
  print(inner_ref)
  # CHECK: "some_module"
  print(inner_ref.module)
  # CHECK: "some_instance"
  print(inner_ref.name)

  global_ref = hw.GlobalRefAttr.get(StringAttr.get("foo"))
  # CHECK: #hw.globalNameRef<@foo>
  print(global_ref)
