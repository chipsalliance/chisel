// RUN: circt-opt %s --msft-partition -verify-diagnostics | FileCheck %s

hw.globalRef @ref1 [#hw.innerNameRef<@top::@b>, #hw.innerNameRef<@B::@unit1>] {
  "loc" = #msft.physloc<M20K, 0, 0, 0>
}

hw.globalRef @ref2 [#hw.innerNameRef<@top::@b>, #hw.innerNameRef<@B::@c>, #hw.innerNameRef<@C::@unit3>] {
  "loc" = #msft.physloc<M20K, 0, 0, 1>
}

hw.globalRef @ref3 [#hw.innerNameRef<@top::@b2>, #hw.innerNameRef<@B2::@unit1>] {
  "loc" = #msft.physloc<M20K, 0, 0, 0>
}

hw.globalRef @ref4 [#hw.innerNameRef<@InnerPartLeafLoc::@inner>, #hw.innerNameRef<@Inner::@leaf>] {
  "loc" = #msft.physloc<M20K, 0, 0, 0>
}

msft.module @top {} (%clk : i1) -> (out1: i2, out2: i2, out3: i2) {
  msft.partition @part1, "dp"

  %res1, %res4 = msft.instance @b @B(%clk) { circt.globalRef = [#hw.globalNameRef<@ref1>, #hw.globalNameRef<@ref2>]} : (i1) -> (i2, i2)
  %res3 = msft.instance @b2 @B2(%clk) { circt.globalRef = [#hw.globalNameRef<@ref3>]} : (i1) -> (i2)

  %c0 = hw.constant 0 : i2
  %res2 = msft.instance @unit1 @Extern(%c0) { targetDesignPartition = @top::@part1 }: (i2) -> (i2)

  msft.output %res1, %res2, %res4 : i2, i2, i2
}

// CHECK:  hw.globalRef @ref1 [#hw.innerNameRef<@top::@part1>, #hw.innerNameRef<@dp::@b.unit1>] {loc = #msft.physloc<M20K, 0, 0, 0>}
// CHECK:  hw.globalRef @ref2 [#hw.innerNameRef<@top::@part1>, #hw.innerNameRef<@dp::@b.c.unit3>] {loc = #msft.physloc<M20K, 0, 0, 1>}
// CHECK:  hw.globalRef @ref3 [#hw.innerNameRef<@top::@part1>, #hw.innerNameRef<@dp::@b2.unit1>] {loc = #msft.physloc<M20K, 0, 0, 0>}
// CHECK:  hw.globalRef @ref4 [#hw.innerNameRef<@InnerPartLeafLoc::@part>, #hw.innerNameRef<@InnerLeafPartition::@inner>, #hw.innerNameRef<@Inner::@leaf>] {loc = #msft.physloc<M20K, 0, 0, 0>}

// CHECK-LABEL:  msft.module @dp {} (%b.seq.compreg.clk: i1) -> (b.unit2.b.unit2.foo_x: i2, unit1.unit1.foo_x: i2) {
// CHECK:    %b.unit1.foo_x = msft.instance @b.unit1 @Extern(%c1_i2)  {circt.globalRef = [#hw.globalNameRef<@ref1>], targetDesignPartition = @top::@part1} : (i2) -> i2
// CHECK:    %b.seq.compreg = seq.compreg %b.unit1.foo_x, %b.seq.compreg.clk {targetDesignPartition = @top::@part1} : i2
// CHECK:    %b.c.unit3.foo_x = msft.instance @b.c.unit3 @Extern(%b.seq.compreg)  {circt.globalRef = [#hw.globalNameRef<@ref2>], targetDesignPartition = @top::@part1} : (i2) -> i2
// CHECK:    %b.unit2.foo_x = msft.instance @b.unit2 @Extern(%b.c.unit3.foo_x)  {targetDesignPartition = @top::@part1} : (i2) -> i2
// CHECK:    %b2.unit1.foo_x = msft.instance @b2.unit1 @Extern(%c1_i2)  {circt.globalRef = [#hw.globalNameRef<@ref3>], targetDesignPartition = @top::@part1} : (i2) -> i2
// CHECK:    %unit1.foo_x = msft.instance @unit1 @Extern(%c0_i2)  {targetDesignPartition = @top::@part1} : (i2) -> i2
// CHECK:    %c1_i2 = hw.constant 1 : i2
// CHECK:    %c0_i2 = hw.constant 0 : i2
// CHECK:    msft.output %b.unit2.foo_x, %unit1.foo_x : i2, i2

// CHECK-LABEL:  msft.module @top {} (%clk: i1) -> (out1: i2, out2: i2, out3: i2) {
// CHECK:    %part1.b.unit2.b.unit2.foo_x, %part1.unit1.unit1.foo_x = msft.instance @part1 @dp(%clk) {circt.globalRef = [#hw.globalNameRef<@ref1>, #hw.globalNameRef<@ref2>, #hw.globalNameRef<@ref3>]} : (i1) -> (i2, i2)
// CHECK:    %b.y = msft.instance @b @B() : () -> i2
// CHECK:    msft.instance @b2 @B2() : () -> ()
// CHECK:    msft.output %part1.b.unit2.b.unit2.foo_x, %part1.unit1.unit1.foo_x, %b.y : i2, i2, i2

// CHECK-LABEL:  msft.module @B {} () -> (y: i2) {
// CHECK:    %c1_i2 = hw.constant 1 : i2
// CHECK:    msft.instance @c @C() : () -> ()
// CHECK:    msft.output %c1_i2 : i2

// CHECK-LABEL:  msft.module @B2 {} () {
// CHECK:    msft.output
// CHECK-LABEL:  msft.module @C {} () {
// CHECK:    msft.output

msft.module.extern @Extern (%foo_a: i2) -> (foo_x: i2)

msft.module @B {} (%clk : i1) -> (x: i2, y: i2)  {
  %c1 = hw.constant 1 : i2
  %0 = msft.instance @unit1 @Extern(%c1) { targetDesignPartition = @top::@part1, circt.globalRef = [#hw.globalNameRef<@ref1>]}: (i2) -> (i2)
  %1 = seq.compreg %0, %clk { targetDesignPartition = @top::@part1 } : i2

  %3 = msft.instance @c @C(%1) { circt.globalRef = [#hw.globalNameRef<@ref2>]}: (i2) -> (i2)

  %2 = msft.instance @unit2 @Extern(%3) { targetDesignPartition = @top::@part1 }: (i2) -> (i2)

  msft.output %2, %c1: i2, i2
}

msft.module @B2 {} (%clk : i1) -> (x: i2) {
  %c1 = hw.constant 1 : i2
  %0 = msft.instance @unit1 @Extern(%c1) { targetDesignPartition = @top::@part1, circt.globalRef = [#hw.globalNameRef<@ref3>]}: (i2) -> (i2)
  msft.output %0 : i2
}

msft.module @C {} (%in : i2) -> (out: i2)  {
  %0 = msft.instance @unit3 @Extern(%in) { targetDesignPartition = @top::@part1, circt.globalRef = [#hw.globalNameRef<@ref2>]} : (i2) -> (i2)
  msft.output %0 : i2
}

msft.module @TopComplex {} (%clk : i1, %arr_in: !hw.array<4xi5>, %datain: i5, %valid: i1) -> (out2: !hw.struct<data: i5, valid: i1>, out2: i5) {
  msft.partition @part2, "dp_complex"

  %mut_arr = msft.instance @b @Array(%arr_in) : (!hw.array<4xi5>) -> (!hw.array<4xi5>)
  %c0 = hw.constant 0 : i2
  %a0 = hw.array_get %mut_arr[%c0] : !hw.array<4xi5>, i2
  %c1 = hw.constant 1 : i2
  %a1 = hw.array_get %mut_arr[%c1] : !hw.array<4xi5>, i2
  %c2 = hw.constant 2 : i2
  %a2 = hw.array_get %mut_arr[%c2] : !hw.array<4xi5>, i2
  %c3 = hw.constant 3 : i2
  %a3 = hw.array_get %mut_arr[%c3] : !hw.array<4xi5>, i2

  %res1 = comb.add %a0, %a1 { targetDesignPartition = @TopComplex::@part2 } : i5

  %din_struct = hw.struct_create (%datain, %valid) : !hw.struct<data: i5, valid: i1>
  %out = msft.instance @structMod @Struct (%din_struct) : (!hw.struct<data: i5, valid: i1>) -> (!hw.struct<data: i5, valid: i1>) 

  msft.output %out, %res1 : !hw.struct<data: i5, valid: i1>, i5
}

msft.module.extern @ExternI5 (%foo_a: i5) -> (foo_x: i5)

msft.module @Array {} (%arr_in: !hw.array<4xi5>) -> (arr_out: !hw.array<4xi5>) {
  %c0 = hw.constant 0 : i2
  %in0 = hw.array_get %arr_in[%c0] : !hw.array<4xi5>, i2
  %out0 = msft.instance @unit2 @ExternI5(%in0) { targetDesignPartition = @TopComplex::@part2 }: (i5) -> (i5)
  %c1 = hw.constant 1 : i2
  %in1 = hw.array_get %arr_in[%c1] : !hw.array<4xi5>, i2
  %out1 = msft.instance @unit2 @ExternI5(%in1) { targetDesignPartition = @TopComplex::@part2 }: (i5) -> (i5)
  %c2 = hw.constant 2 : i2
  %in2 = hw.array_get %arr_in[%c2] : !hw.array<4xi5>, i2
  %out2 = msft.instance @unit2 @ExternI5(%in2) { targetDesignPartition = @TopComplex::@part2 }: (i5) -> (i5)
  %c3 = hw.constant 3 : i2
  %in3 = hw.array_get %arr_in[%c3] : !hw.array<4xi5>, i2
  %out3 = msft.instance @unit2 @ExternI5(%in3) { targetDesignPartition = @TopComplex::@part2 }: (i5) -> (i5)
  %arr_out = hw.array_create %out0, %out1, %out2, %out3 : i5
  msft.output %arr_out : !hw.array<4xi5>
}

msft.module @Struct {} (%in: !hw.struct<data: i5, valid: i1>) -> (out: !hw.struct<data: i5, valid: i1>) {
  %d = hw.struct_extract %in["data"] : !hw.struct<data: i5, valid: i1>
  %valid = hw.struct_extract %in["valid"] : !hw.struct<data: i5, valid: i1>
  %dprime = msft.instance @dataModif @ExternI5(%d) { targetDesignPartition = @TopComplex::@part2 } : (i5) -> (i5)
  %dpp = msft.instance @dataModif @ExternI5(%dprime) { targetDesignPartition = @TopComplex::@part2 } : (i5) -> (i5)
  %inprime = hw.struct_create (%dpp, %valid) : !hw.struct<data: i5, valid: i1>
  msft.output %inprime : !hw.struct<data: i5, valid: i1>
}

// CHECK-LABEL:  msft.module @dp_complex {} (%structMod.dataModif.datain: i5, %hw.array_get.arr_in: !hw.array<4xi5>, %hw.struct_create.valid: i1) -> (comb.add: i5, hw.struct_create: !hw.struct<data: i5, valid: i1>) {
// CHECK:    %b.unit2.foo_x = msft.instance @b.unit2 @ExternI5(%1)  {targetDesignPartition = @TopComplex::@part2} : (i5) -> i5
// CHECK:    %b.unit2.foo_x_0 = msft.instance @b.unit2 @ExternI5(%2)  {targetDesignPartition = @TopComplex::@part2} : (i5) -> i5
// CHECK:    %b.unit2.foo_x_1 = msft.instance @b.unit2 @ExternI5(%3)  {targetDesignPartition = @TopComplex::@part2} : (i5) -> i5
// CHECK:    %b.unit2.foo_x_2 = msft.instance @b.unit2 @ExternI5(%4)  {targetDesignPartition = @TopComplex::@part2} : (i5) -> i5
// CHECK:    %0 = comb.add %b.unit2.foo_x_2, %b.unit2.foo_x_1 {targetDesignPartition = @TopComplex::@part2} : i5
// CHECK:    %structMod.dataModif.foo_x = msft.instance @structMod.dataModif @ExternI5(%structMod.dataModif.datain)  {targetDesignPartition = @TopComplex::@part2} : (i5) -> i5
// CHECK:    %structMod.dataModif.foo_x_3 = msft.instance @structMod.dataModif @ExternI5(%structMod.dataModif.foo_x)  {targetDesignPartition = @TopComplex::@part2} : (i5) -> i5
// CHECK:    %1 = hw.array_get %hw.array_get.arr_in[%c0_i2] : !hw.array<4xi5>
// CHECK:    %2 = hw.array_get %hw.array_get.arr_in[%c1_i2] : !hw.array<4xi5>
// CHECK:    %3 = hw.array_get %hw.array_get.arr_in[%c-2_i2] : !hw.array<4xi5>
// CHECK:    %4 = hw.array_get %hw.array_get.arr_in[%c-1_i2] : !hw.array<4xi5>
// CHECK:    %5 = hw.struct_create (%structMod.dataModif.foo_x_3, %hw.struct_create.valid) : !hw.struct<data: i5, valid: i1>
// CHECK:    %c0_i2 = hw.constant 0 : i2
// CHECK:    %c1_i2 = hw.constant 1 : i2
// CHECK:    %c-2_i2 = hw.constant -2 : i2
// CHECK:    %c-1_i2 = hw.constant -1 : i2
// CHECK:    msft.output %0, %5 : i5, !hw.struct<data: i5, valid: i1>

// CHECK-LABEL:  msft.module @TopComplex {} (%clk: i1, %arr_in: !hw.array<4xi5>, %datain: i5, %valid: i1) -> (out2: !hw.struct<data: i5, valid: i1>, out2: i5) {
// CHECK:    %part2.comb.add, %part2.hw.struct_create = msft.instance @part2 @dp_complex(%datain, %arr_in, %valid)  {circt.globalRef = []} : (i5, !hw.array<4xi5>, i1) -> (i5, !hw.struct<data: i5, valid: i1>)
// CHECK:    msft.instance @b @Array()  : () -> ()
// CHECK:    msft.instance @structMod @Struct()  : () -> ()
// CHECK:    msft.output %part2.hw.struct_create, %part2.comb.add : !hw.struct<data: i5, valid: i1>, i5

// CHECK-LABEL:  msft.module @Array {} () {
// CHECK:    msft.output

// CHECK-LABEL:  msft.module @Struct {} () {
// CHECK:    msft.output

msft.module @InnerPartLeafLoc {} (%clk : i1) -> (out1: i2, out2: i2) {
  msft.partition @part, "InnerLeafPartition"

  %res1, %res2 = msft.instance @inner @Inner(%clk) { targetDesignPartition = @InnerPartLeafLoc::@part, circt.globalRef = [#hw.globalNameRef<@ref4>] } : (i1) -> (i2, i2)

  msft.output %res1, %res2 : i2, i2
}

msft.module @Inner {} (%clk: i1) -> (out1: i2, out2: i2) {
  %c0 = hw.constant 0 : i2

  %0 = msft.instance @leaf @Extern(%c0) { circt.globalRef = [#hw.globalNameRef<@ref4>] }: (i2) -> (i2)

  msft.output %0, %0 : i2, i2
}

// CHECK-LABEL: msft.module @InnerLeafPartition
// CHECK: msft.instance @inner @Inner{{.+}}circt.globalRef = [#hw.globalNameRef<@ref4>]

// CHECK-LABEL: msft.module @InnerPartLeafLoc
// CHECK: msft.instance @part @InnerLeafPartition{{.+}}circt.globalRef = [#hw.globalNameRef<@ref4>]

// CHECK-LABEL: msft.module @Inner
// CHECK: msft.instance @leaf @Extern{{.+}}circt.globalRef = [#hw.globalNameRef<@ref4>]
