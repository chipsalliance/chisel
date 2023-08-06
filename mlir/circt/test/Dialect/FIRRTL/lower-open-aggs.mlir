// RUN: circt-opt --pass-pipeline="builtin.module(firrtl.circuit(firrtl-lower-open-aggs))" %s --split-input-file | FileCheck %s --implicit-check-not=openvector --implicit-check-not=openbundle --implicit-check-not=opensub

// CHECK-LABEL: circuit "Bundle"
firrtl.circuit "Bundle" {
// CHECK-LABEL: module private @Child
  firrtl.module private @Child(in %in: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>,
                               out %r: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>) {
    %0 = firrtl.ref.send %in : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    firrtl.ref.define %r, %0 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
  }
// CHECK-LABEL: module private @Probe
// CHECK-SAME: in %in
// All probes
// CHECK-NOT:  out %r:
// CHECK-SAME: out %r_a: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>,
// CHECK-SAME: out %r_b: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>,
// Mixed HW / nonHW.
// "mixed" has non-hw removed but preserved structure:
// CHECK-SAME: out %mixed: !firrtl.bundle<a: uint<1>, x flip: vector<bundle<data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>,
// CHECK-SAME: out %mixed_x_0_p: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>,
// CHECK-SAME: out %mixed_x_1_p: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>,
// All probes, but interior structure has no HW projection.
// CHECK-NOT:  out %nohw:
// CHECK-SAME: out %nohw_x_0_p: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>,
// CHECK-SAME: out %nohw_x_1_p: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>) {
  firrtl.module private @Probe(in %in: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>,
                               out %r: !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>,
                               out %mixed: !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>,
                               out %nohw: !firrtl.openbundle<x: openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>>) {
    %0 = firrtl.opensubfield %nohw[x] : !firrtl.openbundle<x: openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>>
    %1 = firrtl.opensubindex %0[1] : !firrtl.openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>
    %2 = firrtl.opensubfield %1[p] : !firrtl.openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %3 = firrtl.opensubindex %0[0] : !firrtl.openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>
    %4 = firrtl.opensubfield %3[p] : !firrtl.openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %5 = firrtl.opensubfield %mixed[x] : !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>
    %6 = firrtl.opensubindex %5[1] : !firrtl.openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>
    %7 = firrtl.opensubfield %6[data] : !firrtl.openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>
    %8 = firrtl.opensubfield %6[p] : !firrtl.openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>
    %9 = firrtl.opensubindex %5[0] : !firrtl.openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>
    %10 = firrtl.opensubfield %9[data] : !firrtl.openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>
    %11 = firrtl.opensubfield %9[p] : !firrtl.openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>
    %12 = firrtl.opensubfield %mixed[b] : !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>
    %13 = firrtl.opensubfield %mixed[a] : !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>
    %14 = firrtl.opensubfield %r[b] : !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %15 = firrtl.opensubfield %r[a] : !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %c1_in, %c1_r = firrtl.instance c1 interesting_name @Child(in in: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out r: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>)
    %16 = firrtl.ref.sub %c1_r[1] : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    %17 = firrtl.ref.sub %c1_r[0] : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    %c2_in, %c2_r = firrtl.instance c2 interesting_name @Child(in in: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out r: !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>)
    %18 = firrtl.ref.sub %c2_r[0] : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    firrtl.strictconnect %c1_in, %in : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    firrtl.strictconnect %c2_in, %in : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    firrtl.ref.define %15, %c1_r : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    firrtl.ref.define %14, %c2_r : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    %19 = firrtl.ref.resolve %17 : !firrtl.probe<uint<1>>
    firrtl.strictconnect %13, %19 : !firrtl.uint<1>
    %20 = firrtl.ref.resolve %16 : !firrtl.probe<vector<uint<1>, 2>>
    firrtl.strictconnect %12, %20 : !firrtl.vector<uint<1>, 2>
    firrtl.ref.define %11, %c1_r : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    firrtl.ref.define %8, %c2_r : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    %21 = firrtl.ref.resolve %17 : !firrtl.probe<uint<1>>
    firrtl.strictconnect %10, %21 : !firrtl.uint<1>
    %22 = firrtl.ref.resolve %18 : !firrtl.probe<uint<1>>
    firrtl.strictconnect %7, %22 : !firrtl.uint<1>
    firrtl.ref.define %4, %c1_r : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    firrtl.ref.define %2, %c2_r : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
  }
// CHECK-LABEL: module @Bundle
  firrtl.module @Bundle(in %in: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out1: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out2: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out3: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out4: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out5: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out6: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out %out7: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>) attributes {convention = #firrtl<convention scalarized>} {
    %0 = firrtl.subfield %out7[b] : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %1 = firrtl.subfield %out7[a] : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %p_in, %p_r, %p_mixed, %p_nohw = firrtl.instance p interesting_name @Probe(in in: !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>, out r: !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, out mixed: !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>, out nohw: !firrtl.openbundle<x: openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>>)
    %2 = firrtl.opensubfield %p_mixed[b] : !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>
    %3 = firrtl.opensubfield %p_mixed[a] : !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>
    %4 = firrtl.opensubfield %p_nohw[x] : !firrtl.openbundle<x: openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>>
    %5 = firrtl.opensubindex %4[1] : !firrtl.openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>
    %6 = firrtl.opensubfield %5[p] : !firrtl.openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %7 = firrtl.opensubindex %4[0] : !firrtl.openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>
    %8 = firrtl.opensubfield %7[p] : !firrtl.openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %9 = firrtl.opensubfield %p_mixed[x] : !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>
    %10 = firrtl.opensubindex %9[1] : !firrtl.openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>
    %11 = firrtl.opensubfield %10[p] : !firrtl.openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>
    %12 = firrtl.opensubindex %9[0] : !firrtl.openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>
    %13 = firrtl.opensubfield %12[p] : !firrtl.openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>
    %14 = firrtl.opensubfield %p_r[b] : !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    %15 = firrtl.opensubfield %p_r[a] : !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>
    firrtl.strictconnect %p_in, %in : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %16 = firrtl.ref.resolve %15 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    firrtl.strictconnect %out1, %16 : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %17 = firrtl.ref.resolve %14 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    firrtl.strictconnect %out2, %17 : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %18 = firrtl.ref.resolve %13 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    firrtl.strictconnect %out3, %18 : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %19 = firrtl.ref.resolve %11 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    firrtl.strictconnect %out4, %19 : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %20 = firrtl.ref.resolve %8 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    firrtl.strictconnect %out5, %20 : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    %21 = firrtl.ref.resolve %6 : !firrtl.probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>
    firrtl.strictconnect %out6, %21 : !firrtl.bundle<a: uint<1>, b: vector<uint<1>, 2>>
    firrtl.strictconnect %1, %3 : !firrtl.uint<1>
    firrtl.strictconnect %0, %2 : !firrtl.vector<uint<1>, 2>
  }

// CHECK-LABEL: extmodule @ExtProbes
  firrtl.extmodule @ExtProbes(
    out r: !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>,
    out mixed: !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>,
    out nohw: !firrtl.openbundle<x: openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>>) attributes {convention = #firrtl<convention scalarized>}
}

// -----

// CHECK-LABEL: circuit "RefsOnlyAggFirstLevel"
firrtl.circuit "RefsOnlyAggFirstLevel" {
  // CHECK-LABEL: module private @Child
  // CHECK-SAME: (in %foo: !firrtl.bundle<x: uint<5>, y: uint<1>>, out %foo_refs_x: !firrtl.probe<uint<5>>, out %foo_refs_y: !firrtl.probe<uint<1>>)
  firrtl.module private @Child(in %foo: !firrtl.openbundle<x: uint<5>, refs flip: openbundle<x: probe<uint<5>>, y: probe<uint<1>>>, y: uint<1>>) {
    %0 = firrtl.opensubfield %foo[y] : !firrtl.openbundle<x: uint<5>, refs flip: openbundle<x: probe<uint<5>>, y: probe<uint<1>>>, y: uint<1>>
    %1 = firrtl.opensubfield %foo[x] : !firrtl.openbundle<x: uint<5>, refs flip: openbundle<x: probe<uint<5>>, y: probe<uint<1>>>, y: uint<1>>
    %2 = firrtl.opensubfield %foo[refs] : !firrtl.openbundle<x: uint<5>, refs flip: openbundle<x: probe<uint<5>>, y: probe<uint<1>>>, y: uint<1>>
    %3 = firrtl.opensubfield %2[y] : !firrtl.openbundle<x: probe<uint<5>>, y: probe<uint<1>>>
    %4 = firrtl.opensubfield %2[x] : !firrtl.openbundle<x: probe<uint<5>>, y: probe<uint<1>>>
    %5 = firrtl.ref.send %1 : !firrtl.uint<5>
    firrtl.ref.define %4, %5 : !firrtl.probe<uint<5>>
    %6 = firrtl.ref.send %0 : !firrtl.uint<1>
    firrtl.ref.define %3, %6 : !firrtl.probe<uint<1>>
  }
  // CHECK-LABEL: module @RefsOnlyAggFirstLevel(
  firrtl.module @RefsOnlyAggFirstLevel(in %x: !firrtl.uint<5>, in %y: !firrtl.uint<1>, out %out: !firrtl.openbundle<x: probe<uint<5>>, y: probe<uint<1>>>) attributes {convention = #firrtl<convention scalarized>} {
    %0 = firrtl.opensubfield %out[y] : !firrtl.openbundle<x: probe<uint<5>>, y: probe<uint<1>>>
    %1 = firrtl.opensubfield %out[x] : !firrtl.openbundle<x: probe<uint<5>>, y: probe<uint<1>>>
    // CHECK: firrtl.instance c interesting_name @Child(in foo: !firrtl.bundle<x: uint<5>, y: uint<1>>, out foo_refs_x: !firrtl.probe<uint<5>>, out foo_refs_y: !firrtl.probe<uint<1>>)
    %c_foo = firrtl.instance c interesting_name @Child(in foo: !firrtl.openbundle<x: uint<5>, refs flip: openbundle<x: probe<uint<5>>, y: probe<uint<1>>>, y: uint<1>>)
    %2 = firrtl.opensubfield %c_foo[refs] : !firrtl.openbundle<x: uint<5>, refs flip: openbundle<x: probe<uint<5>>, y: probe<uint<1>>>, y: uint<1>>
    %3 = firrtl.opensubfield %2[y] : !firrtl.openbundle<x: probe<uint<5>>, y: probe<uint<1>>>
    %4 = firrtl.opensubfield %2[x] : !firrtl.openbundle<x: probe<uint<5>>, y: probe<uint<1>>>
    %5 = firrtl.opensubfield %c_foo[y] : !firrtl.openbundle<x: uint<5>, refs flip: openbundle<x: probe<uint<5>>, y: probe<uint<1>>>, y: uint<1>>
    %6 = firrtl.opensubfield %c_foo[x] : !firrtl.openbundle<x: uint<5>, refs flip: openbundle<x: probe<uint<5>>, y: probe<uint<1>>>, y: uint<1>>
    firrtl.strictconnect %6, %x : !firrtl.uint<5>
    firrtl.strictconnect %5, %y : !firrtl.uint<1>
    firrtl.ref.define %1, %4 : !firrtl.probe<uint<5>>
    firrtl.ref.define %0, %3 : !firrtl.probe<uint<1>>
  }
}

// -----

// CHECK-LABEL: circuit "SymbolOnField"
firrtl.circuit "SymbolOnField" {
  // CHECK: @SymbolOnField
  // CHECK-SAME: (out r: !firrtl.bundle<x: uint<1>> sym [<@sym,1,public>],
  // CHECK-SAME:  out r_p: !firrtl.probe<uint<1>>)
  firrtl.extmodule @SymbolOnField(out r : !firrtl.openbundle<p: probe<uint<1>>, x: uint<1>> sym [<@sym,2,public>])
}

// -----

// CHECK-LABEL: circuit "ManySymbols"
firrtl.circuit "ManySymbols" {
   // Innner-syms on everything not a probe.
   // (mixed.x[0].p and mixed.x[1].p are pulled out)
  // CHECK: extmodule @ManySymbols(
  // CHECK-SAME: <@mixed,0,public>,
  // CHECK-SAME: <@a,1,public>,
  // CHECK-SAME: <@xvec,2,public>,
  // CHECK-SAME: <@x0,3,public>,
  // CHECK-SAME: <@x0_data,4,public>,
  // CHECK-SAME: <@x1,5,public>,
  // CHECK-SAME: <@x1_data,6,public>,
  // CHECK-SAME: <@b,7,public>
  firrtl.extmodule @ManySymbols(
    out mixed: !firrtl.openbundle<a: uint<1>,
                                   x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>,
                                                                                      b: vector<uint<1>, 2>>>,
                                                                 data flip: uint<1>
                                                      >, 2>,
                                   b: vector<uint<1>, 2>>
        sym [<@mixed,0,public>,
               <@a,1,public>,
               <@xvec,2,public>,
                 <@x0,3,public>, <@x0_data,5,public>,
                 <@x1,6,public>, <@x1_data,8,public>,
               <@b,9,public>])

  // Similar but with a refs-only agg between HW elements.
  // Same HW-only contents as above.
  // CHECK: extmodule @ManySymbols2(
  // CHECK-SAME: <@mixed,0,public>,
  // CHECK-SAME: <@a,1,public>,
  // CHECK-SAME: <@xvec,2,public>,
  // CHECK-SAME: <@x0,3,public>,
  // CHECK-SAME: <@x0_data,4,public>,
  // CHECK-SAME: <@x1,5,public>,
  // CHECK-SAME: <@x1_data,6,public>,
  // CHECK-SAME: <@b,7,public>
  firrtl.extmodule @ManySymbols2(
    out mixed: !firrtl.openbundle<a: uint<1>,
                                   x flip: openvector<openbundle<refsonly : openbundle<p: probe<bundle<a: uint<1>,
                                                                                                       b: vector<uint<1>, 2>>>>,
                                                                 data flip: uint<1>
                                                      >, 2>,
                                   b: vector<uint<1>, 2>>
        sym [<@mixed,0,public>,
               <@a,1,public>,
               <@xvec,2,public>,
                 <@x0,3,public>, <@x0_data,6,public>,
                 <@x1,7,public>, <@x1_data,10,public>,
               <@b,11,public>])
}
