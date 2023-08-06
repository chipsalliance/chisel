; Check this design makes it through firtool.
; RUN: firtool %s
; RUN: firtool %s --preserve-aggregate=all --scalarize-top-module=false

; Check parsing into open aggregates.
; RUN: circt-translate -import-firrtl -verify-diagnostics -split-input-file %s | circt-opt | FileCheck %s

; CHECK-LABEL: circuit "Bundle"
circuit Bundle :
; CHECK-LABEL: module private @Child
; CHECK-SAME: in %in: !firrtl.bundle
; CHECK-SAME: out %r: !firrtl.probe<
  module Child :
    input in : {a : UInt<1>, b : UInt<1>[2]}
    output r : Probe<{a : UInt<1>, b : UInt<1>[2]}> ; probe of bundle (of UInt, vec)
    define r = probe(in)

; CHECK-LABEL: module private @Probe
; CHECK-SAME: in %in: !firrtl.bundle<
; CHECK-SAME: out %r: !firrtl.openbundle<a: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, b: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>,
; CHECK-SAME: out %mixed: !firrtl.openbundle<a: uint<1>, x flip: openvector<openbundle<p flip: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>, data flip: uint<1>>, 2>, b: vector<uint<1>, 2>>,
; CHECK-SAME: out %nohw: !firrtl.openbundle<x: openvector<openbundle<p: probe<bundle<a: uint<1>, b: vector<uint<1>, 2>>>>, 2>>) {
; CHECK: opensubfield
; CHECK: opensubindex
  module Probe :
    input in : {a : UInt<1>, b : UInt<1>[2]}
    output r : {a : Probe<{a : UInt<1>, b : UInt<1>[2]}>, b : Probe<{a : UInt<1>, b : UInt<1>[2]}>} ; bundle of probes of bundles (of UInt, vec)
    output mixed : {a : UInt<1>,
                    flip x : {flip p: Probe<{a : UInt<1>,
                                             b : UInt<1>[2]
                                             }>,
                              flip data: UInt<1>
                             }[2],
                    b : UInt<1>[2]} ; mixed
    output nohw : {x : {p: Probe<{a : UInt<1>, b : UInt<1>[2]}>}[2]} ; non-hw-only

    inst c1 of Child
    inst c2 of Child
    c1.in <= in
    c2.in <= in

    define r.a = c1.r
    define r.b = c2.r

    mixed.a <= read(c1.r.a)
    mixed.b <= read(c1.r.b)
    define mixed.x[0].p = c1.r
    define mixed.x[1].p = c2.r
    mixed.x[0].data <= read(c1.r.a)
    mixed.x[1].data <= read(c2.r.a)
    define nohw.x[0].p = c1.r
    define nohw.x[1].p = c2.r

; CHECK-LABEL: module @Bundle
; CHECK: opensubfield
; CHECK: opensubindex
  module Bundle:
    input in : {a : UInt<1>, b : UInt<1>[2]}
    output out1 : {a : UInt<1>, b : UInt<1>[2]}
    output out2 : {a : UInt<1>, b : UInt<1>[2]}
    output out3 : {a : UInt<1>, b : UInt<1>[2]}
    output out4 : {a : UInt<1>, b : UInt<1>[2]}
    output out5 : {a : UInt<1>, b : UInt<1>[2]}
    output out6 : {a : UInt<1>, b : UInt<1>[2]}
    output out7 : {a : UInt<1>, b : UInt<1>[2]}

    inst p of Probe
    p.in <= in

   ; Resolve probes
   out1 <= read(p.r.a)
   out2 <= read(p.r.b)
   out3 <= read(p.mixed.x[0].p)
   out4 <= read(p.mixed.x[1].p)
   out5 <= read(p.nohw.x[0].p)
   out6 <= read(p.nohw.x[1].p)
   out7.a <= p.mixed.a
   out7.b <= p.mixed.b

