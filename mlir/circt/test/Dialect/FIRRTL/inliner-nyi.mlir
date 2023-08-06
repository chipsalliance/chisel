// This test is expected to crash.  Use "not --crash" instead of XFAIL to work
// around llvm-symbolizer being slow.  For more information, see:
//   https://discourse.llvm.org/t/llvm-symbolizer-has-gotten-extremely-slow/67262
// RUN: not --crash circt-opt --firrtl-inliner %s

// Inliner does not support running before expand when's,
// here was crash the reference-handling code because it assumes
// the instance and its uses are in the same block.

module {
  firrtl.circuit "InlinerRefs" {
    firrtl.module private @ChildOut(in %in: !firrtl.bundle<a: uint<1>, b: uint<2>>, out %out: !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
      %0 = firrtl.subfield %in[a] : !firrtl.bundle<a: uint<1>, b: uint<2>>
      firrtl.when %0 : !firrtl.uint<1> {
        %1 = firrtl.ref.send %in : !firrtl.bundle<a: uint<1>, b: uint<2>>
        firrtl.ref.define %out, %1 : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>
      }
    }
    firrtl.module @InlinerRefs(in %in: !firrtl.bundle<a: uint<1>, b: uint<2>>, out %out: !firrtl.uint<1>) {
      %0 = firrtl.subfield %in[a] : !firrtl.bundle<a: uint<1>, b: uint<2>>
      %co_in, %co_out = firrtl.instance co interesting_name @ChildOut(in in: !firrtl.bundle<a: uint<1>, b: uint<2>>, out out: !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>)
      %1 = firrtl.ref.sub %co_out[0] : !firrtl.probe<bundle<a: uint<1>, b: uint<2>>>
      firrtl.strictconnect %co_in, %in : !firrtl.bundle<a: uint<1>, b: uint<2>>
      firrtl.when %0 : !firrtl.uint<1> {
        %2 = firrtl.ref.resolve %1 : !firrtl.probe<uint<1>>
        firrtl.strictconnect %out, %2 : !firrtl.uint<1>
      }
    }
  }
}
