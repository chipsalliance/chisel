// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-emit-omir{file=omir.json}))' --verify-diagnostics --split-input-file %s

#loc = loc(unknown)
firrtl.circuit "Top" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    omType = {info = #loc, index = 0, value = ["OMString:OMLazyModule", "OMString:OMSRAM"]},
    finalPath = {info = #loc, index = 1, value = {omir.tracker, id = 0, type = "OMMemberReferenceTarget"}}
  }}]
}]} {
  firrtl.extmodule private @MySRAM()
  firrtl.module private @Submodule() {
    // expected-error @+1 {{OMIR node targets uninstantiated component `mem`}}
    firrtl.instance mem {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0}]} @MySRAM()
  }
  firrtl.module @Top() {}
}

// -----

#loc = loc(unknown)
firrtl.circuit "Top" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    omType = {info = #loc, index = 0, value = ["OMString:OMLazyModule", "OMString:OMSRAM"]},
    finalPath = {info = #loc, index = 1, value = {omir.tracker, id = 0, type = "OMMemberReferenceTarget"}}
  }}]
}]} {
  firrtl.extmodule private @MySRAM()
  firrtl.module private @Submodule() {
    // expected-error @+4 {{OMIR node targets ambiguous component `mem`}}
    // expected-note @+3 {{may refer to the following paths:}}
    // expected-note @+2 {{- $root/sub1:Submodule}}
    // expected-note @+1 {{- $root/sub2:Submodule}}
    firrtl.instance mem {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0}]} @MySRAM()
  }
  firrtl.module @Top() {
    firrtl.instance sub1 @Submodule()
    firrtl.instance sub2 @Submodule()
  }
}

// -----

#loc = loc(unknown)
// expected-error @+3 {{tracked OMIR target of type `OMInstanceTarget` was deleted}}
// expected-note @+2 {{`OMInstanceTarget` should never be deleted}}
// expected-note @+1 {{original path: `~Foo|Bar`}}
firrtl.circuit "Foo" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    a = {info = #loc, index = 1, value = {omir.tracker, id = 0, path = "~Foo|Bar", type = "OMInstanceTarget"}}
  }}]
}]} {
  firrtl.module @Foo() {}
}

// -----

#loc = loc(unknown)
// expected-error @+3 {{tracked OMIR target of type `OMDontTouchedReferenceTarget` was deleted}}
// expected-note @+2 {{`OMDontTouchedReferenceTarget` should never be deleted}}
// expected-note @+1 {{original path: `~Foo|Bar`}}
firrtl.circuit "Foo" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    a = {info = #loc, index = 1, value = {omir.tracker, id = 0, path = "~Foo|Bar", type = "OMDontTouchedReferenceTarget"}}
  }}]
}]} {
  firrtl.module @Foo() {}
}

// -----

#loc = loc(unknown)
firrtl.circuit "Foo" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    a = {info = #loc, index = 1, value = {omir.tracker, id = 0, path = "~Foo|Foo>w", type = "OMMemberReferenceTarget"}}
  }}]
}]} {
  firrtl.module @Foo() {
    // expected-note @below {{tracker with same ID already found here}}
    %w_a = firrtl.wire {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0 : i64, type = "OMMemberReferenceTarget"}]} : !firrtl.uint<1>
    // expected-error @below {{freechips.rocketchip.objectmodel.OMIRTracker annotation with same ID already found, must resolve to single target}}
    %w_b = firrtl.wire {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0 : i64, type = "OMMemberReferenceTarget"}]} : !firrtl.uint<1>
}
}
