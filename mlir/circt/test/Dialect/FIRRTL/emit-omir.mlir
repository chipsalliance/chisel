// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-emit-omir{file=omir.json}))' %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Absence of any OMIR
//===----------------------------------------------------------------------===//

firrtl.circuit "NoOMIR" {
  firrtl.module @NoOMIR() {
  }
}
// CHECK-LABEL: firrtl.circuit "NoOMIR" {
// CHECK-NEXT:    firrtl.module @NoOMIR() {
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.verbatim "[]"
// CHECK-SAME:    #hw.output_file<"omir.json", excludeFromFileList>}
// CHECK-NEXT:  }

//===----------------------------------------------------------------------===//
// Empty OMIR data
//===----------------------------------------------------------------------===//

firrtl.circuit "NoNodes" attributes {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRAnnotation", nodes = []}]}  {
  firrtl.module @NoNodes() {
  }
}
// CHECK-LABEL: firrtl.circuit "NoNodes" {
// CHECK-NEXT:    firrtl.module @NoNodes() {
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.verbatim "[]"
// CHECK-SAME:    excludeFromFileList>}
// CHECK-NEXT:  }

//===----------------------------------------------------------------------===//
// Empty node
//===----------------------------------------------------------------------===//

#loc = loc(unknown)
firrtl.circuit "EmptyNode" attributes {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRAnnotation", nodes = [{fields = {}, id = "OMID:0", info = #loc}]}]}  {
  firrtl.module @EmptyNode() {
  }
}
// CHECK-LABEL: firrtl.circuit "EmptyNode" {
// CHECK-NEXT:    firrtl.module @EmptyNode() {
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.verbatim
// CHECK-SAME:    \22info\22: \22UnlocatableSourceInfo\22
// CHECK-SAME:    \22id\22: \22OMID:0\22
// CHECK-SAME:    \22fields\22: []
// CHECK-SAME:    excludeFromFileList>}
// CHECK-NEXT:  }

//===----------------------------------------------------------------------===//
// Source locator serialization
//===----------------------------------------------------------------------===//

#loc0 = loc("B":2:3)
#loc1 = loc(fused["C":4:5, "D":6:7])
#loc2 = loc("A":0:1)
firrtl.circuit "SourceLocators" attributes {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRAnnotation", nodes = [{fields = {x = {index = 1 : i64, info = #loc0, value = "OMReference:0"}, y = {index = 0 : i64, info = #loc1, value = "OMReference:0"}}, id = "OMID:0", info = #loc2}]}]}  {
  firrtl.module @SourceLocators() {
  }
}
// CHECK-LABEL: firrtl.circuit "SourceLocators" {
// CHECK-NEXT:    firrtl.module @SourceLocators() {
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.verbatim
// CHECK-SAME:    \22info\22: \22@[A 0:1]\22
// CHECK-SAME:    \22id\22: \22OMID:0\22
// CHECK-SAME:    \22fields\22: [
// CHECK-SAME:      {
// CHECK-SAME:        \22info\22: \22@[C 4:5 D 6:7]\22
// CHECK-SAME:        \22name\22: \22y\22
// CHECK-SAME:        \22value\22: \22OMReference:0\22
// CHECK-SAME:      }
// CHECK-SAME:      {
// CHECK-SAME:        \22info\22: \22@[B 2:3]\22
// CHECK-SAME:        \22name\22: \22x\22
// CHECK-SAME:        \22value\22: \22OMReference:0\22
// CHECK-SAME:      }
// CHECK-SAME:    ]
// CHECK-SAME:    excludeFromFileList>}
// CHECK-NEXT:  }

//===----------------------------------------------------------------------===//
// Check that all the OMIR types support serialization
//===----------------------------------------------------------------------===//

firrtl.circuit "AllTypesSupported" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    OMBoolean = {info = #loc, index = 1, value = true},
    OMInt1 = {info = #loc, index = 2, value = 9001 : i32},
    OMInt2 = {info = #loc, index = 3, value = -42 : i32},
    OMDouble = {info = #loc, index = 4, value = 3.14 : f32},
    OMID = {info = #loc, index = 5, value = "OMID:1337"},
    OMReference = {info = #loc, index = 6, value = "OMReference:0"},
    OMBigInt = {info = #loc, index = 7, value = "OMBigInt:42"},
    OMLong = {info = #loc, index = 8, value = "OMLong:ff"},
    OMString = {info = #loc, index = 9, value = "OMString:hello"},
    OMBigDecimal = {info = #loc, index = 10, value = "OMBigDecimal:10.5"},
    OMDeleted = {info = #loc, index = 11, value = "OMDeleted"},
    OMConstant = {info = #loc, index = 12, value = "OMConstant:UInt<2>(\"h1\")"},
    OMArray = {info = #loc, index = 13, value = [true, 9001, "OMString:bar"]},
    OMMap = {info = #loc, index = 14, value = {foo = true, bar = 9001}}
  }}]
}]} {
  firrtl.module @AllTypesSupported() {
  }
}
// CHECK-LABEL: firrtl.circuit "AllTypesSupported" {
// CHECK-NEXT:    firrtl.module @AllTypesSupported() {
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.verbatim
// CHECK-SAME:    \22name\22: \22OMBoolean\22
// CHECK-SAME:    \22value\22: true
// CHECK-SAME:    \22name\22: \22OMInt1\22
// CHECK-SAME:    \22value\22: 9001
// CHECK-SAME:    \22name\22: \22OMInt2\22
// CHECK-SAME:    \22value\22: -42
// CHECK-SAME:    \22name\22: \22OMDouble\22
// CHECK-SAME:    \22value\22: 3.14
// CHECK-SAME:    \22name\22: \22OMID\22
// CHECK-SAME:    \22value\22: \22OMID:1337\22
// CHECK-SAME:    \22name\22: \22OMReference\22
// CHECK-SAME:    \22value\22: \22OMReference:0\22
// CHECK-SAME:    \22name\22: \22OMBigInt\22
// CHECK-SAME:    \22value\22: \22OMBigInt:42\22
// CHECK-SAME:    \22name\22: \22OMLong\22
// CHECK-SAME:    \22value\22: \22OMLong:ff\22
// CHECK-SAME:    \22name\22: \22OMString\22
// CHECK-SAME:    \22value\22: \22OMString:hello\22
// CHECK-SAME:    \22name\22: \22OMBigDecimal\22
// CHECK-SAME:    \22value\22: \22OMBigDecimal:10.5\22
// CHECK-SAME:    \22name\22: \22OMDeleted\22
// CHECK-SAME:    \22value\22: \22OMDeleted\22
// CHECK-SAME:    \22name\22: \22OMConstant\22
// CHECK-SAME:    \22value\22: \22OMConstant:UInt<2>(\\\22h1\\\22)\22
// CHECK-SAME:    \22name\22: \22OMArray\22
// CHECK-SAME:    \22value\22: [
// CHECK-SAME:      true
// CHECK-SAME:      9001
// CHECK-SAME:      \22OMString:bar\22
// CHECK-SAME:    ]
// CHECK-SAME:    \22name\22: \22OMMap\22
// CHECK-SAME:    \22value\22: {
// CHECK-SAME:      \22bar\22: 9001
// CHECK-SAME:      \22foo\22: true
// CHECK-SAME:    }
// CHECK-SAME:    excludeFromFileList>}
// CHECK-NEXT:  }

//===----------------------------------------------------------------------===//
// Trackers as Local Annotations
//===----------------------------------------------------------------------===//

firrtl.circuit "LocalTrackers" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    OMReferenceTarget1 = {info = #loc, index = 1, value = {omir.tracker, id = 0, type = "OMReferenceTarget"}},
    OMReferenceTarget2 = {info = #loc, index = 2, value = {omir.tracker, id = 1, type = "OMReferenceTarget"}},
    OMReferenceTarget3 = {info = #loc, index = 3, value = {omir.tracker, id = 2, type = "OMReferenceTarget"}},
    OMReferenceTarget4 = {info = #loc, index = 4, value = {omir.tracker, id = 3, type = "OMReferenceTarget"}}
  }}]
}]} {
  firrtl.module @A() attributes {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0}]} {
    %c = firrtl.wire {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 1}]} : !firrtl.uint<42>
  }
  firrtl.module @LocalTrackers() {
    firrtl.instance a {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 3}]} @A()
    %b = firrtl.wire {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 2}]} : !firrtl.uint<42>
  }
}
// CHECK-LABEL: firrtl.circuit "LocalTrackers" {
// CHECK-NEXT:    firrtl.module @A() {
// CHECK-NEXT:      %c = firrtl.wire sym [[SYMC:@[a-zA-Z0-9_]+]] : !firrtl.uint<42>
// CHECK-NEXT:    }
// CHECK-NEXT:    firrtl.module @LocalTrackers() {
// CHECK-NEXT:      firrtl.instance a sym [[SYMA:@[a-zA-Z0-9_]+]] @A()
// CHECK-NEXT:      %b = firrtl.wire sym [[SYMB:@[a-zA-Z0-9_]+]] : !firrtl.uint<42>
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.verbatim
// CHECK-SAME:           \22name\22: \22OMReferenceTarget1\22
// CHECK-SAME{LITERAL}:  \22value\22: \22OMReferenceTarget:~LocalTrackers|{{0}}\22
// CHECK-SAME:           \22name\22: \22OMReferenceTarget2\22
// CHECK-SAME{LITERAL}:  \22value\22: \22OMReferenceTarget:~LocalTrackers|{{0}}>{{1}}\22
// CHECK-SAME:           \22name\22: \22OMReferenceTarget3\22
// CHECK-SAME{LITERAL}:  \22value\22: \22OMReferenceTarget:~LocalTrackers|{{2}}>{{3}}\22
// CHECK-SAME:           \22name\22: \22OMReferenceTarget4\22
// CHECK-SAME{LITERAL}:  \22value\22: \22OMReferenceTarget:~LocalTrackers|{{2}}>{{4}}\22
// CHECK-SAME:           symbols = [
// CHECK-SAME:             @A,
// CHECK-SAME:             #hw.innerNameRef<@A::[[SYMC]]>,
// CHECK-SAME:             @LocalTrackers,
// CHECK-SAME:             #hw.innerNameRef<@LocalTrackers::[[SYMB]]>,
// CHECK-SAME:             #hw.innerNameRef<@LocalTrackers::[[SYMA]]>
// CHECK-SAME:           ]}
// CHECK-NEXT:  }

//===----------------------------------------------------------------------===//
// Trackers as Non-Local Annotations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.circuit "NonLocalTrackers"
firrtl.circuit "NonLocalTrackers" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    OMReferenceTarget1 = {info = #loc, index = 1, id = "OMID:1", value = {omir.tracker, id = 0, type = "OMReferenceTarget"}},
    OMReferenceTarget2 = {info = #loc, index = 2, id = "OMID:2", value = {omir.tracker, id = 1, type = "OMReferenceTarget"}}
  }}]
}]} {
  // Both OMReferenceTarget1 and OMReferenceTarget2 share the same NLA.  This
  // NLA should not be deleted.
  hw.hierpath private @nla_0 [@NonLocalTrackers::@b, @B::@a, @A]
  // CHECK: firrtl.module @A
  firrtl.module @A() attributes {annotations = [{circt.nonlocal = @nla_0, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0}]} {
    // CHECK-NEXT: %a = firrtl.wire sym @[[a_sym:[^ ]+]]
    %a = firrtl.wire {annotations = [{circt.nonlocal = @nla_0, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 1}]} : !firrtl.uint<1>
  }
  firrtl.module @B() {
    firrtl.instance a sym @a @A()
  }
  firrtl.module @NonLocalTrackers() {
    firrtl.instance b sym @b @B()
  }
}
// CHECK:       firrtl.instance a sym [[SYMA:@[a-zA-Z0-9_]+]]
// CHECK:       firrtl.instance b sym [[SYMB:@[a-zA-Z0-9_]+]]
// CHECK:       sv.verbatim
// CHECK-SAME:           \22name\22: \22OMReferenceTarget1\22
// CHECK-SAME{LITERAL}:  \22value\22: \22OMReferenceTarget:~NonLocalTrackers|{{0}}/{{1}}:{{2}}/{{3}}:{{4}}\22
// CHECK-SAME:           \22name\22: \22OMReferenceTarget2\22
// CHECK-SAME{LITERAL}:  \22value\22: \22OMReferenceTarget:~NonLocalTrackers|{{0}}/{{1}}:{{2}}/{{3}}:{{4}}>{{5}}\22
// CHECK-SAME:  symbols = [
// CHECK-SAME:    @NonLocalTrackers,
// CHECK-SAME:    #hw.innerNameRef<@NonLocalTrackers::[[SYMB]]>,
// CHECK-SAME:    @B,
// CHECK-SAME:    #hw.innerNameRef<@B::[[SYMA]]>,
// CHECK-SAME:    @A,
// CHECK-SAME:    #hw.innerNameRef<@A::@[[a_sym]]>
// CHECK-SAME:  ]


//===----------------------------------------------------------------------===//
// Trackers support fieldID
//===----------------------------------------------------------------------===//

#loc3 = loc("OM.scala":1:1)
#loc4 = loc("Foo.scala":1:1)
firrtl.circuit "Top"  attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{
    fields = {
      paths = {
        index = 0 : i64,
        info = #loc3,
        value = [{
          id = 0 : i64,
          omir.tracker,
          path = "~Top|Top/a:A>in0.io[0]",
          type = "OMReferenceTarget"
        }, {
          id = 1 : i64,
          omir.tracker,
          path = "~Top|Top/a:A>in0.io[1]",
          type = "OMReferenceTarget"
        }]
      }
    },
    id = "OMID:1",
    info = #loc4
  }, {
    fields = {
      paths = {
        index = 0 : i64,
        info = #loc4,
        value = [{
          id = 2 : i64,
          omir.tracker,
          path = "~Top|Top/a:A>w0[0]",
          type = "OMReferenceTarget"
        }, {
          id = 3 : i64,
          omir.tracker,
          path = "~Top|Top/a:A>w0[1]",
          type = "OMReferenceTarget"
        }]
      }
    },
    id = "OMID:2",
    info = #loc4
  }, {
    fields = {
      paths = {
        index = 0 : i64,
        info = #loc3,
        value = [{
          id = 4 : i64,
          omir.tracker,
          path = "~Top|Top/a:A>in1.io.f0",
          type = "OMReferenceTarget"
        }, {
          id = 5 : i64,
          omir.tracker,
          path = "~Top|Top/a:A>in1.io.f1",
          type = "OMReferenceTarget"
        }]
      }
    },
    id = "OMID:3",
    info = #loc4
  }, {
    fields = {
      paths = {
        index = 0 : i64,
        info = #loc4,
        value = [{
          id = 6 : i64,
          omir.tracker,
          path = "~Top|Top/a:A>w1.f0",
          type = "OMReferenceTarget"
        }, {
          id = 7 : i64,
          omir.tracker,
          path = "~Top|Top/a:A>w1.f1",
          type = "OMReferenceTarget"
        }]
      }
    },
    id = "OMID:4",
    info = #loc4
  }, {
    fields = {
      paths = {
        index = 0 : i64,
        info = #loc4,
        value = [{
          id = 8 : i64,
          omir.tracker,
          path = "~Top|Top/a:A>wf[0]",
          type = "OMReferenceTarget"
        }, {
          id = 9 : i64,
          omir.tracker,
          path = "~Top|Top/a:A>wf[1]",
          type = "OMReferenceTarget"
        }]
      }
    },
    id = "OMID:5",
    info = #loc4
  }
]
}]} {
  hw.hierpath private @nla [@Top::@a, @A]
  firrtl.module @Top(in %in0_0: !firrtl.uint<4>, in %in0_1: !firrtl.uint<4>, in %in1_f0: !firrtl.uint<4>, in %in1_f1: !firrtl.uint<4>, out %out0_0: !firrtl.uint<4>, out %out0_1: !firrtl.uint<4>, out %out1_f0: !firrtl.uint<4>, out %out1_f1: !firrtl.uint<4>) {
    %a_in0, %a_in1, %a_out0, %a_out1 = firrtl.instance a sym @a  @A(in in0: !firrtl.bundle<io: vector<uint<4>, 2>>, in in1: !firrtl.bundle<io: bundle<f0: uint<4>, f1: uint<4>>>, out out0: !firrtl.vector<uint<4>, 2>, out out1: !firrtl.bundle<f0: uint<4>, f1: uint<4>>)
    %0 = firrtl.subfield %a_in1[io] : !firrtl.bundle<io: bundle<f0: uint<4>, f1: uint<4>>>
    %1 = firrtl.subfield %a_in0[io]: !firrtl.bundle<io: vector<uint<4>, 2>>
    %2 = firrtl.subindex %a_out0[0] : !firrtl.vector<uint<4>, 2>
    firrtl.strictconnect %out0_0, %2 : !firrtl.uint<4>
    %3 = firrtl.subindex %a_out0[1] : !firrtl.vector<uint<4>, 2>
    firrtl.strictconnect %out0_1, %3 : !firrtl.uint<4>
    %4 = firrtl.subfield %a_out1[f0] : !firrtl.bundle<f0: uint<4>, f1: uint<4>>
    firrtl.strictconnect %out1_f0, %4 : !firrtl.uint<4>
    %5 = firrtl.subfield %a_out1[f1] : !firrtl.bundle<f0: uint<4>, f1: uint<4>>
    firrtl.strictconnect %out1_f1, %5 : !firrtl.uint<4>
    %6 = firrtl.bundlecreate %in1_f0, %in1_f1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.bundle<f0: uint<4>, f1: uint<4>>
    firrtl.strictconnect %0, %6 : !firrtl.bundle<f0: uint<4>, f1: uint<4>>
    %7 = firrtl.vectorcreate %in0_0, %in0_1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.vector<uint<4>, 2>
    firrtl.strictconnect %1, %7 : !firrtl.vector<uint<4>, 2>
  }
  // CHECK-LABEL: firrtl.module private @A
  // CHECK-SAME:    %in0: !firrtl.bundle<io: vector<uint<4>, 2>> sym @[[in0_sym:[^ ]+]],
  // CHECK-SAME:    %in1: !firrtl.bundle<io: bundle<f0: uint<4>, f1: uint<4>>> sym @[[in1_sym:[^ ]+]],
  firrtl.module private @A(in %in0: !firrtl.bundle<io: vector<uint<4>, 2>> [{circt.fieldID = 3 : i32, circt.nonlocal = @nla, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 1 : i64},
                                                                            {circt.fieldID = 2 : i32, circt.nonlocal = @nla, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0 : i64}],
                           in %in1: !firrtl.bundle<io: bundle<f0: uint<4>, f1: uint<4>>> [{circt.fieldID = 3 : i32, circt.nonlocal = @nla, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 5 : i64},
                                                                                          {circt.fieldID = 2 : i32, circt.nonlocal = @nla, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 4 : i64}],
                           out %out0: !firrtl.vector<uint<4>, 2>,
                           out %out1: !firrtl.bundle<f0: uint<4>, f1: uint<4>>) {
    %0 = firrtl.subfield %in1[io] : !firrtl.bundle<io: bundle<f0: uint<4>, f1: uint<4>>>
    %1 = firrtl.subfield %in0[io] : !firrtl.bundle<io: vector<uint<4>, 2>>
    // CHECK: %w0 = firrtl.wire sym @[[w0_sym:[^ ]+]]
    %w0 = firrtl.wire   {annotations = [{circt.fieldID = 2 : i32, circt.nonlocal = @nla, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 3 : i64},
                                        {circt.fieldID = 1 : i32, circt.nonlocal = @nla, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 2 : i64}]} : !firrtl.vector<uint<4>, 2>
    %2 = firrtl.subindex %1[0] : !firrtl.vector<uint<4>, 2>
    %3 = firrtl.subindex %1[1] : !firrtl.vector<uint<4>, 2>
    // CHECK: %w1 = firrtl.wire sym @[[w1_sym:[^ ]+]]
    %w1 = firrtl.wire   {annotations = [{circt.fieldID = 2 : i32, circt.nonlocal = @nla, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 7 : i64},
                                        {circt.fieldID = 1 : i32, circt.nonlocal = @nla, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 6 : i64}]} : !firrtl.bundle<f0: uint<4>, f1: uint<4>>
    %4 = firrtl.subfield %0[f0] : !firrtl.bundle<f0: uint<4>, f1: uint<4>>
    %5 = firrtl.subfield %0[f1] : !firrtl.bundle<f0: uint<4>, f1: uint<4>>
    %6 = firrtl.subindex %w0[0] : !firrtl.vector<uint<4>, 2>
    %7 = firrtl.subindex %out0[0] : !firrtl.vector<uint<4>, 2>
    firrtl.strictconnect %7, %6 : !firrtl.uint<4>
    %8 = firrtl.subindex %w0[1] : !firrtl.vector<uint<4>, 2>
    %9 = firrtl.subindex %out0[1] : !firrtl.vector<uint<4>, 2>
    firrtl.strictconnect %9, %8 : !firrtl.uint<4>
    %10 = firrtl.subfield %w1[f0] : !firrtl.bundle<f0: uint<4>, f1: uint<4>>
    %11 = firrtl.subfield %out1[f0] : !firrtl.bundle<f0: uint<4>, f1: uint<4>>
    firrtl.strictconnect %11, %10 : !firrtl.uint<4>
    %12 = firrtl.subfield %w1[f1] : !firrtl.bundle<f0: uint<4>, f1: uint<4>>
    %13 = firrtl.subfield %out1[f1] : !firrtl.bundle<f0: uint<4>, f1: uint<4>>
    firrtl.strictconnect %13, %12 : !firrtl.uint<4>
    %14 = firrtl.vectorcreate %2, %3 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.vector<uint<4>, 2>
    firrtl.strictconnect %w0, %14 : !firrtl.vector<uint<4>, 2>
    %15 = firrtl.bundlecreate %4, %5 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.bundle<f0: uint<4>, f1: uint<4>>
    firrtl.strictconnect %w1, %15 : !firrtl.bundle<f0: uint<4>, f1: uint<4>>

    // CHECK: %wf, %wf_ref = firrtl.wire sym @[[wf_sym:[^ ]+]] forceable
    %wf, %wf_ref = firrtl.wire forceable {annotations = [{circt.fieldID = 2 : i32, circt.nonlocal = @nla, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 9 : i64},
                                        {circt.fieldID = 1 : i32, circt.nonlocal = @nla, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 8 : i64}]} : !firrtl.vector<uint<4>, 2>, !firrtl.rwprobe<vector<uint<4>, 2>>
    %wf_0 = firrtl.subindex %wf[0] : !firrtl.vector<uint<4>, 2>
    %wf_1 = firrtl.subindex %wf[1] : !firrtl.vector<uint<4>, 2>
    firrtl.strictconnect %wf_0, %2 : !firrtl.uint<4>
    firrtl.strictconnect %wf_1, %3 : !firrtl.uint<4>
  }
}

// CHECK:      sv.verbatim
// CHECK-SAME{LITERAL}: OMReferenceTarget:~Top|{{0}}/{{1}}:{{2}}>{{3}}.io[0]
// CHECK-SAME{LITERAL}: OMReferenceTarget:~Top|{{0}}/{{1}}:{{2}}>{{3}}.io[1]
// CHECK-SAME{LITERAL}: OMReferenceTarget:~Top|{{0}}/{{1}}:{{2}}>{{4}}[0]
// CHECK-SAME{LITERAL}: OMReferenceTarget:~Top|{{0}}/{{1}}:{{2}}>{{4}}[1]
// CHECK-SAME{LITERAL}: OMReferenceTarget:~Top|{{0}}/{{1}}:{{2}}>{{5}}.io.f0
// CHECK-SAME{LITERAL}: OMReferenceTarget:~Top|{{0}}/{{1}}:{{2}}>{{5}}.io.f1
// CHECK-SAME{LITERAL}: OMReferenceTarget:~Top|{{0}}/{{1}}:{{2}}>{{6}}
// CHECK-SAME{LITERAL}: OMReferenceTarget:~Top|{{0}}/{{1}}:{{2}}>{{6}}
// CHECK-SAME{LITERAL}: OMReferenceTarget:~Top|{{0}}/{{1}}:{{2}}>{{7}}[0]
// CHECK-SAME{LITERAL}: OMReferenceTarget:~Top|{{0}}/{{1}}:{{2}}>{{7}}[1]
// CHECK-SAME: symbols = [
// CHECK-SAME:  @Top,
// CHECK-SAME:  #hw.innerNameRef<@Top::@a>
// CHECK-SAME:  @A
// CHECK-SAME:  #hw.innerNameRef<@A::@[[in0_sym]]>
// CHECK-SAME:  #hw.innerNameRef<@A::@[[w0_sym]]>
// CHECK-SAME:  #hw.innerNameRef<@A::@[[in1_sym]]>
// CHECK-SAME:  #hw.innerNameRef<@A::@[[w1_sym]]>
// CHECK-SAME:  #hw.innerNameRef<@A::@[[wf_sym]]>

//===----------------------------------------------------------------------===//
// Targets that are allowed to lose their tracker
//===----------------------------------------------------------------------===//

firrtl.circuit "DeletedTargets" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    a = {info = #loc, index = 1, value = {omir.tracker, id = 0, type = "OMReferenceTarget"}},
    b = {info = #loc, index = 2, value = {omir.tracker, id = 1, type = "OMMemberReferenceTarget"}},
    c = {info = #loc, index = 3, value = {omir.tracker, id = 2, type = "OMMemberInstanceTarget"}}
  }}]
}]} {
  firrtl.module @DeletedTargets() {}
}
// CHECK-LABEL: firrtl.circuit "DeletedTargets"
// CHECK:       sv.verbatim
// CHECK-SAME:  \22name\22: \22a\22
// CHECK-SAME:  \22value\22: \22OMDeleted:\22
// CHECK-SAME:  \22name\22: \22b\22
// CHECK-SAME:  \22value\22: \22OMDeleted:\22
// CHECK-SAME:  \22name\22: \22c\22
// CHECK-SAME:  \22value\22: \22OMDeleted:\22

//===----------------------------------------------------------------------===//
// Make SRAM Paths Absolute (`SetOMIRSRAMPaths`)
//===----------------------------------------------------------------------===//

firrtl.circuit "SRAMPaths" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [
    {
      info = #loc,
      id = "OMID:0",
      fields = {
        omType = {info = #loc, index = 0, value = ["OMString:OMLazyModule", "OMString:OMSRAM"]},
        // We purposefully pick the wrong `OMMemberTarget` here, to check that
        // it actually gets emitted as a `OMMemberInstanceTarget`. These can
        // change back and forth as the FIRRTL passes work on the IR, and the
        // OMIR output should reflect the final target.
        finalPath = {info = #loc, index = 1, value = {omir.tracker, id = 0, type = "OMMemberReferenceTarget"}}
      }
    },
    {
      info = #loc,
      id = "OMID:1",
      fields = {
        omType = {info = #loc, index = 0, value = ["OMString:OMLazyModule", "OMString:OMSRAM"]},
        finalPath = {info = #loc, index = 1, value = {omir.tracker, id = 1, type = "OMMemberReferenceTarget"}}
      }
    }
  ]
}]} {
  firrtl.extmodule private @MySRAM()
  firrtl.module private @Submodule() {
    firrtl.instance mem1 {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0}]} @MySRAM()
    %mem2_port = firrtl.mem Undefined {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 1}], depth = 8, name = "mem2", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32 } : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<42>>
  }
  firrtl.module @SRAMPaths() {
    firrtl.instance sub @Submodule()
  }
}
// CHECK-LABEL: firrtl.circuit "SRAMPaths" {
// CHECK:         firrtl.extmodule private @MySRAM()
// CHECK-NEXT:    firrtl.module private @Submodule() {
// CHECK-NEXT:      firrtl.instance mem1 sym [[SYMMEM1:@[a-zA-Z0-9_]+]]
// CHECK-SAME:        @MySRAM()
// CHECK-NEXT:      firrtl.mem sym [[SYMMEM2:@[a-zA-Z0-9_]+]]
// CHECK-SAME:        name = "mem2"
// CHECK-SAME:        : !firrtl.bundle
// CHECK-NEXT:    }
// CHECK-NEXT:    firrtl.module @SRAMPaths() {
// CHECK-NEXT:      firrtl.instance sub sym [[SYMSUB:@[a-zA-Z0-9_]+]]
// CHECK-NOT:         circt.nonlocal
// CHECK-SAME:        @Submodule()
// CHECK-NEXT:    }
// CHECK-NEXT:    sv.verbatim

// CHECK-SAME:           \22id\22: \22OMID:0\22
// CHECK-SAME:             \22name\22: \22omType\22
// CHECK-SAME:             \22value\22: [
// CHECK-SAME:               \22OMString:OMLazyModule\22
// CHECK-SAME:               \22OMString:OMSRAM\22
// CHECK-SAME:             ]
// CHECK-SAME:              \22name\22: \22finalPath\22
// CHECK-SAME{LITERAL}:    \22value\22: \22OMMemberInstanceTarget:~SRAMPaths|{{0}}/{{1}}:{{2}}/{{3}}:{{4}}\22

// CHECK-SAME:           \22id\22: \22OMID:1\22
// CHECK-SAME:             \22name\22: \22omType\22
// CHECK-SAME:             \22value\22: [
// CHECK-SAME:               \22OMString:OMLazyModule\22
// CHECK-SAME:               \22OMString:OMSRAM\22
// CHECK-SAME:             ]
// CHECK-SAME:             \22name\22: \22finalPath\22
// CHECK-SAME{LITERAL}:    \22value\22: \22OMMemberInstanceTarget:~SRAMPaths|{{0}}/{{1}}:{{2}}/{{5}}:
// CHECK-NOT:                {{.+}}
// CHECK-SAME:               {{[^\\]+}}\22

// CHECK-SAME:    symbols = [
// CHECK-SAME:      @SRAMPaths,
// CHECK-SAME:      #hw.innerNameRef<@SRAMPaths::[[SYMSUB]]>,
// CHECK-SAME:      @Submodule,
// CHECK-SAME:      #hw.innerNameRef<@Submodule::[[SYMMEM1]]>,
// CHECK-SAME:      @MySRAM,
// CHECK-SAME:      #hw.innerNameRef<@Submodule::[[SYMMEM2]]>
// CHECK-SAME:    ]}
// CHECK-NEXT:  }

//===----------------------------------------------------------------------===//
// Make SRAM Paths Absolute with existing absolute NLA (`SetOMIRSRAMPaths`)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: firrtl.circuit "SRAMPathsWithNLA"
firrtl.circuit "SRAMPathsWithNLA" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [
    {
      info = #loc,
      id = "OMID:1",
      fields = {
        omType = {info = #loc, index = 0, value = ["OMString:OMLazyModule", "OMString:OMSRAM"]},
        finalPath = {info = #loc, index = 1, value = {omir.tracker, id = 1, type = "OMMemberReferenceTarget"}}
      }
    },
    {
      info = #loc,
      id = "OMID:2",
      fields = {
        omType = {info = #loc, index = 0, value = ["OMString:OMLazyModule", "OMString:OMSRAM"]},
        finalPath = {info = #loc, index = 1, value = {omir.tracker, id = 2, type = "OMMemberReferenceTarget"}}
      }
    }
  ]
}]} {
  hw.hierpath private @nla_old [@SRAMPathsWithNLA::@s1, @Submodule::@mem]
  hw.hierpath private @nla_new [@SRAMPathsWithNLA::@s1, @Submodule]
  firrtl.module @Submodule() {
    %mem_port = firrtl.mem sym @mem Undefined {
      annotations = [
        {circt.nonlocal = @nla_old, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 1}
      ],
      depth = 8,
      name = "mem",
      portNames = ["port"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<42>>
    // CHECK: %mem2_port = firrtl.mem sym @[[mem2_sym:.+]] Undefined
    %mem2_port = firrtl.mem Undefined {
      annotations = [
        {circt.nonlocal = @nla_new, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 2}
      ],
      depth = 8,
      name = "mem2",
      portNames = ["port"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<42>>
  }
  firrtl.module @SRAMPathsWithNLA() {
    firrtl.instance sub sym @s1 @Submodule()
    firrtl.instance sub1 sym @s2 @Submodule()
  }
}

// CHECK:         sv.verbatim
// CHECK-SAME:           id\22: \22OMID:1\22
// CHECK-SAME:             \22name\22: \22omType\22
// CHECK-SAME:             \22value\22: [
// CHECK-SAME:               \22OMString:OMLazyModule\22
// CHECK-SAME:               \22OMString:OMSRAM\22
// CHECK-SAME:             ]
// CHECK-SAME:             \22name\22: \22finalPath\22
// CHECK-SAME{LITERAL}:    \22value\22: \22OMMemberInstanceTarget:~SRAMPathsWithNLA|{{0}}/{{1}}:{{2}}/{{3}}:
// CHECK-NOT:                {{.+}}
// CHECK-SAME:               {{[^\\]+}}\22

// CHECK-SAME:           id\22: \22OMID:2\22
// CHECK-SAME:             \22name\22: \22omType\22
// CHECK-SAME:             \22value\22: [
// CHECK-SAME:               \22OMString:OMLazyModule\22
// CHECK-SAME:               \22OMString:OMSRAM\22
// CHECK-SAME:             ]
// CHECK-SAME:             \22name\22: \22finalPath\22
// CHECK-SAME{LITERAL}:    \22value\22: \22OMMemberInstanceTarget:~SRAMPathsWithNLA|{{0}}/{{1}}:{{2}}/{{4}}:
// CHECK-NOT:                {{.+}}
// CHECK-SAME:               {{[^\\]+}}\22

// CHECK-SAME:  symbols = [
// CHECK-SAME:    @SRAMPathsWithNLA,
// CHECK-SAME:    #hw.innerNameRef<@SRAMPathsWithNLA::@s1>,
// CHECK-SAME:    @Submodule,
// CHECK-SAME:    #hw.innerNameRef<@Submodule::@mem>,
// CHECK-SAME:    #hw.innerNameRef<@Submodule::@[[mem2_sym]]>
// CHECK-SAME:  ]

//===----------------------------------------------------------------------===//
// Make SRAM Paths Absolute with existing non-absolute NLAs (`SetOMIRSRAMPaths`)
//===----------------------------------------------------------------------===//

firrtl.circuit "SRAMPathsWithNLA" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [
    {
      info = #loc,
      id = "OMID:0",
      fields = {
        omType = {info = #loc, index = 0, value = ["OMString:OMLazyModule", "OMString:OMSRAM"]},
        instancePath = {info = #loc, index = 1, value = {omir.tracker, id = 0, type = "OMMemberReferenceTarget"}}
      }
    }
  ]
}]} {
  hw.hierpath private @nla [@SRAMPaths::@sub, @Submodule]
  firrtl.extmodule private @MySRAM()
  firrtl.module private @Submodule() {
    firrtl.instance mem1 {annotations = [{circt.nonlocal = @nla, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0}]} @MySRAM()
  }
  firrtl.module private @SRAMPaths() {
    firrtl.instance sub sym @sub @Submodule()
  }
  firrtl.module @SRAMPathsWithNLA() {
    firrtl.instance paths @SRAMPaths()
  }
}

// CHECK-LABEL: firrtl.circuit "SRAMPathsWithNLA"
// CHECK:      symbols = [
// CHECK-SAME:   @SRAMPathsWithNLA,
// CHECK-SAME:   #hw.innerNameRef<@SRAMPathsWithNLA::@{{[^>]+}}>,
// CHECK-SAME:   @SRAMPaths,
// CHECK-SAME:   #hw.innerNameRef<@SRAMPaths::@sub>,
// CHECK-SAME:   @Submodule,
// CHECK-SAME:   #hw.innerNameRef<@Submodule::@{{[^>]+}}>,
// CHECK-SAME:   @MySRAM
// CHECK-SAME: ]

//===----------------------------------------------------------------------===//
// Make SRAM Paths Absolute when SRAM is top-level (invalid for NLA)
//===----------------------------------------------------------------------===//

firrtl.circuit "SRAMPathsTopLevel" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [
    {
      info = #loc,
      id = "OMID:0",
      fields = {
        omType = {info = #loc, index = 0, value = ["OMString:OMLazyModule", "OMString:OMSRAM"]},
        finalPath = {info = #loc, index = 1, value = {omir.tracker, id = 0, type = "OMMemberReferenceTarget"}}
      }
    }
  ]
}]} {
  firrtl.extmodule @MySRAM()
  firrtl.module @SRAMPathsTopLevel() {
    firrtl.instance mem1 {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0}]} @MySRAM()
  }
}

// CHECK-LABEL: firrtl.circuit "SRAMPathsTopLevel"
// CHECK:       firrtl.instance mem1 sym [[SYMMEM1:@[a-zA-Z0-9_]+]]

// CHECK:       sv.verbatim
// CHECK-SAME{LITERAL}:    \22value\22: \22OMMemberInstanceTarget:~SRAMPathsTopLevel|{{0}}/{{1}}:{{2}}\22

// CHECK-SAME:  symbols = [
// CHECK-SAME:    @SRAMPathsTopLevel,
// CHECK-SAME:    #hw.innerNameRef<@SRAMPathsTopLevel::[[SYMMEM1]]>,
// CHECK-SAME:    @MySRAM
// CHECK-SAME:  ]

//===----------------------------------------------------------------------===//
// Add module port information to the OMIR (`SetOMIRPorts`)
//===----------------------------------------------------------------------===//

firrtl.circuit "AddPorts" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [
    {
      info = #loc,
      id = "OMID:0",
      fields = {
        containingModule = {
          info = #loc,
          index = 0,
          value = {
            omir.tracker,
            id = 0,
            path = "~AddPorts|AddPorts",
            type = "OMInstanceTarget"
          }
        }
      }
    },
    {
      info = #loc,
      id = "OMID:1",
      fields = {
        containingModule = {
          info = #loc,
          index = 0,
          value = {
            omir.tracker,
            id = 1,
            path = "~AddPorts|AddPorts>w",
            type = "OMReferenceTarget"
          }
        }
      }
    }
  ]
}]} {
  firrtl.module @AddPorts(in %x: !firrtl.uint<29>, out %y: !firrtl.uint<31>) attributes {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0}]} {
    %w = firrtl.wire {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 1}]} : !firrtl.uint<29>
    firrtl.connect %y, %x : !firrtl.uint<31>, !firrtl.uint<29>
  }
}
// CHECK-LABEL: firrtl.circuit "AddPorts"
// CHECK:       firrtl.module @AddPorts
// CHECK-SAME:    in %x: !firrtl.uint<29> sym [[SYMX:@[a-zA-Z0-9_]+]]
// CHECK-SAME:    out %y: !firrtl.uint<31> sym [[SYMY:@[a-zA-Z0-9_]+]]
// CHECK:       %w = firrtl.wire sym [[SYMW:@[a-zA-Z0-9_]+]]
// CHECK:       sv.verbatim

// CHECK-SAME:           \22id\22: \22OMID:0\22
// CHECK-SAME:             \22name\22: \22containingModule\22
// CHECK-SAME{LITERAL}:    \22value\22: \22OMInstanceTarget:~AddPorts|{{0}}\22
// CHECK-SAME:             \22name\22: \22ports\22
// CHECK-SAME:             \22value\22: [
// CHECK-SAME:               {
// CHECK-SAME{LITERAL}:        \22ref\22: \22OMDontTouchedReferenceTarget:~AddPorts|{{0}}>{{1}}\22
// CHECK-SAME:                 \22direction\22: \22OMString:Input\22
// CHECK-SAME:                 \22width\22: \22OMBigInt:1d\22
// CHECK-SAME:               }
// CHECK-SAME:               {
// CHECK-SAME{LITERAL}:        \22ref\22: \22OMDontTouchedReferenceTarget:~AddPorts|{{0}}>{{2}}\22
// CHECK-SAME:                 \22direction\22: \22OMString:Output\22
// CHECK-SAME:                 \22width\22: \22OMBigInt:1f\22
// CHECK-SAME:               }
// CHECK-SAME:             ]

// CHECK-SAME:           \22id\22: \22OMID:1\22
// CHECK-SAME:             \22name\22: \22containingModule\22
// CHECK-SAME{LITERAL}:    \22value\22: \22OMReferenceTarget:~AddPorts|{{0}}>{{3}}\22
// CHECK-SAME:             \22name\22: \22ports\22
// CHECK-SAME:             \22value\22: [
// CHECK-SAME:               {
// CHECK-SAME{LITERAL}:        \22ref\22: \22OMDontTouchedReferenceTarget:~AddPorts|{{0}}>{{1}}\22
// CHECK-SAME:                 \22direction\22: \22OMString:Input\22
// CHECK-SAME:                 \22width\22: \22OMBigInt:1d\22
// CHECK-SAME:               }
// CHECK-SAME:               {
// CHECK-SAME{LITERAL}:        \22ref\22: \22OMDontTouchedReferenceTarget:~AddPorts|{{0}}>{{2}}\22
// CHECK-SAME:                 \22direction\22: \22OMString:Output\22
// CHECK-SAME:                 \22width\22: \22OMBigInt:1f\22
// CHECK-SAME:               }
// CHECK-SAME:             ]

// CHECK-SAME:  symbols = [
// CHECK-SAME:    @AddPorts,
// CHECK-SAME:    #hw.innerNameRef<@AddPorts::[[SYMX]]>,
// CHECK-SAME:    #hw.innerNameRef<@AddPorts::[[SYMY]]>,
// CHECK-SAME:    #hw.innerNameRef<@AddPorts::[[SYMW]]>
// CHECK-SAME:  ]

firrtl.circuit "AddPortsRelative" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [
    {
      info = #loc,
      id = "OMID:0",
      fields = {
        containingModule = {
          info = #loc,
          index = 0,
          value = {
            omir.tracker,
            id = 0,
            path = "~AddPortsRelative|DUT",
            type = "OMInstanceTarget"
          }
        }
      }
    }
  ]
}]} {
  firrtl.module @AddPortsRelative () {
    %in = firrtl.wire : !firrtl.uint<1>
    %out = firrtl.wire : !firrtl.uint<1>
    %instance_x, %instance_y = firrtl.instance dut @DUT(in x: !firrtl.uint<1>, out y: !firrtl.uint<1>)
    firrtl.connect %instance_x, %in : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %instance_y : !firrtl.uint<1>, !firrtl.uint<1>
  }

  firrtl.module @DUT(in %x: !firrtl.uint<1> sym @x, out %y: !firrtl.uint<1> sym @y) attributes {
    annotations = [
      {class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0},
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation", id = 1}
    ]} {
    firrtl.connect %y, %x : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// CHECK-LABEL: firrtl.circuit "AddPortsRelative"
// CHECK:       firrtl.module @DUT
// CHECK-SAME:    in %x: !firrtl.uint<1> sym [[SYMX:@[a-zA-Z0-9_]+]]
// CHECK-SAME:    out %y: !firrtl.uint<1> sym [[SYMY:@[a-zA-Z0-9_]+]]

// CHECK:       sv.verbatim
// CHECK-SAME:           \22id\22: \22OMID:0\22
// CHECK-SAME:             \22name\22: \22containingModule\22
// CHECK-SAME{LITERAL}:    \22value\22: \22OMInstanceTarget:~DUT|{{0}}\22
// CHECK-SAME:             \22name\22: \22ports\22
// CHECK-SAME:             \22value\22: [
// CHECK-SAME:               {
// CHECK-SAME{LITERAL}:        \22ref\22: \22OMDontTouchedReferenceTarget:~DUT|{{0}}>{{1}}\22
// CHECK-SAME:                 \22direction\22: \22OMString:Input\22
// CHECK-SAME:                 \22width\22: \22OMBigInt:1\22
// CHECK-SAME:               }
// CHECK-SAME:               {
// CHECK-SAME{LITERAL}:        \22ref\22: \22OMDontTouchedReferenceTarget:~DUT|{{0}}>{{2}}\22
// CHECK-SAME:                 \22direction\22: \22OMString:Output\22
// CHECK-SAME:                 \22width\22: \22OMBigInt:1\22
// CHECK-SAME:               }
// CHECK-SAME:             ]

// CHECK-SAME:  symbols = [
// CHECK-SAME:    @DUT,
// CHECK-SAME:    #hw.innerNameRef<@DUT::[[SYMX]]>,
// CHECK-SAME:    #hw.innerNameRef<@DUT::[[SYMY]]>
// CHECK-SAME:  ]


// Check that the Target path is relative to the DUT, except for dutInstance

// Input annotations
// 	{
// 		"class":"sifive.enterprise.firrtl.MarkDUTAnnotation",
// 		"target": "FixPath.C"
// 	},
//   {
//     "class":"freechips.rocketchip.objectmodel.OMIRAnnotation",
//     "nodes": [
//       {
//         "info":"",
//         "id":"OMID:0",
//         "fields":[
//           {
//             "info":"",
//             "name":"dutInstance",
//             "value":"OMMemberInstanceTarget:~FixPath|FixPath/c:C"
//           },
//           {
//             "info":"",
//             "name":"pwm",
//             "value":"OMMemberInstanceTarget:~FixPath|FixPath/c:C>in"
//           },
//           {
//             "info":"",
//             "name":"power",
//             "value":"OMMemberInstanceTarget:~FixPath|FixPath/c:C/cd:D"
//           },
//           {
//             "info":"",
//             "name":"d",
//             "value":"OMMemberInstanceTarget:~FixPath|D"
//           }
//         ]
//       }
//     ]
//   }
// Output OMIR for reference::
// [
//   {
//     "info": "UnlocatableSourceInfo",
//     "id": "OMID:0",
//     "fields": [
//       {
//         "info": "UnlocatableSourceInfo",
//         "name": "dutInstance",
//         "value": "OMMemberInstanceTarget:~FixPath|FixPath/c:C"
//       },
//       {
//         "info": "UnlocatableSourceInfo",
//         "name": "pwm",
//         "value": "OMMemberInstanceTarget:~C|C>in"
//       },
//       {
//         "info": "UnlocatableSourceInfo",
//         "name": "power",
//         "value": "OMMemberInstanceTarget:~C|C/cd:D"
//       },
//       {
//         "info": "UnlocatableSourceInfo",
//         "name": "d",
//         "value": "OMMemberInstanceTarget:~FixPath|D"
//       }
//     ]
//   }
// ]

firrtl.circuit "FixPath"  attributes {annotations = [
  {class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
   nodes = [
     {fields = {
        d = {
          index = 3 : i64,
          info = loc(unknown),
          value = {
            id = 3 : i64,
            omir.tracker,
            path = "~FixPath|D",
            type = "OMMemberInstanceTarget"}},
        dutInstance = {
          index = 0 : i64,
          info = loc(unknown),
          value = {
            id = 0 : i64,
            omir.tracker,
            path = "~FixPath|FixPath/c:C",
            type = "OMMemberInstanceTarget"}},
        power = {
          index = 2 : i64,
          info = loc(unknown),
          value = {
            id = 2 : i64,
            omir.tracker,
            path = "~FixPath|FixPath/c:C/cd:D",
            type = "OMMemberInstanceTarget"}},
        pwm = {
          index = 1 : i64,
          info = loc(unknown),
          value = {
            id = 1 : i64,
            omir.tracker,
            path = "~FixPath|FixPath/c:C>in",
            type = "OMMemberInstanceTarget"}}},
      id = "OMID:0",
      info = loc(unknown)}
    ]}]} {
  hw.hierpath private @nla_3 [@FixPath::@c, @C::@cd, @D]
  hw.hierpath private @nla_2 [@FixPath::@c, @C::@in]
  hw.hierpath private @nla_1 [@FixPath::@c, @C]
  firrtl.module @C(
    in %in: !firrtl.uint<1> sym @in [
      {circt.nonlocal = @nla_2, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 1 : i64}
    ]
  ) attributes {annotations = [
       {circt.nonlocal = @nla_1, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0 : i64},
       {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
     ]}
  {
    firrtl.instance cd sym @cd @D()
  }
  firrtl.module @D() attributes {annotations = [
    {circt.nonlocal = @nla_3, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 2 : i64},
    {class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 3 : i64}
  ]} {}
  firrtl.module @FixPath(in %a: !firrtl.uint<1>) {
    %c_in = firrtl.instance c sym @c @C(in in: !firrtl.uint<1>)
    firrtl.connect %c_in, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.instance d  @D()
  }
  // CHECK-LABEL: firrtl.circuit "FixPath"
  // CHECK:         firrtl.module @FixPath
  // CHECK:           firrtl.instance d  @D()
  // CHECK:         sv.verbatim
  // CHECK-SAME:               name\22: \22dutInstance\22,\0A
  // CHECK-SAME{LITERAL}:      OMMemberInstanceTarget:~FixPath|{{0}}/{{1}}:{{2}}
  // CHECK-SAME:               name\22: \22pwm\22,\0A
  // CHECK-SAME{LITERAL}:      value\22: \22OMMemberInstanceTarget:~C|{{2}}>{{3}}\22\0A
  // CHECK-SAME:               name\22: \22power\22,\0A
  // CHECK-SAME{LITERAL}:      value\22: \22OMMemberInstanceTarget:~C|{{2}}/{{4}}:{{5}}
  // CHECK-SAME:               name\22: \22d\22,\0A
  // CHECK-SAME{LITERAL}:      value\22: \22OMMemberInstanceTarget:~FixPath|{{5}}\22\0A
  // CHECK-SAME:      {output_file = #hw.output_file<"omir.json", excludeFromFileList>, symbols = [@FixPath, #hw.innerNameRef<@FixPath::@c>, @C, #hw.innerNameRef<@C::@in>, #hw.innerNameRef<@C::@cd>, @D]}
}
