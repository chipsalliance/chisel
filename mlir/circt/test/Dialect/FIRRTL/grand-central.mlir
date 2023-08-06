// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-grand-central,symbol-dce))' -split-input-file %s | FileCheck %s

// This is the main test that includes different interfaces of different
// types. All the interfaces share a common, simple circuit that provides two
// RefType signals, "foo" and "bar".

firrtl.circuit "InterfaceGroundType" attributes {
  annotations = [
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "GroundView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "description of foo",
          name = "foo",
          id = 1 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          description = "multi\nline\ndescription\nof\nbar",
          name = "bar",
          id = 2 : i64
        }
      ],
      id = 0 : i64,
      name = "GroundView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "VectorView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "foo",
              id = 4 : i64
            },
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "bar",
              id = 5 : i64
            }
          ],
          name = "vector"
        }
      ],
      id = 3 : i64,
      name = "VectorView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "BundleView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedBundleType",
          defName = "Bundle",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "foo",
              id = 7 : i64
            },
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              name = "bar",
              id = 8 : i64
            }
          ],
          name = "bundle"
        }
      ],
      id = 6 : i64,
      name = "BundleView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "VectorOfBundleView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedBundleType",
              defName = "Bundle2",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "foo",
                  id = 10 : i64
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "bar",
                  id = 11 : i64
                }
              ],
              name = "bundle2"
            }
          ],
          name = "vector"
        }
      ],
      id = 9 : i64,
      name = "VectorOfBundleView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "VectorOfVectorView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedVectorType",
              defName = "Vector2",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "foo",
                  id = 13 : i64
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "bar",
                  id = 14 : i64
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "baz",
                  id = 15 : i64
                }
              ],
              name = "vector2"
            },
            {
              class = "sifive.enterprise.grandcentral.AugmentedVectorType",
              defName = "Vector2",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "foo",
                  id = 16 : i64
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "bar",
                  id = 17 : i64
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "baz",
                  id = 18 : i64
                }
              ],
              name = "vector2"
            }
          ],
          name = "vector"
        }
      ],
      id = 12 : i64,
      name = "VectorOfVectorView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "ZeroWidthView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 20 : i64,
          name = "zerowidth"
        }
      ],
      id = 19 : i64,
      name = "ZeroWidthView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "ConstantView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",

          name = "foo",
          id = 22 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          name = "bar",
          id = 23 : i64
        }
      ],
      id = 21 : i64,
      name = "ConstantView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "UnsupportedView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedStringType",
          name = "string"
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedBooleanType",
          name = "boolean"
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedIntegerType",
          name = "integer"
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedDoubleType",
          name = "double"
        }
      ],
      id = 24 : i64,
      name = "UnsupporteView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "VectorOfVerbatimView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedVectorType",
              defName = "Vector4",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedStringType",
                  name = "baz"
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedStringType",
                  name = "baz"
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedStringType",
                  name = "baz"
                }
              ],
              name = "vector4"
            },
            {
              class = "sifive.enterprise.grandcentral.AugmentedVectorType",
              defName = "Vector4",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedStringType",
                  name = "baz"
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedStringType",
                  name = "baz"
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedStringType",
                  name = "baz"
                }
              ],
              name = "vector4"
            }
          ],
          name = "vectorOfVerbatim"
        }
      ],
      id = 25 : i64,
      name = "VectorOfVerbatimView"
    },
    {
      class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
      directory = "gct-dir",
      filename = "bindings.sv"
    },
    {
      class = "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
      filename = "gct.yaml"
    }
  ]
} {
  firrtl.module @Companion() attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "GroundView",
        id = 0 : i64,
        name = "GroundView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "VectorView",
        id = 3 : i64,
        name = "VectorView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "BundleView",
        id = 6 : i64,
        name = "BundleView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "VectorOfBundleView",
        id = 9 : i64,
        name = "VectorOfBundleView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "VectorOfVectorView",
        id = 12 : i64,
        name = "VectorOfVectorView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "ZeroWidthView",
        id = 19 : i64,
        name = "ZeroWidthView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "ConstantView",
        id = 21 : i64,
        name = "ConstantView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "UnsupportedView",
        id = 24 : i64,
        name = "UnsupportedView"
      },
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "VectorOfVerbatimView",
        id = 25 : i64,
        name = "VectorOfVerbatimView"
      }
    ]
  } {

    // These are dummy references created for the purposes of the test.
    %_ui0 = firrtl.verbatim.expr "???" : () -> !firrtl.uint<0>
    %_ui1 = firrtl.verbatim.expr "???" : () -> !firrtl.uint<1>
    %_ui2 = firrtl.verbatim.expr "???" : () -> !firrtl.uint<2>
    %ref_ui0 = firrtl.ref.send %_ui0 : !firrtl.uint<0>
    %ref_ui1 = firrtl.ref.send %_ui1 : !firrtl.uint<1>
    %ref_ui2 = firrtl.ref.send %_ui2 : !firrtl.uint<2>

    %ui1 = firrtl.ref.resolve %ref_ui1 : !firrtl.probe<uint<1>>
    %foo = firrtl.node %ui1 {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 1 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 4 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 5 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 7 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 10 : i64
        }
      ]
    } : !firrtl.uint<1>

    %ui2 = firrtl.ref.resolve %ref_ui2 : !firrtl.probe<uint<2>>
    %bar = firrtl.node %ui2 {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 2 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 8 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 11 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 13 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 14 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 15 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 16 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 17 : i64
        },
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 18 : i64
        }
      ]
    } : !firrtl.uint<2>

    %ui0 = firrtl.ref.resolve %ref_ui0 : !firrtl.probe<uint<0>>
    %baz = firrtl.node %ui0 {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 20 : i64
        }
      ]
    } : !firrtl.uint<0>

    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c-1_si2 = firrtl.constant -1 : !firrtl.sint<2>

    %node_c0_ui1 = firrtl.node %c0_ui1 {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 22 : i64
        }
      ]
    } : !firrtl.uint<1>

    %node_c-1_si2 = firrtl.node %c-1_si2 {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 23 : i64
        }
      ]
    } : !firrtl.sint<2>

  }
  firrtl.module @InterfaceGroundType() {
    firrtl.instance companion @Companion()
  }
}

// All AugmentedBundleType annotations are removed from the circuit.
//
// CHECK-LABEL: firrtl.circuit "InterfaceGroundType" {{.+}} {annotations =
// CHECK-SAME:    class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation"
// CHECK-NOT:     class = "sifive.enterprise.grandcentral.AugmentedBundleType"
// CHECK-SAME: {

// Check YAML Output.
//
// Note: Built-in vector serialization works slightly differently than
// user-defined vector serialization.  This results in the verbose "[ ]" for the
// empty dimensions vector, and the terse "[]" for the empty instances vector.
//
// CHECK:      sv.verbatim
// CHECK-SAME:   - name: GroundView
// CHECK-SAME:     fields:
// CHECK-SAME:       - name: foo
// CHECK-SAME:         description: description of foo
// CHECK-SAME:         dimensions: [ ]
// CHECK-SAME:         width: 1
// CHECK-SAME:       - name: bar
// CHECK-SAME:         description: \22multi\\nline\\ndescription\\nof\\nbar\22
// CHECK-SAME:         dimensions: [ ]
// CHECK-SAME:         width: 2
// CHECK-SAME:     instances: []
// CHECK-SAME:   - name: VectorView
// CHECK-SAME:     fields:
// CHECK-SAME:       - name: vector
// CHECK-SAME:         dimensions: [ 2 ]
// CHECK-SAME:         width: 1
// CHECK-SAME:     instances: []
// CHECK-SAME:   - name: BundleView
// CHECK-SAME:     fields: []
// CHECK-SAME:     instances:
// CHECK-SAME:       - name: bundle
// CHECK-SAME:         dimensions: [ ]
// CHECK-SAME:         interface:
// CHECK-SAME:           name: Bundle
// CHECK-SAME:           fields:
// CHECK-SAME:             - name: foo
// CHECK-SAME:               dimensions: [ ]
// CHECK-SAME:               width: 1
// CHECK-SAME:             - name: bar
// CHECK-SAME:               dimensions: [ ]
// CHECK-SAME:               width: 2
// CHECK-SAME:           instances: []
// CHECK-SAME:   - name: VectorOfBundleView
// CHECK-SAME:     fields: []
// CHECK-SAME:     instances:
// CHECK-SAME:       - name: vector
// CHECK-SAME:         dimensions: [ 1 ]
// CHECK-SAME:         interface:
// CHECK-SAME:           name: Bundle2
// CHECK-SAME:           fields:
// CHECK-SAME:             - name: foo
// CHECK-SAME:               dimensions: [ ]
// CHECK-SAME:               width: 1
// CHECK-SAME:             - name: bar
// CHECK-SAME:               dimensions: [ ]
// CHECK-SAME:               width: 2
// CHECK-SAME:           instances: []
// CHECK-SAME:   - name: VectorOfVectorView
// CHECK-SAME:     fields:
// CHECK-SAME:       - name: vector
// CHECK-SAME:         dimensions: [ 3, 2 ]
// CHECK-SAME:         width: 2
// CHECK-SAME:     instances: []
// CHECK-SAME:   - name: ZeroWidthView
// CHECK-SAME:     fields:
// CHECK-SAME:       - name: zerowidth
// CHECK-SAME:         dimensions: [ ]
// CHECK-SAME:         width: 0
// CHECK-SAME:     instances: []
// CHECK-SAME:   - name: ConstantView
// CHECK-SAME:     fields:
// CHECK-SAME:       - name: foo
// CHECK-SAME:         dimensions: [ ]
// CHECK-SAME:         width: 1
// CHECK-SAME:       - name: bar
// CHECK-SAME:         dimensions: [ ]
// CHECK-SAME:         width: 2
// CHECK-SAME:     instances: []
// CHECK-SAME:   - name: UnsupportedView
// CHECK-SAME:     fields: []
// CHECK-SAME:     instances: []
// CHECK-SAME:   - name: VectorOfVerbatimView
// CHECK-SAME:     fields: []
// CHECK-SAME:     instances: []

// The shared companion contains all instantiated interfaces.
// AugmentedGroundType annotations are removed.  Interface is driven via XMRs
// directly from ref resolve ops.
//
// CHECK:          firrtl.module @Companion
// CHECK-SAME:       output_file = #hw.output_file<"gct-dir{{/|\\\\}}"
//
// CHECK-NEXT:       %VectorOfVerbatimView = sv.interface.instance sym @[[vectorOfVerbatim:[a-zA-Z0-9_]+]] : !sv.interface<@VectorOfVerbatimView>
// CHECK-NEXT:       %UnsupportedView = sv.interface.instance sym @[[unsupportedSym:[a-zA-Z0-9_]+]] : !sv.interface<@UnsupportedView>
// CHECK-NEXT:       %ConstantView = sv.interface.instance sym @[[constantSym:[a-zA-Z0-9_]+]] : !sv.interface<@ConstantView>
// CHECK-NEXT:       %ZeroWidthView = sv.interface.instance sym @[[zeroWidthSym:[a-zA-Z0-9_]+]] : !sv.interface<@ZeroWidthView>
// CHECK-NEXT:       %VectorOfVectorView = sv.interface.instance sym @[[vectorOfVectorSym:[a-zA-Z0-9_]+]] : !sv.interface<@VectorOfVectorView>
// CHECK-NEXT:       %VectorOfBundleView = sv.interface.instance sym @[[vectorOfBundleSym:[a-zA-Z0-9_]+]] : !sv.interface<@VectorOfBundleView>
// CHECK-NEXT:       %BundleView = sv.interface.instance sym @[[bundleSym:[a-zA-Z0-9_]+]] : !sv.interface<@BundleView>
// CHECK-NEXT:       %VectorView = sv.interface.instance sym @[[vectorSym:[a-zA-Z0-9_]+]] : !sv.interface<@VectorView>
// CHECK-NEXT:       %GroundView = sv.interface.instance sym @[[groundSym:[a-zA-Z0-9_]+]] : !sv.interface<@GroundView>
//
// CHECK:            %[[foo_ref:[a-zA-Z0-9_]+]] = firrtl.ref.resolve {{.+}} : !firrtl.probe<uint<1>>
// CHECK-NOT:        sifive.enterprise.grandcentral.AugmentedGroundType
// CHECK:            %[[bar_ref:[a-zA-Z0-9_]+]] = firrtl.ref.resolve {{.+}} : !firrtl.probe<uint<2>>
// CHECK-NOT:        sifive.enterprise.grandcentral.AugmentedGroundType
//
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.foo = {{0}};"
// CHECK-SAME:         (%[[foo_ref]]) : !firrtl.uint<1>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[groundSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.bar = {{0}};"
// CHECK-SAME:         (%[[bar_ref]]) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[groundSym]]>]}
//
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[0] = {{0}};"
// CHECK-SAME:         (%[[foo_ref]]) : !firrtl.uint<1>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[1] = {{0}};"
// CHECK-SAME:         (%[[foo_ref]]) : !firrtl.uint<1>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorSym]]>]}
//
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.bundle.foo = {{0}};"
// CHECK-SAME:         (%[[foo_ref]]) : !firrtl.uint<1>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[bundleSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.bundle.bar = {{0}};"
// CHECK-SAME:         (%[[bar_ref]]) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[bundleSym]]>]}
//
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[0].foo = {{0}};"
// CHECK-SAME:         (%[[foo_ref]]) : !firrtl.uint<1>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfBundleSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[0].bar = {{0}};"
// CHECK-SAME:         (%[[bar_ref]]) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfBundleSym]]>]}
//
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[0][0] = {{0}};"
// CHECK-SAME:         (%[[bar_ref]]) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfVectorSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[0][1] = {{0}};"
// CHECK-SAME:         (%[[bar_ref]]) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfVectorSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[0][2] = {{0}};"
// CHECK-SAME:         (%[[bar_ref]]) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfVectorSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[1][0] = {{0}};"
// CHECK-SAME:         (%[[bar_ref]]) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfVectorSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[1][1] = {{0}};"
// CHECK-SAME:         (%[[bar_ref]]) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfVectorSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.vector[1][2] = {{0}};"
// CHECK-SAME:         (%[[bar_ref]]) : !firrtl.uint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[vectorOfVectorSym]]>]}
//
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.foo = {{0}};"
// CHECK-SAME:         (%c0_ui1) : !firrtl.uint<1>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[constantSym]]>]}
// CHECK{LITERAL}:   sv.verbatim "assign {{1}}.bar = {{0}};"
// CHECK-SAME:         (%c-1_si2) : !firrtl.sint<2>
// CHECK-SAME:         {symbols = [#hw.innerNameRef<@Companion::@[[constantSym]]>]}
//
// There are no more verbatim assigns after this.  The zero-width view and any
// "unsupported" types, e.g., AugmentedStringType, are not given XMRs.
//
// CHECK-NOT:        sv.verbatim "assign

// The companion instance is marked "lowerToBind" in the parent.  This instance
// gets the correct output file.
//
// CHECK:          firrtl.module @InterfaceGroundType()
// CHECK:            firrtl.instance companion
// CHECK-SAME:         lowerToBind
// CHECK-SAME:         output_file = #hw.output_file<"bindings.sv", excludeFromFileList>}

// The body of all interfaces are populated with the correct signals, names,
// comments, and types.
//
// CHECK:      sv.interface @GroundView
// CHECK-SAME:   comment = "VCS coverage exclude_file"
// CHECK-SAME:   output_file = #hw.output_file<"gct-dir{{/|\\\\}}"
// CHECK-NEXT:   sv.verbatim "// description of foo"
// CHECK-NEXT:   sv.interface.signal @foo : i1
// CHECK-NEXT:   sv.verbatim "// multi\0A// line\0A// description\0A// of\0A// bar"
// CHECK-NEXT:   sv.interface.signal @bar : i2
//
// CHECK:      sv.interface @VectorView
// CHECK-NEXT:   sv.interface.signal @vector : !hw.uarray<2xi1>
//
// CHECK:      sv.interface @BundleView
// CHECK-NEXT:   sv.verbatim "Bundle bundle();"
//
// CHECK:      sv.interface @Bundle
// CHECK-NEXT:   sv.interface.signal @foo : i1
// CHECK-NEXT:   sv.interface.signal @bar : i2
//
// CHECK:      sv.interface @VectorOfBundleView
// CHECK-NEXT:   sv.verbatim "Bundle2 vector[1]();"
//
// CHECK:      sv.interface @Bundle2
// CHECK-NEXT:   sv.interface.signal @foo : i1
// CHECK-NEXT:   sv.interface.signal @bar : i2
//
// CHECK:      sv.interface @VectorOfVectorView
// CHECK-NEXT:   sv.interface.signal @vector : !hw.uarray<2xuarray<3xi2>>
//
// CHECK:      sv.interface @ZeroWidthView
// CHECK-NEXT:   sv.interface.signal @zerowidth : i0
//
// CHECK:      sv.interface @ConstantView
// CHECK-NEXT:   sv.interface.signal @foo : i1
// CHECK-NEXT:   sv.interface.signal @bar : i2
//
// CHECK:      sv.interface @UnsupportedView
// CHECK-NEXT:   sv.verbatim "// <unsupported string type> string;"
// CHECK-NEXT:   sv.verbatim "// <unsupported boolean type> boolean;"
// CHECK-NEXT:   sv.verbatim "// <unsupported integer type> integer;"
// CHECK-NEXT:   sv.verbatim "// <unsupported double type> double;"
//
// CHECK:      sv.interface @VectorOfVerbatimView
// CHECK-NEXT:   sv.verbatim "// <unsupported string type> vectorOfVerbatim[2][3];"

// -----

firrtl.circuit "PrefixInterfacesAnnotation"
  attributes {annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
        defName = "Bar",
        elements = [],
        name = "bar"}],
     id = 0 : i64,
     name = "MyView"},
    {class = "sifive.enterprise.grandcentral.PrefixInterfacesAnnotation",
     prefix = "PREFIX_"}]}  {
  firrtl.module private @MyView_companion()
    attributes {annotations = [{
      class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
      id = 0 : i64,
      name = "MyView"}]} {}
  firrtl.module private @DUT() {
    firrtl.instance MyView_companion  @MyView_companion()
  }
  firrtl.module @PrefixInterfacesAnnotation() {
    firrtl.instance dut @DUT()
  }
}

// CHECK-LABEL: firrtl.circuit "PrefixInterfacesAnnotation"
// The PrefixInterfacesAnnotation was removed from the circuit.
// CHECK-NOT:     sifive.enterprise.grandcentral.PrefixInterfacesAnnotation

// Interface "Foo" is prefixed.
// CHECK:       sv.interface @PREFIX_Foo
// Interface "Bar" is prefixed, but not its name.
// CHECK-NEXT:    PREFIX_Bar bar()

// Interface "Bar" is prefixed.
// CHECK:       sv.interface @PREFIX_Bar

// -----

firrtl.circuit "DirectoryBehaviorWithDUT" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "bindings.sv"}]} {

  // Each of these modules is instantiated in a different location.  A leading
  // "E" indicates that this is an external module.  A leading "M" indicates
  // that this is a module.  The instantiation location is indicated by three
  // binary bits with an "_" indicating the absence of instantiation:
  //   1) "T" indicates this is instantiated in the "Top" (above the DUT)
  //   2) "D" indicates this is instantiated in the "DUT"
  //   3) "C" indicates this is instantiated in the "Companion"
  // E.g., "ET_C" is an external module instantiated above the DUT and in the
  // Companion.
  firrtl.module @MT__() {}
  firrtl.module @M_D_() {}
  firrtl.module @M__C() {}
  firrtl.module @MTD_() {}
  firrtl.module @M_DC() {}
  firrtl.module @MT_C() {}
  firrtl.module @MTDC() {}
  firrtl.extmodule @ET__() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "ET__.v", text = ""}
  ]}
  firrtl.extmodule @E_D_() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "E_D_.v", text = ""}
  ]}
  firrtl.extmodule @E__C() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "E__C.v", text = ""}
  ]}
  firrtl.extmodule @ETD_() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "ETD_.v", text = ""}
  ]}
  firrtl.extmodule @E_DC() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "E_DC.v", text = ""}
  ]}
  firrtl.extmodule @ET_C() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "ET_C.v", text = ""}
  ]}
  firrtl.extmodule @ETDC() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "ETDC.v", text = ""}
  ]}

  // The Grand Central Companion module.
  firrtl.module private @Companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Foo",
       id = 0 : i64,
       name = "View"}]} {

    firrtl.instance m__c @M__C()
    firrtl.instance m_dc @M_DC()
    firrtl.instance mt_c @MT_C()
    firrtl.instance mtdc @MTDC()

    firrtl.instance e__c @E__C()
    firrtl.instance e_dc @E_DC()
    firrtl.instance et_c @ET_C()
    firrtl.instance etdc @ETDC()
  }

  // The Design-under-test as indicated by the MarkDUTAnnotation
  firrtl.module private @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]
  } {
    firrtl.instance companion @Companion()

    firrtl.instance m_d_ @M_D_()
    firrtl.instance mtd_ @MTD_()
    firrtl.instance m_dc @M_DC()
    firrtl.instance mtdc @MTDC()

    firrtl.instance e_d_ @E_D_()
    firrtl.instance etd_ @ETD_()
    firrtl.instance e_dc @E_DC()
    firrtl.instance etdc @ETDC()
  }

  // The Top module that instantiates the DUT
  firrtl.module @DirectoryBehaviorWithDUT() {
    firrtl.instance dut @DUT()

    firrtl.instance mt__ @MT__()
    firrtl.instance mtd_ @MTD_()
    firrtl.instance mt_c @MT_C()
    firrtl.instance mtdc @MTDC()

    firrtl.instance et__ @ET__()
    firrtl.instance etd_ @ETD_()
    firrtl.instance et_c @ET_C()
    firrtl.instance etdc @ETDC()
  }
}

// Any module instantiated by the Companion, but not instantiated by the DUT is
// moved to the same directory as the Companion.  I.e., only "*__C" and "*T_C"
// modules should be moved into the "gct-dir".
//
// CHECK-LABEL: "DirectoryBehaviorWithDUT"
//
// CHECK-NOT:    output_file
// CHECK:      firrtl.module @M__C
// CHECK-SAME:   output_file = #hw.output_file<"gct-dir{{/|\\\\}}"
// CHECK-NOT:    output_file
// CHECK:      firrtl.module @MT_C
// CHECK-SAME:   output_file = #hw.output_file<"gct-dir{{/|\\\\}}"
//
// CHECK-NOT:    output_file
// CHECK:      firrtl.extmodule @E__C
// CHECK-SAME:   output_file = #hw.output_file<"gct-dir{{/|\\\\}}">
// CHECK-NOT:    output_file
// CHECK:      firrtl.extmodule @ET_C
// CHECK-SAME:   output_file = #hw.output_file<"gct-dir{{/|\\\\}}">
// CHECK-NOT:    output_file
//
// CHECK:      firrtl.module

// -----

firrtl.circuit "DirectoryBehaviorWithoutDUT" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [],
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "bindings.sv"}]} {

  // Each of these modules is instantiated in a different location.  A leading
  // "E" indicates that this is an external module.  A leading "M" indicates
  // that this is a module.  The instantiation location is indicated by three
  // binary bits with an "_" indicating the absence of instantiation:
  //   1) "T" indicates this is instantiated in the "Top"
  //   2) "C" indicates this is instantiated in the "Companion"
  // E.g., "E_C" is an external module instantiated only in the Companion.
  firrtl.module @MT_() {}
  firrtl.module @M_C() {}
  firrtl.module @MTC() {}
  firrtl.extmodule @ET_() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "ET_.v", text = ""}
  ]}
  firrtl.extmodule @E_C() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "E_C.v", text = ""}
  ]}
  firrtl.extmodule @ETC() attributes {annotations = [
    {class = "firrtl.transforms.BlackBoxInlineAnno", name = "ETC.v", text = ""}
  ]}

  // The Grand Central Companion module.
  firrtl.module private @Companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Foo",
       id = 0 : i64,
       name = "View"}]} {

    firrtl.instance m__c @M_C()
    firrtl.instance m_dc @MTC()

    firrtl.instance e__c @E_C()
    firrtl.instance e_dc @ETC()
  }

  // This is the DUT in the previous example, but is no longer marked as the
  // DUT.
  firrtl.module @DirectoryBehaviorWithoutDUT() {
    firrtl.instance companion @Companion()

    firrtl.instance m_d_ @MT_()
    firrtl.instance m_dc @MTC()

    firrtl.instance e_d_ @ET_()
    firrtl.instance e_dc @ETC()
  }

}

// Any module instantiated by the Companion, but not instantiated by the DUT is
// moved to the same directory as the Companion.  I.e., only "*_C" modules
// should be moved into the "gct-dir".
//
// CHECK-LABEL: "DirectoryBehaviorWithoutDUT"
//
// CHECK-NOT:    output_file
// CHECK:      firrtl.module @M_C
// CHECK-SAME:   output_file = #hw.output_file<"gct-dir{{/|\\\\}}"
// CHECK-NOT:    output_file
//
// CHECK:      firrtl.extmodule @E_C
// CHECK-SAME:   output_file = #hw.output_file<"gct-dir{{/|\\\\}}">
// CHECK-NOT:    output_file
//
// CHECK:      firrtl.module

// -----

firrtl.circuit "Top" attributes {
  annotations = [
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "MyInterface_w1",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedBundleType",
          defName = "SameName",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              id = 1 : i64,
              name = "uint"
            }
          ],
          name = "SameName"
        }
      ],
      id = 0 : i64,
      name = "View_w1"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "MyInterface_w2",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedBundleType",
          defName = "SameName",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedGroundType",
              id = 3 : i64,
              name = "uint"
            }
          ],
          name = "SameName"
        }
      ],
      id = 2 : i64,
      name = "View_w2"
    },
    {
      class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
      directory = ".",
      filename = "bindings.sv"
    }
  ]
} {
  firrtl.module @Companion_w1(in %_gen_uint: !firrtl.probe<uint<1>>) attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        id = 0 : i64,
        name = "View_w1"
      }
    ]
  } {
    %0 = firrtl.ref.resolve %_gen_uint : !firrtl.probe<uint<1>>
    %view_uintrefPort = firrtl.node  %0  {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 1 : i64
        }
      ]
    } : !firrtl.uint<1>
  }
  firrtl.module @Companion_w2(in %_gen_uint: !firrtl.probe<uint<2>>) attributes {
    annotations = [
      {
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        id = 2 : i64,
        name = "View_w2"
      }
    ]
  } {
    %0 = firrtl.ref.resolve %_gen_uint : !firrtl.probe<uint<2>>
    %view_uintrefPort = firrtl.node  %0  {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 3 : i64
        }
      ]
    } : !firrtl.uint<2>
  }
  firrtl.module private @DUT() {
    %c0_ui2 = firrtl.constant 0 : !firrtl.uint<2>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %a_w1 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    firrtl.strictconnect %a_w1, %c0_ui1 : !firrtl.uint<1>
    %a_w2 = firrtl.wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<2>
    firrtl.strictconnect %a_w2, %c0_ui2 : !firrtl.uint<2>
    %companion_w1__gen_uint = firrtl.instance companion_w1  @Companion_w1(in _gen_uint: !firrtl.probe<uint<1>>)
    %companion_w2__gen_uint = firrtl.instance companion_w2  @Companion_w2(in _gen_uint: !firrtl.probe<uint<2>>)
    %0 = firrtl.ref.send %a_w1 : !firrtl.uint<1>
    firrtl.ref.define %companion_w1__gen_uint, %0 : !firrtl.probe<uint<1>>
    %1 = firrtl.ref.send %a_w2 : !firrtl.uint<2>
    firrtl.ref.define %companion_w2__gen_uint, %1 : !firrtl.probe<uint<2>>
  }
  firrtl.module @Top() {
    firrtl.instance dut  @DUT()
  }
}

// Check that the correct subinterface name is used when aliasing is possible.
// Here, SameName is used twice as a sub-interface name and we need to make sure
// that MyInterface_w2 uses the uniqued name of SameName.
//
// See: https://github.com/llvm/circt/issues/4234

// CHECK-LABEL:  sv.interface @MyInterface_w1 {{.+}} {
// CHECK-NEXT:     sv.verbatim "SameName SameName();"
// CHECK-NEXT:   }
// CHECK-LABEL:  sv.interface @MyInterface_w2 {{.+}} {
// CHECK-NEXT:     sv.verbatim "SameName_0 SameName();"
// CHECK-NEXT:   }

// -----

firrtl.circuit "NoInterfaces" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
     filename = "gct.yaml"}]} {
  firrtl.module @NoInterfaces() {}
}

// CHECK-LABEL: module {
// CHECK:         sv.verbatim
// CHECK-SAME:      []

// -----

// Check that nonlocal duplicate views are dropped.
firrtl.circuit "Top" attributes {
  annotations = [
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "VectorOfBundleView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedBundleType",
              defName = "Bundle2",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "foo",
                  id = 10 : i64
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "bar",
                  id = 11 : i64
                }
              ],
              name = "bundle2"
            }
          ],
          name = "vector"
        }
      ],
      id = 9 : i64,
      name = "VectorOfBundleView"
    },
    {
      class = "sifive.enterprise.grandcentral.AugmentedBundleType",
      defName = "VectorOfBundleView",
      elements = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedVectorType",
          elements = [
            {
              class = "sifive.enterprise.grandcentral.AugmentedBundleType",
              defName = "Bundle2",
              elements = [
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "foo",
                  id = 110 : i64
                },
                {
                  class = "sifive.enterprise.grandcentral.AugmentedGroundType",
                  name = "bar",
                  id = 111 : i64
                }
              ],
              name = "bundle2"
            }
          ],
          name = "vector"
        }
      ],
      id = 19 : i64,
      name = "VectorOfBundleView"
    },
    {
      class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
      directory = "gct-dir",
      filename = "bindings.sv"
    },
    {
      class = "sifive.enterprise.grandcentral.GrandCentralHierarchyFileAnnotation",
      filename = "gct.yaml"
    }
  ]
} {
  hw.hierpath private @nla_0 [@Top::@t1, @Dut::@s1]
  hw.hierpath private @nla [@Top::@t1, @Dut::@s1]
  firrtl.module @Companion() attributes {
    annotations = [
      {
        circt.nonlocal = @nla,
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "VectorOfBundleView",
        id = 9 : i64,
        name = "VectorOfBundleView"
      },
      {
        circt.nonlocal = @nla_0,
        class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
        defName = "VectorOfBundleView",
        id = 19 : i64,
        name = "VectorOfBundleView"
      }
    ]
  } {
      // These are dummy references created for the purposes of the test.
      %_ui1 = firrtl.verbatim.expr "???" : () -> !firrtl.uint<1>
      %_ui2 = firrtl.verbatim.expr "???" : () -> !firrtl.uint<2>
      %ref_ui1 = firrtl.ref.send %_ui1 : !firrtl.uint<1>
      %ref_ui2 = firrtl.ref.send %_ui2 : !firrtl.uint<2>

      %ui1 = firrtl.ref.resolve %ref_ui1 : !firrtl.probe<uint<1>>
      %foo = firrtl.node %ui1 {
        annotations = [
          {
            circt.nonlocal = @nla,
            class = "sifive.enterprise.grandcentral.AugmentedGroundType",
            id = 10 : i64
          },{
            circt.nonlocal = @nla_0,
            class = "sifive.enterprise.grandcentral.AugmentedGroundType",
            id = 110 : i64
          }
        ]
      } : !firrtl.uint<1>
      %ui2 = firrtl.ref.resolve %ref_ui2 : !firrtl.probe<uint<2>>
      %bar = firrtl.node %ui2 {
        annotations = [
          {
            circt.nonlocal = @nla,
            class = "sifive.enterprise.grandcentral.AugmentedGroundType",
            id = 11 : i64
          },
          {
            circt.nonlocal = @nla_0,
            class = "sifive.enterprise.grandcentral.AugmentedGroundType",
            id = 111 : i64
          }
        ]
      } : !firrtl.uint<2>
      // CHECK: sv.interface.instance sym
      // CHECK-SAME: !sv.interface<@[[VectorOfBundleView:[a-zA-Z0-9_]+]]>
      // CHECK-NOT: sv.interface.instance
    }
  firrtl.module public @Dut() {
    firrtl.instance s1 sym @s1 @Companion()
  }
  firrtl.module public @Top() {
    firrtl.instance t1 sym @t1 @Dut()
  }

  // CHECK:      sv.interface @[[VectorOfBundleView]] attributes
  // CHECK-NOT:    sv.interface @VectorOfBundleView_0
  // CHECK:      sv.interface @Bundle2
  // CHECK-NEXT:   sv.interface.signal @foo : i1
  // CHECK-NEXT:   sv.interface.signal @bar : i2
  // CHECK-NOT:    sv.interface @Bundle2_0
}
