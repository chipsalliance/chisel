// RUN: circt-opt -firrtl-lower-annotations -split-input-file -verify-diagnostics %s

// The "nodes" member in OMIRAnnotation is required.
//
// expected-error @+3 {{Unable to apply annotation}}
// expected-error @+2 {{did not contain required key 'nodes'}}
// expected-note @+1 {{The full Annotation is reproduced here}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "freechips.rocketchip.objectmodel.OMIRAnnotation"
  }
]} {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----

// The "info" member in an OMNode is required.
//
// expected-error @+3 {{Unable to apply annotation}}
// expected-error @+2 {{did not contain required key 'info'}}
// expected-note @+1 {{The full Annotation is reproduced here}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
    nodes = [
      {
        id = "OMID:0"
      }
    ]
  }
]} {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----

// The "id" member in an OMNode is required.
//
// expected-error @+3 {{Unable to apply annotation}}
// expected-error @+2 {{did not contain required key 'id'}}
// expected-note @+1 {{The full Annotation is reproduced here}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
    nodes = [
      {
        info = ""
      }
    ]
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// The "info" member in an OMField is required.
//
// expected-error @+3 {{Unable to apply annotation}}
// expected-error @+2 {{did not contain required key 'info'}}
// expected-note @+1 {{The full Annotation is reproduced here}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
    nodes = [
      {
        fields = [
          {
            name = "x",
            value = "OMReference:0"
          }
        ],
        id = "OMID:0",
        info = "@[A 0:1]"
      }
    ]
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// The "name" member in an OMField is required.
//
// expected-error @+3 {{Unable to apply annotation}}
// expected-error @+2 {{did not contain required key 'name'}}
// expected-note @+1 {{The full Annotation is reproduced here}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
    nodes = [
      {
        fields = [
          {
            info = "",
            value = "OMReference:0"
          }
        ],
        id = "OMID:0",
        info = "@[A 0:1]"
      }
    ]
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// The "value" member in an OMField is required.
//
// expected-error @+3 {{Unable to apply annotation}}
// expected-error @+2 {{did not contain required key 'value'}}
// expected-note @+1 {{The full Annotation is reproduced here}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
    nodes = [
      {
        fields = [
          {
            info = "",
            name = "x"
          }
        ],
        id = "OMID:0",
        info = "@[A 0:1]"
      }
    ]
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// Test the behavior of an OMIR string-encoded type that is never supposed to be
// seen shows up in the OMIR.  This test is checking the behavior for one such
// type, OMInt, which is should show up as an actual integer.
//
// expected-error @+3 {{Unable to apply annotation}}
// expected-error @+2 {{found known string-encoded OMIR type "OMInt"}}
// expected-note @+1 {{the problematic OMIR is reproduced here}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
    nodes = [
      {
        fields = [
          {
            info = "",
            name = "x",
            value = "OMInt:0"
          }
        ],
        id = "OMID:0",
        info = "@[A 0:1]"
      }
    ]
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// Test that an unknown OMIR string-encoded type produces an error.
//
// expected-error @+3 {{Unable to apply annotation}}
// expected-error @+2 {{found unknown string-encoded OMIR type "OMFoo"}}
// expected-note @+1 {{the problematic OMIR is reproduced here}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
    nodes = [
      {
        fields = [
          {
            info = "",
            name = "x",
            value = "OMFoo:Bar"
          }
        ],
        id = "OMID:0",
        info = "@[A 0:1]"
      }
    ]
  }
]} {
  firrtl.module @Foo() {}
}

// -----

// Test that an unexpected MLIR attribute throws an error.
//
// expected-error @+2 {{Unable to apply annotation}}
// expected-error @+1 {{found unexpected MLIR attribute "unit"}}
firrtl.circuit "Foo"  attributes {rawAnnotations = [
  {
    class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
    nodes = [
      {
        fields = [
          {
            info = "",
            name = "x", value
          }
        ],
        id = "OMID:0",
        info = "@[A 0:1]"
      }
    ]
  }
]} {
  firrtl.module @Foo() {}
}
