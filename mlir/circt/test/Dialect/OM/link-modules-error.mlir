// RUN: circt-opt %s --verify-diagnostics -pass-pipeline='builtin.module(om-link-modules)' --split-input-file

module {
  // Raise an error if there is no definition.
  module {
    // expected-error @+1 {{class "A" is declared as an external class but there is no definition}}
    om.class.extern @A() {}
  }
  module {
    // expected-note @+1 {{class "A" is declared here as well}}
    om.class.extern @A() {}
  }
}

// -----

module {
  // Raise an error if there are multiple definitions.
  module {
    // expected-error @+1 {{class "A" is declared as an external class but there are multiple definitions}}
    om.class.extern @A() {}
  }
  module {
    // expected-note @+1 {{class "A" is declared here as well}}
    om.class.extern @A() {}
  }
  module {
    // expected-note @+1 {{class "A" is defined here}}
    om.class @A() {}
  }
  module {
    // expected-note @+1 {{class "A" is defined here}}
    om.class @A() {}
  }
}

// -----

module {
  // Check types mismatch.
  module {
    // expected-error @+1 {{failed to link class "A" since declaration doesn't match the definition: 0-th argument type is not equal, 'i2' vs 'i1'}}
    om.class.extern @A(%arg: i1) {
      om.class.extern.field @a: i1
    }
    om.class @UseA(%arg: i1) {
      %0 = om.object @A(%arg) : (i1) -> !om.class.type<@A>
    }
  }
  module {
    // expected-note @+1 {{definition is here}}
    om.class @A(%arg: i2) {
      om.class.field @a, %arg: i2
    }
  }
}

// -----

module {
  module {
    // expected-error @+1 {{failed to link class "A" since declaration doesn't match the definition: declaration has a field "a" but not found in its definition}}
    om.class.extern @A() {
      om.class.extern.field @a : i1
    }
  }
  module {
    // expected-note @+1 {{definition is here}}
    om.class @A() {
    }
  }
}

// -----

module {
  module {
    // expected-error @+1 {{failed to link class "A" since declaration doesn't match the definition: definition has a field "a" but not found in this declaration}}
    om.class.extern @A() {
    }
  }
  module {
    // expected-note @+1 {{definition is here}}
    om.class @A() {
      %0 = om.constant false
      om.class.field @a, %0 : i1
    }
  }
}

// -----

module {
  module {
    // expected-error @+1 {{failed to link class "A" since declaration doesn't match the definition: declaration has a field "a" but types don't match, 'i1' vs 'i2'}}
    om.class.extern @A() {
      om.class.extern.field @a : i2
    }
  }
  module {
    // expected-note @+1 {{definition is here}}
    om.class @A() {
      %0 = om.constant false
      om.class.field @a, %0 : i1
    }
  }
}
