// RUN: circt-opt --firrtl-emit-metadata="repl-seq-mem=true repl-seq-mem-file='dut.conf'" --verify-diagnostics -split-input-file %s

//===----------------------------------------------------------------------===//
// RetimeModules
//===----------------------------------------------------------------------===//

// expected-error @+1 {{sifive.enterprise.firrtl.RetimeModulesAnnotation requires a filename}}
firrtl.circuit "NoFilename" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.RetimeModulesAnnotation"
  }]} {
  firrtl.module @NoFilename() { }
}

// -----

// expected-error @+1 {{sifive.enterprise.firrtl.RetimeModulesAnnotation requires a non-empty filename}}
firrtl.circuit "EmptyFilename" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.RetimeModulesAnnotation",
    filename = ""
  }]} {
  firrtl.module @EmptyFilename() { }
}

// -----

// expected-error @+1 {{more than one sifive.enterprise.firrtl.RetimeModulesAnnotation annotation attached}}
firrtl.circuit "MultipleAnnotations" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.RetimeModulesAnnotation",
    filename = "test0.json"
  }, {
    class = "sifive.enterprise.firrtl.RetimeModulesAnnotation",
    filename = "test1.json"
  }]} {
  firrtl.module @MultipleAnnotations() { }
}

// -----

//===----------------------------------------------------------------------===//
// SitestBlackbox
//===----------------------------------------------------------------------===//

// expected-error @+1 {{sifive.enterprise.firrtl.SitestBlackBoxAnnotation requires a filename}}
firrtl.circuit "sitest" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.SitestBlackBoxAnnotation"
  }]} {
  firrtl.module @sitest() { }
}

// -----

// expected-error @+1 {{sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation requires a filename}}
firrtl.circuit "sitest" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation"
  }]} {
  firrtl.module @sitest() { }
}

// -----

// expected-error @+1 {{sifive.enterprise.firrtl.SitestBlackBoxAnnotation requires a non-empty filename}}
firrtl.circuit "sitest" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
    filename = ""
  }]} {
  firrtl.module @sitest() { }
}

// -----

// expected-error @+1 {{sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation requires a non-empty filename}}
firrtl.circuit "sitest" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
    filename = ""
  }]} {
  firrtl.module @sitest() { }
}

// -----

// expected-error @+1 {{more than one sifive.enterprise.firrtl.SitestBlackBoxAnnotation annotation attached}}
firrtl.circuit "sitest" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
    filename = "test.json"
  }, {
    class = "sifive.enterprise.firrtl.SitestBlackBoxAnnotation",
    filename = "test.json"
  }]} {
  firrtl.module @sitest() { }
}

// -----

// expected-error @+1 {{more than one sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation annotation attached}}
firrtl.circuit "sitest" attributes { annotations = [{
    class = "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
    filename = "test.json"
  }, {
    class = "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation",
    filename = "test.json"
  }]} {
  firrtl.module @sitest() { }
}

