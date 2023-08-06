// RUN: circt-opt -firrtl-add-seqmem-ports -verify-diagnostics -split-input-file %s

// expected-error@below {{MetadataDirAnnotation requires field 'dirname' of string type}}
firrtl.circuit "Simple" attributes {annotations = [{
    class = "sifive.enterprise.firrtl.MetadataDirAnnotation"
  }]} {
  firrtl.module @Simple() {}
}

// -----

// expected-error@below {{AddSeqMemPortsFileAnnotation requires field 'filename' of string type}}
firrtl.circuit "Simple" attributes {annotations = [{
    class = "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation"
  }]} {
  firrtl.module @Simple() {}
}

// -----

// expected-error@below {{circuit has two AddSeqMemPortsFileAnnotation annotations}}
firrtl.circuit "Simple" attributes {annotations = [
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation",
    filename = "test"
  },
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation",
    filename = "test"
  }]} {
  firrtl.module @Simple() {}
}

// -----

// expected-error@below {{AddSeqMemPortAnnotation requires field 'name' of string type}}
firrtl.circuit "Simple" attributes {annotations = [{
  class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
  input = true,
  width = 5
 }]} {
  firrtl.module @Simple() { }
}

// -----

// expected-error@below {{AddSeqMemPortAnnotation requires field 'input' of boolean type}}
firrtl.circuit "Simple" attributes {annotations = [{
  class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
  name = "user_input",
  width = 5
 }]} {
  firrtl.module @Simple() { }
}

// -----

// expected-error@below {{AddSeqMemPortAnnotation requires field 'width' of integer type}}
firrtl.circuit "Simple" attributes {annotations = [{
  class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
  name = "user_input",
  input = true
 }]} {
  firrtl.module @Simple() { }
}
