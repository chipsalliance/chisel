// RUN: circt-opt --allow-unregistered-dialect -verify-diagnostics -split-input-file -pass-pipeline='builtin.module(firrtl.circuit(firrtl-dedup))' %s

// expected-error@below {{MustDeduplicateAnnotation missing "modules" member}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation"
    }]} {
  firrtl.module @MustDedup() { }
}

// -----

// expected-error@below {{MustDeduplicateAnnotation references module "Simple0" which does not exist}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Simple0"]
    }]} {
  firrtl.module @MustDedup() { }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{module marked NoDedup}}
  firrtl.module @Test0() attributes {annotations = [{class = "firrtl.transforms.NoDedupAnnotation"}]} { }
  firrtl.module @Test1() { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
  firrtl.module @Test0() {
    // expected-note@below {{first operation is a firrtl.wire}}
    %w = firrtl.wire : !firrtl.uint<8>
  }
  firrtl.module @Test1() {
    // expected-note@below {{second operation is a firrtl.constant}}
    %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
  }
}

// -----

// expected-error@+2 {{module "Mid1" not deduplicated with "Mid0"}}
// expected-note@+1 {{in instance "test0" of "Test0", and instance "test1" of "Test1"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Mid0", "~MustDedup|Mid1"]
    }]} {
  firrtl.module @MustDedup() {
    firrtl.instance mid0 @Mid0()
    firrtl.instance mid1 @Mid1()
  }
  firrtl.module @Mid0() {
    firrtl.instance test0 @Test0()
  }
  firrtl.module @Mid1() {
    firrtl.instance test1 @Test1()
  }
  firrtl.module @Test0() {
    // expected-note@below {{first operation is a firrtl.wire}}
    %w = firrtl.wire : !firrtl.uint<8>
  }
  firrtl.module @Test1() {
    // expected-note@below {{second operation is a firrtl.constant}}
    %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0() {
    // expected-note@below {{operations have different number of results}}
    "test"() : () -> ()
  }
  firrtl.module @Test1() {
    // expected-note@below {{second operation here}}
    %0 = "test"() : () -> (i32)
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0() {
    // expected-note@below {{operation result types don't match, first type is '!firrtl.uint<1>'}}
    %w = firrtl.wire : !firrtl.uint<1>
  }
  firrtl.module @Test1() {
    // expected-note@below {{second type is '!firrtl.uint<2>'}}
    %w = firrtl.wire : !firrtl.uint<2>
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0() {
    // expected-note@below {{operation result bundle type has different number of elements}}
    %w = firrtl.wire : !firrtl.bundle<a : uint<1>>
  }
  firrtl.module @Test1() {
    // expected-note@below {{second operation here}}
    %w = firrtl.wire : !firrtl.bundle<>
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0() {
    // expected-note@below {{operation result bundle element "a" flip does not match}}
    %w = firrtl.wire : !firrtl.bundle<a : uint<1>>
  }
  firrtl.module @Test1() {
    // expected-note@below {{second operation here}}
    %w = firrtl.wire : !firrtl.bundle<a flip : uint<1>>
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}
// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0() {
    // expected-note@below {{bundle element 'a' types don't match, first type is '!firrtl.uint<1>'}}
    %w = firrtl.wire : !firrtl.bundle<a : uint<1>>
  }
  firrtl.module @Test1() {
    // expected-note@below {{second type is '!firrtl.uint<2>'}}
    %w = firrtl.wire : !firrtl.bundle<b : uint<2>>
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0(in %a : !firrtl.uint<1>) {
    // expected-note@below {{operations have different number of operands}}
    "test"(%a) : (!firrtl.uint<1>) -> ()
  }
  firrtl.module @Test1(in %a : !firrtl.uint<1>) {
    // expected-note@below {{second operation here}}
    "test"() : () -> ()
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(in a : !firrtl.uint<1>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0(in %a : !firrtl.uint<1>, in %b : !firrtl.uint<1>) {
    // expected-note@below {{operations use different operands, first operand is 'a'}}
    %n = firrtl.node %a : !firrtl.uint<1>
  }
  firrtl.module @Test1(in %c : !firrtl.uint<1>, in %d : !firrtl.uint<1>) {
    // expected-note@below {{second operand is 'd', but should have been 'c'}}
    %n = firrtl.node %d : !firrtl.uint<1>
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>, in b : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(in c : !firrtl.uint<1>, in d : !firrtl.uint<1>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0(in %a : !firrtl.uint<1>) {
    // expected-note@below {{operations have different number of regions}}
    "test"()({}) : () -> ()
  }
  firrtl.module @Test1(in %a : !firrtl.uint<1>) {
    // expected-note@below {{second operation here}}
    "test"() : () -> ()
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(in a : !firrtl.uint<1>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0(in %a : !firrtl.uint<1>) {
    // expected-note@below {{operation regions have different number of blocks}}
    "test"()({
      ^bb0:
        "return"() : () -> ()
    }) : () -> ()
  }
  firrtl.module @Test1(in %a : !firrtl.uint<1>) {
    // expected-note@below {{second operation here}}
    "test"() ({
      ^bb0:
        "return"() : () -> ()
      ^bb1:
        "return"() : () -> ()
    }): () -> ()
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(in a : !firrtl.uint<1>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{port 'a' only exists in one of the modules}}
  firrtl.module @Test0(in %a : !firrtl.uint<1>) { }
  // expected-note@below {{second module to be deduped that does not have the port}}
  firrtl.module @Test1() { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{module port 'a' types don't match, first type is '!firrtl.uint<1>'}}
  firrtl.module @Test0(in %a : !firrtl.uint<1>) { }
  // expected-note@below {{second type is '!firrtl.uint<2>'}}
  firrtl.module @Test1(in %a : !firrtl.uint<2>) { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(in a : !firrtl.uint<2>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{module port 'a' directions don't match, first direction is 'in'}}
  firrtl.module @Test0(in %a : !firrtl.uint<1>) { }
  // expected-note@below {{second direction is 'out'}}
  firrtl.module @Test1(out %a : !firrtl.uint<1>) { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(out a : !firrtl.uint<1>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0() {
    // expected-note@below {{first block has more operations}}
    %w = firrtl.wire : !firrtl.uint<8>
  }
  // expected-note@below {{second block here}}
  firrtl.module @Test1() { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{first block here}}
  firrtl.module @Test0() { }
  firrtl.module @Test1() {
    // expected-note@below {{second block has more operations}}
    %w = firrtl.wire : !firrtl.uint<8>
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{second operation is missing attribute "test1"}}
  firrtl.module @Test0() attributes {test1} { }
  // expected-note@below {{second operation here}}
  firrtl.module @Test1() attributes {} { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{first operation has attribute 'test' with value "a"}}
  firrtl.module @Test0() attributes {test = "a"} { }
  // expected-note@below {{second operation has value "b"}}
  firrtl.module @Test1() attributes {test = "b"} { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{first operation has attribute 'test' with value 0x21}}
  firrtl.module @Test0() attributes {test = 33 : i8} { }
  // expected-note@below {{second operation has value 0x20}}
  firrtl.module @Test1() attributes {test = 32 : i8} { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// This test is checking that we don't crash when the two modules we want
// deduped were actually deduped with another module.

// expected-error@below {{module "Test3" not deduplicated with "Test1"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test1", "~MustDedup|Test3"]
    }]} {
  
  // expected-note@below {{first operation has attribute 'test' with value "a"}}
  firrtl.module @Test0() attributes {test = "a"} { }
  firrtl.module @Test1() attributes {test = "a"} { }
  // expected-note@below {{second operation has value "b"}}
  firrtl.module @Test2() attributes {test = "b"} { }
  firrtl.module @Test3() attributes {test = "b"} { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
    firrtl.instance test2 @Test2()
    firrtl.instance test3 @Test3()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{module port 'a', has a RefType with a different base type '!firrtl.uint<1>' in the same position of the two modules marked as 'must dedup'. (This may be due to Grand Central Taps or Views being different between the two modules.)}}
  firrtl.module @Test0(in %a : !firrtl.probe<uint<1>>, in %b : !firrtl.probe<uint<2>>) { }
  // expected-note@below {{the second module has a different base type '!firrtl.uint<2>'}}
  firrtl.module @Test1(in %a : !firrtl.probe<uint<2>>, in %b : !firrtl.probe<uint<1>>) { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.probe<uint<1>>, in b : !firrtl.probe<uint<2>>)
    firrtl.instance test1 @Test1(in a : !firrtl.probe<uint<2>>, in b : !firrtl.probe<uint<1>>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{contains a RefType port named 'b' that only exists in one of the modules (can be due to difference in Grand Central Tap or View of two modules marked with must dedup)}}
  firrtl.module @Test0(in %a : !firrtl.probe<uint<1>>, in %b : !firrtl.probe<uint<2>>) { }
  // expected-note@below {{second module to be deduped that does not have the RefType port}}
  firrtl.module @Test1(in %a : !firrtl.probe<uint<1>>) { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.probe<uint<1>>, in b : !firrtl.probe<uint<2>>)
    firrtl.instance test1 @Test1(in a : !firrtl.probe<uint<1>>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{contains a RefType port named 'b' that only exists in one of the modules (can be due to difference in Grand Central Tap or View of two modules marked with must dedup)}}
  firrtl.module @Test1(in %a : !firrtl.probe<uint<1>>, in %b : !firrtl.probe<uint<2>>) { }
  // expected-note@below {{second module to be deduped that does not have the RefType port}}
  firrtl.module @Test0(in %a : !firrtl.probe<uint<1>>) { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test1(in a : !firrtl.probe<uint<1>>, in b : !firrtl.probe<uint<2>>)
    firrtl.instance test1 @Test0(in a : !firrtl.probe<uint<1>>)
  }
}
