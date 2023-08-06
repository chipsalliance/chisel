// RUN: circt-opt %s --export-split-verilog='dir-name=%t.dir' && cat %t.dir/circt_header.svh | FileCheck %s --check-prefix=HEADER
// RUN: circt-opt %s --export-split-verilog='dir-name=%t.dir' && cat %t.dir/file1.sv | FileCheck %s --check-prefix=FILE1
// RUN: circt-opt %s --export-split-verilog='dir-name=%t.dir' && cat %t.dir/file2.sv | FileCheck %s --check-prefix=FILE2
// RUN: circt-opt %s --export-split-verilog='dir-name=%t.dir' && cat %t.dir/Foo.sv | FileCheck %s --check-prefix=FOO


module attributes {circt.loweringOptions = "emitReplicatedOpsToHeader"}{

// HEADER: `define HEADER
// HEADER-EMPTY:
sv.verbatim "`define HEADER"

// FILE1: `include "circt_header.svh"
// FILE1-NEXT: File1 should include header
sv.verbatim "File1 should include header" {output_file = #hw.output_file<"file1.sv", includeReplicatedOps>}

// FILE2-NOT: `include "circt_header.svh"
// FILE2: File2 should not include header
sv.verbatim "File2 should not include header" {output_file = #hw.output_file<"file2.sv"> }

// FOO: `include "circt_header.svh"
// FOO-NEXT: module Foo
hw.module @Foo() -> () {
  hw.output
}
}
