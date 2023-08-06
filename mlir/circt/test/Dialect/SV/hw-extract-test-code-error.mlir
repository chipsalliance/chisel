// RUN:  circt-opt --sv-extract-test-code %s -verify-diagnostics
module attributes {firrtl.extract.assert =  #hw.output_file<"dir3/", excludeFromFileList, includeReplicatedOps>, firrtl.extract.assume.bindfile = #hw.output_file<"file4", excludeFromFileList>} {
  hw.module.extern @foo_cover(%a : i1) -> (b : i1) attributes {"firrtl.extract.cover.extra"} 
  hw.module @extract_return(%clock: i1) -> (c : i1) {
// expected-error @+1 {{Extracting op with result}}
    %b = hw.instance "bar_cover" @foo_cover(a: %clock : i1) -> (b : i1)
    hw.output %b : i1
  }
}
