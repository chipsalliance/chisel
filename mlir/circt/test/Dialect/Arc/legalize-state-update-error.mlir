// RUN: circt-opt %s --arc-legalize-state-update --split-input-file --verify-diagnostics

arc.model "Memory" {
^bb0(%arg0: !arc.storage):
  %false = hw.constant false
  arc.clock_tree %false attributes {ct4} {
    %r1 = arc.state_read %s1 : <i32>
    scf.if %false {
      // expected-error @+1 {{could not be moved to be after all reads to the same memory}}
      arc.memory_write %mem2[%false], %r1 : <2 x i32, i1>
      %mr1 = arc.memory_read %mem1[%false] : <2 x i32, i1>
    }
    scf.if %false {
      arc.memory_write %mem1[%false], %r1 : <2 x i32, i1>
      // expected-note @+1 {{could not be moved after this read}}
      %mr1 = arc.memory_read %mem2[%false] : <2 x i32, i1>
    }
  }
  %mem1 = arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<2 x i32, i1>
  %mem2 = arc.alloc_memory %arg0 : (!arc.storage) -> !arc.memory<2 x i32, i1>
  %s1 = arc.alloc_state %arg0 : (!arc.storage) -> !arc.state<i32>
}
