; RUN: firtool %s -disable-all-randomization -lowering-options=emitBindComments | FileCheck %s --check-prefixes=CHECK,INIT_NONE
; RUN: firtool %s -annotation-file %s.inline.anno.json -disable-all-randomization -lowering-options=emitBindComments | FileCheck %s --check-prefixes=CHECK,INIT_INLINE
; RUN: firtool %s -annotation-file %s.outline.anno.json -disable-all-randomization -lowering-options=emitBindComments | FileCheck %s --check-prefixes=CHECK,INIT_OUTLINE

circuit Foo :
  module Foo :
    input clock : Clock
    input reset : UInt<1>
    output read : { flip en : UInt<1>, data : UInt<8>, flip addr : UInt<5>}
    output write : { flip en : UInt<1>, flip data : UInt<8>, flip addr : UInt<5>}

    smem m : UInt<8> [32]
    read.data is invalid
    when read.en :
      read mport read_data_MPORT = m[read.addr], clock
      read.data <= read_data_MPORT
    when write.en :
      write mport MPORT = m[write.addr], clock
      MPORT <= write.data

      ; CHECK:             module [[memoryModule:[a-zA-Z0-9_]+]](
      ; CHECK:               reg [7:0] [[memory:[a-zA-Z0-9_]+]][0:31];

      ; INIT_NONE-NOT:       readmem

      ; INIT_INLINE:         initial
      ; INIT_INLINE-NEXT:      $readmemh("mem1.txt", [[memory]]);

      ; INIT_OUTLINE:        /* This instance is elsewhere emitted as a bind statement.
      ; INIT_OUTLINE-NEXT:     [[bindModule:[a-zA-Z0-9_]+]] [[bindInstance:[a-zA-Z0-9_]+]] ();
      ; INIT_OUTLINE-NEXT;   */
      ; INIT_OUTLINE:        endmodule
      ;
      ; INIT_OUTLINE:      FILE "[[memoryModule]]_init.sv"
      ; INIT_OUTLINE-NOT:  FILE
      ; INIT_OUTLINE:      module [[bindModule]]();
      ; INIT_OUTLINE:        initial
      ; INIT_OUTLINE-NEXT:     $readmemb("mem2.txt", [[memoryModule]].[[memory]]);
      ; INIT_OUTLINE:      endmodule
      ;
      ; INIT_OUTLINE:      bind [[memoryModule]] [[bindModule]] [[bindInstance]] ();
