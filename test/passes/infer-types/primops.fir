; RUN: firrtl %s abcde ct | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module top :
      wire io : {input x : UInt, output y : UInt}
      node a = io.x
;CHECK: wire io : {input x : UInt@<t:UInt>, output y : UInt@<t:UInt>}@<t:{input x : UInt@<t:UInt>, output y : UInt@<t:UInt>}>
;CHECK: node a = io@<t:{input x : UInt@<t:UInt>, output y : UInt@<t:UInt>}>.x@<t:UInt>
;CHECK: Finished Infer Types
