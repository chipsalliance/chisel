; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: FIRRTL Parsing Error: Expected a statement here.

circuit Nested-Module : 
  module Top : 
    module Child :
