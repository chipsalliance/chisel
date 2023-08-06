; RUN: firtool %s --verbose-pass-executions 2>&1 | FileCheck %s
;
; CHECK: [firtool] Running

circuit Empty:
  module Empty:
