; Test annotation parse error.

; Setup:
; RUN: echo 'invalid' > %t.json

; RUN: firtool %s --parse-only --annotation-file %t.json --verify-diagnostics

; expected-error @below {{Failed to parse JSON}}
; expected-note @below {{}}
circuit Test:
  module Test:
    output o : UInt<1>
    o <= UInt<1>(0)
    
