; Test interaction between include directory and annotation files.

; Setup:
; RUN: rm -rf %t && mkdir -p %t
; RUN: echo '[{ "class": "circt.test", "target": "~Test" }]' > %t/test.anno.json

; Confirm annotation file not found without include directory provided:
; RUN: not firtool %s --parse-only --annotation-file test.anno.json
; Confirm annotation file is found.
; RUN: firtool %s --parse-only --annotation-file test.anno.json -I %t | grep circt.test

circuit Test:
  module Test:
    output o : UInt<1>
    o <= UInt<1>(0)
    
