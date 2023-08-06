; Test using "include directory" options for resolving source locators in diagnostics.

; Setup:
; RUN: rm -rf %t && mkdir -p %t/subdir %t/alt/subdir
; RUN: echo -e "foo\ngenerate_bad_circuit" > %t/subdir/source.dummy
; RUN: echo -e "foo\nalternate" > %t/alt/subdir/source.dummy

; Check behavior when unable to find the mentioned file.
; RUN: not firtool %s 2>&1 | FileCheck %s --check-prefixes=COMMON,MISSING
; RUN: not firtool %s -I %t/subdir 2>&1 | FileCheck %s --check-prefixes=COMMON,MISSING

; Check referenced file is found when its directory is specified.
; RUN: not firtool %s -I %t -I %t/subdir 2>&1 | FileCheck %s --check-prefixes=COMMON,FOUND
; Same check, but checking -Idir works (no space) and that `--include-dir` works.
; RUN: not firtool %s --include-dir %t -I%t/subdir 2>&1 | FileCheck %s --check-prefixes=COMMON,FOUND

; Check search order.
; RUN: not firtool %s -I %t/alt -I %t 2>&1 | FileCheck %s --check-prefixes=COMMON,ALTERNATE
; RUN: not firtool %s -I %t -I %t/alt 2>&1 | FileCheck %s --check-prefixes=COMMON,FOUND

; Include paths are extra, searched after normal resolution (matters for relative paths).
; Expect this to resolve to %t/subdir/source.dummy.
; RUN: cd %t && not firtool %s -Ialt 2>&1 | FileCheck %s --check-prefixes=COMMON,FOUND

; Check relative path (look in '../alt' from 'subdir').
; RUN: cd %t/subdir && not firtool %s -I../alt 2>&1 | FileCheck %s --check-prefixes=COMMON,ALTERNATE


; COMMON: subdir/source.dummy:2:3:
; MISSING-NOT: alternate
; MISSING-NOT: generate
; ALTERNATE-NEXT: alternate
; FOUND-NEXT: generate_bad_circuit

; Invalid circuit that generates errors with the specified source locator:
circuit Test: @[subdir/source.dummy 2:3]
  module Oops:
