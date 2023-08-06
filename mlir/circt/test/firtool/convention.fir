; RUN: firtool %s --format=fir --parse-only --scalarize-top-module=false --scalarize-ext-modules=false --preserve-aggregate=all | FileCheck --check-prefix=SCALARIZE_NONE %s
; RUN: firtool %s --format=fir --parse-only --scalarize-top-module=true  --scalarize-ext-modules=false --preserve-aggregate=all | FileCheck --check-prefix=SCALARIZE_TOP  %s
; RUN: firtool %s --format=fir --parse-only --scalarize-top-module=false --scalarize-ext-modules=true  --preserve-aggregate=all | FileCheck --check-prefix=SCALARIZE_EXT  %s
; RUN: firtool %s --format=fir --parse-only --scalarize-top-module=true  --scalarize-ext-modules=true  --preserve-aggregate=all | FileCheck --check-prefix=SCALARIZE_BOTH %s

; Ensure that top module and ext modules are marked scalarized.

circuit Top :
  ; SCALARIZE_NONE-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TOP:      attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_EXT-NOT:  attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_BOTH:     attributes {convention = #firrtl<convention scalarized>}
  module Top :
    output port: UInt<8>[2]
    port[0] <= UInt<8>(0)
    port[1] <= UInt<8>(0)
  
  ; SCALARIZE_NONE-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TOP-NOT:  attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_EXT:      attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_BOTH:     attributes {convention = #firrtl<convention scalarized>}
  extmodule External :
    output port: UInt<8>[2]

  ; SCALARIZE_NONE-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TOP-NOT:  attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_EXT-NOT:  attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_BOTH-NOT: attributes {convention = #firrtl<convention scalarized>}
  module Internal :
    output port: UInt<8>[2]
    port[0] <= UInt<8>(0)
    port[1] <= UInt<8>(0)
