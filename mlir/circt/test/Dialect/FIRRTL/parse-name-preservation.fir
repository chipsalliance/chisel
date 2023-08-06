; RUN: firtool --parse-only %s | FileCheck %s
circuit Bar :
  module Bar :
    input in: UInt<1>

    ; Should create a "tap" node with the same type and a symbol when the type
    ; is passive.
    wire a: UInt<1>
    ; CHECK: %a = firrtl.wire interesting_name : !firrtl.uint<1>

    ; Should use the non-tap wire in expressions.
    a <= in
    ; CHECK: firrtl.strictconnect %a, %in

    ; When the type is not passive, the tap should be a wire with the passive
    ; type of the original wire.
    wire flip: {flip a: UInt<1>}
    ; CHECK: %flip = firrtl.wire interesting_name : !firrtl.bundle<a flip: uint<1>>

    ; Analog values should be tapped with a node.
    wire analog: Analog<1>
    ; CHECK: %analog = firrtl.wire interesting_name : !firrtl.analog<1>

    ; Should create attaches for analog typed elements between the tap wire and
    ; the original wire.
    ; https://github.com/llvm/circt/issues/2718
    wire w: {flip a: UInt<1>, b: Analog<1>}[1]
    ; CHECK: %w = firrtl.wire interesting_name : !firrtl.vector<bundle<a flip: uint<1>, b: analog<1>>, 1>
