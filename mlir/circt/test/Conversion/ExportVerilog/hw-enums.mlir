// RUN: circt-opt %s --test-apply-lowering-options='options=emittedLineLength=100,emitBindComments' -export-verilog -split-input-file -o %t.mlir | FileCheck %s

// CHECK: typedef enum bit [0:0] {enum0_T} enum0;
// CHECK: // typedef enum bit [0:0] {} enum1;
// CHECK: typedef enum bit [0:0] {enum2_U} enum2;
// CHECK-LABEL: module EnumCheck
hw.module @EnumCheck(%a : !hw.enum<T>, %b: !hw.enum<>, %c : !hw.enum<U>)
    -> (d : !hw.enum<T>, e: !hw.enum<>, f: !hw.enum<U>) {
  // CHECK: assign d = a;
  // CHECK: // Zero width: assign e = b;
  // CHECK: assign f = c;
  hw.output %a, %b, %c : !hw.enum<T>, !hw.enum<>, !hw.enum<U>
}

// -----

// CHECK:       typedef enum bit [0:0] {enum0_A, enum0_B} enum0;
// CHECK-LABEL: module EnumCmp(
// CHECK-NEXT:   input  enum0 test,
// CHECK-NEXT:   output       result
// CHECK-NEXT:  )
// CHECK-EMPTY:
// CHECK-NEXT:   assign result = test == enum0_A;
// CHECK-NEXT: endmodule
hw.module @EnumCmp(%test: !hw.enum<A, B>) -> (result: i1) {
  %A = hw.enum.constant A : !hw.enum<A, B>
  %0 = hw.enum.cmp %test, %A : !hw.enum<A, B>, !hw.enum<A, B>
  hw.output %0 : i1
}

// -----

// Mixing !hw.enum types which alias in their fields - one anonymous enum
// and two aliasing named enum
// CHECK: `ifndef _TYPESCOPE_Enums
// CHECK: `define _TYPESCOPE_Enums
// CHECK: typedef enum bit [0:0] {enum0_A, enum0_B} enum0;
// CHECK: `endif // _TYPESCOPE_Enums
// CHECK: `ifndef _TYPESCOPE___AnFSMTypedecl
// CHECK: `define _TYPESCOPE___AnFSMTypedecl
// CHECK: typedef enum bit [0:0] {_state1_A, _state1_B} _state1;
// CHECK: typedef enum bit [0:0] {_state2_A, _state2_B} _state2;
// CHECK: `endif // _TYPESCOPE___AnFSMTypedecl
// CHECK-LABEL: module AnFSM
// CHECK:   enum0 reg_0;
// OLD:   _state1     reg_state1;
// OLD:   _state2     reg_state2;
// CHECK:   always @(posedge clock) begin
// CHECK:     case (reg_0)
// CHECK:       A:
// CHECK:         reg_0 <= enum0_B;
// CHECK:       default:
// CHECK:         reg_0 <= enum0_A;
// CHECK:     endcase
// CHECK:   end
// NEW:   _state1     reg_state1;
// CHECK:   always @(posedge clock) begin
// CHECK:     case (reg_state1)
// CHECK:       _state1_A:
// CHECK:         reg_state1 <= _state1_B;
// CHECK:       default:
// CHECK:         reg_state1 <= _state1_A;
// CHECK:     endcase
// CHECK:   end
// NEW:   _state2     reg_state2;
// CHECK:   always @(posedge clock) begin
// CHECK:     case (reg_state2)
// CHECK:       _state2_A:
// CHECK:         reg_state2 <= _state2_B;
// CHECK:       default:
// CHECK:         reg_state2 <= _state2_A;
// CHECK:     endcase
// CHECK:   end
// CHECK: endmodule

hw.type_scope @__AnFSMTypedecl {
  hw.typedecl @_state1 : !hw.enum<A, B>
  hw.typedecl @_state2 : !hw.enum<A, B>
}

hw.module @AnFSM(%clock : i1) {
  // Anonymous enum
  %reg = sv.reg : !hw.inout<!hw.enum<A, B>>
  %reg_read = sv.read_inout %reg : !hw.inout<!hw.enum<A, B>>
  %A = hw.enum.constant A : !hw.enum<A, B>
  %B = hw.enum.constant B : !hw.enum<A, B>
  sv.always posedge %clock {
    sv.case case %reg_read : !hw.enum<A, B>
      case A : { sv.passign %reg, %B : !hw.enum<A, B> }
      default : { sv.passign %reg, %A : !hw.enum<A, B> }
  }

  // typedecl'd # 1
  %reg_state1 = sv.reg : !hw.inout<!hw.typealias<@__AnFSMTypedecl::@_state1,!hw.enum<A, B>>>
  %reg_read_state1 = sv.read_inout %reg_state1 : !hw.inout<!hw.typealias<@__AnFSMTypedecl::@_state1,!hw.enum<A, B>>>

  %A_state1 = hw.enum.constant A : !hw.typealias<@__AnFSMTypedecl::@_state1,!hw.enum<A, B>>
  %B_state1 = hw.enum.constant B : !hw.typealias<@__AnFSMTypedecl::@_state1,!hw.enum<A, B>>
  sv.always posedge %clock {
    sv.case case %reg_read_state1 : !hw.typealias<@__AnFSMTypedecl::@_state1,!hw.enum<A, B>>
      case A : { sv.passign %reg_state1, %B_state1 : !hw.typealias<@__AnFSMTypedecl::@_state1,!hw.enum<A, B>> }
      default : { sv.passign %reg_state1, %A_state1 : !hw.typealias<@__AnFSMTypedecl::@_state1,!hw.enum<A, B>> }
  }

  // typedecl'd # 2
  %reg_state2 = sv.reg : !hw.inout<!hw.typealias<@__AnFSMTypedecl::@_state2,!hw.enum<A, B>>>
  %reg_read_state2 = sv.read_inout %reg_state2 : !hw.inout<!hw.typealias<@__AnFSMTypedecl::@_state2,!hw.enum<A, B>>>

  %A_state2 = hw.enum.constant A : !hw.typealias<@__AnFSMTypedecl::@_state2,!hw.enum<A, B>>
  %B_state2 = hw.enum.constant B : !hw.typealias<@__AnFSMTypedecl::@_state2,!hw.enum<A, B>>
  sv.always posedge %clock {
    sv.case case %reg_read_state2 : !hw.typealias<@__AnFSMTypedecl::@_state2,!hw.enum<A, B>>
      case A : { sv.passign %reg_state2, %B_state2 : !hw.typealias<@__AnFSMTypedecl::@_state2,!hw.enum<A, B>> }
      default : { sv.passign %reg_state2, %A_state2 : !hw.typealias<@__AnFSMTypedecl::@_state2,!hw.enum<A, B>> }
  }
}
