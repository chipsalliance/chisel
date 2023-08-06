// RUN: circt-opt %s | FileCheck %s
// RUN: circt-opt %s | circt-opt | FileCheck %s

sv.macro.decl @RANDOM
sv.macro.decl @PRINTF_COND_

// CHECK-LABEL: hw.module @test1(%arg0: i1, %arg1: i1, %arg8: i8) {
hw.module @test1(%arg0: i1, %arg1: i1, %arg8: i8) {
  // CHECK: [[FD:%.*]] = hw.constant -2147483646 : i32
  %fd = hw.constant 0x80000002 : i32

  // CHECK: %param_x = sv.localparam {value = 11 : i42} : i42
  %param_x = sv.localparam {value = 11 : i42} : i42


  // This corresponds to this block of system verilog code:
  //    always @(posedge arg0) begin
  //      `ifndef SYNTHESIS
  //         if (`PRINTF_COND_ && arg1) $fwrite(32'h80000002, "Hi\n");
  //      `endif
  //    end // always @(posedge)

  sv.always posedge  %arg0 {
    sv.ifdef.procedural "SYNTHESIS" {
    } else {
      %tmp = sv.macro.ref @PRINTF_COND_() : () -> i1
      %tmpx = sv.constantX : i1
      %tmpz = sv.constantZ : i1
      %tmp2 = comb.and %tmp, %tmpx, %tmpz, %arg1 : i1
      sv.if %tmp2 {
        sv.fwrite %fd, "Hi\n"
      }
      sv.if %tmp2 {
        // Test fwrite with operands.
        sv.fwrite %fd, "%x"(%tmp2) : i1
      } else {
        sv.fwrite %fd, "There\n"
      }
    }
  }

  // CHECK-NEXT: sv.always posedge %arg0 {
  // CHECK-NEXT:   sv.ifdef.procedural "SYNTHESIS" {
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %PRINTF_COND_ = sv.macro.ref @PRINTF_COND
  // CHECK-NEXT:     %x_i1 = sv.constantX : i1
  // CHECK-NEXT:     %z_i1 = sv.constantZ : i1
  // CHECK-NEXT:     [[COND:%.*]] = comb.and %PRINTF_COND_, %x_i1, %z_i1, %arg1 : i1
  // CHECK-NEXT:     sv.if [[COND]] {
  // CHECK-NEXT:       sv.fwrite [[FD]], "Hi\0A"
  // CHECK-NEXT:     }
  // CHECK-NEXT:     sv.if [[COND]] {
  // CHECK-NEXT:       sv.fwrite [[FD]], "%x"([[COND]]) : i1
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       sv.fwrite [[FD]], "There\0A"
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  sv.alwaysff(posedge %arg0) {
    sv.fwrite %fd, "Yo\n"
  }

  // CHECK-NEXT: sv.alwaysff(posedge %arg0)  {
  // CHECK-NEXT:   sv.fwrite [[FD]], "Yo\0A"
  // CHECK-NEXT: }

  sv.alwaysff(posedge %arg0) {
    sv.fwrite %fd, "Sync Main Block\n"
  } ( syncreset : posedge %arg1) {
    sv.fwrite %fd, "Sync Reset Block\n"
  }

  // CHECK-NEXT: sv.alwaysff(posedge %arg0) {
  // CHECK-NEXT:   sv.fwrite [[FD]], "Sync Main Block\0A"
  // CHECK-NEXT:  }(syncreset : posedge %arg1) {
  // CHECK-NEXT:   sv.fwrite [[FD]], "Sync Reset Block\0A"
  // CHECK-NEXT: }

  sv.alwaysff (posedge %arg0) {
    sv.fwrite %fd, "Async Main Block\n"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite %fd, "Async Reset Block\n"
  }

  // CHECK-NEXT: sv.alwaysff(posedge %arg0) {
  // CHECK-NEXT:   sv.fwrite [[FD]], "Async Main Block\0A"
  // CHECK-NEXT:  }(asyncreset : negedge %arg1) {
  // CHECK-NEXT:   sv.fwrite [[FD]], "Async Reset Block\0A"
  // CHECK-NEXT: }

// Smoke test generic syntax.
  sv.initial {
    "sv.if"(%arg0) ( {
      ^bb0:
    }, {
    }) : (i1) -> ()
  }

  // CHECK-NEXT:     sv.initial {
  // CHECK-NEXT:       sv.if %arg0 {
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }

  // CHECK-NEXT: sv.initial {
  // CHECK-NEXT:   sv.case casez %arg8 : i8
  // CHECK-NEXT:   case b0000001x: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "x"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   case b000000x1: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "y"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   default: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "z"
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  sv.initial {
    sv.case casez %arg8 : i8
    case b0000001x: {
      sv.fwrite %fd, "x"
    }
    case b000000x1: {
      sv.fwrite %fd, "y"
    }
    default: {
      sv.fwrite %fd, "z"
    }
  }

  // CHECK-NEXT: sv.initial {
  // CHECK-NEXT:   sv.case %arg1 : i1
  // CHECK-NEXT:   case b0: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "zero"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   case b1: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "one"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   sv.case %arg1 : i1
  // CHECK-NEXT:   case b0: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "zero"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   case b1: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "one"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   sv.case casex %arg1 : i1
  // CHECK-NEXT:   case b0: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "zero"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   case b1: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "one"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   sv.case casez %arg1 : i1
  // CHECK-NEXT:   case b0: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "zero"
  // CHECK-NEXT:   }
  // CHECK-NEXT:   case b1: {
  // CHECK-NEXT:     sv.fwrite [[FD]], "one"
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  sv.initial {
    sv.case %arg1 : i1
    case b0: {
      sv.fwrite %fd, "zero"
    }
    case b1: {
      sv.fwrite %fd, "one"
    }
    sv.case case %arg1 : i1
    case b0: {
      sv.fwrite %fd, "zero"
    }
    case b1: {
      sv.fwrite %fd, "one"
    }
    sv.case casex %arg1 : i1
    case b0: {
      sv.fwrite %fd, "zero"
    }
    case b1: {
      sv.fwrite %fd, "one"
    }
    sv.case casez %arg1 : i1
    case b0: {
      sv.fwrite %fd, "zero"
    }
    case b1: {
      sv.fwrite %fd, "one"
    }
  }

  // CHECK-NEXT: %combWire = sv.reg {sv.attributes = [#sv.attribute<"dont_merge">]} : !hw.inout<i1>
  %combWire = sv.reg {sv.attributes=[#sv.attribute<"dont_merge">]} : !hw.inout<i1>
  // CHECK-NEXT: %selReg = sv.reg {sv.attributes = [#sv.attribute<"dont_merge">, #sv.attribute<"dont_retime" = "true">]} : !hw.inout<i10>
  %selReg = sv.reg {sv.attributes = [#sv.attribute<"dont_merge">, #sv.attribute<"dont_retime" ="true">]} : !hw.inout<i10>
  // CHECK-NEXT: %combWire2 = sv.wire : !hw.inout<i1>
  %combWire2 = sv.wire : !hw.inout<i1>
  // CHECK-NEXT: %regForce = sv.reg : !hw.inout<i1>
  %regForce = sv.reg  : !hw.inout<i1>
  // CHECK-NEXT: sv.alwayscomb {
  sv.alwayscomb {
    // CHECK-NEXT: %x_i1 = sv.constantX : i1
    %tmpx = sv.constantX : i1
    // CHECK-NEXT: sv.passign %combWire, %x_i1 : i1
    sv.passign %combWire, %tmpx : i1
    // CHECK-NEXT: %[[c2_i3:.+]] = hw.constant 2 : i3
    // CHECK-NEXT: %[[v0:.+]] = sv.indexed_part_select_inout %selReg[%[[c2_i3]] : 1] : !hw.inout<i10>, i3
    // CHECK-NEXT: sv.passign %[[v0]], %x_i1 : i1
    %c2 = hw.constant 2 : i3
    %xx1 = sv.indexed_part_select_inout %selReg[%c2:1] :  !hw.inout<i10>, i3
    sv.passign %xx1, %tmpx : i1
    // CHECK-NEXT: sv.force %combWire2, %x_i1 : i1
    sv.force %combWire2, %tmpx : i1
    // CHECK-NEXT: sv.force %regForce, %x_i1 : i1
    sv.force %regForce, %tmpx : i1
    sv.release %combWire2 : !hw.inout<i1>
    sv.release %regForce : !hw.inout<i1>
    // CHECK-NEXT: sv.release %combWire2 : !hw.inout<i1>
    // CHECK-NEXT: sv.release %regForce : !hw.inout<i1>
    // CHECK-NEXT: }
  }

  // CHECK-NEXT: %reg23 = sv.reg : !hw.inout<i23>
  // CHECK-NEXT: %regStruct23 = sv.reg : !hw.inout<struct<foo: i23>>
  // CHECK-NEXT: %reg24 = sv.reg sym @regSym1 : !hw.inout<i23>
  // CHECK-NEXT: %wire25 = sv.wire sym @wireSym1 : !hw.inout<i23>
  %reg23       = sv.reg  : !hw.inout<i23>
  %regStruct23 = sv.reg  : !hw.inout<struct<foo: i23>>
  %reg24       = sv.reg sym @regSym1 : !hw.inout<i23>
  %wire25      = sv.wire sym @wireSym1 : !hw.inout<i23>

  // Simulation Control Tasks
  // CHECK-NEXT: sv.initial {
  // CHECK-NEXT: sv.stop 1
  // CHECK-NEXT: sv.finish 1
  // CHECK-NEXT: sv.exit
  // CHECK-NEXT: }
  sv.initial {
    sv.stop 1
    sv.finish 1
    sv.exit
  }

  // Severity Message Tasks
  // CHECK-NEXT: sv.initial {
  // CHECK-NEXT: sv.fatal 1
  // CHECK-NEXT: sv.fatal 1, "hello"
  // CHECK-NEXT: sv.fatal 1, "hello %d"(%arg0) : i1
  // CHECK-NEXT: sv.error
  // CHECK-NEXT: sv.error "hello"
  // CHECK-NEXT: sv.error "hello %d"(%arg0) : i1
  // CHECK-NEXT: sv.warning
  // CHECK-NEXT: sv.warning "hello"
  // CHECK-NEXT: sv.warning "hello %d"(%arg0) : i1
  // CHECK-NEXT: sv.info
  // CHECK-NEXT: sv.info "hello"
  // CHECK-NEXT: sv.info "hello %d"(%arg0) : i1
  // CHECK-NEXT: }
  sv.initial {
    sv.fatal 1
    sv.fatal 1, "hello"
    sv.fatal 1, "hello %d"(%arg0) : i1
    sv.error
    sv.error "hello"
    sv.error "hello %d"(%arg0) : i1
    sv.warning
    sv.warning "hello"
    sv.warning "hello %d"(%arg0) : i1
    sv.info
    sv.info "hello"
    sv.info "hello %d"(%arg0) : i1
  }

  // Tests for ReadMemOp ($readmemb/$readmemh)
  // CHECK-NEXT: sv.initial {
  // CHECK-NEXT:   %memForReadMem = sv.reg
  // CHECK-NEXT:   sv.readmem %memForReadMem, "file1.txt", MemBaseBin
  // CHECK-NEXT:   sv.readmem %memForReadMem, "file2.txt", MemBaseHex
  // CHECK-NEXT: }
  sv.initial {
    %memForReadMem = sv.reg sym @MemForReadMem : !hw.inout<uarray<8xi32>>
    sv.readmem %memForReadMem, "file1.txt", MemBaseBin : !hw.inout<uarray<8xi32>>
    sv.readmem %memForReadMem, "file2.txt", MemBaseHex : !hw.inout<uarray<8xi32>>
  }

  // CHECK-NEXT: hw.output
  hw.output
}

//CHECK-LABEL: sv.bind <@AB::@a1>
//CHECK-NEXT: sv.bind <@AB::@b1>
sv.bind <@AB::@a1>
sv.bind <@AB::@b1>


hw.module.extern @ExternDestMod(%a: i1, %b: i2)
hw.module @InternalDestMod(%a: i1, %b: i2) {}
//CHECK-LABEL: hw.module @AB(%a: i1, %b: i2) {
//CHECK-NEXT:   hw.instance "whatever" sym @a1 @ExternDestMod(a: %a: i1, b: %b: i2) -> () {doNotPrint = 1 : i64}
//CHECK-NEXT:   hw.instance "yo" sym @b1 @InternalDestMod(a: %a: i1, b: %b: i2) -> () {doNotPrint = 1 : i64}

hw.module @AB(%a: i1, %b: i2) {
  hw.instance "whatever" sym @a1 @ExternDestMod(a: %a: i1, b: %b: i2) -> () {doNotPrint=1}
  hw.instance "yo" sym @b1 @InternalDestMod(a: %a: i1, b: %b: i2) -> () {doNotPrint=1}
}

//CHECK-LABEL: hw.module @XMR_src
hw.module @XMR_src(%a : i23) {
  //CHECK-NEXT:   sv.xmr isRooted "a", "b", "c" : !hw.inout<i23>
  %xmr1 = sv.xmr isRooted a,b,c : !hw.inout<i23>
  //CHECK-NEXT:   sv.xmr "a", "b", "c" : !hw.inout<i3>
  %xmr2 = sv.xmr "a",b,c : !hw.inout<i3>
  %r = sv.read_inout %xmr1 : !hw.inout<i23>
  sv.assign %xmr1, %a : i23
}

  hw.module @part_select(%in4 : i4, %in8 : i8) -> (a : i3, b : i5) {
  // CHECK-LABEL: hw.module @part_select

    %myReg2 = sv.reg : !hw.inout<i18>
    %c2_i3 = hw.constant 7 : i4
    // CHECK: = sv.indexed_part_select_inout %myReg2[%[[c7_i4:.+]] : 8] : !hw.inout<i18>, i4
    %a1 = sv.indexed_part_select_inout %myReg2 [%c2_i3:8] : !hw.inout<i18>, i4
    sv.assign %a1, %in8 : i8
    // CHECK:  = sv.indexed_part_select_inout %myReg2[%[[c7_i4]] decrement : 8] : !hw.inout<i18>, i4
    %b1 = sv.indexed_part_select_inout %myReg2 [%c2_i3 decrement:8] : !hw.inout<i18>, i4
    sv.assign %b1, %in8 : i8
    %c3_i3 = hw.constant 3 : i4
    %rc = sv.read_inout %myReg2 : !hw.inout<i18>
    %c = sv.indexed_part_select %rc [%c3_i3:3] : i18, i4
    // CHECK: %[[v2:.+]] = sv.read_inout %myReg2 : !hw.inout<i18>
    // CHECK: = sv.indexed_part_select %[[v2]][%[[c3_i4:.+]] : 3] : i18, i4
    %rd = sv.read_inout %myReg2 : !hw.inout<i18>
    %d = sv.indexed_part_select %rd [%in4 decrement:5] : i18, i4
    // CHECK: %[[v4:.+]] = sv.read_inout %myReg2 : !hw.inout<i18>
    // CHECK: = sv.indexed_part_select %[[v4]][%[[in4:.+]]  decrement : 5] : i18, i4
    hw.output %c, %d : i3, i5
}

// CHECK-LABEL: hw.module @nested_wire
hw.module @nested_wire(%a: i1) {
  // CHECK: sv.ifdef "foo"
  sv.ifdef "foo" {
    // CHECK: sv.wire
    %wire = sv.wire : !hw.inout<i1>
    // CHECK: sv.assign
    sv.assign %wire, %a : i1
  }
}


// CHECK-LABEL: hw.module @ordered_region
hw.module @ordered_region(%a: i1) {
  // CHECK: sv.ordered
  sv.ordered {
    // CHECK: sv.ifdef "foo"
    sv.ifdef "foo" {
      // CHECK: sv.wire
      %wire = sv.wire : !hw.inout<i1>
      // CHECK: sv.assign
      sv.assign %wire, %a : i1
    }
    sv.ifdef "bar" {
      // CHECK: sv.wire
      %wire = sv.wire : !hw.inout<i1>
      // CHECK: sv.assign
      sv.assign %wire, %a : i1
    }
  }
}

// CHECK-LABEL: hw.module @XMRRefOp
hw.hierpath private @ref [@XMRRefOp::@foo, @XMRRefFoo::@a]
hw.hierpath @ref2 [@XMRRefOp::@bar]
hw.module.extern @XMRRefBar()
hw.module @XMRRefFoo() {
  %a = sv.wire sym @a : !hw.inout<i2>
}
hw.module @XMRRefOp() {
  hw.instance "foo" sym @foo @XMRRefFoo() -> ()
  hw.instance "bar" sym @bar @XMRRefBar() -> ()
  // CHECK: %0 = sv.xmr.ref @ref : !hw.inout<i2>
  %0 = sv.xmr.ref @ref : !hw.inout<i2>
  // CHECK: %1 = sv.xmr.ref @ref2 ".x.y.z[42]" : !hw.inout<i8>
  %1 = sv.xmr.ref @ref2 ".x.y.z[42]" : !hw.inout<i8>
}
