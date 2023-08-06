// RUN: circt-opt -pass-pipeline='builtin.module(calyx.component(calyx-remove-groups))' %s | FileCheck %s

module attributes {calyx.entrypoint = "main"} {
  calyx.component @Z(%go: i1 {go}, %reset: i1 {reset}, %clk: i1 {clk}) -> (%flag: i1, %out :i2, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @main
  calyx.component @main(%go: i1 {go}, %reset: i1 {reset}, %clk: i1 {clk}) -> (%done: i1 {done}) {
    %z.go, %z.reset, %z.clk, %z.flag, %z.out, %z.done = calyx.instance @z of @Z : i1, i1, i1, i1, i2, i1
    %fsm.in, %fsm.write_en, %fsm.clk, %fsm.reset, %fsm.out, %fsm.done = calyx.register @fsm : i2, i1, i1, i1, i2, i1

    calyx.wires {
      // CHECK: %[[FSM_IS_GROUP_A_BEGIN_STATE:.+]] = comb.icmp eq %fsm.out, {{.+}} : i2
      // CHECK: %[[GROUP_A_GO_GUARD:.+]] = comb.and %[[FSM_IS_GROUP_A_BEGIN_STATE]], {{.+}} : i1
      // CHECK: %[[GROUP_A_ASSIGN_GUARD:.+]] = comb.and %[[FSM_IS_GROUP_A_BEGIN_STATE]], %z.done : i1
      // CHECK: %[[SEQ_GROUP_DONE_GUARD:.+]] = comb.icmp eq %fsm.out, {{.+}} : i2
      // CHECK: %[[A_GO_AND_COMPONENT_GO:.+]] = comb.and %[[GROUP_A_GO_GUARD]], %go : i1
      %signal_on = hw.constant true
      %group_A_fsm_begin = hw.constant 0 : i2
      %fsm_is_group_A_begin_state = comb.icmp eq %fsm.out, %group_A_fsm_begin : i2
      %group_A_not_done = comb.xor %z.done, %signal_on : i1
      %group_A_go_guard = comb.and %fsm_is_group_A_begin_state, %group_A_not_done : i1

      // Verify the group, and its respective DoneOp, GoOp are removed.
      // CHECK-NOT: calyx.group
      // CHECK-NOT: calyx.group_go
      // CHECK-NOT: calyx.group_done

      // Verify that assignments are guarded by the group's GoOp and the component's go signal.
      // CHECK: calyx.assign %z.go = %[[A_GO_AND_COMPONENT_GO]] ? %z.flag : i1
      calyx.group @A {
        %A.go = calyx.group_go %group_A_go_guard ? %signal_on : i1
        calyx.assign %z.go = %A.go ? %z.flag : i1
        calyx.group_done %z.done : i1
      }

      %group_A_assign_guard = comb.and %fsm_is_group_A_begin_state, %z.done : i1
      %fsm_step_1 = hw.constant 1 : i2
      %seq_group_done_guard = comb.icmp eq %fsm.out, %fsm_step_1 : i2

      // CHECK: %[[UPDATED_A_ASSIGN_GUARD:.+]] = comb.and %[[GROUP_A_ASSIGN_GUARD]], %go : i1
      // Verify that the component's done signal is assigned the top-level group's DoneOp.
      // CHECK: calyx.assign %done = %[[SEQ_GROUP_DONE_GUARD]] ? {{.+}} : i1

      // Verify that the assignments in the top-level group use the updated guard.
      // CHECK: calyx.assign %fsm.in = %[[UPDATED_A_ASSIGN_GUARD]] ? {{.+}} : i2
      calyx.group @seq {
        calyx.assign %fsm.in = %group_A_assign_guard ? %fsm_step_1 : i2
        calyx.assign %fsm.write_en = %group_A_assign_guard ? %signal_on : i1
        calyx.group_done %seq_group_done_guard ? %signal_on : i1
      }
    }

    // CHECK-LABEL: calyx.control {
    // CHECK-NEXT:  }
    calyx.control {
      calyx.enable @seq {compiledGroups = [@A]}
    }
  }
}
