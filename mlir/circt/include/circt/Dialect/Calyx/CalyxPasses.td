//===-- CalyxPasses.td - Calyx pass definition file --------*- tablegen -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the Calyx dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_CALYXPASSES_TD
#define CIRCT_DIALECT_CALYX_CALYXPASSES_TD

include "mlir/Pass/PassBase.td"

def RemoveGroups : Pass<"calyx-remove-groups", "calyx::ComponentOp"> {
  let summary = "Inlines the groups in a Calyx component.";
  let description = [{
    This pass removes the Group interface from the Calyx program, and inlines all
    assignments. This is done in the following manner:
    1. Assign values to the 'done' signal of the component, corresponding with the
       top-level control group's DoneOp. Add the 'go' signal of the component to
        all assignments.
    2. TODO(Calyx): If there are multiple writes to a signal, replace the reads
       with the disjunction.
    3. Remove all groups.
  }];
  let dependentDialects = ["comb::CombDialect"];
  let constructor = "circt::calyx::createRemoveGroupsPass()";
}

def ClkInsertion : Pass<"calyx-clk-insertion", "calyx::ComponentOp"> {
  let summary = "Inserts assignments from component clock to sub-component clock.";
  let description = "";
  let dependentDialects = [];
  let constructor = "circt::calyx::createClkInsertionPass()";
}

def ResetInsertion : Pass<"calyx-reset-insertion", "calyx::ComponentOp"> {
  let summary = "Connect component reset to sub-component reset for applicable components.";
  let description = "";
  let dependentDialects = [];
  let constructor = "circt::calyx::createResetInsertionPass()";
}

def GroupInvariantCodeMotion : Pass<"calyx-gicm", "calyx::ComponentOp"> {
  let summary = "Lift group-invariant operations to wire-scope.";
  let description = [{
    This pass performs GICM (group-invariant code motion) of operations which are
    deemed to be invariant of the group in which they are placed. In practice,
    this amounts to anything which is not a `calyx.group_done/assign/group_go`
    operation. GICM'd operations are lifted to wire-scope.

    After GICM, a Calyx component has the following properties:
    * No values are being defined within groups (excluding `calyx.group_go`).
      As such, groups will only contain group-level assignments
      (calyx.assign/group_done).
    * Any value referenced by operations within the group may safely be
      referenced by other groups, or operations in wire scope.
    * A group does not define anything structural; it exclusively describes
      wiring between existing structures.
  }];
  let dependentDialects = [];
  let constructor = "circt::calyx::createGroupInvariantCodeMotionPass()";
}

def RemoveCombGroups : Pass<"calyx-remove-comb-groups", "calyx::ComponentOp"> {
  let summary = "Removes combinational groups from a Calyx component.";
  let description = [{
    Transforms combinational groups, which have a constant done condition,
    into proper groups by registering the values read from the ports of cells
    used within the combinational group.

    It also transforms (invoke,if,while)-with into semantically equivalent
    control programs that first enable a group that calculates and registers the
    ports defined by the combinational group execute the respective cond group
    and then execute the control operator.

    ## Example
    ```
    group comb_cond<"static"=0> {
        lt.right = 32'd10;
        lt.left = 32'd1;
        eq.right = r.out;
        eq.left = x.out;
        comb_cond[done] = 1'd1;
    }
    control {
        invoke comp(left = lt.out, ..)(..) with comb_cond;
        if lt.out with comb_cond {
            ...
        }
        while eq.out with comb_cond {
            ...
        }
    }
    ```
    into:
    ```
    group comb_cond<"static"=1> {
        lt.right = 32'd10;
        lt.left = 32'd1;
        eq.right = r.out;
        eq.left = x.out;
        lt_reg.in = lt.out
        lt_reg.write_en = 1'd1;
        eq_reg.in = eq.out;
        eq_reg.write_en = 1'd1;
        comb_cond[done] = lt_reg.done & eq_reg.done ? 1'd1;
    }
    control {
        seq {
          comb_cond;
          invoke comp(left = lt_reg.out, ..)(..);
        }
        seq {
          comb_cond;
          if lt_reg.out {
              ...
          }
        }
        seq {
          comb_cond;
          while eq_reg.out {
              ...
              comb_cond;
          }
        }
    }
    ```
  }];
  let dependentDialects = ["comb::CombDialect"];
  let constructor = "circt::calyx::createRemoveCombGroupsPass()";
}

def CompileControl : Pass<"calyx-compile-control", "calyx::ComponentOp"> {
  let summary = "Generates latency-insensitive finite state machines to realize control.";
  let description = [{
    This pass performs a bottom-up traversal of the control program and does the following:
    1. For each control statement such as "calyx.seq", create a new GroupOp to contain all
       the structure to realize the schedule.
    2. Implement the schedule by setting the constituent groups' GoOp and DoneOp.
    3. Replace the control statement in the control program with the corresponding
       compilation group.
  }];
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect"];
  let constructor = "circt::calyx::createCompileControlPass()";
}

def GoInsertion : Pass<"calyx-go-insertion", "calyx::ComponentOp"> {
  let summary = "Insert go signals into the guards of a group's non-hole assignments";
  let description = [{
    This pass inserts the operation "calyx.group_go" into the guards of all
    assignments housed in the group, with the exception of the "calyx.group_done"
    terminator. For example,

    Before:
    ```mlir
    calyx.group @Group1 {
      calyx.assign %in = %out1, %guard ? : i8
      %done = calyx.group_done %out2 : i1
    }
    ```

    After:
    ```mlir
    // The `go` assignment takes on an undefined
    // value until the Compile Control pass.
    %undef = calyx.undef : i1
    ...
    calyx.group @Group1 {
      %go = calyx.group_go %undef : i1

      %and = comb.and %guard, %go : i1
      calyx.assign %in = %out1, %and ? : i8

      %done = calyx.group_done %out2 : i1
    }
    ```
  }];
  let dependentDialects = ["comb::CombDialect"];
  let constructor = "circt::calyx::createGoInsertionPass()";
}

#endif // CIRCT_DIALECT_CALYX_CALYXPASSES_TD
