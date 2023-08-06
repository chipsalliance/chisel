// RUN: circt-opt -handshake-insert-buffers=strategy=cycles %s | circt-opt -handshake-insert-buffers=strategy=cycles | FileCheck %s -check-prefix=CHECK


// CHECK-LABEL:   handshake.func @simple_loop(
// CHECK-SAME:                                %[[VAL_0:.*]]: none, ...) -> none
// CHECK:           %[[VAL_1:.*]] = br %[[VAL_0]] : none
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = control_merge %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]] = buffer [2] fifo %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = buffer [2] fifo %[[VAL_2]] : none
// CHECK:           %[[VAL_6:.*]]:3 = fork [3] %[[VAL_5]] : none
// CHECK:           sink %[[VAL_4]] : index
// CHECK:           %[[VAL_7:.*]] = constant %[[VAL_6]]#1 {value = 1 : index} : index
// CHECK:           %[[VAL_8:.*]] = constant %[[VAL_6]]#0 {value = 42 : index} : index
// CHECK:           %[[VAL_9:.*]] = br %[[VAL_6]]#2 : none
// CHECK:           %[[VAL_10:.*]] = br %[[VAL_7]] : index
// CHECK:           %[[VAL_11:.*]] = br %[[VAL_8]] : index
// CHECK:           %[[VAL_12:.*]] = mux %[[VAL_13:.*]]#1 {{\[}}%[[VAL_14:.*]], %[[VAL_11]]] : index, index
// CHECK:           %[[VAL_15:.*]] = buffer [2] seq %[[VAL_12]] : index
// CHECK:           %[[VAL_16:.*]]:2 = fork [2] %[[VAL_15]] : index
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = control_merge %[[VAL_19:.*]], %[[VAL_9]] : none, index
// CHECK:           %[[VAL_20:.*]] = buffer [2] seq %[[VAL_18]] : index
// CHECK:           %[[VAL_21:.*]] = buffer [2] seq %[[VAL_17]] : none
// CHECK:           %[[VAL_13]]:2 = fork [2] %[[VAL_20]] : index
// CHECK:           %[[VAL_22:.*]] = mux %[[VAL_16]]#0 {{\[}}%[[VAL_23:.*]], %[[VAL_10]]] : index, index
// CHECK:           %[[VAL_24:.*]] = buffer [2] seq %[[VAL_22]] : index
// CHECK:           %[[VAL_25:.*]]:2 = fork [2] %[[VAL_24]] : index
// CHECK:           %[[VAL_26:.*]] = arith.cmpi slt, %[[VAL_25]]#1, %[[VAL_16]]#1 : index
// CHECK:           %[[VAL_27:.*]]:3 = fork [3] %[[VAL_26]] : i1
// CHECK:           %[[VAL_28:.*]], %[[VAL_29:.*]] = cond_br %[[VAL_27]]#2, %[[VAL_16]]#0 : index
// CHECK:           sink %[[VAL_29]] : index
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = cond_br %[[VAL_27]]#1, %[[VAL_21]] : none
// CHECK:           %[[VAL_32:.*]], %[[VAL_33:.*]] = cond_br %[[VAL_27]]#0, %[[VAL_25]]#0 : index
// CHECK:           sink %[[VAL_33]] : index
// CHECK:           %[[VAL_34:.*]] = merge %[[VAL_32]] : index
// CHECK:           %[[VAL_35:.*]] = buffer [2] fifo %[[VAL_34]] : index
// CHECK:           %[[VAL_36:.*]] = merge %[[VAL_28]] : index
// CHECK:           %[[VAL_37:.*]] = buffer [2] fifo %[[VAL_36]] : index
// CHECK:           %[[VAL_38:.*]], %[[VAL_39:.*]] = control_merge %[[VAL_30]] : none, index
// CHECK:           %[[VAL_40:.*]] = buffer [2] fifo %[[VAL_39]] : index
// CHECK:           %[[VAL_41:.*]] = buffer [2] fifo %[[VAL_38]] : none
// CHECK:           %[[VAL_42:.*]]:2 = fork [2] %[[VAL_41]] : none
// CHECK:           sink %[[VAL_40]] : index
// CHECK:           %[[VAL_43:.*]] = constant %[[VAL_42]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_44:.*]] = arith.addi %[[VAL_35]], %[[VAL_43]] : index
// CHECK:           %[[VAL_14]] = br %[[VAL_37]] : index
// CHECK:           %[[VAL_19]] = br %[[VAL_42]]#1 : none
// CHECK:           %[[VAL_23]] = br %[[VAL_44]] : index
// CHECK:           %[[VAL_45:.*]], %[[VAL_46:.*]] = control_merge %[[VAL_31]] : none, index
// CHECK:           %[[VAL_47:.*]] = buffer [2] fifo %[[VAL_46]] : index
// CHECK:           %[[VAL_48:.*]] = buffer [2] fifo %[[VAL_45]] : none
// CHECK:           sink %[[VAL_47]] : index
// CHECK:           return %[[VAL_48]] : none
// CHECK:         }
module {
  handshake.func @simple_loop(%arg0: none, ...) -> none {
    %0 = br %arg0 : none
    %1:2 = control_merge %0 : none, index
    %2:3 = fork [3] %1#0 : none
    sink %1#1 : index
    %3 = constant %2#1 {value = 1 : index} : index
    %4 = constant %2#0 {value = 42 : index} : index
    %5 = br %2#2 : none
    %6 = br %3 : index
    %7 = br %4 : index
    %8 = mux %11#1 [%22, %7] : index, index
    %9:2 = fork [2] %8 : index
    %10:2 = control_merge %23, %5 : none, index
    %11:2 = fork [2] %10#1 : index
    %12 = mux %9#0 [%24, %6] : index, index
    %13:2 = fork [2] %12 : index
    %14 = arith.cmpi slt, %13#1, %9#1 : index
    %15:3 = fork [3] %14 : i1
    %trueResult, %falseResult = cond_br %15#2, %9#0 : index
    sink %falseResult : index
    %trueResult_0, %falseResult_1 = cond_br %15#1, %10#0 : none
    %trueResult_2, %falseResult_3 = cond_br %15#0, %13#0 : index
    sink %falseResult_3 : index
    %16 = merge %trueResult_2 : index
    %17 = merge %trueResult : index
    %18:2 = control_merge %trueResult_0 : none, index
    %19:2 = fork [2] %18#0 : none
    sink %18#1 : index
    %20 = constant %19#0 {value = 1 : index} : index
    %21 = arith.addi %16, %20 : index
    %22 = br %17 : index
    %23 = br %19#1 : none
    %24 = br %21 : index
    %25:2 = control_merge %falseResult_1 : none, index
    sink %25#1 : index
    return %25#0 : none
  }
}
