;RUN: firrtl %s abcefghipjklmno c | tee %s.out | FileCheck %s
;CHECK: To Flo
circuit Counter : 
  module Counter : 
    input inc : UInt(1)
    output tot : UInt(8)
    input amt : UInt(4)
    
    node T_13 = UInt(255, 8)
    node T_14 = UInt(0, 8)
    reg T_15 : UInt(8)
    T_15.init := T_14
    when inc : 
      node T_16 = add-wrap(T_15, amt)
      node T_17 = gt(T_16, T_13)
      node T_18 = UInt(0, 1)
      node T_19 = mux(T_17, T_18, T_16)
      T_15 := T_19
    tot := T_15
;CHECK: Finished To Flo
