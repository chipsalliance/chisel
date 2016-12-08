circuit RightShiftTester : 
  module RightShift : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip i : UInt<1>, flip j : SInt<1>, i_shifted : UInt, j_shifted : SInt, k_shifted : UInt, l_shifted : UInt, m_shifted : SInt, n_shifted : SInt, o_shifted : UInt}
    
    io is invalid
    wire k : UInt<16>
    k is invalid
    k <= UInt<1>("h01")
    wire o : UInt<32>
    o is invalid
    o <= UInt<21>("h012d687")
    node T_19 = shr(io.i, 1)
    io.i_shifted <= T_19
    node T_20 = shr(io.j, 1)
    io.j_shifted <= T_20
    node T_21 = shr(k, 18)
    io.k_shifted <= T_21
    node T_23 = shr(UInt<4>("h0f"), 4)
    io.l_shifted <= T_23
    node T_25 = shr(asSInt(UInt<1>("h01")), 5)
    io.m_shifted <= T_25
    node T_27 = shr(asSInt(UInt<3>("h03")), 4)
    io.n_shifted <= T_27
    node T_28 = shr(o, 16)
    io.o_shifted <= T_28
    
  module RightShiftTester : 
    input clock : Clock
    input reset : UInt<1>
    output io : {}
    
    io is invalid
    inst dut of RightShift
    dut.io is invalid
    dut.clock <= clock
    dut.reset <= reset
    reg T_6 : UInt<2>, clock with : (reset => (reset, UInt<2>("h00")))
    when UInt<1>("h01") :
      node T_8 = eq(T_6, UInt<2>("h03"))
      node T_10 = and(UInt<1>("h00"), T_8)
      node T_13 = add(T_6, UInt<1>("h01"))
      node T_14 = tail(T_13, 1)
      node T_15 = mux(T_10, UInt<1>("h00"), T_14)
      T_6 <= T_15
      skip
    node done = and(UInt<1>("h01"), T_8)
    when done :
      node T_18 = eq(reset, UInt<1>("h00"))
      when T_18 :
        stop(clock, UInt<1>(1), 0)
        skip
      skip
    dut.io.i <= UInt<1>("h01")
    dut.io.j <= asSInt(UInt<1>("h01"))
    node T_22 = eq(dut.io.i_shifted, UInt<1>("h00"))
    node T_24 = eq(reset, UInt<1>("h00"))
    when T_24 :
      node T_26 = eq(T_22, UInt<1>("h00"))
      when T_26 :
        node T_28 = eq(reset, UInt<1>("h00"))
        when T_28 :
          printf(clock, UInt<1>(1), "Assertion failed\n    at RightShift.scala:47 assert(dut.io.i_shifted === UInt(0))\n")
          skip
        stop(clock, UInt<1>(1), 1)
        skip
      skip
    node T_30 = eq(dut.io.j_shifted, asSInt(UInt<1>("h01")))
    node T_32 = eq(reset, UInt<1>("h00"))
    when T_32 :
      node T_34 = eq(T_30, UInt<1>("h00"))
      when T_34 :
        node T_36 = eq(reset, UInt<1>("h00"))
        when T_36 :
          printf(clock, UInt<1>(1), "Assertion failed\n    at RightShift.scala:48 assert(dut.io.j_shifted === SInt(-1))\n")
          skip
        stop(clock, UInt<1>(1), 1)
        skip
      skip
    node T_38 = eq(dut.io.k_shifted, UInt<1>("h00"))
    node T_40 = eq(reset, UInt<1>("h00"))
    when T_40 :
      node T_42 = eq(T_38, UInt<1>("h00"))
      when T_42 :
        node T_44 = eq(reset, UInt<1>("h00"))
        when T_44 :
          printf(clock, UInt<1>(1), "Assertion failed\n    at RightShift.scala:49 assert(dut.io.k_shifted === UInt(0))\n")
          skip
        stop(clock, UInt<1>(1), 1)
        skip
      skip
    node T_46 = eq(dut.io.l_shifted, UInt<1>("h00"))
    node T_48 = eq(reset, UInt<1>("h00"))
    when T_48 :
      node T_50 = eq(T_46, UInt<1>("h00"))
      when T_50 :
        node T_52 = eq(reset, UInt<1>("h00"))
        when T_52 :
          printf(clock, UInt<1>(1), "Assertion failed\n    at RightShift.scala:50 assert(dut.io.l_shifted === UInt(0))\n")
          skip
        stop(clock, UInt<1>(1), 1)
        skip
      skip
    node T_54 = eq(dut.io.m_shifted, asSInt(UInt<1>("h01")))
    node T_56 = eq(reset, UInt<1>("h00"))
    when T_56 :
      node T_58 = eq(T_54, UInt<1>("h00"))
      when T_58 :
        node T_60 = eq(reset, UInt<1>("h00"))
        when T_60 :
          printf(clock, UInt<1>(1), "Assertion failed\n    at RightShift.scala:51 assert(dut.io.m_shifted === SInt(-1))\n")
          skip
        stop(clock, UInt<1>(1), 1)
        skip
      skip
    node T_62 = eq(dut.io.n_shifted, asSInt(UInt<1>("h00")))
    node T_64 = eq(reset, UInt<1>("h00"))
    when T_64 :
      node T_66 = eq(T_62, UInt<1>("h00"))
      when T_66 :
        node T_68 = eq(reset, UInt<1>("h00"))
        when T_68 :
          printf(clock, UInt<1>(1), "Assertion failed\n    at RightShift.scala:52 assert(dut.io.n_shifted === SInt(0))\n")
          skip
        stop(clock, UInt<1>(1), 1)
        skip
      skip
    node T_70 = eq(dut.io.o_shifted, UInt<5>("h012"))
    node T_72 = eq(reset, UInt<1>("h00"))
    when T_72 :
      node T_74 = eq(T_70, UInt<1>("h00"))
      when T_74 :
        node T_76 = eq(reset, UInt<1>("h00"))
        when T_76 :
          printf(clock, UInt<1>(1), "Assertion failed\n    at RightShift.scala:53 assert(dut.io.o_shifted === UInt(18))\n")
          skip
        stop(clock, UInt<1>(1), 1)
        skip
      skip
    
