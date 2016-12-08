circuit GCDTester : 
  module DecoupledGCD : 
    input clock : Clock
    input reset : UInt<1>
    output io : { flip in : { flip ready : UInt<1>, valid : UInt<1>, bits : { a : UInt<32>, b : UInt<32>}}, out : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<32>}}
    io is invalid
    reg busy : UInt<1>, clock with :
      reset => (reset, UInt<1>("h0"))
    reg done : UInt<1>, clock with :
      reset => (reset, UInt<1>("h0"))
    reg x : UInt<32>, clock with :
      reset => (UInt<1>("h0"), x)
    reg y : UInt<32>, clock with :
      reset => (UInt<1>("h0"), y)
    node T_40 = eq(busy, UInt<1>("h0"))
    io.in.ready <= T_40
    io.out.valid <= done
    node T_42 = eq(y, UInt<1>("h0"))
    node T_43 = and(busy, T_42)
    when T_43 : 
      done <= UInt<1>("h1")
      skip
    node T_45 = and(done, io.out.ready)
    when T_45 : 
      busy <= UInt<1>("h0")
      skip
    node start = and(io.in.valid, io.in.ready)
    when start : 
      busy <= UInt<1>("h1")
      done <= UInt<1>("h0")
      skip
    node T_50 = gt(x, y)
    when T_50 : 
      node T_51 = sub(x, y)
      node T_52 = tail(T_51, 1)
      x <= T_52
      skip
    node T_54 = eq(T_50, UInt<1>("h0"))
    when T_54 : 
      node T_55 = sub(y, x)
      node T_56 = tail(T_55, 1)
      y <= T_56
      skip
    when start : 
      x <= io.in.bits.a
      y <= io.in.bits.b
      skip
    io.out.bits <= x
  
  module GCDTester : 
    input clock : Clock
    input reset : UInt<1>
    output io : { }
    io is invalid
    inst dut of DecoupledGCD
    dut.io is invalid
    dut.clock <= clock
    dut.reset <= reset
    reg count : UInt<4>, clock with :
      reset => (reset, UInt<4>("h9"))
    wire a : UInt<7>[10]
    a[0] <= UInt<6>("h2e")
    a[1] <= UInt<7>("h5f")
    a[2] <= UInt<5>("h1a")
    a[3] <= UInt<6>("h3d")
    a[4] <= UInt<5>("h12")
    a[5] <= UInt<6>("h33")
    a[6] <= UInt<6>("h2d")
    a[7] <= UInt<7>("h42")
    a[8] <= UInt<7>("h47")
    a[9] <= UInt<7>("h47")
    wire b : UInt<7>[10]
    b[0] <= UInt<6>("h27")
    b[1] <= UInt<6>("h2c")
    b[2] <= UInt<6>("h25")
    b[3] <= UInt<7>("h60")
    b[4] <= UInt<6>("h30")
    b[5] <= UInt<6>("h27")
    b[6] <= UInt<7>("h55")
    b[7] <= UInt<7>("h54")
    b[8] <= UInt<4>("h8")
    b[9] <= UInt<7>("h50")
    wire z : UInt<3>[10]
    z[0] <= UInt<1>("h1")
    z[1] <= UInt<1>("h1")
    z[2] <= UInt<1>("h1")
    z[3] <= UInt<1>("h1")
    z[4] <= UInt<3>("h6")
    z[5] <= UInt<2>("h3")
    z[6] <= UInt<3>("h5")
    z[7] <= UInt<3>("h6")
    z[8] <= UInt<1>("h1")
    z[9] <= UInt<1>("h1")
    dut.io.out.ready <= UInt<1>("h0")
    reg en : UInt<1>, clock with :
      reset => (reset, UInt<1>("h1"))
    dut.io.in.bits.a <= a[count]
    dut.io.in.bits.b <= b[count]
    dut.io.in.valid <= en
    node T_80 = and(en, dut.io.in.ready)
    when T_80 : 
      en <= UInt<1>("h0")
      skip
    node T_83 = eq(en, UInt<1>("h0"))
    node T_84 = and(dut.io.out.valid, T_83)
    when T_84 : 
      dut.io.out.ready <= UInt<1>("h1")
      node T_87 = eq(dut.io.out.bits, z[count])
      node T_89 = eq(reset, UInt<1>("h0"))
      when T_89 : 
        node T_91 = eq(T_87, UInt<1>("h0"))
        when T_91 : 
          node T_93 = eq(reset, UInt<1>("h0"))
          when T_93 : 
            printf(clock, UInt<1>("h1"), "Assertion failed\n    at GCDTester.scala:38 assert( dut.io.out.bits === z(count) )\n")
            skip
          stop(clock, UInt<1>("h1"), 1)
          skip
        skip
      node T_95 = eq(count, UInt<1>("h0"))
      when T_95 : 
        node T_97 = eq(reset, UInt<1>("h0"))
        when T_97 : 
          stop(clock, UInt<1>("h1"), 0)
          skip
        skip
      node T_99 = eq(T_95, UInt<1>("h0"))
      when T_99 : 
        en <= UInt<1>("h1")
        node T_102 = sub(count, UInt<1>("h1"))
        node T_103 = tail(T_102, 1)
        count <= T_103
        skip
      skip

