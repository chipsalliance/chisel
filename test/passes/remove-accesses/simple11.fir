; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Done!

circuit DecoupledAdderTests : 
  module NewDecoupledAdder : 
    input clock : Clock
    input reset : UInt<1>
    output io : {flip in : {flip ready : UInt<1>, valid : UInt<1>, bits : {a : UInt<16>, b : UInt<16>}}, out : {flip ready : UInt<1>, valid : UInt<1>, bits : {c : UInt<16>}}}

    io.out.bits.c <= UInt<1>("h00")
    io.out.valid <= UInt<1>("h00")
    io.in.ready <= UInt<1>("h00")
    reg ready : UInt<1>, clock, reset, ready
    reg busy : UInt<1>, clock, reset, busy
    reg a_reg : UInt<16>, clock, reset, a_reg
    reg b_reg : UInt<16>, clock, reset, b_reg
    io.in.ready <= ready
    when io.in.valid :
      a_reg <= io.in.bits.a
      b_reg <= io.in.bits.b
      io.in.ready <= UInt<1>("h00")
      ready <= UInt<1>("h00")
      busy <= UInt<1>("h01")
      skip
    node T_45 = and(busy, io.out.ready)
    when T_45 :
      node T_46 = addw(a_reg, b_reg)
      io.out.bits.c <= T_46
      io.out.valid <= UInt<1>("h01")
      io.in.ready <= UInt<1>("h01")
      busy <= UInt<1>("h00")
      skip

  module DecoupledAdderTests : 
    input clock : Clock
    input reset : UInt<1>
    output io : {}

    inst device_under_test of NewDecoupledAdder
    device_under_test.io.out.ready <= UInt<1>("h00")
    device_under_test.io.in.bits.b <= UInt<1>("h00")
    device_under_test.io.in.bits.a <= UInt<1>("h00")
    device_under_test.io.in.valid <= UInt<1>("h00")
    device_under_test.clock <= clock
    device_under_test.reset <= reset
    reg T_10 : UInt<33>, clock, reset, UInt<33>("h00")
    node T_12 = addw(T_10, UInt<1>("h01"))
    T_10 <= T_12
    node T_14 = eq(reset, UInt<1>("h00"))
    when T_14 :
      printf(clock,UInt(1), "ticker %d", T_10)
      skip
    node T_16 = gt(T_10, UInt<7>("h064"))
    when T_16 :
      node T_18 = eq(reset, UInt<1>("h00"))
      when T_18 :
        stop(clock,UInt(1), 0)
        skip
      skip
    node T_20 = eq(reset, UInt<1>("h00"))
    when T_20 :
      printf(clock,UInt(1), "device out ready %d, valid %d", device_under_test.io.out.ready, device_under_test.io.out.valid)
      skip
    node T_22 = eq(reset, UInt<1>("h00"))
    when T_22 :
      printf(clock,UInt(1), "device in ready %d, valid %d", device_under_test.io.in.ready, device_under_test.io.in.valid)
      skip
    reg T_24 : UInt<1>, clock, reset, UInt<1>("h00")
    reg T_26 : UInt<1>, clock, reset, UInt<1>("h00")
    node T_27 = and(T_24, T_26)
    when T_27 :
      node T_29 = eq(reset, UInt<1>("h00"))
      when T_29 :
        stop(clock,UInt(1), 0)
        skip
      skip
    reg T_31 : UInt<1>, clock, reset, UInt<1>("h00")
    reg T_33 : UInt<1>, clock, reset, UInt<1>("h00")
    wire T_43 : {flip ready : UInt<1>, valid : UInt<1>, bits : {a : UInt<16>, b : UInt<16>}}[1]
    T_43[0].bits.b <= UInt(0)
    T_43[0].bits.a <= UInt(1)
    T_43[0].valid <= UInt(1)
    device_under_test.io.in <- T_43[0]
    wire T_64 : UInt<16>[1]
    T_64[0] <= UInt<16>("h04")
    node T_68 = eq(T_24, UInt<1>("h00"))
    node T_78 = and(T_68, T_43[T_31].ready)
    when T_78 :
      node T_80 = eq(reset, UInt<1>("h00"))
      when T_80 :
        printf(clock,UInt(1), "input_event_counter %d", T_31)
        skip
      device_under_test.io.in.bits.a <= T_64[T_31]
      skip
    wire T_84 : UInt<16>[1]
    T_84[0] <= UInt<16>("h07")
    node T_88 = eq(T_24, UInt<1>("h00"))
    node T_98 = and(T_88, T_43[T_31].ready)
    when T_98 :
      node T_100 = eq(reset, UInt<1>("h00"))
      when T_100 :
        printf(clock,UInt(1), "input_event_counter %d", T_31)
        skip
      device_under_test.io.in.bits.b <= T_84[T_31]
      skip
    node T_103 = eq(T_24, UInt<1>("h00"))
    node T_113 = and(T_103, T_43[T_31].ready)
    when T_113 :
      T_43[T_31].valid <= UInt<1>("h01")
      node T_125 = addw(T_31, UInt<1>("h01"))
      T_31 <= T_125
      node T_127 = geq(T_31, UInt<1>("h00"))
      T_24 <= T_127
      skip
    node T_129 = eq(T_26, UInt<1>("h00"))
    when T_129 :
      node T_131 = addw(T_33, UInt<1>("h01"))
      T_33 <= T_131
      node T_133 = geq(T_33, UInt<1>("h00"))
      T_26 <= T_133
      skip
    wire T_136 : UInt<1>[1]
    T_136[0] <= UInt<1>("h01")
    wire T_141 : UInt<16>[1]
    T_141[0] <= UInt<16>("h03")
    node T_145 = eq(T_26, UInt<1>("h00"))
    node T_147 = and(T_145, T_136[T_33])
    node T_148 = and(T_147, device_under_test.io.out.valid)
    when T_148 :
      node T_150 = eq(reset, UInt<1>("h00"))
      when T_150 :
        printf(clock,UInt(1), "output_event_counter %d", T_33)
        skip
      node T_152 = neq(device_under_test.io.out.bits.c, T_141[T_33])
      when T_152 :
        node T_155 = eq(reset, UInt<1>("h00"))
        when T_155 :
          printf(clock,UInt(1), "Error: event %d out.bits.c was %x should be %x", T_33, device_under_test.io.out.bits.c, T_141[T_33])
          skip
        skip
      skip
