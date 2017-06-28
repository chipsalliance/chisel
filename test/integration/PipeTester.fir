
circuit PipeTester :
  ; This module should simply delay a signal by 2 cycles
  ; Internal registers reset to 0
  module Pipe :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<4>
    output out : UInt<4>

    ;reg r : UInt<4>, clock with : (reset => (reset, UInt(0)))
    ;r <= in
    ; This is equivalent to the above

    reg r : UInt<4>, clock
    r <= mux(reset, UInt(0), in)

    reg s : UInt<4>, clock with : (reset => (reset, UInt(0)))
    s <= r
    out <= s

  module PipeTester :
    input clock : Clock
    input reset : UInt<1>

    inst pipe of Pipe
    pipe.clock <= clock
    pipe.reset <= reset
    pipe.in <= UInt(3)

    reg cycle : UInt<4>, clock with : (reset => (reset, UInt<4>(0)))
    cycle <= tail(add(cycle, UInt(1)), 1)

    wire fail : UInt<1>
    fail <= UInt(0)

    when fail :
      printf(clock, not(reset), "Assertion failed!\n")
      stop(clock, not(reset), 1)

    when not(reset) :
      when lt(cycle, UInt(2)) :
        when neq(pipe.out, UInt(0)) :
          fail <= UInt(1)
      when eq(cycle, UInt(2)) :
        when neq(pipe.out, UInt(3)) :
          fail <= UInt(1)
      when eq(cycle, UInt(3)) :
        printf(clock, UInt(1), "Success!\n")
        stop(clock, UInt(1), 0)

