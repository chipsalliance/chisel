; See LICENSE for license details.
circuit ExpandWhens :
  module ExpandWhens :
    input clock : Clock
    input reset : UInt<1>

    node done = UInt(6)

    reg x : UInt<16>, clock with :
      reset => (reset, UInt(0))
    x <= x

    x <= UInt(2)

    ; Tricky assignment only in else
    when geq(x, UInt(1)) :
    else :
      x <= UInt(1)

    when geq(x, UInt(2)) :
      when lt(x, UInt(3)) :
        x <= UInt(3)
      else :
        x <= UInt(4)

    when reset :
    else :
      when geq(x, UInt(4)) :
        x <= UInt(6)
        when lt(x, UInt(5)) :
          x <= UInt(5)

    reg count : UInt<16>, clock with :
      reset => (reset, UInt(0))
    when neq(count, done) :
      count <= add(count, UInt(1))

    printf(clock, UInt(1), "count = %d, x = %d\n", count, x)

    when not(reset) :
      when neq(x, count) : 
        printf(clock, UInt(1), "Assertion failed! x =/= count!\n")
        stop(clock, UInt(1), 1)
      when eq(x, done) :
        stop(clock, UInt(1), 0)

     
