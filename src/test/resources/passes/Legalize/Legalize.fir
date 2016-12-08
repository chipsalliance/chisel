; See LICENSE for license details.
circuit Legalize :
  module Legalize :
    input clock : Clock
    input reset : UInt<1>

    ; Count till done
    node done = UInt(6)
    reg count : UInt<16>, clock with :
      reset => (reset, UInt(0))
    when neq(count, done) :
      count <= add(count, UInt(1))
    when not(reset) :
      when eq(count, done) :
        stop(clock, UInt(1), 0)

    ; Begin Test
    ; Check assignment to smaller width
    node x = UInt<32>("hdeadbeef")
    wire y : UInt<16>
    y <- x
    when neq(y, UInt("hbeef")) :
      printf(clock, UInt(1), "Assertion failed!\n y != beef\n")
      stop(clock, UInt(1), 1)

    ; Check bit select of literal
    node b = bits(UInt("hd0"), 7, 5)
    node b2 = bits(UInt("h9"), 3, 3)
    when neq(b, UInt(6)) :
      printf(clock, UInt(1), "Assertion failed!\n b != 6\n")
      stop(clock, UInt(1), 1)
    when neq(b2, UInt(1)) :
      printf(clock, UInt(1), "Assertion failed!\n b2 != 1\n")
      stop(clock, UInt(1), 1)

    ; Check padding of literal
    node bar = pad(SInt(-1), 16)
    node bar_15 = bits(bar, 15, 15)
    when neq(bar_15, UInt(1)) :
      printf(clock, UInt(1), "Assertion failed!\n bar_15 != 0\n")
      stop(clock, UInt(1), 1)
