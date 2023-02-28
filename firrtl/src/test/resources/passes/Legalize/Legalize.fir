; SPDX-License-Identifier: Apache-2.0
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

    ; Check neg of literals
    node negUInt0 = neg(UInt(123))
    when neq(negUInt0, SInt(-123)) :
      printf(clock, UInt(1), "Assertion failed!\n negUInt0 != -123\n")
      stop(clock, UInt(1), 1)
    node negUInt1 = neg(UInt<8>(0))
    when neq(negUInt1, SInt<8>(0)) :
      printf(clock, UInt(1), "Assertion failed!\n negUInt1 != 0\n")
      stop(clock, UInt(1), 1)
    node negSInt0 = neg(SInt(123))
    when neq(negSInt0, SInt(-123)) :
      printf(clock, UInt(1), "Assertion failed!\n negSInt0 != -123\n")
      stop(clock, UInt(1), 1)
    node negSInt1 = neg(SInt(-123))
    when neq(negSInt1, SInt(123)) :
      printf(clock, UInt(1), "Assertion failed!\n negSInt1 != 123\n")
      stop(clock, UInt(1), 1)
    node negSInt2 = neg(SInt(0))
    when neq(negSInt2, SInt(0)) :
      printf(clock, UInt(1), "Assertion failed!\n negSInt2 != 0\n")
      stop(clock, UInt(1), 1)
