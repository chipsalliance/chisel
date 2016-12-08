; See LICENSE for license details.
circuit ZeroPortMem :
  module ZeroPortMem :
    input clock : Clock
    input reset : UInt<1>

    mem mymem :
      data-type => UInt<32>
      depth => 128
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    wire foo : UInt<32>
    foo <= UInt<32>("hdeadbeef")

    when not(reset) :
      when eq(foo, UInt<32>("hdeadbeef")) :
        stop(clock, UInt(1), 0) ; Success !
      else :
        printf(clock, UInt(1), "Assertion failed!\n")
        stop(clock, UInt(1), 1) ; Failure!

