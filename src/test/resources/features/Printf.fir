; See LICENSE for license details.
circuit Printf : 
  module Printf : 
    input clock : Clock
    input reset : UInt<1>

    reg count : UInt<10>, clock with :
      reset => (reset, UInt<6>(0))
    reg const : UInt<32> clock with :
      reset => (reset, UInt(123456))

    node notReset = not(reset)
    count <= add(count, UInt(1))
    printf(clock, notReset, "\tcount = %d 0x%x b%b\\\'%d%%\'\n", count, count, count, const)

    when eq(count, UInt(255)) :
      stop(clock, UInt(1), 0)
