circuit Printf : 
  module Printf : 
    input clk : Clock
    input reset : UInt<1>

    reg count : UInt<10>, clk with :
      reset => (reset, UInt<6>(0))
    reg const : UInt<32> clk with :
      reset => (reset, UInt(123456))

    node notReset = not(reset)
    count <= add(count, UInt(1))
    printf(clk, notReset, "\tcount = %d 0x%x b%b\\\'%d%%\'\n", count, count, count, const)

    when eq(count, UInt(255)) :
      stop(clk, UInt(1), 0)
