
circuit MemTester :
  module ReadWrite :
    input clk : Clock
    input reset : UInt<1>

    reg n : UInt<32>, clk with :
      reset => (reset, UInt(0))

    reg wmode : UInt<1>, clk with : 
      reset => (reset, UInt(1))
    wmode <= not(wmode)

    reg addr : UInt<5>, clk with :
      reset => (reset, UInt(0))

    when eq(wmode, UInt(0)) :
      n <= add(n, UInt(1))
      addr <= add(addr, UInt(1))

    mem m :
      data-type => UInt<32>
      depth => 32
      read-latency => 0
      write-latency => 1
      readwriter => rw
      read-under-write => undefined
    m.rw.clk <= clk
    m.rw.addr <= addr
    m.rw.wmode <= wmode
    m.rw.wdata <= n
    m.rw.wmask <= UInt(1)
    m.rw.en <= UInt(1)

    when not(reset) :
      when eq(wmode, UInt(0)) :
        when neq(m.rw.rdata, n) :
          printf(clk, UInt(1), "Assertion failed! m.rw.rdata has the wrong value!\n")
          stop(clk, UInt(1), 1)

  module MemTester :
    input clk : Clock
    input reset : UInt<1>

    reg count : UInt<32>, clk with :
      reset => (reset, UInt(100))
    count <= tail(sub(count, UInt(1)), 1)

    inst rwMod of ReadWrite
    rwMod.clk <= clk
    rwMod.reset <= reset

    when eq(count, UInt(0)) :
      stop(clk, UInt(1), 0)

