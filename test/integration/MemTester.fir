
circuit MemTester :
  module ReadWrite :
    input clock : Clock
    input reset : UInt<1>

    reg n : UInt<32>, clock with :
      reset => (reset, UInt(0))

    reg wmode : UInt<1>, clock with :
      reset => (reset, UInt(1))
    wmode <= not(wmode)

    reg addr : UInt<5>, clock with :
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
    m.rw.clk <= clock
    m.rw.addr <= addr
    m.rw.wmode <= wmode
    m.rw.wdata <= n
    m.rw.wmask <= UInt(1)
    m.rw.en <= UInt(1)

    when not(reset) :
      when eq(wmode, UInt(0)) :
        when neq(m.rw.rdata, n) :
          printf(clock, UInt(1), "Assertion failed! m.rw.rdata has the wrong value!\n")
          stop(clock, UInt(1), 1)

  module MemTester :
    input clock : Clock
    input reset : UInt<1>

    reg count : UInt<32>, clock with :
      reset => (reset, UInt(100))
    count <= tail(sub(count, UInt(1)), 1)

    inst rwMod of ReadWrite
    rwMod.clock <= clock
    rwMod.reset <= reset

    when eq(count, UInt(0)) :
      stop(clock, UInt(1), 0)

