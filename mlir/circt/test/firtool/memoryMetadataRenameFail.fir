; RUN: firtool %s --format=fir  --annotation-file %s.anno.json --verilog |  FileCheck %s
; XFAIL: true

circuit test:
  module memoryTest1:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>

    mem memory:
      data-type => UInt<8>
      depth => 16
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    ; All of these are unified together
    memory.r.clk <= clock
    memory.r.en <= rEn
    memory.r.addr <= rAddr
    rData <= memory.r.data

    memory.w.clk <= clock
    memory.w.en <= rEn
    memory.w.addr <= rAddr
    ; These two are split
    memory.w.mask <= wMask
    memory.w.data <= wData

  module memoryTest2:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>

    mem memory:
      data-type => UInt<8>
      depth => 16
      reader => r
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined

    ; All of these are unified together
    memory.r.clk <= clock
    memory.r.en <= rEn
    memory.r.addr <= rAddr
    rData <= memory.r.data

    memory.w.clk <= clock
    memory.w.en <= rEn
    memory.w.addr <= rAddr
    ; These two are split
    memory.w.mask <= wMask
    memory.w.data <= wData


  module test: 
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>


    inst m of memoryTest1
    m.clock <= clock
    m.rAddr <= rAddr
    m.rEn <= rEn
    rData <= m.rData
    m.wMask <= wMask
    m.wData <= wData

    inst signed of memoryP
    signed.clock <= clock
    signed.rAddr <= rAddr
    signed.rEn <= rEn
    rData <= signed.rData
    signed.wMask <= wMask
    signed.wData <= wData


  module memoryP:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>
    input wMask: UInt<1>
    input wData: UInt<8>


    inst m of memoryTest2
    m.clock <= clock
    m.rAddr <= rAddr
    m.rEn <= rEn
    rData <= m.rData
    m.wMask <= wMask
    m.wData <= wData

; CHECK-LABEL: module test
; CHECK: memoryP [[signed:.+]] (

; CHECK-LABEL: module memoryP
; CHECK: memoryTest2 [[m:.+]] (

; CHECK:      FILE "metadata/tb_seq_mems.json"
; CHECK:      [{"module_name":"memory","depth":16,"width":8,"masked":"true",
; CHECK-SAME: "read":"true","write":"true","readwrite":"false",
; CHECK-SAME: "mask_granularity":8,"extra_ports":[],
; CHECK-SAME: "hierarchy":["test.[[m]]"]}]

; CHECK:       FILE "metadata/seq_mems.json"
; CHECK:       [{"module_name":"memory","depth":16,"width":8,"masked":"true",
; CHECK-SAME:  "read":"true","write":"true","readwrite":"false",
; CHECK-SAME:  "mask_granularity":8,"extra_ports":[],
; CHECK-SAME:  "hierarchy":["test.[[signed]].[[m]]"],
; CHECK-SAME:  "verification_only_data":{
; CHECK-SAME:  "test.[[signed]].[[m]]":{"baseAddress":1073741824,
; CHECK-SAME:  "dataBits":8,"eccBits":0,"eccIndices":[],"eccScheme":"none"}}}]

