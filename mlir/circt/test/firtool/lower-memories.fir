; RUN: firtool %s --format=fir | FileCheck %s  --check-prefixes COMMON,NOLOWER
; RUN: firtool %s --format=fir -lower-memories | FileCheck %s --check-prefixes COMMON,LOWER

circuit ByteEnableMemory:
  module ByteEnableMemory :
    input clock : Clock
    input reset : UInt<1>
    output io : { flip readAddr : UInt<16>, dataOut : UInt<8>[2], flip readEnable : UInt<1>, flip dataIn : UInt<8>[2], flip writeAddr : UInt<16>, flip writeMask : UInt<1>[2]}

    smem mem : UInt<8>[2] [2]
    node _T = bits(io.writeAddr, 0, 0)
    write mport MPORT = mem[_T], clock
    when io.writeMask[0] :
      MPORT[0] <= io.dataIn[0]
    when io.writeMask[1] :
      MPORT[1] <= io.dataIn[1]
    wire _WIRE : UInt
    _WIRE is invalid
    when io.readEnable :
      _WIRE <= io.readAddr
      node _T_1 = or(_WIRE, UInt<1>("h0"))
      node _T_2 = bits(_T_1, 0, 0)
      read mport MPORT_1 = mem[_T_2], clock
    io.dataOut <= MPORT_1

; COMMON:        module [[memory:[a-zA-Z0-9_]+]]
; COMMON:        module ByteEnableMemory
; LOWER-COUNT-2:   [[memory]] {{[a-zA-Z0-9_]+}}
; NOLOWER:         [[memory]] {{[a-zA-Z0-9_]+}}
; NOLOWER-NOT:     [[memory]] {{[a-zA-Z0-9_]+}}
