// RUN: circt-translate --export-firrtl --verify-diagnostics %s -o %t
// RUN: cat %t | FileCheck %s --strict-whitespace
// RUN: circt-translate --import-firrtl %t --mlir-print-debuginfo | circt-translate --export-firrtl | diff - %t

// Check emission at various widths, ensuring still parses and round-trips back to same FIRRTL as default width (inc debug info).
// RUN: circt-translate --export-firrtl %s --target-line-length=10 | circt-translate --import-firrtl --mlir-print-debuginfo | circt-translate --export-firrtl | diff - %t
// RUN: circt-translate --export-firrtl %s --target-line-length=1000 | circt-translate --import-firrtl --mlir-print-debuginfo | circt-translate --export-firrtl | diff - %t

// Sanity-check line length control:
// Check if printing with very long line length, no line ends with a comma.
// RUN: circt-translate --export-firrtl %s --target-line-length=1000 | FileCheck %s --implicit-check-not "{{,$}}" --check-prefix PRETTY
// Check if printing with very short line length, removing info locators (@[...]), no line is longer than 5x line length.
// RUN: circt-translate --export-firrtl %s --target-line-length=10 | sed -e 's/ @\[.*\]//' | FileCheck %s --implicit-check-not "{{^(.{50})}}" --check-prefix PRETTY

// CHECK-LABEL: FIRRTL version 3.1.0
// CHECK-LABEL: circuit Foo :
// PRETTY-LABEL: circuit Foo :
firrtl.circuit "Foo" {
  // CHECK-LABEL: module Foo :
  firrtl.module @Foo() {}

  // CHECK-LABEL: module PortsAndTypes :
  firrtl.module @PortsAndTypes(
    // CHECK-NEXT: input a00 : Clock
    // CHECK-NEXT: input a01 : Reset
    // CHECK-NEXT: input a02 : AsyncReset
    // CHECK-NEXT: input a03 : UInt
    // CHECK-NEXT: input a04 : SInt
    // CHECK-NEXT: input a05 : Analog
    // CHECK-NEXT: input a06 : UInt<42>
    // CHECK-NEXT: input a07 : SInt<42>
    // CHECK-NEXT: input a08 : Analog<42>
    // CHECK-NEXT: input a09 : { a : UInt, flip b : UInt }
    // CHECK-NEXT: input a10 : UInt[42]
    // CHECK-NEXT: output b0 : UInt
    // CHECK-NEXT: output b1 : Probe<UInt<1>>
    // CHECK-NEXT: output b2 : RWProbe<UInt<1>>
    // CHECK-NEXT: input string : String
    // CHECK-NEXT: input integer : Integer
    // CHECK-NEXT: input path : Path
    in %a00: !firrtl.clock,
    in %a01: !firrtl.reset,
    in %a02: !firrtl.asyncreset,
    in %a03: !firrtl.uint,
    in %a04: !firrtl.sint,
    in %a05: !firrtl.analog,
    in %a06: !firrtl.uint<42>,
    in %a07: !firrtl.sint<42>,
    in %a08: !firrtl.analog<42>,
    in %a09: !firrtl.bundle<a: uint, b flip: uint>,
    in %a10: !firrtl.vector<uint, 42>,
    out %b0: !firrtl.uint,
    out %b1: !firrtl.probe<uint<1>>,
    out %b2: !firrtl.rwprobe<uint<1>>,
    in %string: !firrtl.string,
    in %integer: !firrtl.integer,
    in %path : !firrtl.path
  ) {}

  // CHECK-LABEL: module Simple :
  // CHECK:         input someIn : UInt<1>
  // CHECK:         output someOut : UInt<1>
  firrtl.module @Simple(in %someIn: !firrtl.uint<1>, out %someOut: !firrtl.uint<1>) {
    firrtl.skip
  }

  // CHECK-LABEL: module Statements :
  firrtl.module @Statements(in %ui1: !firrtl.uint<1>, in %someAddr: !firrtl.uint<8>, in %someClock: !firrtl.clock, in %someReset: !firrtl.reset, out %someOut: !firrtl.uint<1>, out %ref: !firrtl.probe<uint<1>>) {
    // CHECK: when ui1 :
    // CHECK:   skip
    firrtl.when %ui1 : !firrtl.uint<1> {
      firrtl.skip
    }
    // CHECK: when ui1 :
    // CHECK:   skip
    // CHECK: else :
    // CHECK:   skip
    firrtl.when %ui1 : !firrtl.uint<1> {
      firrtl.skip
    } else {
      firrtl.skip
    }
    // CHECK: when ui1 :
    // CHECK:   skip
    // CHECK: else when ui1 :
    // CHECK:   skip
    firrtl.when %ui1 : !firrtl.uint<1> {
      firrtl.skip
    } else {
      firrtl.when %ui1 : !firrtl.uint<1> {
        firrtl.skip
      }
    }
    // CHECK: wire someWire : UInt<1>
    %someWire = firrtl.wire : !firrtl.uint<1>
    // CHECK: reg someReg : UInt<1>, someClock
    %someReg = firrtl.reg %someClock : !firrtl.clock, !firrtl.uint<1>
    // CHECK: regreset someReg2 : UInt<1>, someClock, someReset, ui1
    %someReg2 = firrtl.regreset %someClock, %someReset, %ui1 : !firrtl.clock, !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: node someNode = ui1
    %someNode = firrtl.node %ui1 : !firrtl.uint<1>
    // CHECK: stop(someClock, ui1, 42) : foo
    firrtl.stop %someClock, %ui1, 42 {name = "foo"} : !firrtl.clock, !firrtl.uint<1>
    // CHECK: skip
    firrtl.skip
    // CHECK: printf(someClock, ui1, "some\n magic\"stuff\"", ui1, someReset) : foo
    firrtl.printf %someClock, %ui1, "some\n magic\"stuff\"" {name = "foo"} (%ui1, %someReset) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.reset
    // CHECK: assert(someClock, ui1, ui1, "msg") : foo
    // CHECK: assume(someClock, ui1, ui1, "msg") : foo
    // CHECK: cover(someClock, ui1, ui1, "msg") : foo
    firrtl.assert %someClock, %ui1, %ui1, "msg" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {name = "foo"}
    firrtl.assume %someClock, %ui1, %ui1, "msg" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {name = "foo"}
    firrtl.cover %someClock, %ui1, %ui1, "msg" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {name = "foo"}
    // CHECK: connect someOut, ui1
    firrtl.connect %someOut, %ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: inst someInst of Simple
    // CHECK: connect someInst.someIn, ui1
    // CHECK: connect someOut, someInst.someOut
    %someInst_someIn, %someInst_someOut = firrtl.instance someInst @Simple(in someIn: !firrtl.uint<1>, out someOut: !firrtl.uint<1>)
    firrtl.connect %someInst_someIn, %ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %someOut, %someInst_someOut : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NOT: _invalid
    // CHECK: invalidate someOut
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.connect %someOut, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NOT: _invalid
    // CHECK: invalidate someOut
    %invalid_ui2 = firrtl.invalidvalue : !firrtl.uint<1>
    firrtl.strictconnect %someOut, %invalid_ui2 : !firrtl.uint<1>

    // CHECK: connect unknownReset, knownReset
    %knownReset = firrtl.wire : !firrtl.asyncreset
    %unknownReset = firrtl.wire : !firrtl.reset
    %resetCast = firrtl.resetCast %knownReset :
      (!firrtl.asyncreset) -> !firrtl.reset
    firrtl.strictconnect %unknownReset, %resetCast : !firrtl.reset

    // CHECK: attach(an0, an1)
    %an0 = firrtl.wire : !firrtl.analog<1>
    %an1 = firrtl.wire : !firrtl.analog<1>
    firrtl.attach %an0, %an1 : !firrtl.analog<1>, !firrtl.analog<1>

    // CHECK: node k0 = UInt<19>(42)
    // CHECK: node k1 = SInt<19>(42)
    // CHECK: node k2 = UInt(42)
    // CHECK: node k3 = SInt(42)
    %0 = firrtl.constant 42 : !firrtl.uint<19>
    %1 = firrtl.constant 42 : !firrtl.sint<19>
    %2 = firrtl.constant 42 : !firrtl.uint
    %3 = firrtl.constant 42 : !firrtl.sint
    %k0 = firrtl.node %0 : !firrtl.uint<19>
    %k1 = firrtl.node %1 : !firrtl.sint<19>
    %k2 = firrtl.node %2 : !firrtl.uint
    %k3 = firrtl.node %3 : !firrtl.sint

    // CHECK: node k4 = asClock(UInt<1>(0))
    // CHECK: node k5 = asAsyncReset(UInt<1>(0))
    // CHECK: node k6 = UInt<1>(0)
    %4 = firrtl.specialconstant 0 : !firrtl.clock
    %5 = firrtl.specialconstant 0 : !firrtl.asyncreset
    %6 = firrtl.specialconstant 0 : !firrtl.reset
    %k4 = firrtl.node %4 : !firrtl.clock
    %k5 = firrtl.node %5 : !firrtl.asyncreset
    %k6 = firrtl.node %6 : !firrtl.reset

    // CHECK: wire bundle : { a : UInt, flip b : UInt }
    // CHECK: wire vector : UInt[42]
    // CHECK: node subfield = bundle.a
    // CHECK: node subindex = vector[19]
    // CHECK: node subaccess = vector[ui1]
    %bundle = firrtl.wire : !firrtl.bundle<a: uint, b flip: uint>
    %vector = firrtl.wire : !firrtl.vector<uint, 42>
    %subfield_tmp = firrtl.subfield %bundle[a] : !firrtl.bundle<a: uint, b flip: uint>
    %subindex_tmp = firrtl.subindex %vector[19] : !firrtl.vector<uint, 42>
    %subaccess_tmp = firrtl.subaccess %vector[%ui1] : !firrtl.vector<uint, 42>, !firrtl.uint<1>
    %subfield = firrtl.node %subfield_tmp : !firrtl.uint
    %subindex = firrtl.node %subindex_tmp : !firrtl.uint
    %subaccess = firrtl.node %subaccess_tmp : !firrtl.uint

    %x = firrtl.node %2 : !firrtl.uint
    %y = firrtl.node %2 : !firrtl.uint

    // CHECK: node addPrimOp = add(x, y)
    // CHECK: node subPrimOp = sub(x, y)
    // CHECK: node mulPrimOp = mul(x, y)
    // CHECK: node divPrimOp = div(x, y)
    // CHECK: node remPrimOp = rem(x, y)
    // CHECK: node andPrimOp = and(x, y)
    // CHECK: node orPrimOp = or(x, y)
    // CHECK: node xorPrimOp = xor(x, y)
    // CHECK: node leqPrimOp = leq(x, y)
    // CHECK: node ltPrimOp = lt(x, y)
    // CHECK: node geqPrimOp = geq(x, y)
    // CHECK: node gtPrimOp = gt(x, y)
    // CHECK: node eqPrimOp = eq(x, y)
    // CHECK: node neqPrimOp = neq(x, y)
    // CHECK: node catPrimOp = cat(x, y)
    // CHECK: node dShlPrimOp = dshl(x, y)
    // CHECK: node dShlwPrimOp = dshlw(x, y)
    // CHECK: node dShrPrimOp = dshr(x, y)
    %addPrimOp_tmp = firrtl.add %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %subPrimOp_tmp = firrtl.sub %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %mulPrimOp_tmp = firrtl.mul %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %divPrimOp_tmp = firrtl.div %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %remPrimOp_tmp = firrtl.rem %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %andPrimOp_tmp = firrtl.and %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %orPrimOp_tmp = firrtl.or %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %xorPrimOp_tmp = firrtl.xor %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %leqPrimOp_tmp = firrtl.leq %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %ltPrimOp_tmp = firrtl.lt %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %geqPrimOp_tmp = firrtl.geq %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %gtPrimOp_tmp = firrtl.gt %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %eqPrimOp_tmp = firrtl.eq %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %neqPrimOp_tmp = firrtl.neq %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
    %catPrimOp_tmp = firrtl.cat %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %dShlPrimOp_tmp = firrtl.dshl %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %dShlwPrimOp_tmp = firrtl.dshlw %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %dShrPrimOp_tmp = firrtl.dshr %x, %y : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %addPrimOp = firrtl.node %addPrimOp_tmp : !firrtl.uint
    %subPrimOp = firrtl.node %subPrimOp_tmp : !firrtl.uint
    %mulPrimOp = firrtl.node %mulPrimOp_tmp : !firrtl.uint
    %divPrimOp = firrtl.node %divPrimOp_tmp : !firrtl.uint
    %remPrimOp = firrtl.node %remPrimOp_tmp : !firrtl.uint
    %andPrimOp = firrtl.node %andPrimOp_tmp : !firrtl.uint
    %orPrimOp = firrtl.node %orPrimOp_tmp : !firrtl.uint
    %xorPrimOp = firrtl.node %xorPrimOp_tmp : !firrtl.uint
    %leqPrimOp = firrtl.node %leqPrimOp_tmp : !firrtl.uint<1>
    %ltPrimOp = firrtl.node %ltPrimOp_tmp : !firrtl.uint<1>
    %geqPrimOp = firrtl.node %geqPrimOp_tmp : !firrtl.uint<1>
    %gtPrimOp = firrtl.node %gtPrimOp_tmp : !firrtl.uint<1>
    %eqPrimOp = firrtl.node %eqPrimOp_tmp : !firrtl.uint<1>
    %neqPrimOp = firrtl.node %neqPrimOp_tmp : !firrtl.uint<1>
    %catPrimOp = firrtl.node %catPrimOp_tmp : !firrtl.uint
    %dShlPrimOp = firrtl.node %dShlPrimOp_tmp : !firrtl.uint
    %dShlwPrimOp = firrtl.node %dShlwPrimOp_tmp : !firrtl.uint
    %dShrPrimOp = firrtl.node %dShrPrimOp_tmp : !firrtl.uint

    // CHECK: node asSIntPrimOp = asSInt(x)
    // CHECK: node asUIntPrimOp = asUInt(x)
    // CHECK: node asAsyncResetPrimOp = asAsyncReset(x)
    // CHECK: node asClockPrimOp = asClock(x)
    // CHECK: node cvtPrimOp = cvt(x)
    // CHECK: node negPrimOp = neg(x)
    // CHECK: node notPrimOp = not(x)
    // CHECK: node andRPrimOp = andr(x)
    // CHECK: node orRPrimOp = orr(x)
    // CHECK: node xorRPrimOp = xorr(x)
    %asSIntPrimOp_tmp = firrtl.asSInt %x : (!firrtl.uint) -> !firrtl.sint
    %asUIntPrimOp_tmp = firrtl.asUInt %x : (!firrtl.uint) -> !firrtl.uint
    %asAsyncResetPrimOp_tmp = firrtl.asAsyncReset %x : (!firrtl.uint) -> !firrtl.asyncreset
    %asClockPrimOp_tmp = firrtl.asClock %x : (!firrtl.uint) -> !firrtl.clock
    %cvtPrimOp_tmp = firrtl.cvt %x : (!firrtl.uint) -> !firrtl.sint
    %negPrimOp_tmp = firrtl.neg %x : (!firrtl.uint) -> !firrtl.sint
    %notPrimOp_tmp = firrtl.not %x : (!firrtl.uint) -> !firrtl.uint
    %andRPrimOp_tmp = firrtl.andr %x : (!firrtl.uint) -> !firrtl.uint<1>
    %orRPrimOp_tmp = firrtl.orr %x : (!firrtl.uint) -> !firrtl.uint<1>
    %xorRPrimOp_tmp = firrtl.xorr %x : (!firrtl.uint) -> !firrtl.uint<1>
    %asSIntPrimOp = firrtl.node %asSIntPrimOp_tmp : !firrtl.sint
    %asUIntPrimOp = firrtl.node %asUIntPrimOp_tmp : !firrtl.uint
    %asAsyncResetPrimOp = firrtl.node %asAsyncResetPrimOp_tmp : !firrtl.asyncreset
    %asClockPrimOp = firrtl.node %asClockPrimOp_tmp : !firrtl.clock
    %cvtPrimOp = firrtl.node %cvtPrimOp_tmp : !firrtl.sint
    %negPrimOp = firrtl.node %negPrimOp_tmp : !firrtl.sint
    %notPrimOp = firrtl.node %notPrimOp_tmp : !firrtl.uint
    %andRPrimOp = firrtl.node %andRPrimOp_tmp : !firrtl.uint<1>
    %orRPrimOp = firrtl.node %orRPrimOp_tmp : !firrtl.uint<1>
    %xorRPrimOp = firrtl.node %xorRPrimOp_tmp : !firrtl.uint<1>

    // CHECK: node bitsPrimOp = bits(x, 4, 2)
    // CHECK: node headPrimOp = head(x, 4)
    // CHECK: node tailPrimOp = tail(x, 4)
    // CHECK: node padPrimOp = pad(x, 16)
    // CHECK: node muxPrimOp = mux(ui1, x, y)
    // CHECK: node shlPrimOp = shl(x, 4)
    // CHECK: node shrPrimOp = shr(x, 4)
    %bitsPrimOp_tmp = firrtl.bits %x 4 to 2 : (!firrtl.uint) -> !firrtl.uint<3>
    %headPrimOp_tmp = firrtl.head %x, 4 : (!firrtl.uint) -> !firrtl.uint<4>
    %tailPrimOp_tmp = firrtl.tail %x, 4 : (!firrtl.uint) -> !firrtl.uint
    %padPrimOp_tmp = firrtl.pad %x, 16 : (!firrtl.uint) -> !firrtl.uint
    %muxPrimOp_tmp = firrtl.mux(%ui1, %x, %y) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %shlPrimOp_tmp = firrtl.shl %x, 4 : (!firrtl.uint) -> !firrtl.uint
    %shrPrimOp_tmp = firrtl.shr %x, 4 : (!firrtl.uint) -> !firrtl.uint
    %bitsPrimOp = firrtl.node %bitsPrimOp_tmp : !firrtl.uint<3>
    %headPrimOp = firrtl.node %headPrimOp_tmp : !firrtl.uint<4>
    %tailPrimOp = firrtl.node %tailPrimOp_tmp : !firrtl.uint
    %padPrimOp = firrtl.node %padPrimOp_tmp : !firrtl.uint
    %muxPrimOp = firrtl.node %muxPrimOp_tmp : !firrtl.uint
    %shlPrimOp = firrtl.node %shlPrimOp_tmp : !firrtl.uint
    %shrPrimOp = firrtl.node %shrPrimOp_tmp : !firrtl.uint

    %MyMem_a, %MyMem_b, %MyMem_c = firrtl.mem Undefined {depth = 8, name = "MyMem", portNames = ["a", "b", "c"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<4>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<4>, mask: uint<1>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint<4>, wmode: uint<1>, wdata: uint<4>, wmask: uint<1>>
    %MyMem_a_clk = firrtl.subfield %MyMem_a[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<4>>
    %MyMem_b_clk = firrtl.subfield %MyMem_b[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<4>, mask: uint<1>>
    %MyMem_c_clk = firrtl.subfield %MyMem_c[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint<4>, wmode: uint<1>, wdata: uint<4>, wmask: uint<1>>
    firrtl.connect %MyMem_a_clk, %someClock : !firrtl.clock, !firrtl.clock
    firrtl.connect %MyMem_b_clk, %someClock : !firrtl.clock, !firrtl.clock
    firrtl.connect %MyMem_c_clk, %someClock : !firrtl.clock, !firrtl.clock
    // CHECK:       mem MyMem :
    // CHECK-NEXT:    data-type => UInt<4>
    // CHECK-NEXT:    depth => 8
    // CHECK-NEXT:    read-latency => 0
    // CHECK-NEXT:    write-latency => 1
    // CHECK-NEXT:    reader => a
    // CHECK-NEXT:    writer => b
    // CHECK-NEXT:    readwriter => c
    // CHECK-NEXT:    read-under-write => undefined
    // CHECK-NEXT:  connect MyMem.a.clk, someClock
    // CHECK-NEXT:  connect MyMem.b.clk, someClock
    // CHECK-NEXT:  connect MyMem.c.clk, someClock

    %combmem = chirrtl.combmem : !chirrtl.cmemory<uint<3>, 256>
    %port0_data, %port0_port = chirrtl.memoryport Infer %combmem {name = "port0"} : (!chirrtl.cmemory<uint<3>, 256>) -> (!firrtl.uint<3>, !chirrtl.cmemoryport)
    firrtl.when %ui1 : !firrtl.uint<1> {
      chirrtl.memoryport.access %port0_port[%someAddr], %someClock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
    }
    // CHECK:      cmem combmem : UInt<3>[256]
    // CHECK-NEXT: when ui1 :
    // CHECK-NEXT:   infer mport port0 = combmem[someAddr], someClock

    %seqmem = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<3>, 256>
    %port1_data, %port1_port = chirrtl.memoryport Infer %seqmem {name = "port1"} : (!chirrtl.cmemory<uint<3>, 256>) -> (!firrtl.uint<3>, !chirrtl.cmemoryport)
    firrtl.when %ui1 : !firrtl.uint<1> {
      chirrtl.memoryport.access %port1_port[%someAddr], %someClock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
    }
    // CHECK:      smem seqmem : UInt<3>[256] undefined
    // CHECK-NEXT: when ui1 :
    // CHECK-NEXT:   infer mport port1 = seqmem[someAddr], someClock

    firrtl.connect %port0_data, %port1_data : !firrtl.uint<3>, !firrtl.uint<3>
    // CHECK: connect port0, port1

    %invalid_clock = firrtl.invalidvalue : !firrtl.clock
    %dummyReg = firrtl.reg %invalid_clock : !firrtl.clock, !firrtl.uint<42>
    // CHECK: wire [[INV:_invalid.*]] : Clock
    // CHECK-NEXT: invalidate [[INV]]
    // CHECK-NEXT: reg dummyReg : UInt<42>, [[INV]]
  }

  // CHECK-LABEL: module RefSource
  firrtl.module @RefSource(out %a_ref: !firrtl.probe<uint<1>>,
                           out %a_rwref: !firrtl.rwprobe<uint<1>>) {
    %a, %_a_rwref = firrtl.wire forceable : !firrtl.uint<1>,
      !firrtl.rwprobe<uint<1>>
    // CHECK: define a_ref = probe(a)
    // CHECK: define a_rwref = rwprobe(a)
    %a_ref_send = firrtl.ref.send %a : !firrtl.uint<1>
    firrtl.ref.define %a_ref, %a_ref_send : !firrtl.probe<uint<1>>
    firrtl.ref.define %a_rwref, %_a_rwref : !firrtl.rwprobe<uint<1>>
  }

  // CHECK-LABEL: module RefSink
  firrtl.module @RefSink(
    in %clock: !firrtl.clock,
    in %enable: !firrtl.uint<1>
  ) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: node b = read(refSource.a_ref)
    %refSource_a_ref, %refSource_a_rwref =
      firrtl.instance refSource @RefSource(
        out a_ref: !firrtl.probe<uint<1>>,
        out a_rwref: !firrtl.rwprobe<uint<1>>
      )
    %a_ref_resolve =
      firrtl.ref.resolve %refSource_a_ref : !firrtl.probe<uint<1>>
    %b = firrtl.node %a_ref_resolve : !firrtl.uint<1>
    // CHECK-NEXT: force_initial(refSource.a_rwref, UInt<1>(0))
    firrtl.ref.force_initial %c1_ui1, %refSource_a_rwref, %c0_ui1 :
      !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: release_initial(refSource.a_rwref)
    firrtl.ref.release_initial %c1_ui1, %refSource_a_rwref :
      !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    // CHECK-NEXT: when enable :
    // CHECK-NEXT:   force_initial(refSource.a_rwref, UInt<1>(0))
    firrtl.when %enable : !firrtl.uint<1> {
      firrtl.ref.force_initial %c1_ui1, %refSource_a_rwref, %c0_ui1 :
        !firrtl.uint<1>, !firrtl.uint<1>
    }
    // CHECK-NEXT: when enable :
    // CHECK-NEXT:   release_initial(refSource.a_rwref)
    firrtl.when %enable : !firrtl.uint<1> {
      firrtl.ref.release_initial %c1_ui1, %refSource_a_rwref :
        !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    }
    // CHECK-NEXT: force(clock, enable, refSource.a_rwref, UInt<1>(1))
    firrtl.ref.force %clock, %enable, %refSource_a_rwref, %c1_ui1 :
      !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: release(clock, enable, refSource.a_rwref)
    firrtl.ref.release %clock, %enable, %refSource_a_rwref :
      !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
  }

  // CHECK-LABEL: module RefExport
  firrtl.module @RefExport(out %a_ref: !firrtl.probe<uint<1>>,
                           out %a_rwref: !firrtl.rwprobe<uint<1>>) {
    // CHECK: define a_ref = refSource.a_ref
    // CHECK: define a_rwref = refSource.a_rwref
    %refSource_a_ref, %refSource_a_rwref =
      firrtl.instance refSource @RefSource(
        out a_ref: !firrtl.probe<uint<1>>,
        out a_rwref: !firrtl.rwprobe<uint<1>>
      )
    firrtl.ref.define %a_ref, %refSource_a_ref : !firrtl.probe<uint<1>>
    firrtl.ref.define %a_rwref, %refSource_a_rwref : !firrtl.rwprobe<uint<1>>
  }

  // CHECK-LABEL: extmodule ExtOpenAgg
  // CHECK-NEXT:  output out : { a : { data : UInt<1> },
  // CHECK-NEXT:                 b : { x : UInt<2>, y : Probe<UInt<2>[3]> }[2] }
  firrtl.extmodule @ExtOpenAgg(
      out out: !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<vector<uint<2>, 3>>>, 2>>)

  // CHECK-LABEL: module OpenAggTest
  firrtl.module @OpenAggTest(
  // CHECK-NEXT: output out_b_0_y_2 : Probe<UInt<2>>
  // CHECK-EMPTY:
      out %out_b_0_y_2 : !firrtl.probe<uint<2>>) {

    // CHECK-NEXT: inst oa of ExtOpenAgg
    %oa_out = firrtl.instance oa @ExtOpenAgg(out out: !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<vector<uint<2>, 3>>>, 2>>)

    %a = firrtl.opensubfield %oa_out[a] : !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<vector<uint<2>, 3>>>, 2>>
    %data = firrtl.subfield %a[data] : !firrtl.bundle<data: uint<1>>
    // CHECK-NEXT:  node n_data = oa.out.a.data
    %n_data = firrtl.node %data : !firrtl.uint<1>
    %b = firrtl.opensubfield %oa_out[b] : !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<vector<uint<2>, 3>>>, 2>>
    %b_0 = firrtl.opensubindex %b[0] : !firrtl.openvector<openbundle<x: uint<2>, y: probe<vector<uint<2>, 3>>>, 2>
    %b_0_y = firrtl.opensubfield %b_0[y] : !firrtl.openbundle<x : uint<2>, y: probe<vector<uint<2>, 3>>>
    %b_0_y_2 = firrtl.ref.sub %b_0_y[2] : !firrtl.probe<vector<uint<2>, 3>>
    // openagg indexing + ref.sub
    // CHECK-NEXT: define out_b_0_y_2 = oa.out.b[0].y[2]
    firrtl.ref.define %out_b_0_y_2, %b_0_y_2 : !firrtl.probe<uint<2>>
  }

  firrtl.extmodule @MyParameterizedExtModule<DEFAULT: i64 = 0, DEPTH: f64 = 3.242000e+01, FORMAT: none = "xyz_timeout=%d\0A", WIDTH: i8 = 32>(in in: !firrtl.uint, out out: !firrtl.uint<8>) attributes {defname = "name_thing"}
  // CHECK-LABEL: extmodule MyParameterizedExtModule :
  // CHECK-NEXT:    input in : UInt
  // CHECK-NEXT:    output out : UInt<8>
  // CHECK-NEXT:    defname = name_thing
  // CHECK-NEXT:    parameter DEFAULT = 0
  // CHECK-NEXT:    parameter DEPTH = 32.42
  // CHECK-NEXT:    parameter FORMAT = "xyz_timeout=%d\n"
  // CHECK-NEXT:    parameter WIDTH = 32

  firrtl.intmodule private @MyIntModule<
    FORMAT: none = "xyz_timeout=%d\0A",
    DEFAULT: ui32 = 0,
    WIDTH: ui32 = 32,
    DEPTH: f64 = 3.242000e+01
  >(
    in in: !firrtl.uint,
    out out: !firrtl.uint<8>
  ) attributes {intrinsic = "testIntrinsic1"}
  // CHECK-LABEL: intmodule MyIntModule :
  // CHECK-NEXT:    input in : UInt
  // CHECK-NEXT:    output out : UInt<8>
  // CHECK-NEXT:    intrinsic = testIntrinsic1
  // CHECK-NEXT:    parameter FORMAT = "xyz_timeout=%d\n"
  // CHECK-NEXT:    parameter DEFAULT = 0
  // CHECK-NEXT:    parameter WIDTH = 32
  // CHECK-NEXT:    parameter DEPTH = 32.42

  // CHECK-LABEL: module ConstTypes :
  firrtl.module @ConstTypes(
    // CHECK-NEXT: input a00 : const Clock
    // CHECK-NEXT: input a01 : const Reset
    // CHECK-NEXT: input a02 : const AsyncReset
    // CHECK-NEXT: input a03 : const UInt
    // CHECK-NEXT: input a04 : const SInt
    // CHECK-NEXT: input a05 : const Analog
    // CHECK-NEXT: input a06 : const UInt<42>
    // CHECK-NEXT: input a07 : const SInt<42>
    // CHECK-NEXT: input a08 : const Analog<42>
    // CHECK-NEXT: input a09 : const { a : UInt, flip b : UInt }
    // CHECK-NEXT: input a10 : { a : const UInt, flip b : UInt }
    // CHECK-NEXT: input a11 : const UInt[42]
    // CHECK-NEXT: output b0 : const UInt<42>
    in %a00: !firrtl.const.clock,
    in %a01: !firrtl.const.reset,
    in %a02: !firrtl.const.asyncreset,
    in %a03: !firrtl.const.uint,
    in %a04: !firrtl.const.sint,
    in %a05: !firrtl.const.analog,
    in %a06: !firrtl.const.uint<42>,
    in %a07: !firrtl.const.sint<42>,
    in %a08: !firrtl.const.analog<42>,
    in %a09: !firrtl.const.bundle<a: uint, b flip: uint>,
    in %a10: !firrtl.bundle<a: const.uint, b flip: uint>,
    in %a11: !firrtl.const.vector<uint, 42>,
    out %b0: !firrtl.const.uint<42>
  ) {
    // Make sure literals strip the 'const' prefix
    // CHECK: connect b0, UInt<42>(1)
    %c = firrtl.constant 1 : !firrtl.const.uint<42>
    firrtl.strictconnect %b0, %c : !firrtl.const.uint<42>
  }

  // Test that literal identifiers work.
  // CHECK-LABEL: module `0Bar` :
  firrtl.module @"0Bar"(
    // CHECK-NEXT: input `0` : UInt<1>
    in %_0: !firrtl.uint<1>
  ) attributes {portNames = ["0"]} {}
  // CHECK-LABEL: module `0Foo` :
  firrtl.module @"0Foo"(
    // CHECK-NEXT: input `0` : Clock
    // CHECK-NEXT: input `1` : Reset
    // CHECK-NEXT: input `2` : AsyncReset
    // CHECK-NEXT: input `3` : UInt<1>
    // CHECK-NEXT: input `4` : SInt
    // CHECK-NEXT: input `5` : Analog
    // CHECK-NEXT: input `6` : UInt<42>
    // CHECK-NEXT: input `7` : SInt<42>
    // CHECK-NEXT: input `8` : Analog<42>
    // CHECK-NEXT: input `9` : { `0` : UInt, flip `1` : UInt }
    // CHECK-NEXT: input `10` : UInt[42]
    // CHECK-NEXT: output `11` : UInt
    // CHECK-NEXT: output `12` : Probe<UInt<1>>
    // CHECK-NEXT: output `13` : RWProbe<UInt<1>>
    in %_0: !firrtl.clock,
    in %_1: !firrtl.reset,
    in %_2: !firrtl.asyncreset,
    in %_3: !firrtl.uint<1>,
    in %_4: !firrtl.sint,
    in %_5: !firrtl.analog,
    in %_6: !firrtl.uint<42>,
    in %_7: !firrtl.sint<42>,
    in %_8: !firrtl.analog<42>,
    in %_9: !firrtl.bundle<"0": uint, "1" flip: uint>,
    in %_10: !firrtl.vector<uint, 42>,
    out %_11: !firrtl.uint,
    out %_12: !firrtl.probe<uint<1>>,
    out %_13: !firrtl.rwprobe<uint<1>>
  ) attributes {
    portNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]
  } {
    %0 = firrtl.subfield %_9["0"] : !firrtl.bundle<"0": uint, "1" flip: uint>
    %1 = firrtl.subfield %_9["1"] : !firrtl.bundle<"0": uint, "1" flip: uint>

    // CHECK:      wire `14` : UInt<1>
    // CHECK-NEXT: reg `15` : UInt<1>, `0`
    // CHECK-NEXT: regreset `16` : UInt<1>, `0`, `1`, `3`
    // CHECK-NEXT: node `17` = `3`
    %_14 = firrtl.wire interesting_name {name = "14"} : !firrtl.uint<1>
    %_15, %_15_ref = firrtl.reg %_0 forceable {name = "15"} :
      !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    %_16 = firrtl.regreset %_0, %_1, %_3 {name = "16"} :
      !firrtl.clock, !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>
    %_17 = firrtl.node %_3 {name = "17"} : !firrtl.uint<1>

    // CHECK:      connect `9`.`1`, `9`.`0`
    firrtl.connect %1, %0 : !firrtl.uint, !firrtl.uint

    // CHECK:      invalidate `11`
    %invalid_ui = firrtl.invalidvalue : !firrtl.uint
    firrtl.connect %_11, %invalid_ui : !firrtl.uint, !firrtl.uint

    // CHECK:      inst `0bar` of `0Bar`
    // CHECK-NEXT: connect `0bar`.`0`, `3`
    %_0bar_0 = firrtl.instance "0bar" @"0Bar"(in "0": !firrtl.uint<1>)
    firrtl.strictconnect %_0bar_0, %_3 : !firrtl.uint<1>

    // CHECK:      mem `18` :
    // CHECK-NEXT:   data-type => UInt<8>
    // CHECK-NEXT:   depth => 32
    // CHECK-NEXT:   read-latency => 1
    // CHECK-NEXT:   write-latency => 1
    // CHECK-NEXT:   reader => `0`
    // CHECK-NEXT:   writer => `1`
    // CHECK-NEXT:   readwriter => `2`
    // CHECK-NEXT:   read-under-write => undefined
    %_18_0, %_18_1, %_18_2 = firrtl.mem Undefined {
      depth = 32 : i64,
      name = "18",
      portNames = ["0", "1", "2"],
      readLatency = 1 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>,
        !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, rdata flip: uint<8>, wmode: uint<1>, wdata: uint<8>, wmask: uint<1>>

    // CHECK:      connect `18`.`0`.clk, `0`
    // CHECK-NEXT: connect `18`.`0`.en, `3`
    // CHECK-NEXT: connect `18`.`0`.addr, pad(`3`, 5)
    // CHECK-NEXT: connect `11`, `18`.`0`.data

    // CHECK-NEXT: connect `18`.`1`.clk, `0`
    // CHECK-NEXT: connect `18`.`1`.en, `3`
    // CHECK-NEXT: connect `18`.`1`.data, pad(`3`, 8)
    // CHECK-NEXT: connect `18`.`1`.addr, pad(`3`, 5)
    // CHECK-NEXT: connect `18`.`1`.mask, `3`
    %3 = firrtl.subfield %_18_1[mask] : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %4 = firrtl.subfield %_18_1[addr] : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %5 = firrtl.subfield %_18_1[data] : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %6 = firrtl.subfield %_18_1[en] : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %7 = firrtl.subfield %_18_1[clk] : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %8 = firrtl.subfield %_18_0[data] : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>
    %9 = firrtl.subfield %_18_0[addr] : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>
    %10 = firrtl.subfield %_18_0[en] : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>
    %11 = firrtl.subfield %_18_0[clk] : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>

    firrtl.strictconnect %11, %_0 : !firrtl.clock
    firrtl.strictconnect %10, %_3 : !firrtl.uint<1>
    %12 = firrtl.pad %_3, 5 : (!firrtl.uint<1>) -> !firrtl.uint<5>
    firrtl.strictconnect %9, %12 : !firrtl.uint<5>
    firrtl.connect %_11, %8 : !firrtl.uint, !firrtl.uint<8>
    firrtl.strictconnect %7, %_0 : !firrtl.clock
    firrtl.strictconnect %6, %_3 : !firrtl.uint<1>
    %14 = firrtl.pad %_3, 8 : (!firrtl.uint<1>) -> !firrtl.uint<8>
    firrtl.strictconnect %5, %14 : !firrtl.uint<8>
    %15 = firrtl.pad %_3, 5 : (!firrtl.uint<1>) -> !firrtl.uint<5>
    firrtl.strictconnect %4, %15 : !firrtl.uint<5>
    firrtl.strictconnect %3, %_3 : !firrtl.uint<1>

    // CHECK-NEXT: cmem `19` : { `0` : UInt<8> }[32]
    // CHECK-NEXT: smem `20` : { `0` : UInt<8> }[32]

    // CHECK-NEXT: write mport `21` = `19`[UInt<5>(8)], `0`
    // CHECK-NEXT: connect `21`.`0`, UInt<8>(0)
    %_19 = chirrtl.combmem {name = "19"} : !chirrtl.cmemory<bundle<"0": uint<8>>, 32>
    %_21_data, %_21_port = chirrtl.memoryport Write %_19 {name = "21"} : (!chirrtl.cmemory<bundle<"0": uint<8>>, 32>) -> (!firrtl.bundle<"0": uint<8>>, !chirrtl.cmemoryport)
    %16 = firrtl.subfield %_21_data["0"] : !firrtl.bundle<"0": uint<8>>
    %_20 = chirrtl.seqmem Undefined  {name = "20"} : !chirrtl.cmemory<bundle<"0": uint<8>>, 32>
    %c8_ui5 = firrtl.constant 8 : !firrtl.const.uint<5>
    chirrtl.memoryport.access %_21_port[%c8_ui5], %_0 : !chirrtl.cmemoryport, !firrtl.const.uint<5>, !firrtl.clock
    %c0_ui8 = firrtl.constant 0 : !firrtl.const.uint<8>
    %17 = firrtl.constCast %c0_ui8 : (!firrtl.const.uint<8>) -> !firrtl.uint<8>
    firrtl.strictconnect %16, %17 : !firrtl.uint<8>

    // CHECK-NEXT: stop(`0`, `3`, 1) : `22`
    // CHECK-NEXT: assert(`0`, `3`, `3`, "message") : `23`
    firrtl.stop %_0, %_3, 1 {name = "22"} : !firrtl.clock, !firrtl.uint<1>
    firrtl.assert %_0, %_3, %_3, "message" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>  {eventControl = 0 : i32, isConcurrent = false, name = "23"}

    // CHECK-NEXT: define `12` = probe(`14`)
    // CHECK-NEXT: define `13` = rwprobe(`15`)
    %18 = firrtl.ref.send %_14 : !firrtl.uint<1>
    firrtl.ref.define %_12, %18 : !firrtl.probe<uint<1>>
    firrtl.ref.define %_13, %_15_ref : !firrtl.rwprobe<uint<1>>
  }
  
  // CHECK-LABEL: module Properties :
  firrtl.module @Properties(out %string : !firrtl.string,
                            out %integer : !firrtl.integer) {
    // CHECK: propassign string, String("hello")
    %0 = firrtl.string "hello"
    firrtl.propassign %string, %0 : !firrtl.string

    // CHECK: propassign integer, Integer(99)
    %1 = firrtl.integer 99
    firrtl.propassign %integer, %1 : !firrtl.integer
  }

  // Test optional group declaration and definition emission.
  //
  // CHECK-LABEL: declgroup GroupA, bind :
  // CHECK-NEXT:    declgroup GroupB, bind :
  // CHECK-NEXT:      declgroup GroupC, bind :
  // CHECK-NEXT:      declgroup GroupD, bind :
  // CHECK-NEXT:        declgroup GroupE, bind :
  // CHECK-NEXT:    declgroup GroupF, bind :
  firrtl.declgroup @GroupA bind {
    firrtl.declgroup @GroupB bind {
      firrtl.declgroup @GroupC bind {
      }
      firrtl.declgroup @GroupD bind {
        firrtl.declgroup @GroupE bind {
        }
      }
    }
    firrtl.declgroup @GroupF bind {
    }
  }
  // CHECK:      module ModuleWithGroups :
  // CHECK-NEXT:   group GroupA :
  // CHECK-NEXT:     group GroupB :
  // CHECK-NEXT:       group GroupC :
  // CHECK-NEXT:       group GroupD :
  // CHECK-NEXT:         group GroupE :
  // CHECK-NEXT:     group GroupF :
  firrtl.module @ModuleWithGroups() {
    firrtl.group @GroupA {
      firrtl.group @GroupA::@GroupB {
        firrtl.group @GroupA::@GroupB::@GroupC {
        }
        firrtl.group @GroupA::@GroupB::@GroupD {
          firrtl.group @GroupA::@GroupB::@GroupD::@GroupE {
          }
        }
      }
      firrtl.group @GroupA::@GroupF {
      }
    }
  }

}
