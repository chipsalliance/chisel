// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-chirrtl)))'  %s | FileCheck %s

firrtl.circuit "Empty" {

// Memories with no ports should be deleted.
firrtl.module @Empty(in %clock: !firrtl.clock) {
  %ram = chirrtl.combmem : !chirrtl.cmemory<uint<1>, 2>
}
// CHECK:      firrtl.module @Empty(in %clock: !firrtl.clock) {
// CHECK-NEXT: }

// Unused ports should be deleted.
firrtl.module @UnusedMemPort(in %clock: !firrtl.clock, in %addr : !firrtl.uint<1>) {
  %ram = chirrtl.combmem : !chirrtl.cmemory<vector<uint<1>, 2>, 2>
  // This port should be deleted.
  %port0_data, %port0_port = chirrtl.memoryport Infer %ram {name = "port0"} : (!chirrtl.cmemory<vector<uint<1>, 2>, 2>) -> (!firrtl.vector<uint<1>, 2>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %port0_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<1>, !firrtl.clock
  // Subindexing a port should not count as a "use".
  %port1_data, %port1_port = chirrtl.memoryport Infer %ram {name = "port1"} : (!chirrtl.cmemory<vector<uint<1>, 2>, 2>) -> (!firrtl.vector<uint<1>, 2>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %port1_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<1>, !firrtl.clock
  %0 = firrtl.subindex %port1_data[1] : !firrtl.vector<uint<1>, 2>
}
// CHECK:      firrtl.module @UnusedMemPort(in %clock: !firrtl.clock, in %addr: !firrtl.uint<1>) {
// CHECK-NEXT: }

firrtl.module @InferRead(in %cond: !firrtl.uint<1>, in %clock: !firrtl.clock, in %addr: !firrtl.uint<8>, out %out : !firrtl.uint<1>, in %vec : !firrtl.vector<uint<1>, 2>) {
  // CHECK: %ram_ramport = firrtl.mem sym @s1 Undefined {depth = 256 : i64, name = "ram", portNames = ["ramport"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<8>, en: uint<1>, clk: clock, data flip: uint<1>>
  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_ramport[addr]
  // CHECK: firrtl.strictconnect [[ADDR]], %invalid_ui8
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_ramport[en]
  // CHECK: firrtl.strictconnect [[EN]], %c0_ui1
  // CHECK: [[CLOCK:%.*]] = firrtl.subfield %ram_ramport[clk]
  // CHECK: firrtl.strictconnect [[CLOCK]], %invalid_clock
  // CHECK: [[DATA:%.*]] = firrtl.subfield %ram_ramport[data]
  %ram = chirrtl.combmem  sym @s1 : !chirrtl.cmemory<uint<1>, 256>
  %ramport_data, %ramport_port = chirrtl.memoryport Infer %ram {name = "ramport"} : (!chirrtl.cmemory<uint<1>, 256>) -> (!firrtl.uint<1>, !chirrtl.cmemoryport)

  // CHECK: firrtl.when %cond : !firrtl.uint<1> {
  // CHECK:   firrtl.strictconnect [[ADDR]], %addr
  // CHECK:   firrtl.strictconnect [[EN]], %c1_ui1
  // CHECK:   firrtl.strictconnect [[CLOCK]], %clock
  // CHECK: }
  firrtl.when %cond : !firrtl.uint<1> {
    chirrtl.memoryport.access %ramport_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  }

  // CHECK: %node = firrtl.node [[DATA]]
  %node = firrtl.node %ramport_data : !firrtl.uint<1>

  // CHECK: firrtl.connect %out, [[DATA]]
  firrtl.connect %out, %ramport_data : !firrtl.uint<1>, !firrtl.uint<1>

  // TODO: How do you get FileCheck to accept "[[[DATA]]]"?
  // CHECK: firrtl.subaccess %vec{{\[}}[[DATA]]{{\]}} : !firrtl.vector<uint<1>, 2>, !firrtl.uint<1>
  firrtl.subaccess %vec[%ramport_data] : !firrtl.vector<uint<1>, 2>, !firrtl.uint<1>
}

firrtl.module @InferWrite(in %cond: !firrtl.uint<1>, in %clock: !firrtl.clock, in %addr: !firrtl.uint<8>, in %in : !firrtl.uint<1>) {
  // CHECK: %ram_ramport = firrtl.mem Undefined {depth = 256 : i64, name = "ram", portNames = ["ramport"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<8>, en: uint<1>, clk: clock, data: uint<1>, mask: uint<1>>
  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_ramport[addr]
  // CHECK: firrtl.strictconnect [[ADDR]], %invalid_ui8
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_ramport[en]
  // CHECK: firrtl.strictconnect [[EN]], %c0_ui1
  // CHECK: [[CLOCK:%.*]] = firrtl.subfield %ram_ramport[clk]
  // CHECK: firrtl.strictconnect [[CLOCK]], %invalid_clock
  // CHECK: [[DATA:%.*]] = firrtl.subfield %ram_ramport[data]
  // CHECK: firrtl.strictconnect [[DATA]], %invalid_ui1
  // CHECK: [[MASK:%.*]] = firrtl.subfield %ram_ramport[mask]
  // CHECK: firrtl.strictconnect [[MASK]], %invalid_ui1
  %ram = chirrtl.combmem : !chirrtl.cmemory<uint<1>, 256>
  %ramport_data, %ramport_port = chirrtl.memoryport Infer %ram {name = "ramport"} : (!chirrtl.cmemory<uint<1>, 256>) -> (!firrtl.uint<1>, !chirrtl.cmemoryport)

  // CHECK: firrtl.when %cond : !firrtl.uint<1> {
  // CHECK:   firrtl.strictconnect [[ADDR]], %addr
  // CHECK:   firrtl.strictconnect [[EN]], %c1_ui1
  // CHECK:   firrtl.strictconnect [[CLOCK]], %clock
  // CHECK:   firrtl.strictconnect [[MASK]], %c0_ui1
  // CHECK: }
  firrtl.when %cond : !firrtl.uint<1> {
    chirrtl.memoryport.access %ramport_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  }

  // CHECK: firrtl.strictconnect [[MASK]], %c1_ui1
  // CHECK: firrtl.connect [[DATA]], %in
  firrtl.connect %ramport_data, %in : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.strictconnect [[MASK]], %c1_ui1
  // CHECK: firrtl.connect [[DATA]], %in
  firrtl.connect %ramport_data, %in : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @InferReadWrite(in %clock: !firrtl.clock, in %addr: !firrtl.uint<8>, in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
  // CHECK: %ram_ramport = firrtl.mem Undefined {depth = 256 : i64, name = "ram", portNames = ["ramport"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<8>, en: uint<1>, clk: clock, rdata flip: uint<1>, wmode: uint<1>, wdata: uint<1>, wmask: uint<1>>
  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_ramport[addr]
  // CHECK: firrtl.strictconnect [[ADDR]], %invalid
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_ramport[en]
  // CHECK: firrtl.strictconnect [[EN]], %c0_ui1
  // CHECK: [[CLOCK:%.*]] = firrtl.subfield %ram_ramport[clk]
  // CHECK: firrtl.strictconnect [[CLOCK]], %invalid_clock
  // CHECK: [[RDATA:%.*]] = firrtl.subfield %ram_ramport[rdata]
  // CHECK: [[WMODE:%.*]] = firrtl.subfield %ram_ramport[wmode]
  // CHECK: firrtl.strictconnect [[WMODE]], %c0_ui1
  // CHECK: [[WDATA:%.*]] = firrtl.subfield %ram_ramport[wdata]
  // CHECK: firrtl.strictconnect [[WDATA]], %invalid_ui1
  // CHECK: [[WMASK:%.*]] = firrtl.subfield %ram_ramport[wmask]
  // CHECK: firrtl.strictconnect [[WMASK]], %invalid
  %ram = chirrtl.combmem : !chirrtl.cmemory<uint<1>, 256>

  // CHECK: firrtl.strictconnect [[ADDR]], %addr : !firrtl.uint<8>
  // CHECK: firrtl.strictconnect [[EN]], %c1_ui1
  // CHECK: firrtl.strictconnect [[CLOCK]], %clock
  // CHECK: firrtl.strictconnect [[WMASK]], %c0_ui1
  %ramport_data, %ramport_port = chirrtl.memoryport Read %ram {name = "ramport"} : (!chirrtl.cmemory<uint<1>, 256>) -> (!firrtl.uint<1>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %ramport_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock

  // CHECK: firrtl.strictconnect [[WMASK]], %c1_ui1
  // CHECK: firrtl.strictconnect [[WMODE]], %c1_ui1
  // CHECK: firrtl.connect [[WDATA]], %in
  firrtl.connect %ramport_data, %in : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: firrtl.connect %out, [[RDATA]] 
  firrtl.connect %out, %ramport_data : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @WriteToSubfield(in %clock: !firrtl.clock, in %addr: !firrtl.uint<8>, in %value: !firrtl.uint<1>) {
  %ram = chirrtl.combmem : !chirrtl.cmemory<bundle<a: uint<1>, b: uint<1>>, 256>
  %ramport_data, %ramport_port = chirrtl.memoryport Infer %ram {name = "ramport"} : (!chirrtl.cmemory<bundle<a: uint<1>, b: uint<1>>, 256>) -> (!firrtl.bundle<a: uint<1>, b: uint<1>>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %ramport_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock

  %ramport_b = firrtl.subfield %ramport_data[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
  // Check that only the subfield of the mask is written to.
  // CHECK: [[DATA:%.*]] = firrtl.subfield %ram_ramport[data]
  // CHECK: [[MASK:%.*]] = firrtl.subfield %ram_ramport[mask]
  // CHECK: [[DATA_B:%.*]] = firrtl.subfield [[DATA]][b]
  // CHECK: [[MASK_B:%.*]] = firrtl.subfield [[MASK]][b]
  // CHECK: firrtl.strictconnect [[MASK_B]], %c1_ui1
  // CHECK: firrtl.connect [[DATA_B]], %value
  firrtl.connect %ramport_b, %value : !firrtl.uint<1>, !firrtl.uint<1>
}

// Read and write from different subfields of the memory.  The memory as a
// whole should be inferred to read+write.
firrtl.module @ReadAndWriteToSubfield(in %clock: !firrtl.clock, in %addr: !firrtl.uint<8>, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  %ram = chirrtl.combmem : !chirrtl.cmemory<bundle<a: uint<1>, b: uint<1>>, 256>
  %ramport_data, %ramport_port = chirrtl.memoryport Infer %ram {name = "ramport"} : (!chirrtl.cmemory<bundle<a: uint<1>, b:uint<1>>, 256>) -> (!firrtl.bundle<a: uint<1>, b: uint<1>>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %ramport_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock

  // CHECK: [[RDATA:%.*]] = firrtl.subfield %ram_ramport[rdata]
  // CHECK: [[WMODE:%.*]] = firrtl.subfield %ram_ramport[wmode]
  // CHECK: [[WDATA:%.*]] = firrtl.subfield %ram_ramport[wdata]
  // CHECK: [[WMASK:%.*]] = firrtl.subfield %ram_ramport[wmask]
  // CHECK: [[WDATA_A:%.*]] = firrtl.subfield [[WDATA]][a]
  // CHECK: [[WMASK_A:%.*]] = firrtl.subfield [[WMASK]][a]
  // CHECK: firrtl.strictconnect [[WMASK_A]], %c1_ui1
  // CHECK: firrtl.strictconnect [[WMODE]], %c1_ui1
  // CHECK: firrtl.connect [[WDATA_A]], %in
  %port_a = firrtl.subfield %ramport_data[a] : !firrtl.bundle<a: uint<1>, b: uint<1>>
  firrtl.connect %port_a, %in : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[RDATA_B:%.*]] = firrtl.subfield [[RDATA]][b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK: firrtl.connect %out, [[RDATA_B]] : !firrtl.uint<1>, !firrtl.uint<1>
  %port_b = firrtl.subfield %ramport_data[b] : !firrtl.bundle<a: uint<1>, b: uint<1>>
  firrtl.connect %out, %port_b : !firrtl.uint<1>, !firrtl.uint<1>
}

// Read and write from different subindex and subaccess of the memory.  The
// memory as a whole should be inferred to read+write.
firrtl.module @ReadAndWriteToSubindex(in %clock: !firrtl.clock, in %addr: !firrtl.uint<8>, in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  %ram = chirrtl.combmem : !chirrtl.cmemory<vector<uint<1>, 10>, 256>
  %ramport_data, %ramport_port = chirrtl.memoryport Infer %ram {name = "ramport"} : (!chirrtl.cmemory<vector<uint<1>, 10>, 256>) -> (!firrtl.vector<uint<1>, 10>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %ramport_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock

  // CHECK: [[RDATA:%.*]] = firrtl.subfield %ram_ramport[rdata]
  // CHECK: [[WMODE:%.*]] = firrtl.subfield %ram_ramport[wmode]
  // CHECK: [[WDATA:%.*]] = firrtl.subfield %ram_ramport[wdata]
  // CHECK: [[WMASK:%.*]] = firrtl.subfield %ram_ramport[wmask]
  // CHECK: [[WDATA_0:%.*]] = firrtl.subindex [[WDATA]][0]
  // CHECK: [[WMASK_0:%.*]] = firrtl.subindex [[WMASK]][0]
  // CHECK: firrtl.strictconnect [[WMASK_0]], %c1_ui1 : !firrtl.uint<1>
  // CHECK: firrtl.strictconnect [[WMODE]], %c1_ui1 : !firrtl.uint<1>
  // CHECK: firrtl.connect [[WDATA_0]], %in : !firrtl.uint<1>, !firrtl.uint<1>
  %port_a = firrtl.subindex %ramport_data[0] : !firrtl.vector<uint<1>, 10>
  firrtl.connect %port_a, %in : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[RDATA_I:%.*]] = firrtl.subaccess [[RDATA]][%addr]
  // CHECK: firrtl.connect %out, [[RDATA_I]]
  %port_b = firrtl.subaccess %ramport_data[%addr] : !firrtl.vector<uint<1>, 10>, !firrtl.uint<8>
  firrtl.connect %out, %port_b : !firrtl.uint<1>, !firrtl.uint<1>
}

// Check that ports are sorted in alphabetical order.
firrtl.module @SortedPorts(in %clock: !firrtl.clock, in %addr : !firrtl.uint<8>) {
  // CHECK: portNames = ["a", "b", "c"]
  %ram = chirrtl.combmem : !chirrtl.cmemory<vector<uint<1>, 2>, 256>
  %c_data, %c_port = chirrtl.memoryport Read %ram {name = "c"} : (!chirrtl.cmemory<vector<uint<1>, 2>, 256>) -> (!firrtl.vector<uint<1>, 2>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %c_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  %a_data, %a_port = chirrtl.memoryport Write %ram {name = "a"} : (!chirrtl.cmemory<vector<uint<1>, 2>, 256>) -> (!firrtl.vector<uint<1>, 2>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %a_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  %b_data, %b_port = chirrtl.memoryport ReadWrite %ram {name = "b"} : (!chirrtl.cmemory<vector<uint<1>, 2>, 256>) -> (!firrtl.vector<uint<1>, 2>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %b_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
}

// Check that annotations are preserved.
firrtl.module @Annotations(in %clock: !firrtl.clock, in %addr : !firrtl.uint<8>) {
  // CHECK: firrtl.mem Undefined
  // CHECK-SAME: annotations = [{a = "a"}]
  // CHECK-SAME: portAnnotations = [
  // CHECK-SAME:   [{b = "b"}],
  // CHECK-SAME:   [{c = "c"}]
  // CHECK-SAME: ]
  // CHECK-SAME: portNames = ["port0", "port1"]
  %ram = chirrtl.combmem {annotations = [{a = "a"}]} : !chirrtl.cmemory<vector<uint<1>, 2>, 256>
  %port0_data, %port0_port = chirrtl.memoryport Read %ram  {annotations = [{b = "b"}], name = "port0"} : (!chirrtl.cmemory<vector<uint<1>, 2>, 256>) -> (!firrtl.vector<uint<1>, 2>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %port0_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  %port1_data, %port1_port = chirrtl.memoryport Read %ram {annotations = [{c = "c"}], name = "port1"} : (!chirrtl.cmemory<vector<uint<1>, 2>, 256>) -> (!firrtl.vector<uint<1>, 2>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %port1_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
}

// When the address is a wire, the enable should be inferred where the address
// is driven.
firrtl.module @EnableInference0(in %p: !firrtl.uint<1>, in %addr: !firrtl.uint<4>, in %clock: !firrtl.clock, out %v: !firrtl.uint<32>) {
  %w = firrtl.wire  : !firrtl.uint<4>
  // This connect should not count as "driving" a value.  If it accidentally
  // inserts an enable here, we will get a use-before-def error, so it is
  // enough of a check that this compiles.
  %invalid_ui4 = firrtl.invalidvalue : !firrtl.uint<4>
  firrtl.connect %w, %invalid_ui4 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_ramport[addr]
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_ramport[en]
  %ram = chirrtl.seqmem Undefined  : !chirrtl.cmemory<uint<32>, 16>
  %ramport_data, %ramport_port = chirrtl.memoryport Read %ram  {name = "ramport"}: (!chirrtl.cmemory<uint<32>, 16>) -> (!firrtl.uint<32>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %ramport_port[%w], %clock : !chirrtl.cmemoryport, !firrtl.uint<4>, !firrtl.clock

  // CHECK: firrtl.when %p : !firrtl.uint<1> {
  firrtl.when %p : !firrtl.uint<1> {
    // CHECK-NEXT: firrtl.strictconnect [[EN]], %c1_ui1
    // CHECK-NEXT: firrtl.connect %w, %addr
    firrtl.connect %w, %addr : !firrtl.uint<4>, !firrtl.uint<4>
  }
  firrtl.connect %v, %ramport_data : !firrtl.uint<32>, !firrtl.uint<32>
}

// When the address is a node, the enable should be inferred where the address is declared.
firrtl.module @EnableInference1(in %p: !firrtl.uint<1>, in %addr: !firrtl.uint<4>, in %clock: !firrtl.clock, out %v: !firrtl.uint<32>) {
  %ram = chirrtl.seqmem Undefined  : !chirrtl.cmemory<uint<32>, 16>
  %invalid_ui32 = firrtl.invalidvalue : !firrtl.uint<32>
  firrtl.connect %v, %invalid_ui32 : !firrtl.uint<32>, !firrtl.uint<32>
  // CHECK: [[ADDR:%.*]] = firrtl.subfield %ram_ramport[addr]
  // CHECK: [[EN:%.*]] = firrtl.subfield %ram_ramport[en]
  // CHECK: firrtl.when %p : !firrtl.uint<1>
  firrtl.when %p : !firrtl.uint<1> {
   // CHECK-NEXT: firrtl.strictconnect [[EN]], %c1_ui1
   // CHECK-NEXT: %n = firrtl.node %addr
   // CHECK-NEXT: firrtl.strictconnect [[ADDR]], %n
   // CHECK-NEXT: firrtl.strictconnect %2, %clock
   // CHECK-NEXT: firrtl.connect %v, %3
    %n = firrtl.node %addr : !firrtl.uint<4>
    %ramport_data, %ramport_port = chirrtl.memoryport Read %ram {name = "ramport"} : (!chirrtl.cmemory<uint<32>, 16>) -> (!firrtl.uint<32>, !chirrtl.cmemoryport)
    chirrtl.memoryport.access %ramport_port[%n], %clock : !chirrtl.cmemoryport, !firrtl.uint<4>, !firrtl.clock
    firrtl.connect %v, %ramport_data : !firrtl.uint<32>, !firrtl.uint<32>
  }
}  

// When the address is not something with a name, including a subfield of an
// aggregate, we do not perform regular enable inference.
// CHECK-LABEL: firrtl.module @EnableInference2
firrtl.module @EnableInference2(in %clock: !firrtl.clock, in %io: !firrtl.bundle<addr: uint<3>>, out %out: !firrtl.uint<8>) {
  %0 = firrtl.subfield %io[addr] : !firrtl.bundle<addr: uint<3>>
  %mem = chirrtl.seqmem Undefined  : !chirrtl.cmemory<uint<8>, 8>
  %read_data, %read_port = chirrtl.memoryport Infer %mem  {name = "read"} : (!chirrtl.cmemory<uint<8>, 8>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %read_port[%0], %clock : !chirrtl.cmemoryport, !firrtl.uint<3>, !firrtl.clock
  firrtl.connect %out, %read_data : !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK: [[EN:%.*]] = firrtl.subfield %mem_read[en]
  // CHECK: firrtl.strictconnect [[EN]], %c0_ui1
  // CHECK: firrtl.strictconnect [[EN]], %c1_ui1
}

// When the address line is larger than the size of the address port, the port
// connection should be made using a truncation and connect.
firrtl.module @AddressLargerThanPort(in %clock: !firrtl.clock, in %addr: !firrtl.uint<3>, out %out: !firrtl.uint<1>) {
  // CHECK-LABEL: @AddressLargerThanPort
  %mem = chirrtl.seqmem Undefined  : !chirrtl.cmemory<uint<1>, 4>
  %r_data, %r_port = chirrtl.memoryport Infer %mem  {name = "r"} : (!chirrtl.cmemory<uint<1>, 4>) -> (!firrtl.uint<1>, !chirrtl.cmemoryport)
  // CHECK: [[ADDR:%.+]] = firrtl.subfield %mem_r[addr]
  %addr_node = firrtl.node %addr  : !firrtl.uint<3>
  // CHECK: [[TRUNC:%.+]] = firrtl.tail %addr_node, 1
  // CHECK: firrtl.strictconnect [[ADDR]], [[TRUNC]]
  chirrtl.memoryport.access %r_port[%addr_node], %clock : !chirrtl.cmemoryport, !firrtl.uint<3>, !firrtl.clock
  // CHECK: firrtl.connect
  firrtl.connect %out, %r_data : !firrtl.uint<1>, !firrtl.uint<1>
}

// Ensure that larger than 32-bit memories work
firrtl.module @LargeMem(in %clock: !firrtl.clock, in %addr: !firrtl.uint<35>, out %out: !firrtl.uint<1>) {
  // CHECK-LABEL: @LargeMem
  %testharness = chirrtl.seqmem Undefined  : !chirrtl.cmemory<uint<1>, 34359738368>
  // CHECK: %testharness_r = firrtl.mem Undefined  {depth = 34359738368 : i64, name = "testharness"
  %r_data, %r_port = chirrtl.memoryport Infer %testharness  {name = "r"} : (!chirrtl.cmemory<uint<1>, 34359738368>) -> (!firrtl.uint<1>, !chirrtl.cmemoryport)
  %addr_node = firrtl.node %addr  : !firrtl.uint<35>
  chirrtl.memoryport.access %r_port[%addr_node], %clock : !chirrtl.cmemoryport, !firrtl.uint<35>, !firrtl.clock
  firrtl.connect %out, %r_data : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @DbgsMemPort(in %clock: !firrtl.clock, in %addr : !firrtl.uint<1>, out %_a: !firrtl.probe<vector<uint<1>, 2>>, in %cond : !firrtl.uint<1>) {
  %ram = chirrtl.combmem : !chirrtl.cmemory<uint<1>, 2>
  // This port should be deleted.
  %port0_data = chirrtl.debugport %ram {name = "port0"} : (!chirrtl.cmemory<uint<1>, 2>) -> !firrtl.probe<vector<uint<1>, 2>>
  %ramport_data, %ramport_port = chirrtl.memoryport Read %ram {name = "ramport"} : (!chirrtl.cmemory<uint<1>, 2>) -> (!firrtl.uint<1>, !chirrtl.cmemoryport)

  firrtl.when %cond : !firrtl.uint<1> {
    chirrtl.memoryport.access %ramport_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<1>, !firrtl.clock
  }
  firrtl.ref.define %_a, %port0_data : !firrtl.probe<vector<uint<1>, 2>>
  // CHECK:    %[[ram_port0:.+]], %ram_ramport = firrtl.mem Undefined {depth = 2 : i64, name = "ram", portNames = ["port0", "ramport"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.probe<vector<uint<1>, 2>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
  // CHECK:    firrtl.ref.define %_a, %[[ram_port0]] : !firrtl.probe<vector<uint<1>, 2>>
}

}
