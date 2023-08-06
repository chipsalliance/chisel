// RUN: circt-opt %s --verify-diagnostics | circt-opt --verify-diagnostics | FileCheck %s

// CHECK-LABEL: hw.module @Basic
hw.module @Basic() {
  // CHECK-NEXT: seq.firmem 0, 1, undefined, undefined : <3 x 19>
  %0 = seq.firmem 0, 1, undefined, undefined : <3 x 19>

  // CHECK-NEXT: seq.firmem sym @someMem 0, 1, undefined, undefined : <3 x 19>
  %1 = seq.firmem sym @someMem 0, 1, undefined, undefined : <3 x 19>

  // CHECK-NEXT: %myMem1 = seq.firmem 0, 1, undefined, undefined : <3 x 19>
  %2 = seq.firmem name "myMem1" 0, 1, undefined, undefined : <3 x 19>

  // CHECK-NEXT: %myMem2 = seq.firmem 0, 1, undefined, undefined : <3 x 19>
  %myMem2 = seq.firmem 0, 1, undefined, undefined : <3 x 19>

  // CHECK-NEXT: %myMem3 = seq.firmem 0, 1, undefined, undefined : <3 x 19>
  %ignoredName = seq.firmem name "myMem3" 0, 1, undefined, undefined : <3 x 19>

  // CHECK-NEXT: seq.firmem 0, 1, undefined, undefined {init = #seq.firmem.init<"mem.txt", false, false>} : <3 x 19>
  %3 = seq.firmem 0, 1, undefined, undefined {init = #seq.firmem.init<"mem.txt", false, false>} : <3 x 19>
}

// CHECK-LABEL: hw.module @Ports
hw.module @Ports(%clock: i1, %enable: i1, %address: i4, %data: i20, %mode: i1, %mask: i4) {
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: %mem2 = seq.firmem 0, 1, undefined, undefined : <12 x 20, mask 4>
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  %mem2 = seq.firmem 0, 1, undefined, undefined : <12 x 20, mask 4>

  // Read ports
  // CHECK-NEXT: [[R0:%.+]] = seq.firmem.read_port %mem[%address], clock %clock : <12 x 20>
  // CHECK-NEXT: [[R1:%.+]] = seq.firmem.read_port %mem[%address], clock %clock enable %enable : <12 x 20>
  %0 = seq.firmem.read_port %mem[%address], clock %clock : <12 x 20>
  %1 = seq.firmem.read_port %mem[%address], clock %clock enable %enable : <12 x 20>

  // Write ports
  // CHECK-NEXT: seq.firmem.write_port %mem[%address] = %data, clock %clock : <12 x 20>
  // CHECK-NEXT: seq.firmem.write_port %mem[%address] = %data, clock %clock enable %enable : <12 x 20>
  // CHECK-NEXT: seq.firmem.write_port %mem2[%address] = %data, clock %clock mask %mask : <12 x 20, mask 4>, i4
  seq.firmem.write_port %mem[%address] = %data, clock %clock : <12 x 20>
  seq.firmem.write_port %mem[%address] = %data, clock %clock enable %enable : <12 x 20>
  seq.firmem.write_port %mem2[%address] = %data, clock %clock mask %mask : <12 x 20, mask 4>, i4

  // Read-write ports
  // CHECK-NEXT: [[R2:%.+]] = seq.firmem.read_write_port %mem[%address] = %data if %mode, clock %clock : <12 x 20>
  // CHECK-NEXT: [[R3:%.+]] = seq.firmem.read_write_port %mem[%address] = %data if %mode, clock %clock enable %enable : <12 x 20>
  // CHECK-NEXT: [[R4:%.+]] = seq.firmem.read_write_port %mem2[%address] = %data if %mode, clock %clock mask %mask : <12 x 20, mask 4>, i4
  %2 = seq.firmem.read_write_port %mem[%address] = %data if %mode, clock %clock : <12 x 20>
  %3 = seq.firmem.read_write_port %mem[%address] = %data if %mode, clock %clock enable %enable : <12 x 20>
  %4 = seq.firmem.read_write_port %mem2[%address] = %data if %mode, clock %clock mask %mask : <12 x 20, mask 4>, i4

  // CHECK-NEXT: comb.xor [[R0]], [[R1]], [[R2]], [[R3]], [[R4]]
  comb.xor %0, %1, %2, %3, %4 : i20
}
