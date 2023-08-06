// RUN: circt-opt -split-input-file -handshake-lower-extmem-to-hw="wrap-esi=true" %s | FileCheck %s
// RUN: circt-opt -split-input-file -handshake-lower-extmem-to-hw="wrap-esi=true" -handshake-materialize-forks-sinks -lower-handshake-to-hw %s | FileCheck %s --check-prefix=LOWERED


//CHECK-LABEL: hw.module.extern @__main_hw(%arg0: !esi.channel<i64>, %arg1: !esi.channel<i64>, %v: !esi.channel<i32>, %mem_ld0.data: !esi.channel<i32>, %mem_st0.done: !esi.channel<i0>, %argCtrl: !esi.channel<i0>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i0>, mem_ld0.addr: !esi.channel<i4>, mem_st0: !esi.channel<!hw.struct<address: i4, data: i32>>)

//CHECK-LABEL: esi.mem.ram @mem i32 x 10

//CHECK-LABEL: hw.module @main_esi_wrapper(%arg0: !esi.channel<i64>, %arg1: !esi.channel<i64>, %v: !esi.channel<i32>, %argCtrl: !esi.channel<i0>, %clock: i1, %reset: i1) -> (out0: !esi.channel<i0>) {
//CHECK-NEXT:   %0 = esi.service.req.inout %main.mem_ld0.addr -> <@mem::@read>([]) : !esi.channel<i4> -> !esi.channel<i32>
//CHECK-NEXT:   %1 = esi.service.req.inout %main.mem_st0 -> <@mem::@write>([]) : !esi.channel<!hw.struct<address: i4, data: i32>> -> !esi.channel<i0>
//CHECK-NEXT:   %main.out0, %main.mem_ld0.addr, %main.mem_st0 = hw.instance "main" @__main_hw(arg0: %arg0: !esi.channel<i64>, arg1: %arg1: !esi.channel<i64>, v: %v: !esi.channel<i32>, mem_ld0.data: %0: !esi.channel<i32>, mem_st0.done: %1: !esi.channel<i0>, argCtrl: %argCtrl: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>, mem_ld0.addr: !esi.channel<i4>, mem_st0: !esi.channel<!hw.struct<address: i4, data: i32>>)
//CHECK-NEXT:   hw.output %main.out0 : !esi.channel<i0>
//CHECK-NEXT: }

handshake.func @main(%arg0: index, %arg1: index, %v: i32, %mem : memref<10xi32>, %argCtrl: none) -> none {
  %ldData, %stCtrl, %ldCtrl = handshake.extmemory[ld=1, st=1](%mem : memref<10xi32>)(%storeData, %storeAddr, %loadAddr) {id = 0 : i32} : (i32, index, index) -> (i32, none, none)
  %fCtrl:2 = fork [2] %argCtrl : none
  %loadData, %loadAddr = load [%arg0] %ldData, %fCtrl#0 : index, i32
  %storeData, %storeAddr = store [%arg1] %v, %fCtrl#1 : index, i32
  sink %loadData : i32
  %finCtrl = join %stCtrl, %ldCtrl : none, none
  return %finCtrl : none
}

// LOWERED:   %main.out0, %main.mem_ld0.addr, %main.mem_st0 = hw.instance "main" @main(arg0: %arg0: !esi.channel<i64>, arg1: %arg1: !esi.channel<i64>, v: %v: !esi.channel<i32>, mem_ld0.data: %0: !esi.channel<i32>, mem_st0.done: %1: !esi.channel<i0>, argCtrl: %argCtrl: !esi.channel<i0>, clock: %clock: i1, reset: %reset: i1) -> (out0: !esi.channel<i0>, mem_ld0.addr: !esi.channel<i4>, mem_st0: !esi.channel<!hw.struct<address: i4, data: i32>>)

// -----

//CHECK-LABEL: hw.module @singleLoad_esi_wrapper
handshake.func @singleLoad(%arg0: index, %mem : memref<10xi32>, %argCtrl: none) {
  %ldData, %ldCtrl = handshake.extmemory[ld=1, st=0](%mem : memref<10xi32>)(%loadAddr) {id = 0 : i32} : (index) -> (i32, none)
  %loadData, %loadAddr = load [%arg0] %ldData, %argCtrl : index, i32
  return
}

// -----

//CHECK-LABEL: hw.module @singleStore_esi_wrapper
handshake.func @singleStore(%arg0: index, %v : i32, %mem : memref<10xi32>, %argCtrl: none) {
  %stCtrl = handshake.extmemory[ld=0, st=1](%mem : memref<10xi32>)(%storeData, %storeAddr) {id = 0 : i32} : (i32, index) -> (none)
  %storeData, %storeAddr = store [%arg0] %v, %argCtrl : index, i32
  return
}

// -----

//CHECK-LABEL: hw.module @multipleMemories_esi_wrapper
handshake.func @multipleMemories(%arg0: index, %v : i32, %mem1 : memref<10xi32>, %mem2 : memref<10xi8>, %argCtrl: none) {
  %stCtrl = handshake.extmemory[ld=0, st=1](%mem1 : memref<10xi32>)(%storeData, %storeAddr) {id = 0 : i32} : (i32, index) -> (none)
  %storeData, %storeAddr = store [%arg0] %v, %argCtrl : index, i32
  %ldData, %ldCtrl = handshake.extmemory[ld=1, st=0](%mem2 : memref<10xi8>)(%loadAddr) {id = 1 : i32} : (index) -> (i8, none)
  %loadData, %loadAddr = load [%arg0] %ldData, %argCtrl : index, i8
  return
}

// -----

//CHECK-LABEL: hw.module @multipleMemories2_esi_wrapper
handshake.func @multipleMemories2(%a : memref<10xi32>, %b : memref<10xi32>, %c : memref<1xi32>) {
  %0 = source
  %addr = constant %0 {value = 0 : index} : index
  %data = constant %0 {value = 0 : i32} : i32
  %2 = extmemory[ld = 0, st = 1] (%c : memref<1xi32>) (%data, %addr) {id = 2 : i32} : (i32, index) -> none
  %3:2 = extmemory[ld = 1, st = 0] (%b : memref<10xi32>) (%addr) {id = 1 : i32} : (index) -> (i32, none)
  %4:2 = extmemory[ld = 1, st = 0] (%a : memref<10xi32>) (%addr) {id = 0 : i32} : (index) -> (i32, none)  return
}
