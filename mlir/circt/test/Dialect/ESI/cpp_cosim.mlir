// RUN: circt-opt %s --esi-emit-cpp-api="to-stderr=true" -o %t 2>&1 | FileCheck %s

!TWrite = !esi.channel<!hw.struct<addr: i32, data: i8>>

esi.service.decl @BSP {
  esi.service.to_client @Recv : !esi.channel<!TWrite>
  esi.service.to_server @Send : !esi.channel<i8>
}

hw.module @Top(%clk: i1, %rst: i1) {
  %0 = esi.null : !esi.channel<i1>
  %1 = esi.cosim %clk, %rst, %0, "m1.loopback_tohw" : !esi.channel<i1> -> !esi.channel<i8>
  %2 = esi.cosim %clk, %rst, %m1.loopback_fromhw, "m1.loopback_fromhw" : !esi.channel<i8> -> !esi.channel<i1>
  esi.service.hierarchy.metadata path [] implementing @BSP impl as "cosim"
    clients [
      {client_name = ["m1", "loopback_tohw"], port = #hw.innerNameRef<@BSP::@Recv>, to_client_type = !esi.channel<i8>},
      {client_name = ["m1", "loopback_fromhw"], port = #hw.innerNameRef<@BSP::@Send>, to_server_type = !TWrite}]
  %m1.loopback_fromhw = hw.instance "m1" @Loopback(clk: %clk: i1, loopback_tohw: %1: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>)
  hw.output
}

hw.module @Loopback(%clk: i1, %loopback_tohw: !esi.channel<i8>) -> (loopback_fromhw: !esi.channel<i8>) {
  hw.output %loopback_tohw : !esi.channel<i8>
}

// =============================================================================
// Verify header
// =============================================================================

// CHECK: #pragma once
// CHECK: #include "refl.hpp"
// CHECK: #include <cstdint>
// CHECK: #include "esi/backends/capnp.h"
// CHECK: #include ESI_COSIM_CAPNP_H

// CHECK: namespace esi {
// CHECK: namespace runtime {

// CHECK: class ESITypes {
// CHECK: public:

// =============================================================================
// Verify types - we're expecting an i8 type and a struct<addr: i32, data: i8> type
// =============================================================================

// CHECK: struct I8 {
// CHECK:   uint8_t i;  // MLIR type is i8
// CHECK:   I8() = default;
// CHECK:   I8(uint8_t i) : i(i) {}
// CHECK:   operator uint8_t() const { return i; }
// CHECK:   auto operator==(const I8 &other) const {
// CHECK:     return (i == other.i);
// CHECK:   }
// CHECK:   auto operator!=(const I8 &other) const {
// CHECK:     return !(*this == other);
// CHECK:   }
// CHECK:   friend std::ostream &operator<<(std::ostream &os, const I8 &val) {
// CHECK:     os << "I8(";
// CHECK:     os << "i: ";
// CHECK:     os << (uint32_t)val.i;
// CHECK:     os << ")";
// CHECK:     return os;
// CHECK:   }
// CHECK:   using CPType = ::I8;

// CHECK: struct Struct17656501409672388976 {
// CHECK:   uint32_t addr;      // MLIR type is i32
// CHECK:   uint8_t data;       // MLIR type is i8
// CHECK:   auto operator==(const Struct17656501409672388976 &other) const {
// CHECK:     return (addr == other.addr) && (data == other.data);
// CHECK:   }
// CHECK:   auto operator!=(const Struct17656501409672388976 &other) const {
// CHECK:     return !(*this == other);
// CHECK:   }
// CHECK:   friend std::ostream &operator<<(std::ostream &os, const Struct17656501409672388976 &val) {
// CHECK:     os << "Struct17656501409672388976(";
// CHECK:     os << "addr: ";
// CHECK:     os << val.addr;
// CHECK:     os << ", ";
// CHECK:     os << "data: ";
// CHECK:     os << (uint32_t)val.data;
// CHECK:     os << ")";
// CHECK:     return os;
// CHECK:   }
// CHECK:   using CPType = ::Struct17656501409672388976;

// =============================================================================
// Verify service declarations
// =============================================================================

// CHECK: template <typename TBackend>
// CHECK: class BSP : public esi::runtime::Module<TBackend> {
// CHECK:     using Port = esi::runtime::Port<TBackend>;
// CHECK: public:
// CHECK:   using TSend = esi::runtime::ReadPort</*readType=*/ ESITypes::I8, TBackend>;
// CHECK:   using TSendPtr = std::shared_ptr<TSend>;
// CHECK:   using TRecv = esi::runtime::WritePort</*writeType=*/ ESITypes::Struct17656501409672388976, TBackend>;
// CHECK:   using TRecvPtr = std::shared_ptr<TRecv>;
// CHECK:   BSP(TSendPtr Send, TRecvPtr Recv)
// CHECK:     : esi::runtime::Module<TBackend>({Send, Recv}),
// CHECK:     Send(Send), Recv(Recv) {}
// CHECK:   TSendPtr Send;
// CHECK:   TRecvPtr Recv;

// =============================================================================
// Verify top modules
// =============================================================================

// CHECK: template <typename TBackend>
// CHECK: class Top {
// CHECK: public:
// CHECK:   std::unique_ptr<BSP<TBackend>> bsp;
// CHECK:   Top(TBackend& backend) {
// CHECK:     auto m1_loopback_tohw = std::make_shared<esi::runtime::WritePort</*writeType=*/ ESITypes::Struct17656501409672388976, TBackend>>(std::vector<std::string>{"m1", "loopback_tohw"}, backend, "cosim");
// CHECK:     auto m1_loopback_fromhw = std::make_shared<esi::runtime::ReadPort</*readType=*/ ESITypes::I8, TBackend>>(std::vector<std::string>{"m1", "loopback_fromhw"}, backend, "cosim");
// CHECK:     bsp = std::make_unique<BSP<TBackend>>(m1_loopback_fromhw, m1_loopback_tohw);
// CHECK:     bsp->init();

// =============================================================================
// Verify type reflection
// =============================================================================

// CHECK: REFL_AUTO (
// CHECK:   type(esi::runtime::ESITypes::I8)  
// CHECK: , field(i)
// CHECK: )
// CHECK: REFL_AUTO (
// CHECK:   type(esi::runtime::ESITypes::Struct17656501409672388976)  
// CHECK: , field(addr)  
// CHECK: , field(data)
// CHECK: )
