
// RUN: circt-opt %s -split-input-file -verify-diagnostics

firrtl.circuit "test" {
// expected-note @below {{the destination was defined here}}
firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.uint<1>) {
  // expected-error @below {{connect has invalid flow: the destination expression "a" has source flow, expected sink or duplex flow}}
  firrtl.connect %a, %b : !firrtl.uint<1>, !firrtl.uint<1>
}
}

/// Analog types cannot be connected and must be attached.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.analog, out %b : !firrtl.analog) {
  // expected-error @+1 {{analog types may not be connected}}
  firrtl.connect %b, %a : !firrtl.analog, !firrtl.analog
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.bundle<a: analog>, out %b : !firrtl.bundle<a: analog>) {
  // expected-error @+1 {{analog types may not be connected}}
  firrtl.connect %b, %a : !firrtl.bundle<a: analog>, !firrtl.bundle<a: analog>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.analog, out %b : !firrtl.uint<1>) {
  // expected-error @+1 {{analog types may not be connected}}
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.analog
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.analog) {
  // expected-error @+1 {{analog types may not be connected}}
  firrtl.connect %b, %a : !firrtl.analog, !firrtl.uint<1>
}
}

/// Reset types can be connected to Reset, UInt<1>, or AsyncReset types.

// Reset source.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.reset, out %b : !firrtl.uint<2>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.uint<2>' and source '!firrtl.reset'}}
  firrtl.connect %b, %a : !firrtl.uint<2>, !firrtl.reset
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.reset, out %b : !firrtl.sint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.sint<1>' and source '!firrtl.reset'}}
  firrtl.connect %b, %a : !firrtl.sint<1>, !firrtl.reset
}
}

// Reset destination.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<2>, out %b : !firrtl.reset) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.reset' and source '!firrtl.uint<2>'}}
  firrtl.connect %b, %a : !firrtl.reset, !firrtl.uint<2>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.sint<1>, out %b : !firrtl.reset) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.reset' and source '!firrtl.sint<1>'}}
  firrtl.connect %b, %a : !firrtl.reset, !firrtl.sint<1>
}
}

/// Ground types can be connected if they are the same ground type.

// UInt<> source.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.sint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.sint<1>' and source '!firrtl.uint<1>'}}
  firrtl.connect %b, %a : !firrtl.sint<1>, !firrtl.uint<1>
}
}

// -----

firrtl.circuit "test" {

firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.clock) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.clock' and source '!firrtl.uint<1>'}}
  firrtl.connect %b, %a : !firrtl.clock, !firrtl.uint<1>
}

}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.asyncreset) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.asyncreset' and source '!firrtl.uint<1>'}}
  firrtl.connect %b, %a : !firrtl.asyncreset, !firrtl.uint<1>
}
}

// SInt<> source.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.sint<1>, out %b : !firrtl.uint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.uint<1>' and source '!firrtl.sint<1>'}}
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.sint<1>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.sint<1>, out %b : !firrtl.clock) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.clock' and source '!firrtl.sint<1>'}}
  firrtl.connect %b, %a : !firrtl.clock, !firrtl.sint<1>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.sint<1>, out %b : !firrtl.asyncreset) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.asyncreset' and source '!firrtl.sint<1>'}}
  firrtl.connect %b, %a : !firrtl.asyncreset, !firrtl.sint<1>
}
}

// Clock source.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.clock, out %b : !firrtl.uint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.uint<1>' and source '!firrtl.clock'}}
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.clock
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.clock, out %b : !firrtl.sint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.sint<1>' and source '!firrtl.clock'}}
  firrtl.connect %b, %a : !firrtl.sint<1>, !firrtl.clock
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.clock, out %b : !firrtl.asyncreset) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.asyncreset' and source '!firrtl.clock'}}
  firrtl.connect %b, %a : !firrtl.asyncreset, !firrtl.clock
}
}

// AsyncReset source.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.asyncreset, out %b : !firrtl.uint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.uint<1>' and source '!firrtl.asyncreset'}}
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.asyncreset
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.asyncreset, out %b : !firrtl.sint<1>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.sint<1>' and source '!firrtl.asyncreset'}}
  firrtl.connect %b, %a : !firrtl.sint<1>, !firrtl.asyncreset
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.asyncreset, out %b : !firrtl.clock) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.clock' and source '!firrtl.asyncreset'}}
  firrtl.connect %b, %a : !firrtl.clock, !firrtl.asyncreset
}
}

/// Vector types can be connected if they have the same size and element type.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.vector<uint<1>, 3>, out %b : !firrtl.vector<uint<1>, 2>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.vector<uint<1>, 2>' and source '!firrtl.vector<uint<1>, 3>'}}
  firrtl.connect %b, %a : !firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 3>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.vector<uint<1>, 3>, out %b : !firrtl.vector<sint<1>, 3>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.vector<sint<1>, 3>' and source '!firrtl.vector<uint<1>, 3>'}}
  firrtl.connect %b, %a : !firrtl.vector<sint<1>, 3>, !firrtl.vector<uint<1>, 3>
}
}

/// Bundle types can be connected if they have the same size, element names, and
/// element types.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.bundle<f1: uint<1>>, in %b : !firrtl.bundle<f1 flip: uint<1>, f2: sint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.bundle<f1 flip: uint<1>, f2: sint<1>>' and source '!firrtl.bundle<f1: uint<1>>'}}
  firrtl.connect %b, %a : !firrtl.bundle<f1 flip: uint<1>, f2: sint<1>>, !firrtl.bundle<f1: uint<1>>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.bundle<f1: uint<1>>, in %b : !firrtl.bundle<f2 flip: uint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.bundle<f2 flip: uint<1>>' and source '!firrtl.bundle<f1: uint<1>>'}}
  firrtl.connect %b, %a : !firrtl.bundle<f2 flip: uint<1>>, !firrtl.bundle<f1: uint<1>>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.bundle<f1: uint<1>>, in %b : !firrtl.bundle<f1 flip: sint<1>>) {
  // expected-error @+1 {{type mismatch between destination '!firrtl.bundle<f1 flip: sint<1>>' and source '!firrtl.bundle<f1: uint<1>>'}}
  firrtl.connect %b, %a : !firrtl.bundle<f1 flip: sint<1>>, !firrtl.bundle<f1: uint<1>>
}
}

// -----

firrtl.circuit "test" {
// expected-note @below {{the destination was defined here}}
firrtl.module @test(in %a : !firrtl.bundle<f1: uint<1>>, out %b : !firrtl.bundle<f1: uint<1>>) {
  %0 = firrtl.subfield %a[f1] : !firrtl.bundle<f1: uint<1>>
  %1 = firrtl.subfield %b[f1] : !firrtl.bundle<f1: uint<1>>
  // expected-error @below {{connect has invalid flow: the destination expression "a.f1" has source flow, expected sink or duplex flow}}
  firrtl.connect %0, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.uint<1>) {
  // expected-note @below {{the destination was defined here}}
  %0 = firrtl.and %a, %a: (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // expected-error @below {{connect has invalid flow: the destination expression has source flow, expected sink or duplex flow}}
  firrtl.connect %0, %b : !firrtl.uint<1>, !firrtl.uint<1>
}
}

/// Destination bitwidth must be greater than or equal to source bitwidth.

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<2>, out %b : !firrtl.uint<1>) {
  // expected-error @+1 {{destination '!firrtl.uint<1>' is not as wide as the source '!firrtl.uint<2>'}}
  firrtl.connect %b, %a : !firrtl.uint<1>, !firrtl.uint<2>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {a: {flip a: UInt<1>}}
///     wire   ax: {a: {flip a: UInt<1>}}
///     a.a.a <= ax.a.a

firrtl.circuit "test"  {
// expected-note @below {{the destination was defined here}}
firrtl.module @test(out %a: !firrtl.bundle<a: bundle<a flip: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a: bundle<a flip: uint<1>>>
  %a_a = firrtl.subfield %a[a] : !firrtl.bundle<a: bundle<a flip: uint<1>>>
  %a_a_a = firrtl.subfield %a_a[a] : !firrtl.bundle<a flip: uint<1>>
  %ax_a = firrtl.subfield %ax[a] : !firrtl.bundle<a: bundle<a flip: uint<1>>>
  %ax_a_a = firrtl.subfield %ax_a[a] : !firrtl.bundle<a flip: uint<1>>
  // expected-error @below {{connect has invalid flow: the destination expression "a.a.a" has source flow}}
  firrtl.connect %a_a_a, %ax_a_a : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {flip a: {a: UInt<1>}}
///     wire   ax: {flip a: {a: UInt<1>}}
///     a.a <= ax.a

firrtl.circuit "test"  {
// expected-note @below {{the destination was defined here}}
firrtl.module @test(out %a: !firrtl.bundle<a flip: bundle<a: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  %a_a = firrtl.subfield %a[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  %ax_a = firrtl.subfield %ax[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  // expected-error @+1 {{the destination expression "a.a" has source flow}}
  firrtl.connect %a_a, %ax_a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {flip a: {a: UInt<1>}}
///     wire   ax: {flip a: {a: UInt<1>}}
///     a.a.a <= ax.a.a

firrtl.circuit "test"  {
// expected-note @below {{the destination was defined here}}
firrtl.module @test(out %a: !firrtl.bundle<a flip: bundle<a: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  %a_a = firrtl.subfield %a[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  %a_a_a = firrtl.subfield %a_a[a] : !firrtl.bundle<a: uint<1>>
  %ax_a = firrtl.subfield %ax[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  %ax_a_a = firrtl.subfield %ax_a[a] : !firrtl.bundle<a: uint<1>>
  // expected-error @+1 {{connect has invalid flow: the destination expression "a.a.a" has source flow}}
  firrtl.connect %a_a_a, %ax_a_a : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {flip a: {flip a: UInt<1>}}
///     wire   ax: {flip a: {flip a: UInt<1>}}
///     a.a <= ax.a

firrtl.circuit "test"  {
// expected-note @below {{the destination was defined here}}
firrtl.module @test(out %a: !firrtl.bundle<a flip: bundle<a flip: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
  %a_a = firrtl.subfield %a[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
  %ax_a = firrtl.subfield %ax[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
  // expected-error @below {{connect has invalid flow: the destination expression "a.a" has source flow}}
  firrtl.connect %a_a, %ax_a : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
}
}

// -----

// Check that different labels cause the enumeration to not match.

firrtl.circuit "test"  {
firrtl.module @test(in %a: !firrtl.enum<a: uint<1>>, out %b: !firrtl.enum<a: uint<2>>) {
  // expected-error @below {{type mismatch between destination '!firrtl.enum<a: uint<2>>' and source '!firrtl.enum<a: uint<1>>'}}
  firrtl.connect %b, %a : !firrtl.enum<a: uint<2>>, !firrtl.enum<a: uint<1>>
}
}

// -----

// Check that different data types causes the enumeration to not match.

firrtl.circuit "test"  {
firrtl.module @test(in %a: !firrtl.enum<a: uint<0>>, out %b: !firrtl.enum<b: uint<0>>) {
  // expected-error @below {{type mismatch between destination '!firrtl.enum<b: uint<0>>' and source '!firrtl.enum<a: uint<0>>'}}
  firrtl.connect %b, %a : !firrtl.enum<b: uint<0>>, !firrtl.enum<a: uint<0>>
}
}

// -----

/// Check that the following is an invalid sink flow source.  This has to use a
/// memory because all other sinks (module outputs or instance inputs) can
/// legally be used as sources.
///
///     output a: UInt<1>
///
///     mem memory:
///       data-type => UInt<1>
///       depth => 2
///       reader => r
///       read-latency => 0
///       write-latency => 1
///       read-under-write => undefined
///
///     a <= memory.r.en

firrtl.circuit "test" {
firrtl.module @test(out %a: !firrtl.uint<1>) {
  // expected-note @below {{the source was defined here}}
  %memory_r = firrtl.mem Undefined  {depth = 2 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
  %memory_r_en = firrtl.subfield %memory_r[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
  // expected-error @below {{connect has invalid flow: the source expression "memory.r.en" has sink flow}}
  firrtl.connect %a, %memory_r_en : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {a: {flip a: UInt<1>}}
///     wire   ax: {a: {flip a: UInt<1>}}
///     a.a.a <- ax.a.a

firrtl.circuit "test"  {
// expected-note @below {{the destination was defined here}}
firrtl.module @test(out %a: !firrtl.bundle<a: bundle<a flip: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a: bundle<a flip: uint<1>>>
  %a_a = firrtl.subfield %a[a] : !firrtl.bundle<a: bundle<a flip: uint<1>>>
  %a_a_a = firrtl.subfield %a_a[a] : !firrtl.bundle<a flip: uint<1>>
  %ax_a = firrtl.subfield %ax[a] : !firrtl.bundle<a: bundle<a flip: uint<1>>>
  %ax_a_a = firrtl.subfield %ax_a[a] : !firrtl.bundle<a flip: uint<1>>
  // expected-error @below {{connect has invalid flow: the destination expression "a.a.a" has source flow}}
  firrtl.connect %a_a_a, %ax_a_a : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {flip a: {a: UInt<1>}}
///     wire   ax: {flip a: {a: UInt<1>}}
///     a.a <- ax.a

firrtl.circuit "test"  {
// expected-note @below {{the destination was defined here}}
firrtl.module @test(out %a: !firrtl.bundle<a flip: bundle<a: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  %a_a = firrtl.subfield %a[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  %ax_a = firrtl.subfield %ax[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  // expected-error @+1 {{connect has invalid flow: the destination expression "a.a" has source flow}}
  firrtl.connect %a_a, %ax_a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {flip a: {a: UInt<1>}}
///     wire   ax: {flip a: {a: UInt<1>}}
///     a.a.a <- ax.a.a

firrtl.circuit "test"  {
// expected-note @below {{the destination was defined here}}
firrtl.module @test(out %a: !firrtl.bundle<a flip: bundle<a: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  %a_a = firrtl.subfield %a[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  %a_a_a = firrtl.subfield %a_a[a] : !firrtl.bundle<a: uint<1>>
  %ax_a = firrtl.subfield %ax[a] : !firrtl.bundle<a flip: bundle<a: uint<1>>>
  %ax_a_a = firrtl.subfield %ax_a[a] : !firrtl.bundle<a: uint<1>>
  // expected-error @below {{connect has invalid flow: the destination expression "a.a.a" has source flow}}
  firrtl.connect %a_a_a, %ax_a_a : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

/// Check that the following is an invalid source flow destination:
///
///     output a:  {flip a: {flip a: UInt<1>}}
///     wire   ax: {flip a: {flip a: UInt<1>}}
///     a.a <- ax.a

firrtl.circuit "test"  {
// expected-note @below {{the destination was defined here}}
firrtl.module @test(out %a: !firrtl.bundle<a flip: bundle<a flip: uint<1>>>) {
  %ax = firrtl.wire  : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
  %a_a = firrtl.subfield %a[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
  %ax_a = firrtl.subfield %ax[a] : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
  // expected-error @below {{connect has invalid flow: the destination expression "a.a" has source flow}}
  firrtl.connect %a_a, %ax_a : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
}
}

// -----

/// Check that the following is an invalid sink flow source.  This has to use a
/// memory because all other sinks (module outputs or instance inputs) can
/// legally be used as sources.
///
///     output a: UInt<1>
///
///     mem memory:
///       data-type => UInt<1>
///       depth => 2
///       reader => r
///       read-latency => 0
///       write-latency => 1
///       read-under-write => undefined
///
///     a <- memory.r.en

firrtl.circuit "test" {
firrtl.module @test(out %a: !firrtl.uint<1>) {
  // expected-note @below {{the source was defined here}}
  %memory_r = firrtl.mem Undefined  {depth = 2 : i64, name = "memory", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
  %memory_r_en = firrtl.subfield %memory_r[en] : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<1>>
  // expected-error @below {{connect has invalid flow: the source expression "memory.r.en" has sink flow}}
  firrtl.connect %a, %memory_r_en : !firrtl.uint<1>, !firrtl.uint<1>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.bundle<a: uint<1>>, out %b : !firrtl.bundle<a flip: uint<1>>) {
  // expected-error @+1 {{type mismatch}}
  firrtl.connect %b, %a : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a: uint<1>>
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.sint<1>) {
  // expected-error @+1 {{type mismatch}}
  firrtl.connect %b, %a : !firrtl.sint<1>, !firrtl.uint<1>
}
}

// -----

// Non-const types cannot be connected to const types.

firrtl.circuit "test" {
firrtl.module @test(in %a : !firrtl.uint<1>, out %b : !firrtl.const.uint<1>) {
  // expected-error @+1 {{type mismatch}}
  firrtl.connect %b, %a : !firrtl.const.uint<1>, !firrtl.uint<1>
}
}

// -----

// Non-const aggregates cannot be connected to const types.

firrtl.circuit "test" {
firrtl.module @test(in %in   : !firrtl.bundle<a: uint<1>>,
                    out %out : !firrtl.const.bundle<a: uint<1>>) {
  // expected-error @+1 {{type mismatch}}
  firrtl.connect %out, %in : !firrtl.const.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
}
}

// -----

/// Const flip types cannot be connected to non-const flip types.

firrtl.circuit "test" {
firrtl.module @test(in %in   : !firrtl.const.bundle<a flip: uint<1>>,
                    out %out : !firrtl.bundle<a flip: uint<1>>) {
  // expected-error @+1 {{type mismatch}}
  firrtl.connect %out, %in : !firrtl.bundle<a flip: uint<1>>, !firrtl.const.bundle<a flip: uint<1>>
}
}

// -----

// Nested const flip types cannot be connected to non-const flip types.

firrtl.circuit "test" {
firrtl.module @test(in %in   : !firrtl.bundle<a flip: const.uint<1>>,
                    out %out : !firrtl.bundle<a flip: uint<1>>) {
  // expected-error @+1 {{type mismatch}}
  firrtl.connect %out, %in : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: const.uint<1>>
}
}

// -----

/// Non-const double flip types cannot be connected to const.

firrtl.circuit "test" {
firrtl.module @test(in %in   : !firrtl.bundle<a flip: bundle<a flip: uint<1>>>,
                    out %out : !firrtl.const.bundle<a flip: bundle<a flip: uint<1>>>) {
  // expected-error @+1 {{type mismatch}}
  firrtl.connect %out, %in : !firrtl.const.bundle<a flip: bundle<a flip: uint<1>>>, 
                             !firrtl.bundle<a flip: bundle<a flip: uint<1>>>
}
}

// -----

// Test that non-const subaccess of a const vector disallows assignment.
firrtl.circuit "test" {
firrtl.module @test(in %index: !firrtl.uint<1>, out %out: !firrtl.const.vector<uint<1>, 1>) {
  %c = firrtl.constant 0 : !firrtl.uint<1>
  %d = firrtl.subaccess %out[%index] : !firrtl.const.vector<uint<1>, 1>, !firrtl.uint<1>
  // expected-error @+1 {{assignment to non-'const' subaccess of 'const' type is disallowed}}
  firrtl.strictconnect %d, %c : !firrtl.uint<1>
}
}

// -----

// Test that non-const subaccess of a const vector disallows assignment, even if the source is const.
firrtl.circuit "test" {
firrtl.module @test(in %index: !firrtl.uint<1>, out %out: !firrtl.const.vector<uint<1>, 1>) {
  %c = firrtl.constant 0 : !firrtl.const.uint<1>
  %d = firrtl.subaccess %out[%index] : !firrtl.const.vector<uint<1>, 1>, !firrtl.uint<1>
  // expected-error @+1 {{assignment to non-'const' subaccess of 'const' type is disallowed}}
  firrtl.connect %d, %c : !firrtl.uint<1>, !firrtl.const.uint<1>
}
}

// -----

// Test that non-const subaccess of a flipped const vector disallows assignment.
firrtl.circuit "test" {
firrtl.module @test(in %index: !firrtl.uint<1>, in %in: !firrtl.const.vector<bundle<a flip: uint<1>>, 1>, out %out: !firrtl.bundle<a flip: uint<1>>) {
  %element = firrtl.subaccess %in[%index] : !firrtl.const.vector<bundle<a flip: uint<1>>, 1>, !firrtl.uint<1>
  // expected-error @+1 {{assignment to non-'const' subaccess of 'const' type is disallowed}}
  firrtl.connect %out, %element : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>
}
}

// -----

// Test that non-const subaccess of a flipped const vector disallows assignment, even if the source is const.
firrtl.circuit "test" {
firrtl.module @test(in %index: !firrtl.uint<1>, in %in: !firrtl.const.vector<bundle<a flip: uint<1>>, 1>, out %out: !firrtl.bundle<a flip: const.uint<1>>) {
  %element = firrtl.subaccess %in[%index] : !firrtl.const.vector<bundle<a flip: uint<1>>, 1>, !firrtl.uint<1>
  // expected-error @+1 {{assignment to non-'const' subaccess of 'const' type is disallowed}}
  firrtl.connect %out, %element : !firrtl.bundle<a flip: const.uint<1>>, !firrtl.bundle<a flip: uint<1>>
}
}

// -----
firrtl.circuit "test" {
firrtl.module @test(in %p: !firrtl.uint<1>, in %in: !firrtl.const.uint<2>, out %out: !firrtl.const.uint<2>) {
  firrtl.when %p : !firrtl.uint<1> {
    // expected-error @+1 {{assignment to 'const' type '!firrtl.const.uint<2>' is dependent on a non-'const' condition}}
    firrtl.connect %out, %in : !firrtl.const.uint<2>, !firrtl.const.uint<2>
  }
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %p: !firrtl.uint<1>, in %in: !firrtl.const.uint<2>, out %out: !firrtl.const.uint<2>) {
  firrtl.when %p : !firrtl.uint<1> {
  } else {
    // expected-error @+1 {{assignment to 'const' type '!firrtl.const.uint<2>' is dependent on a non-'const' condition}}
    firrtl.connect %out, %in : !firrtl.const.uint<2>, !firrtl.const.uint<2>
  }
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %constP: !firrtl.const.uint<1>, in %p: !firrtl.uint<1>, in %in: !firrtl.const.uint<2>, out %out: !firrtl.const.uint<2>) {
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.when %constP : !firrtl.const.uint<1> {
      // expected-error @+1 {{assignment to 'const' type '!firrtl.const.uint<2>' is dependent on a non-'const' condition}}
      firrtl.connect %out, %in : !firrtl.const.uint<2>, !firrtl.const.uint<2>
    }
  }
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %p: !firrtl.uint<1>, in %in: !firrtl.bundle<a: const.uint<2>>, out %out: !firrtl.bundle<a: const.uint<2>>) {
  firrtl.when %p : !firrtl.uint<1> {
    // expected-error @+1 {{assignment to nested 'const' member of type '!firrtl.bundle<a: const.uint<2>>' is dependent on a non-'const' condition}}
    firrtl.connect %out, %in : !firrtl.bundle<a: const.uint<2>>, !firrtl.bundle<a: const.uint<2>>
  }
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %p: !firrtl.uint<1>, in %in: !firrtl.const.bundle<a flip: uint<2>>, out %out: !firrtl.const.bundle<a flip: uint<2>>) {
  firrtl.when %p : !firrtl.uint<1> {
    // expected-error @+1 {{assignment to 'const' type '!firrtl.const.bundle<a flip: uint<2>>' is dependent on a non-'const' condition}}
    firrtl.connect %out, %in : !firrtl.const.bundle<a flip: uint<2>>, !firrtl.const.bundle<a flip: uint<2>>
  }
}
}

// -----

firrtl.circuit "test" {
firrtl.module @test(in %p: !firrtl.uint<1>, in %in: !firrtl.bundle<a flip: const.uint<2>>, out %out: !firrtl.bundle<a flip: const.uint<2>>) {
  firrtl.when %p : !firrtl.uint<1> {
    // expected-error @+1 {{assignment to nested 'const' member of type '!firrtl.bundle<a flip: const.uint<2>>' is dependent on a non-'const' condition}}
    firrtl.connect %out, %in : !firrtl.bundle<a flip: const.uint<2>>, !firrtl.bundle<a flip: const.uint<2>>
  }
}
}

// -----

// Test that the declaration location of the bundle containing the field is checked.
firrtl.circuit "test" {
firrtl.module @test(in %p: !firrtl.uint<1>, out %out: !firrtl.const.bundle<a: uint<1>>) {
  firrtl.when %p : !firrtl.uint<1> {
    %f = firrtl.subfield %out[a] : !firrtl.const.bundle<a: uint<1>>
    %c = firrtl.constant 0 : !firrtl.const.uint<1>
    // expected-error @+1 {{assignment to 'const' type '!firrtl.const.uint<1>' is dependent on a non-'const' condition}}
    firrtl.strictconnect %f, %c : !firrtl.const.uint<1>
  }
}
}

// -----

// Test that the declaration location of the vector containing the field is checked.
firrtl.circuit "test" {
firrtl.module @test(in %p: !firrtl.uint<1>, out %out: !firrtl.const.vector<uint<1>, 1>) {
  firrtl.when %p : !firrtl.uint<1> {
    %e = firrtl.subindex %out[0] : !firrtl.const.vector<uint<1>, 1>
    %c = firrtl.constant 0 : !firrtl.const.uint<1>
    // expected-error @+1 {{assignment to 'const' type '!firrtl.const.uint<1>' is dependent on a non-'const' condition}}
    firrtl.strictconnect %e, %c : !firrtl.const.uint<1>
  }
}
}
