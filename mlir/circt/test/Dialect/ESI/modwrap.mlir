// RUN: esi-tester %s --test-mod-wrap | FileCheck %s

hw.module.extern @OutputChannel(%clk: i1, %bar_ready: i1) -> (bar: i42, bar_valid: i1)

// CHECK-LABEL:  hw.module @OutputChannel_esi(%clk: i1) -> (bar: !esi.channel<i42>) {
// CHECK:          %chanOutput, %ready = esi.wrap.vr %pearl.bar, %pearl.bar_valid : i42
// CHECK:          %pearl.bar, %pearl.bar_valid = hw.instance "pearl" @OutputChannel(clk: %clk: i1, bar_ready: %ready: i1) -> (bar: i42, bar_valid: i1)
// CHECK:          hw.output %chanOutput : !esi.channel<i42>

hw.module.extern @InputChannel(%clk: i1, %foo_data: i23, %foo_valid: i1) -> (foo_ready: i1, rawOut: i99)

// CHECK-LABEL:  hw.module @InputChannel_esi(%clk: i1, %foo_data: !esi.channel<i23>) -> (rawOut: i99) {
// CHECK:          %rawOutput, %valid = esi.unwrap.vr %foo_data, %pearl.foo_ready : i23
// CHECK:          %pearl.foo_ready, %pearl.rawOut = hw.instance "pearl" @InputChannel(clk: %clk: i1, foo_data: %rawOutput: i23, foo_valid: %valid: i1) -> (foo_ready: i1, rawOut: i99)
// CHECK:          hw.output %pearl.rawOut : i99

hw.module.extern @Mixed(%clk: i1, %foo: i23, %foo_valid: i1, %bar_ready: i1) ->
                          (bar: i42, bar_valid: i1, foo_ready: i1, rawOut: i99)

// CHECK-LABEL:  hw.module @Mixed_esi(%clk: i1, %foo: !esi.channel<i23>) -> (bar: !esi.channel<i42>, rawOut: i99) {
// CHECK:          %rawOutput, %valid = esi.unwrap.vr %foo, %pearl.foo_ready : i23
// CHECK:          %chanOutput, %ready = esi.wrap.vr %pearl.bar, %pearl.bar_valid : i42
// CHECK:          %pearl.bar, %pearl.bar_valid, %pearl.foo_ready, %pearl.rawOut = hw.instance "pearl" @Mixed(clk: %clk: i1, foo: %rawOutput: i23, foo_valid: %valid: i1, bar_ready: %ready: i1) -> (bar: i42, bar_valid: i1, foo_ready: i1, rawOut: i99)
// CHECK:          hw.output %chanOutput, %pearl.rawOut : !esi.channel<i42>, i99
