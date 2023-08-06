// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: systemc.cpp.func externC @basic(%name0: i32, %name1: i1) {
systemc.cpp.func externC @basic(%name0: i32, %name1: i1) {
  // CHECK-NEXT: systemc.cpp.return
  systemc.cpp.return
}

// CHECK-LABEL: systemc.cpp.func private @foo(i32)
systemc.cpp.func private @foo(i32)

// CHECK-LABEL: systemc.cpp.func private @foo2(i32, i32) -> i32
systemc.cpp.func private @foo2(i32, i32) -> i32

// CHECK-LABEL: systemc.cpp.func externC private @foobar(%abc: i32, %func1: (i32) -> (), %func2: (i32, i32) -> i32) -> i32 {
systemc.cpp.func externC private @foobar(%abc: i32, %func1: (i32) -> (), %func2 : (i32, i32) -> i32) -> i32 {
  // CHECK-NEXT: systemc.cpp.call @foo(%abc) : (i32) -> ()
  systemc.cpp.call @foo(%abc) : (i32) -> ()
  // CHECK-NEXT: {{%.+}} = systemc.cpp.call @foo2(%abc, %abc) : (i32, i32) -> i32
  %0 = systemc.cpp.call @foo2(%abc, %abc) : (i32, i32) -> i32
  // CHECK-NEXT: systemc.cpp.call_indirect %func1(%abc) : (i32) -> ()
  systemc.cpp.call_indirect %func1(%abc) : (i32) -> ()
  // CHECK-NEXT: {{%.+}} = systemc.cpp.call_indirect %func2(%abc, %abc) : (i32, i32) -> i32
  %1 = systemc.cpp.call_indirect %func2(%abc, %abc) : (i32, i32) -> i32
  // CHECK-NEXT: systemc.cpp.return %abc : i32
  systemc.cpp.return %abc : i32
}
