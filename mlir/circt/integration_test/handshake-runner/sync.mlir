// RUN: handshake-runner %s | FileCheck %s
// CHECK: 0 42

handshake.func @main(%ctrl: none) -> (i64, i64, none) {
  %ctrlF:3 = fork [3] %ctrl : none
  %c0 = constant %ctrlF#0 {value = 0 : i64} : i64
  %c42 = constant %ctrlF#1 {value = 42 : i64} : i64
  %out:3 = sync %c0, %c42, %ctrlF#2 : i64, i64, none
  return %out#0, %out#1, %out#2 : i64, i64, none
}
