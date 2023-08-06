// RUN: circt-opt %s --split-input-file --verify-diagnostics


handshake.func @invalid_operand_count(%data:f32, %staddress: index, %ldaddress: index) -> (f32, none, none) {
  // expected-error @+1 {{number of operands 3 does not match number expected of 2 with 1 address inputs per port}}
  %0:3 = memory [ld = 0, st = 1](%data, %staddress, %ldaddress) {id = 0 : i32, lsq = false} :  memref<10xf32>, (f32, index, index) -> (f32, none, none)
  return %0#0, %0#1, %0#2 : f32, none, none
}

// -----

handshake.func @invalid_result_count(%data:f32, %staddress: index, %ldaddress: index) -> (f32, none) {
  // expected-error @+1 {{number of results 2 does not match number expected of 3 with 1 address inputs per port}}
  %0:2 = memory [ld = 1, st = 1](%data, %staddress, %ldaddress) {id = 0 : i32, lsq = false} : memref<10xf32>, (f32, index, index) -> (f32, none)
  return %0#0, %0#1 : f32, none
}

// -----

handshake.func @invalid_address_type(%data:f32, %staddress: f32, %ldaddress: index) -> (f32, none, none) {
  // expected-error @+1 {{address type for load port 0:'index' doesn't match address type 'f32'}}
  %0:3 = memory [ld = 1, st = 1](%data, %staddress, %ldaddress) {id = 0 : i32, lsq = false} : memref<10xf32>, (f32, f32, index) -> (f32, none, none)
  return %0#0, %0#1, %0#2 : f32, none, none
}

// -----

handshake.func @invalid_store_data_type(%data:i32, %staddress: index, %ldaddress: index) -> (f32, none, none) {
  // expected-error @+1 {{data type for store port 0:'i32' doesn't match memory type 'f32'}}
  %0:3 = memory [ld = 1, st = 1](%data, %staddress, %ldaddress) {id = 0 : i32, lsq = false} : memref<10xf32>, (i32, index, index) -> (f32, none, none)
  return %0#0, %0#1, %0#2 : f32, none, none
}

// -----

handshake.func @invalid_load_data_type(%data:f32, %staddress: index, %ldaddress: index) -> (f32, none, none) {
  // expected-error @+1 {{data type for load port 0:'i32' doesn't match memory type 'f32'}}
  %0:3 = memory[ld = 1, st = 1](%data, %staddress, %ldaddress) {id = 0 : i32, lsq = false} : memref<10xf32>, (f32, index, index) -> (i32, none, none)
  return %0#0, %0#1, %0#2 : i32, none, none
}
