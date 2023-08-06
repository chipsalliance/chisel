// RUN: circt-opt %s --split-input-file --verify-diagnostics


handshake.func @invalid_merge_like_wrong_type(%arg0: i1, %arg1: i32, %arg2: i64) { // expected-note {{prior use here}}
  // expected-error @+1 {{use of value '%arg2' expects different type than prior uses: 'i32' vs 'i64'}}
  %0 = mux %arg0 [%arg1, %arg2] : i1, i32
  return %0 : i32
}

// -----

handshake.func @invalid_mux_unsupported_select(%arg0: tensor<i1>, %arg1: i32, %arg2: i32) {
  // expected-error @+1 {{unsupported type for indexing value: 'tensor<i1>'}}
  %0 = mux %arg0 [%arg1, %arg2] : tensor<i1>, i32
  return %0 : i32
}

// -----

handshake.func @invalid_mux_narrow_select(%arg0: i1, %arg1: i32, %arg2: i32, %arg3: i32) {
  // expected-error @+1 {{bitwidth of indexing value is 1, which can index into 2 operands, but found 3 operands}}
  %0 = mux %arg0 [%arg1, %arg2, %arg3] : i1, i32
  return %0 : i32
}

// -----

handshake.func @invalid_cmerge_unsupported_index(%arg1: i32, %arg2: i32) -> tensor<i1> {
  // expected-error @below {{unsupported type for indexing value: 'tensor<i1>'}}
  %result, %index = control_merge %arg1, %arg2 : i32, tensor<i1>
  return %index : tensor<i1>
}

// -----

handshake.func @invalid_cmerge_narrow_index(%arg1: i32, %arg2: i32, %arg3: i32) -> i1 {
  // expected-error @below {{bitwidth of indexing value is 1, which can index into 2 operands, but found 3 operands}}
  %result, %index = control_merge %arg1, %arg2, %arg3 : i32, i1
  return %index : i1
}

// -----

handshake.func @foo(%ctrl : none) -> none{
  return %ctrl : none
}

handshake.func @invalid_instance_op(%arg0 : i32, %ctrl : none) -> none {
  // expected-error @+1 {{'handshake.instance' op incorrect number of operands for the referenced handshake function}}
  instance @foo(%arg0, %ctrl) : (i32, none) -> ()
  return %ctrl : none
}

// -----

handshake.func @foo(%arg0 : i32, %ctrl : none) -> (none) {
  return %ctrl : none
}

handshake.func @invalid_instance_op(%arg0 : i64, %ctrl : none) -> none {
  // expected-error @+1 {{'handshake.instance' op operand type mismatch: expected operand type 'i32', but provided 'i64' for operand number 0}}
  instance @foo(%arg0, %ctrl) : (i64, none) -> (none)
  return %ctrl : none
}

// -----

handshake.func @foo(%ctrl : none) -> (i32, none) {
  %0 = constant %ctrl {value = 1 : i32} : i32
  return %0, %ctrl : i32, none
}

handshake.func @invalid_instance_op(%arg0 : i32, %ctrl : none) -> none {
  // expected-error @+1 {{'handshake.instance' op incorrect number of results for the referenced handshake function}}
  instance @foo(%ctrl) : (none) -> (none)
  return %ctrl : none
}

// -----

handshake.func @foo(%ctrl : none) -> (i32, none) {
  %0 = constant %ctrl {value = 1 : i32} : i32
  return %0, %ctrl : i32, none
}

handshake.func @invalid_instance_op(%arg0 : i32, %ctrl : none) -> none {
  // expected-error @+1 {{'handshake.instance' op result type mismatch: expected result type 'i32', but provided 'i64' for result number 0}}
  %0, %outCtrl = instance @foo(%ctrl) : (none) -> (i64, none)
  return %ctrl : none
}

// -----

func.func @foo() {
  return
}

handshake.func @invalid_instance_op(%ctrl : none) -> none {
  // expected-error @+1 {{'handshake.instance' op 'foo' does not reference a valid handshake function}}
  instance @foo(%ctrl) : (none) -> (none)
  return %ctrl : none
}

// -----

handshake.func @invalid_multidim_memory(%ctrl : none) -> none {
  // expected-error @+1 {{'handshake.memory' op memref must have only a single dimension.}}
  memory [ld = 0, st = 0] () {id = 0 : i32, lsq = false} : memref<10x10xi8>, () -> ()
  return %ctrl : none
}

// -----

handshake.func @invalid_missing_lsq(%ctrl : none) -> none {
  // expected-error @+1 {{'handshake.memory' op requires attribute 'lsq'}}
  memory [ld = 0, st = 0] () {id = 0 : i32} : memref<10xi8>, () -> ()
  return %ctrl : none
}

// -----

handshake.func @invalid_dynamic_memory(%ctrl : none) -> none {
  // expected-error @+1 {{'handshake.memory' op memref dimensions for handshake.memory must be static.}}
  memory [ld = 0, st = 0] () {id = 0 : i32, lsq = false} : memref<?xi8>, () -> ()
  return %ctrl : none
}

// -----

// expected-error @+1 {{'handshake.func' op attribute 'argNames' has 2 entries but is expected to have 3.}}
handshake.func @invalid_num_argnames(%a : i32, %b : i32, %c : none) -> none attributes {argNames = ["a", "b"]} {
  return %c : none
}

// -----

// expected-error @+1 {{'handshake.func' op expected all entries in attribute 'argNames' to be strings.}}
handshake.func @invalid_type_argnames(%a : i32, %b : none) -> none attributes {argNames = ["a", 2 : i32]} {
  return %b : none
}

// -----

// expected-error @+1 {{'handshake.func' op attribute 'resNames' has 1 entries but is expected to have 2.}}
handshake.func @invalid_num_resnames(%a : i32, %b : i32, %c : none) -> (i32, none) attributes {resNames = ["a"]} {
  return %a, %c : i32, none
}

// -----

// expected-error @+1 {{'handshake.func' op expected all entries in attribute 'resNames' to be strings.}}
handshake.func @invalid_type_resnames(%a : i32, %b : none) -> none attributes {resNames = [2 : i32]} {
  return %b : none
}

// -----

handshake.func @invalid_constant_value(%ctrl : none) -> none {
  // expected-error @+1 {{'handshake.constant' op constant value type 'i31' differs from operation result type 'i32'}}
  %0 = constant %ctrl {value = 1 : i31} : i32
  return %ctrl : none
}

// -----

handshake.func @invalid_buffer_init1(%arg0 : i32, %ctrl : none) -> (i32, none) {
  // expected-error @+1 {{'handshake.buffer' op expected 2 init values but got 1.}}
  %0 = buffer [2] seq %arg0 {initValues = [1]} : i32
  return %0, %ctrl : i32, none
}

// -----

handshake.func @invalid_buffer_init2(%arg0 : i32, %ctrl : none) -> (i32, none) {
  // expected-error @+1 {{'handshake.buffer' op only bufferType buffers are allowed to have initial values.}}
  %0 = buffer [1] fifo %arg0 {initValues = [1]} : i32
  return %0, %ctrl : i32, none
}

// -----

handshake.func @invalid_buffer_init3(%arg0 : i32, %ctrl : none) -> (i32, none) {
  // expected-error @+2 {{'handshake.buffer' expected valid keyword}}
  // expected-error @+1 {{'handshake.buffer' failed to parse BufferTypeEnumAttr parameter 'value' which is to be a `::BufferTypeEnum`}}
  %0 = buffer [1]  %arg0 {initValues = [1]} : i32
  return %0, %ctrl : i32, none
}

// -----

handshake.func @invalid_buffer_init4(%arg0 : i32, %ctrl : none) -> (i32, none) {
  // expected-error @+2 {{'handshake.buffer' expected ::BufferTypeEnum to be one of: seq, fifo}}
  // expected-error @+1 {{'handshake.buffer' failed to parse BufferTypeEnumAttr parameter 'value' which is to be a `::BufferTypeEnum`}}
  %0 = buffer [1] SEQ %arg0 {initValues = [1]} : i32
  return %0, %ctrl : i32, none
}

// -----

handshake.func @invalid_unpack_types_not_matching(%arg0 : tuple<i64, i32>, %ctrl : none) -> (i64, none) {
  // expected-error @+1 {{'handshake.unpack' op failed to verify that result types match element types of 'tuple'}}
  %0:2 = "handshake.unpack"(%arg0) : (tuple<i64, i32>) -> (i64, i64)
  return %0#0, %ctrl : i64, none
}

// -----

handshake.func @invalid_pack_types_not_matching(%arg0 : i32, %arg1 : i64, %ctrl : none) -> (tuple<i64, i32>, none) {
  // expected-error @+1 {{'handshake.pack' op failed to verify that input types match element types of 'tuple'}}
  %0 = "handshake.pack"(%arg0, %arg1) : (i32, i64) -> tuple<i64, i32>
  return %0#0, %ctrl : tuple<i64, i32>, none
}

// -----

handshake.func @invalid_unpack_type(%arg0 : i64, %ctrl : none) -> (i64, none) {
  // expected-note @-1 {{prior use here}}
  // expected-error @+1 {{use of value '%arg0' expects different type than prior uses: 'tuple<i64>' vs 'i64'}}
  %0 = handshake.unpack %arg0 : tuple<i64>
  return %0, %ctrl : i64, none
}

// -----

handshake.func @invalid_unpack_wrong_types(%arg0 : tuple<i64>, %ctrl : none) -> (i64, none) {
  // expected-error @+1 {{'handshake.unpack' invalid kind of type specified}}
  %0 = handshake.unpack %arg0 : i64
  return %0, %ctrl : i64, none
}

// -----

handshake.func @invalid_pack_wrong_types(%arg0 : i64, %arg1 : i32, %ctrl : none) -> (tuple<i64, i32>, none) {
  // expected-error @+1 {{'handshake.pack' invalid kind of type specified}}
  %0 = handshake.pack %arg0, %arg1 : i64, i32
  return %0, %ctrl : tuple<i64, i32>, none
}

// -----

handshake.func @invalid_memref_block_arg(%arg0 : memref<2xi64>, %ctrl : none) -> none {
  // expected-error @-1 {{'handshake.func' op expected that block argument #0 is used by an 'extmemory' operation}}
  return %ctrl : none
}

// -----

handshake.func @invalid_sost_op_zero_size(%ctrl : none) -> (none, none) {
  // expected-error @+1 {{'handshake.merge' op must have at least one data operand}}
  %0 = merge : none
  return %0, %ctrl : none, none
}

// -----

handshake.func @invalid_sost_op_wrong_operands(%arg0 : i64, %arg1 : i32, %ctrl : none) -> (i64, none) { // expected-note {{prior use here}}
  // expected-error @+1 {{use of value '%arg1' expects different type than prior uses: 'i64' vs 'i32'}}
  %0, %1 = control_merge %arg0, %arg1 : i64, index
  return %0, %ctrl : i64, none
}

