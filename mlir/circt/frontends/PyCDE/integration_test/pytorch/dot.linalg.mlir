func.func @forward(%arg0: tensor<5xi32>, %arg1: tensor<5xi32>) -> tensor<i32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<i32>
  %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<i32>) -> tensor<i32>
  %2 = linalg.dot ins(%arg0, %arg1 : tensor<5xi32>, tensor<5xi32>) outs(%1 : tensor<i32>) -> tensor<i32>
  return %2 : tensor<i32>
}
