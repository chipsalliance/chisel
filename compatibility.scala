package Chisel

package object iotesters {
  type PeekPokeTester[+T <: chisel3.Module] = chisel3.iotesters.PeekPokeTester[T]
}
