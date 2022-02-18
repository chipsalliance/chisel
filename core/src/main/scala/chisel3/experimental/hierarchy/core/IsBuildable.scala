package chisel3.experimental.hierarchy.core

trait Buildable[T] {
  def apply(f: => T): Underlying[T]
}

trait Stampable[T] {
  def apply(definition: Definition[T]): Underlying[T]
}