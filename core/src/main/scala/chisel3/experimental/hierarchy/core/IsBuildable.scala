package chisel3.experimental.hierarchy.core

trait Buildable[T] {
  def apply(f: => T): Proxy[T]
}

trait Stampable[T] {
  def apply(definition: Definition[T]): Proxy[T]
}