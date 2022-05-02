package chisel3.experimental.hierarchy.core

import java.util.IdentityHashMap

trait Underlying[+T] {
  def proto: T
  def get[X](query: Query[T, X]): X
}

case class Raw[+T](proto: T) extends Underlying[T] {
  def get[X](query: Query[T, X]): X = query match {
    case f: FunctionQuery[T, X] => f.func(proto)
    case n: NameQuery[T, X] => throw new Exception("Cannot have a name query on a raw underlying")
  }
}

case class Freezable[+T](proto: T, nameToFunc: Map[String, Any => Any]){
  def get[X](query: Query[T, X]): X = query match {
    case f: FunctionQuery[T, X] => f.func(proto)
    case NameQuery(name)     => nameToFunc(name)(proto).asInstanceOf[X]
  }
}

case class Dethawed[+T](nameToValue: Map[String, Any]) extends Underlying[T] {
  def proto: T = ???
  def get[X](query: Query[T, X]): X = query match {
    case FunctionQuery(func) => throw new Exception("Cannot have a function query on a dethawed underlying")
    case NameQuery(name)     => nameToValue(name).asInstanceOf[X]
  }
}

trait ToUnderlying[T] {
  def toUnderlying(proto: T): Underlying[T]
}

case class ToRaw[T]() extends ToUnderlying[T] {
  def toUnderlying(proto: T): Raw[T] = Raw(proto)
}

trait ToFreezable[T] {
  type X = T
  def toUnderlying(proto: X): Freezable[X]
}


trait Query[-T, X]
case class FunctionQuery[-T, X](func: T => X) extends Query[T, X]
case class NameQuery[-T, X](name: String) extends Query[T, X]