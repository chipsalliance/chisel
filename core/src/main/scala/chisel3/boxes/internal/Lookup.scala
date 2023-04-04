package chisel3.boxes.internal

import chisel3.Data
import chisel3.experimental.BaseModule
import chisel3.internal.{Context, CloneToContext, HasId}

trait Lookup[X, H <: HasId] {
  type R
  def contextualize(item: R, context: Context): R
  def toWrapped(item: => X, from: H): R
}
object Lookup {
  implicit def lookupInt[H <: HasId] = new Lookup[Int, H] {
    type R = Int
    def toWrapped(item: => Int, from: H): Int = item
    def contextualize(item: Int, context: Context): Int = item
  }
  //implicit def lookupType[T <: Data, H <: HasId] = new Lookup[Type[T], H] {
  //  type R = Type[T]
  //  def toWrapped(item: => Type[T], from: H): Type[T] = item
  //  def contextualize(item: Type[T], context: Context): Type[T] = {
  //    // Previously, would contextualize Type, but I think now we should just return it
  //    // context.contextualize(item.context, c => Type(item.proto, c))
  //    item
  //  }
  //}
  //implicit def lookupField[T <: Data, H <: HasId] = new Lookup[Field[T], H] {
  //  type R = Field[T]
  //  def toWrapped(item: => Field[T], from: H): Field[T] = item
  //  def contextualize(item: Field[T], context: Context): Field[T] = item
  //}
  implicit def lookupInstance[T <: Module, H <: HasId] = new Lookup[Instance[T], H] {
    type R = Instance[T]
    def toWrapped(item: => Instance[T], from: H): Instance[T] = item
    def contextualize(item: Instance[T], context: Context): Instance[T] = context.contextualize(item.context, c => Instance(item.proto, c))
  }
  //implicit def lookupPort[T <: Data, H <: HasId] = new Lookup[Port[T], H] {
  //  type R = Port[T]
  //  def toWrapped(item: => Port[T], from: H): Port[T] = item
  //  def contextualize(item: Port[T], context: Context): Port[T] = context.contextualize(item.context, c => Port(item.tpe, c, !item.outgoing, item.isDefinition || c.isTemplate, !item.isWritable))
  //}
  //implicit def lookupFieldFromPort[T <: Data, H <: Port[_]] = new Lookup[Field[T], H] {
  //  type R = Port[T]
  //  def toWrapped(item: => Field[T], from: H): Port[T] = Port(item, item.context, item.aligned ^ from.outgoing, false, from.isWritable ^ !item.aligned)
  //  def contextualize(item: Port[T], context: Context): Port[T] = context.contextualize(item.context, c => Port(item.tpe, c, item.outgoing, item.isDefinition || c.isTemplate, item.isWritable))
  //}
}


