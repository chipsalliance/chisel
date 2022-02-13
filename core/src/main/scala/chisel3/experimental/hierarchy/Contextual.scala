package chisel3.experimental.hierarchy
import chisel3.internal.Builder


/** Contextual represent context-dependent values which can be passed around in various datastructures
  * Contextual values can only be accessed through calling @public vals from the Hierarchy[_]
  *
  * @param values
  */
case class Contextual[T, V] private[chisel3] (private[chisel3] val values: Seq[(Hierarchy[T], V)]) {
  // Only ever called in lookupable
  private[chisel3] def get(context: Hierarchy[T]): V = {
    val matchingValues = values.collect {
      //case (h: Hierarchy[T], value: V) if Contextual.viewableFrom(h, context) => value
      case (h: Hierarchy[T], value: V) => value
    }
    require(matchingValues.size == 1)
    matchingValues.head
  }
  //def ++ (other: Contextual[T, V]): Contextual[T, V] = Contextual(values ++ other.values)
}

object Contextual {
  //def collapse[T <: IsInstantiable, V](seq: Seq[Contextual[T, V]]): Contextual[T, V] = seq.foldLeft(Contextual.empty[T, V]) { case (agg, c) => agg ++ c }

  def apply[V](value: V) = {
    val currentModule = Builder.currentModule.get
    new Contextual(Seq(currentModule.toDefinition -> value))
  }
  def apply[I <: IsInstantiable, V](context: I, value: V) = new Contextual(Seq((context.toInstance, value)))
  def empty[T, V]: Contextual[T, V] = new Contextual(Seq.empty[(Hierarchy[T], V)])

  // This needs to be a derived typeclass w.r.t. the hierarchical type in Contextual
  def viewableFrom[T](h: Hierarchy[T], context: Hierarchy[T]): Boolean = ???
}

