// SPDX-License-Identifier: Apache-2.0

package chisel3

import firrtl.annotations.Annotation
import firrtl.ir.{CircuitWithAnnos, Serializer}
import firrtl.options.{CustomFileEmission, Unserializable}
import chisel3.experimental.{BaseModule, SourceInfo, UnlocatableSourceInfo}
import chisel3.experimental.hierarchy.Definition
import chisel3.internal.firrtl.ir.Circuit
import chisel3.internal.firrtl.Converter

/** The result of running Chisel elaboration
  *
  * Provides limited APIs for inspection of the resulting circuit.
  */
// This is an important interface so let's keep it separate from the implementation.
sealed trait ElaboratedCircuit {

  /** The name of the circuit, also the name of the top public module */
  def name: String

  /** The circuit and annotations as a string of FIRRTL IR
    *
    * This will include annotations passed to Chisel to build the circuit and those created during elaboration.
    * For large circuits (> 2 GiB of text) use [[lazilySerialize]].
    */
  def serialize: String

  /** The circuit and annotations as a string of FIRRTL IR
    *
    * For large circuits (> 2 GiB of text) use [[lazilySerialize]]
    *
    * @param annotations annotations to include in the FIRRTL IR. No other annotations will be included.
    */
  def serialize(annotations: Iterable[Annotation]): String

  /** The circuit and annotations as a lazy buffer of strings of FIRRTL IR
    *
    * This will include annotations passed to Chisel to build the circuit and those created during elaboration.
    * Serialized lazily to reduce peak memory use and support cicuits larger than 2 GiB.
    */
  def lazilySerialize: Iterable[String]

  /** The circuit and annotations as a lazy buffer of strings of FIRRTL IR
    *
    * Serialized lazily to reduce peak memory use and support cicuits larger than 2 GiB.
    *
    * @param annotations annotations to include in the FIRRTL IR. No other annotations will be included.
    */
  def lazilySerialize(annotations: Iterable[Annotation]): Iterable[String]

  /** The annotations created during elaboration of this circuit
    *
    * This does not include annotations passed to elaboration.
    */
  def annotations: Iterable[Annotation]

  /** The Definition of the top module in the elaborated circuit */
  def topDefinition: Definition[BaseModule]

  /** The underlying circuit, for private use only */
  private[chisel3] def _circuit: Circuit
}

private class ElaboratedCircuitImpl(circuit: Circuit, initialAnnotations: Seq[Annotation]) extends ElaboratedCircuit {

  // Source locator needed for toDefinition
  private implicit def sourceInfo: SourceInfo = UnlocatableSourceInfo

  override def name: String = circuit.name

  override def serialize: String = lazilySerialize.mkString

  override def serialize(annotations: Iterable[Annotation]): String = lazilySerialize(annotations).mkString

  override def lazilySerialize: Iterable[String] = {
    val annotations = (initialAnnotations.view ++ circuit.firrtlAnnotations).flatMap {
      case _: Unserializable     => None
      case _: CustomFileEmission => None
      case a => Some(a)
    }.toVector
    lazilySerialize(annotations)
  }

  override def lazilySerialize(annotations: Iterable[Annotation]): Iterable[String] = {
    val prelude = {
      val dummyCircuit = circuit.copy(components = Nil)
      val converted = Converter.convert(dummyCircuit)
      val withAnnos = CircuitWithAnnos(converted, annotations.toVector)
      Serializer.lazily(withAnnos)
    }
    val typeAliases: Seq[String] = circuit.typeAliases.map(_.name)
    val modules = circuit.components.iterator.map(c => Converter.convert(c, typeAliases))
    val moduleStrings = modules.flatMap { m =>
      Serializer.lazily(m, 1) ++ Seq("\n\n")
    }
    prelude ++ moduleStrings
  }

  override def annotations: Iterable[Annotation] = circuit.firrtlAnnotations

  // TODO come up with a better way to figure this out than "last"
  override def topDefinition: Definition[BaseModule] = circuit.components.last.id.toDefinition

  private[chisel3] override def _circuit: Circuit = circuit
}

object ElaboratedCircuit {
  private[chisel3] def apply(circuit: Circuit, initialAnnotations: Seq[Annotation]): ElaboratedCircuit =
    new ElaboratedCircuitImpl(circuit, initialAnnotations)
}
