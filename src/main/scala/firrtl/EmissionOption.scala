// SPDX-License-Identifier: Apache-2.0

package firrtl

/**
  * Base type for emission customization options
  * NOTE: all the following traits must be mixed with SingleTargetAnnotation[T <: Named]
  *       in order to be taken into account in the Emitter
  */
trait EmissionOption

/** Emission customization options for memories */
trait MemoryEmissionOption extends EmissionOption {
  def initValue: MemoryInitValue = MemoryRandomInit
}

sealed trait MemoryInitValue
case object MemoryRandomInit extends MemoryInitValue
case class MemoryScalarInit(value: BigInt) extends MemoryInitValue
case class MemoryArrayInit(values: Seq[BigInt]) extends MemoryInitValue

/** default Emitter behavior for memories */
case object MemoryEmissionOptionDefault extends MemoryEmissionOption

/** Emission customization options for registers */
trait RegisterEmissionOption extends EmissionOption {

  /** when true the reset init value will be used to emit a bitstream preset */
  def useInitAsPreset: Boolean = false

  /** when true the initial randomization is disabled for this register */
  def disableRandomization: Boolean = false
}

/** default Emitter behavior for registers */
case object RegisterEmissionOptionDefault extends RegisterEmissionOption

/** Emission customization options for IO ports */
trait PortEmissionOption extends EmissionOption

/** default Emitter behavior for IO ports */
case object PortEmissionOptionDefault extends PortEmissionOption

/** Emission customization options for wires */
trait WireEmissionOption extends EmissionOption

/** default Emitter behavior for wires */
case object WireEmissionOptionDefault extends WireEmissionOption

/** Emission customization options for nodes */
trait NodeEmissionOption extends EmissionOption

/** default Emitter behavior for nodes */
case object NodeEmissionOptionDefault extends NodeEmissionOption

/** Emission customization options for connect */
trait ConnectEmissionOption extends EmissionOption

/** default Emitter behavior for connect */
case object ConnectEmissionOptionDefault extends ConnectEmissionOption
