// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.hierarchy.core.{Clone, IsClone, Proto}
import chisel3.experimental.hierarchy.{Definition, Instance, Instantiate}
import chisel3.experimental.{
  BaseModule,
  ExtModule,
  Generator,
  IntrinsicModule,
  Param,
  SerializableModule,
  SerializableModuleGenerator,
  SerializableModuleParameter,
  StringParam
}
import upickle.default.Writer

import scala.reflect.runtime.universe

/** A module or external module whose IO is generated from a specific generator.
  * This module may have no additional IO created other than what is specified
  * by its `ioGenerator` abstract member.
  */
sealed trait FixedIOBaseModule[A <: Data] extends BaseModule {

  /** A generator of IO */
  protected def ioGenerator: A

  final val io = FlatIO(ioGenerator)
  endIOCreation()

}

/** A Chisel module whose IO is determined by an IO generator.  This module
  * cannot have additional IO created by modules that extend it.
  *
  * @param ioGenerator
  */
class FixedIORawModule[A <: Data](final val ioGenerator: A) extends RawModule with FixedIOBaseModule[A]

/** A Chisel blackbox whose IO is determined by an IO generator.  This module
  * cannot have additional IO created by modules that extend it.
  *
  * @param ioGenerator
  * @param params
  */
class FixedIOExtModule[A <: Data](final val ioGenerator: A, params: Map[String, Param] = Map.empty[String, Param])
    extends ExtModule(params)
    with FixedIOBaseModule[A]

// TODO: pickle it.
trait SerializableData extends Data

/** Three ways to generate the FixedIOIntrinsicModuleParameter
  *  - directly construct it via IO, class name and parameter.
  *  - construct it via [[SerializableModule]] without IO defined, this will elaborate Module,
  *    but only get `io` from [[FixedIOBaseModule]], notice: after elaboration, CIRCT will elaborate it again.
  *  - construct it via [[SerializableModule]] with IO defined
  */
object FixedIOIntrinsicModuleParameter {

  /** generate a FixedIOIntrinsicModuleParameter unsafely. */
  def apply[D <: SerializableData](
    _generatorClass:     String,
    _generatorParameter: String,
    _io:                 D
  ): FixedIOIntrinsicModuleParameter[D] = new FixedIOIntrinsicModuleParameter[D] {
    val generatorClass:     String = _generatorClass
    val generatorParameter: String = _generatorParameter
    val io:                 D = _io
  }

  /** generate a FixedIOIntrinsicModuleParameter without elaborating a Definition. */
  def apply[
    D <: SerializableData,
    M <: SerializableModule[P] with FixedIOBaseModule[D],
    P <: SerializableModuleParameter
  ](serializableModuleGenerator: SerializableModuleGenerator[M, P],
    io:                          D
  )(
    implicit pRW: upickle.default.Writer[P]
  ): FixedIOIntrinsicModuleParameter[D] =
    apply[D](
      serializableModuleGenerator.generator.getName,
      upickle.default.write(serializableModuleGenerator.parameter),
      io
    )

  /** generate a FixedIOIntrinsicModuleParameter by elaborating a Definition.
    * TODO: implement me!
    */
  def apply[
    D <: SerializableData,
    M <: SerializableModule[P] with FixedIOBaseModule[D],
    P <: SerializableModuleParameter
  ](serializableModuleGenerator: SerializableModuleGenerator[M, P]
  )(
    implicit pRW: upickle.default.Writer[P]
  ): FixedIOIntrinsicModuleParameter[D] =
    apply[D](
      serializableModuleGenerator.generator.getName,
      upickle.default.write(serializableModuleGenerator.parameter),
      ???
    )
}

trait FixedIOIntrinsicModuleParameter[D <: SerializableData] {
  val generatorClass:     String
  val generatorParameter: String
  val io:                 D
}

class FixedIOIntrinsicModule[D <: SerializableData, T <: FixedIOIntrinsicModuleParameter[D]](intrinsicParameter: T)
    extends IntrinsicModule("SerializableGenerator")
    with FixedIOBaseModule[D] {

  override val params: Map[String, Param] = Map(
    "generator" -> StringParam(intrinsicParameter.generatorClass),
    "parameter" -> StringParam(upickle.default.write(intrinsicParameter.generatorParameter))
    // TODO: pass down io for validation.
  )

  /** A generator of IO */
  override protected def ioGenerator: D = io
}

class FixedIOIntrinsicModuleGenerator[D <: SerializableData, T <: FixedIOIntrinsicModuleParameter[D]](parameter: T)
    extends Generator[FixedIOIntrinsicModule[D, T]] {

  /** elaborate a module from this generator. */
  override def module(): FixedIOIntrinsicModule[D, T] = new FixedIOIntrinsicModule(parameter)

  /** get the definition from this generator. */
  override def definition(): Definition[FixedIOIntrinsicModule[D, T]] = Definition(
    new FixedIOIntrinsicModule[D, T](parameter)
  )

  /** get an instance of from this generator. */
  override def instance(): Instance[FixedIOIntrinsicModule[D, T]] = Instantiate(new FixedIOIntrinsicModule(parameter))
}

object FixedIOIntrinsicModuleGenerator {
  def apply[D <: SerializableData, P <: FixedIOIntrinsicModuleParameter[D]](
    fixedIOIntrinsicModuleParameter: P
  ): FixedIOIntrinsicModuleGenerator[D, FixedIOIntrinsicModuleParameter[D]] =
    new FixedIOIntrinsicModuleGenerator[D, FixedIOIntrinsicModuleParameter[D]](fixedIOIntrinsicModuleParameter)
}
