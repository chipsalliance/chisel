package chisel3.experimental.util

import chisel3.RawModule
import chisel3.experimental.{SerializableModule, SerializableModuleGenerator, SerializableModuleParameter}

import scala.reflect.runtime.universe
import scala.reflect.runtime.universe.{runtimeMirror, typeOf}

/** The target design emitted by [[SerializableModuleElaborator]], which contains
  * @param fir the firrtl circuit
  * @param annos annotations which is serializable
  */
case class SerializableModuleDesign(fir: firrtl.ir.Circuit, annos: Seq[firrtl.annotations.Annotation]) {
  def firFile = fir.serialize
  def annosFile = firrtl.annotations.JsonProtocol.serializeRecover(annos)
}

/** Mixin this trait to produce elaborators for [[SerializableModule]]
  */
trait SerializableModuleElaborator {

  /**
    * Implementation of a config API to serialize the [[SerializableModuleParameter]]
    * @example
    * {{{
    *  def config(parameter: MySerializableModuleParameter) =
    *    os.write.over(os.pwd / "config.json", configImpl(parameter))
    * }}}
    */
  def configImpl[P <: SerializableModuleParameter: universe.TypeTag](
    parameter: P
  )(
    implicit rwP: upickle.default.Writer[P]
  ) = upickle.default.write(parameter)

  /**
    * Implementation of a design API to elaborate [[SerializableModule]]
    *
    * @return [[SerializableModuleDesign]]
    * @example
    * {{{
    *  def design(parameter: os.Path) = {
    *    val design = designImpl[MySerializableModule, MySerializableModuleParameter](os.read(parameter))
    *    os.write.over(os.pwd / s"\${design.fir.main}.fir", design.firFile)
    *    os.write.over(os.pwd / s"\${design.fir.main}.anno.json", design.annosFile)
    *  }
    * }}}
    */
  def designImpl[M <: SerializableModule[P]: universe.TypeTag, P <: SerializableModuleParameter: universe.TypeTag](
    parameter: String
  )(
    implicit
    rwP: upickle.default.Reader[P]
  ): SerializableModuleDesign = {
    var fir: firrtl.ir.Circuit = null
    val annos = Seq(
      new chisel3.stage.phases.Elaborate,
      new chisel3.stage.phases.Convert
    ).foldLeft(
      Seq(
        chisel3.stage.ChiselGeneratorAnnotation(() =>
          SerializableModuleGenerator(
            runtimeMirror(getClass.getClassLoader)
              .runtimeClass(typeOf[M].typeSymbol.asClass)
              .asInstanceOf[Class[M]],
            upickle.default.read[P](parameter)
          ).module().asInstanceOf[RawModule]
        )
      ): firrtl.AnnotationSeq
    ) { case (annos, stage) => stage.transform(annos) }
      .flatMap {
        case firrtl.stage.FirrtlCircuitAnnotation(circuit) =>
          fir = circuit
          None
        case _: firrtl.options.Unserializable => None
        case a => Some(a)
      }
    SerializableModuleDesign(fir, annos)
  }
}
