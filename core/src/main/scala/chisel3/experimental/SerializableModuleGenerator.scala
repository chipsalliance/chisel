package chisel3.experimental

import ujson.Obj
import upickle.default._

import scala.reflect.macros.whitebox
import scala.reflect.runtime.universe._
import scala.language.experimental.macros

trait SerializableModule { this: BaseModule =>
  type SerializableModuleParameter
  val parameter: SerializableModuleParameter
}

abstract class SerializableModuleGenerator {
  type M <: SerializableModule
  // TODO: use macro to construct moduleClass
  protected[chisel3] val moduleClass: Class[M]
  // user should implement the serialization for M#SerializableModuleParameter
  // compiler can observe the implicit function for serializedParameter
  implicit val parameterRW: ReadWriter[M#SerializableModuleParameter]
  val parameter:            M#SerializableModuleParameter
  protected[chisel3] def moduleClassName = moduleClass.getName
  protected[chisel3] def serializedParameter = upickle.default.writeJs(parameter)
}

object SerializableModuleGenerator {
  import SerializableModuleGeneratorImpl._
  implicit def rw: upickle.default.ReadWriter[SerializableModuleGenerator] =
    upickle.default
      .readwriter[ujson.Value]
      .bimap[SerializableModuleGenerator](
        x =>
          ujson.Obj(
            "parameter" -> x.serializedParameter,
            "module" -> x.moduleClassName
          ),
        json => {
          val module = json.obj("module").toString
          val parameter = json.obj("parameter").toString
          import scala.reflect.runtime.{universe => ru}
          val m = ru.runtimeMirror(getClass.getClassLoader)
          val classSymbol = ru.typeOf[SerializableModuleGenerator].typeSymbol.asClass
          val classMirror = m.reflectClass(classSymbol)
          val decl = ru.typeOf[SerializableModuleGenerator].decl(ru.termNames.CONSTRUCTOR).asMethod
          val ctorm = classMirror.reflectConstructor(decl)
          // reflect to concrete SerializableGenerator.SomeSerializableModuleImp to construct a SerializableModuleGenerator
          ???
        }
      )
  def apply[M <: SerializableModule](
    parameter: M#SerializableModuleParameter
  )(
    implicit rwP: upickle.default.ReadWriter[M#SerializableModuleParameter]
  ): SerializableModuleGenerator = macro applyImpl[M]
}

private[chisel3] object SerializableModuleGeneratorImpl {

  // TODO: for each concrete SerializableModule, generate SerializableGenerator.SomeSerializableModuleImp at compile time.
  def applyImpl[M <: SerializableModule: c.WeakTypeTag](
    c:         whitebox.Context
  )(parameter: c.Expr[M#SerializableModuleParameter]
  )(rwP:       c.Expr[upickle.default.ReadWriter[M#SerializableModuleParameter]]
  ): c.Expr[SerializableModuleGenerator] = {
    import c.universe._
    // the asInstanceOf static upper cast SerializableModuleGenerator{type M = chiselTests.experimental.GCDSerializableModule}
    // to SerializableModuleGenerator for upickle being able to observe the implicit parameter rwP
    c.Expr[SerializableModuleGenerator](q"""
      new _root_.chisel3.experimental.SerializableModuleGenerator {
        override type M = ${weakTypeOf[M]}
        val parameter: ${weakTypeOf[M]}#SerializableModuleParameter = $parameter
        override implicit val parameterRW: upickle.default.ReadWriter[${weakTypeOf[M]}#SerializableModuleParameter] = $rwP
        override val moduleClass = classOf[M]
      }.asInstanceOf[_root_.chisel3.experimental.SerializableModuleGenerator]""")
  }
}
