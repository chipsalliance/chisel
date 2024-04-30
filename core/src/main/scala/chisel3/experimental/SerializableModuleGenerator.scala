package chisel3.experimental

import chisel3.experimental.hierarchy.{Definition, Instance}
import chisel3.internal.firrtl.Converter
import chisel3.internal.{Builder, BuilderContextCache, DynamicContext, instantiable}
import firrtl.annotations.JsonProtocol
import firrtl.ir.Circuit
import firrtl.options.Unserializable
import logger.LogLevelAnnotation
import mainargs._
import upickle.default._

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime.universe
import scala.reflect.runtime.universe.{runtimeMirror, typeOf}

/** Parameter for SerializableModule, it should be serializable via upickle API.
  * For more information, please refer to [[https://com-lihaoyi.github.io/upickle/]]
  *
  * user should define their own rw to `FooSerializableModuleParameter.rw`, otherwise compiler plugin will do this:
  * {{{
  *   object FooSerializableModuleParameter {
  *     implicit val rw: upickle.default.RW[SerializableModuleParameter] = macroRW
  *   }
  * }}}
  * But if `upickle.default.RW[SerializableModuleParameter]` doesn't compile, scalac would complain.
  */
@main
trait SerializableModuleParameter

/** Mixin this trait to let chisel auto serialize module, it has these constraints:
  * 1. Module should not be any inner class of other class, since serializing outer class is impossible.
  * 2. Module should have and only have one parameter with type `T`:
  * {{{
  * class FooSerializableModule[FooSerializableModuleParameter](val parameter: FooSerializableModuleParameter)
  * }}}
  * 3. user should guarantee the module is reproducible on their own.
  */
@instantiable
trait SerializableModule[T <: SerializableModuleParameter] extends BaseModule {
  val parameter: T
}
object SerializableModuleGenerator {

  /** serializer for SerializableModuleGenerator. */
  implicit def rw[P <: SerializableModuleParameter, M <: SerializableModule[P]](
    implicit rwP: ReadWriter[P],
    pTypeTag:     universe.TypeTag[P],
    mTypeTag:     universe.TypeTag[M]
  ): ReadWriter[SerializableModuleGenerator[M, P]] = readwriter[ujson.Value].bimap[SerializableModuleGenerator[M, P]](
    { (x: SerializableModuleGenerator[M, P]) =>
      ujson
        .Obj(
          "parameter" -> upickle.default.writeJs[P](x.parameter),
          "generator" -> x.generator.getName
        )
    },
    { (json: ujson.Value) =>
      SerializableModuleGenerator[M, P](
        Class.forName(json.obj("generator").str).asInstanceOf[Class[M]],
        upickle.default.read[P](json.obj("parameter"))
      )
    }
  )

  /** cache instance of a generator. */
  private case class CacheKey[P <: SerializableModuleParameter, M <: SerializableModule[P]](
    parameter: P,
    mTypeTag:  universe.TypeTag[M])
      extends BuilderContextCache.Key[Definition[M]]

  private[chisel3] def construct[M <: SerializableModule[P], P <: SerializableModuleParameter](
    generator: Class[M],
    parameter: P
  ): M = {
    require(
      generator.getConstructors.length == 1,
      s"""only allow constructing SerializableModule from SerializableModuleParameter via class Module(val parameter: Parameter),
         |you have ${generator.getConstructors.length} constructors
         |""".stripMargin
    )
    require(
      !generator.getConstructors.head.getParameterTypes.last.toString.contains("$"),
      s"""You define your ${generator.getConstructors.head.getParameterTypes.last} inside other class.
         |This is a forbidden behavior, since we cannot serialize the out classes,
         |for debugging, these are full parameter types of constructor:
         |${generator.getConstructors.head.getParameterTypes.mkString("\n")}
         |""".stripMargin
    )
    require(
      generator.getConstructors.head.getParameterCount == 1,
      s"""You define multiple constructors:
         |${generator.getConstructors.head.getParameterTypes.mkString("\n")}
         |""".stripMargin
    )
    generator.getConstructors.head.newInstance(parameter).asInstanceOf[M with BaseModule]
  }

  def apply[M <: SerializableModule[P]: universe.TypeTag, P <: SerializableModuleParameter](parameter: P): SerializableModuleGenerator[M, P] =
    new SerializableModuleGenerator(runtimeMirror(getClass.getClassLoader).runtimeClass(typeOf[M].typeSymbol.asClass).asInstanceOf[Class[M]], parameter)
}

/** the serializable module generator:
  * @param generator a non-inner class of module, which should be a subclass of [[SerializableModule]]
  * @param parameter the parameter of `generator`
  */
case class SerializableModuleGenerator[M <: SerializableModule[P], P <: SerializableModuleParameter](
  generator: Class[M],
  parameter: P
)(
  implicit moduleTypeTag: universe.TypeTag[M])
    extends Generator[M] {

  /** elaborate a module from this generator. */
  def module(): M = SerializableModuleGenerator.construct(generator, parameter)

  /** get the definition from this generator. */
  def definition(): Definition[M] = Builder.contextCache
    .getOrElseUpdate(
      SerializableModuleGenerator.CacheKey(
        parameter,
        moduleTypeTag
      ),
      Definition.do_apply(SerializableModuleGenerator.construct(generator, parameter))(UnlocatableSourceInfo)
    )

  /** get an instance of from this generator. */
  def instance(): Instance[M] = Instance.do_apply(definition())(UnlocatableSourceInfo)
}

/** a container for arbitrary generator. */
private[chisel3] trait Generator[M <: BaseModule] {

  /** elaborate a module from this generator. */
  def module(): M

  /** get the definition from this generator. */
  def definition(): Definition[M]

  /** get an instance of from this generator. */
  def instance(): Instance[M]
}

/** Mix-in this trait to create a main function generating the mlirbc file.
  *
  * @note from jiuyang
  *   - don't want to depend on the Stage API! just directly convert it.
  *   - w/o directly emit Verilog, it generate mlirbc instead, delegate later process to `firtool`.
  *     - only one concern is circt version should be aligned for parsing and using.
  *   - user can use @arg() API to add documentation to parameter fields and expose them in the command line.
  */
trait SerializableModuleMain[P <: SerializableModuleParameter, M <: SerializableModule[P]] {
  // In the future, we will use plugin to fill these three options.
  /** fill it with: `implicit val pRW: upickle.default.ReadWriter[P] = P.rw`
    * In the future, P.rw must be implemented, and guarded by the compiler Plugin.
    */
  implicit val pRW: upickle.default.ReadWriter[P]
  // Traits cannot have type parameters with context bounds
  /** fill it with: `implicit val mTypeTag: universe.TypeTag[M] = implicitly[universe.TypeTag[M]]` */
  implicit val mTypeTag: universe.TypeTag[M]
  /** fill it with: `implicit val pTypeTag: universe.TypeTag[M] = implicitly[universe.TypeTag[P]]` */
  implicit val pTypeTag: universe.TypeTag[P]

  private implicit val gRW: upickle.default.ReadWriter[SerializableModuleGenerator[M, P]] = SerializableModuleGenerator.rw[P,M]

  implicit object PathRead extends TokensReader.Simple[os.Path]{
    def shortName = "path"
    def read(strs: Seq[String]) = Right(os.Path(strs.head, os.pwd))
  }

  // This is the example about how do we add user command-line APIs.
  // Annotation are not friendly to users! let's purge it!!!
  @main
  private[chisel3] case class Context(
                     throwOnFirstError: Boolean = false,
                     legacyShiftRightWidth: Boolean = false,
                     firtoolBinary: Option[os.Path] = None,
                     outPutDirectory: os.Path = os.pwd
                     // TODO: source roots,
                     // TODO: warning filter.
                     // TODO: Log Level
                    )

  private def build(module: => M, context: Context): Unit = {
    // TODO: in the far future, we can think about refactor Builder talking to CIRCT.
    val cir = Builder.build(
      module,
      // TODO: expose Builder options to cmdline via mainargs
      new DynamicContext(
        Nil,
        context.throwOnFirstError,
        context.legacyShiftRightWidth,
        Nil,
        Nil,
        None,
        logger.LoggerOptionsView.view(Seq(LogLevelAnnotation())),
        ArrayBuffer.empty,
        BuilderContextCache.empty
      )
    )._1

    // TODO: use scala reflect to optionally select panama backend.
    val fir: Circuit = Converter.convert(cir)
    val annotations = cir.annotations.map(_.toFirrtl).flatMap {
      case _: Unserializable          => None
      case a => Some(a)
    }

    os.proc(
      context.firtoolBinary.map(_.toString).getOrElse("firtool"),
      os.temp(fir.serialize).toString,
      "--annotation-file", os.temp(fir.serialize).toString,
      "--parse-only", os.temp(JsonProtocol.serialize(annotations)),
      "--emit-bytecode",
      "-o", (context.outPutDirectory / s"${fir.main}.mlirbc").toString
    ).call()

  }
  /** get the MLIRBC file from parameter. */
  @main
  def mlirbc(parameter: P, context: Context): Unit = {
    build(SerializableModuleGenerator[M, P](parameter).module(), context)
  }

  /** get the MLIRBC file from a json file. */
  @main
  def mlirbc(parameterFile: os.Path, context: Context): Unit = {
    build(SerializableModuleGenerator[M, P](upickle.default.read(os.read(parameterFile))).module(), context)
  }

  def main(args: Array[String]): Unit = ParserForMethods(this).runOrExit(args)
}
