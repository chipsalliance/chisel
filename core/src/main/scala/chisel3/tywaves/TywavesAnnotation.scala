package chisel3.tywaves

import chisel3.{Aggregate, Bits, Bundle, Data, Record, Vec, VecLike}
import chisel3.experimental.{annotate, BaseModule, ChiselAnnotation}
import chisel3.internal.firrtl.Converter.convert
import chisel3.internal.firrtl.ir._
import firrtl.annotations.{Annotation, IsMember, ReferenceTarget, SingleTargetAnnotation}

// TODO: if the code touches a lot of Chisel internals, it might be better to put it into
//    - core
//  otherwise:
//    - src

/**
  * TywavesAnnotation is a custom annotation that is used to store Chisel high-level information in the FIRRTL for the
  * Tywaves waveform viewer.
  *
  *  This case class is not intended to be used by the user.
  *
  * @param target  The target of the annotation
  * @param typeName
  */
private[chisel3] case class TywavesAnnotation[T <: IsMember](target: T, typeName: String)
    extends SingleTargetAnnotation[T] {
  def duplicate(n: T) = this.copy(n)
}

object TywavesChiselAnnotation {
  def generate(circuit: Circuit): Seq[ChiselAnnotation] = {
    // TODO: iterate over a circuit and generate TywavesAnnotation
    val typeAliases: Seq[String] = circuit.typeAliases.map(_.name)

    circuit.components.flatMap(c => generate(c, typeAliases))
    //    circuit.layers
    //    circuit.options

//    ???
  }

  def generate(component: Component, typeAliases: Seq[String]): Seq[ChiselAnnotation] = component match {
    case ctx @ DefModule(id, name, public, layers, ports, cmds) =>
      // TODO: Add tywaves annotation: components, ports, commands, layers
      Seq(createAnno(id)) ++ (ports ++ ctx.secretPorts).flatMap(p =>
        generate(p, typeAliases)
      ) ++ (cmds ++ ctx.secretCommands).flatMap(c => generate(c, typeAliases))
    case ctx @ DefBlackBox(id, name, ports, topDir, params) =>
      // TODO: Add tywaves annotation, ports, ?params?
      Seq(createAnno(id)) ++ (ports ++ ctx.secretPorts).flatMap(p => generate(p, typeAliases))
    case ctx @ DefIntrinsicModule(id, name, ports, topDir, params) =>
      // TODO: Add tywaves annotation: ports, ?params?
      Seq(createAnno(id)) ++ (ports ++ ctx.secretPorts).flatMap(p => generate(p, typeAliases))
    case ctx @ DefClass(id, name, ports, cmds) =>
      // TODO: Add tywaves annotation: ports, commands
      Seq(createAnno(id)) ++ (ports ++ ctx.secretPorts).flatMap(p => generate(p, typeAliases)) ++ cmds.flatMap(c =>
        generate(c, typeAliases)
      )
    case ctx => throw new Exception(s"Failed to generate TywavesAnnotation. Unknown component type: $ctx")
  }

  // TODO: Add tywaves annotation
  def generate(port: Port, typeAliases: Seq[String]): Seq[ChiselAnnotation] = createAnno(port.id)

  def generate(command: Command, strings: Seq[String]): Seq[ChiselAnnotation] = {

    // TODO: Add tywaves annotation
    Seq.empty
//    ???
  }

  /**
    * Create the annotation
    * @param target
    */
  private def createAnno(target: Data): Seq[ChiselAnnotation] = {
//    val name = target.toString
    val name = target match {
//      case t: Bundle =>
      //        // t.className
      //        t.toString.split(" ").last
      case t: Vec[?] =>
        t.toString.split(" ").last
      // This is a workaround to pretty print anonymous bundles and other records
      case t: Record =>
        // t.prettyPrint
        t.topBindingOpt match {
          case Some(binding) =>
            s"${t._bindingToString(binding)}[${t.className}]" // t._bindingToString(binding) + "[" + t.className + "]"
          case None => t.className
        }
//      case t: Bits =>
//        // t.typeName
//        t.topBindingOpt match {
//          case Some(binding) =>
//            s"${t._bindingToString(binding)}[Bits${t.width.toString}]" // t._bindingToString(binding) + "[" + t.className + "]"
//          case None => s"Bits${t.width.toString}"
//        }
      case t =>
        // t.typeName
        target.toString.split(" ").last
    }

    var annotations: Seq[ChiselAnnotation] = Seq.empty
    target match {
      case record: Record =>
        record.elements.foreach {
          case (name, element) => annotations = annotations ++ createAnno(element)
        }
      case vecLike: VecLike[_] =>
        // Warning: this assumes all the elements have the same type
        vecLike.collectFirst { element =>
          annotations = annotations ++ createAnno(element); true
        }
      case _ => ()
    }
    annotations :+ new ChiselAnnotation {
      override def toFirrtl: Annotation = TywavesAnnotation(target.toTarget, name)
    }
  }

  private def createAnno(target: BaseModule): ChiselAnnotation = {
    val name = target.desiredName
    //    val name = target.getClass.getTypeName
    new ChiselAnnotation {
      override def toFirrtl: Annotation = TywavesAnnotation(target.toTarget, name)
    }
  }

}
