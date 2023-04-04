package chisel3.boxes

import chisel3.boxes.internal._
import chisel3.internal.{HasId, Context, CloneToContext}

trait HasSerialize {
  def serialize: String
}

trait Box[+T] extends HasSerialize with HasId with CloneToContext {
  val proto: Proto[T]
  val context: Option[Context]

  override def toString = serialize
}

trait ContextLookup[T, H <: Box[T]] {
  final def lookup[X](thunk: T => X, name: String, h: H)(implicit l: Lookup[X, H]): l.R = {
    h.proto.protoOpt.map {
      p => l.contextualize(l.toWrapped(thunk(p), h), h.context)
    }.orElse {
      h.proto.protoMap.map(x => l.contextualize(x(name).asInstanceOf[l.R], h.context))
    }.get
  }
}
object Box {
  implicit def contextLookup[T, H <: Box[T]]: ContextLookup[T, H] = new ContextLookup[T, H] {}
  implicit class BoxExtension[T, H <: Box[T]] (h: H) {
    def lookup[X](item: T => X, name: String)(implicit l: Lookup[X, H], cl: ContextLookup[T, H]): l.R = cl.lookup(item, name, h)
  }
}

case class Definition[+T <: Module] (proto: Proto[T], context: Context) extends Box[T] {
  def cloneTo(c: Context) = new Definition(proto, c)
}
object Definition {
  def apply[T <: Module](thing: => T)(implicit valName: ValName): Definition[T] = Definition[T, T](thing, None)
  def apply[T <: Module, S <: Module](thing: => T, impl: Implementation[S])(implicit valName: ValName, ev: T =:= S): Definition[T] = {
    Definition(thing, Some(impl))
  }
  def apply[T <: Module, S <: Module](thing: => T, implOpt: Option[Implementation[S]])(implicit valName: ValName, ev: T =:= S): Definition[T] = {
    val context = Runtime.priorConstructor(valName.name)
    val evaluated = thing
    Runtime.afterConstructor(context)
    context.setValue(Definition(new Proto(Some(evaluated), None), implOpt.asInstanceOf[Option[Implementation[T]]], context))
  }
  def apply[T <: Module](map: Map[String, () => Any], DefinitionName: String): Definition[T] = {
    val context = Runtime.priorConstructor(DefinitionName)
    val proto = new Proto[T](None, Some(map.view.mapValues(x => x()).toMap))
    Runtime.afterConstructor(context)
    context.setValue(Definition(proto, None, context))
  }
}

//trait ImpBuilder[-T <: Module] {
//  def build(d: Definition[T]): String
//}
//case class Implementation[+T <: Module] private (i: Int)//ib: Either[ImpBuilder[T], String])
//object Implementation {
//  def apply[T <: Module](ib: ImpBuilder[T]): Implementation[T] = Implementation(Left(ib))
//  def apply[T <: Module](ir: String): Implementation[T] = Implementation(Right(ir))
//}

trait Component[+T] extends Box[T] {
  //def isDeclaration: Boolean = context.parent.flatMap { p => p.getValueOpt.flatMap { case c: Component[_] => }  }
  //def reference: String = context.parentContextPath.reverse.dropWhile(p => p.isOrigin).map(_.key).mkString(".")
}

case class Instance[+T <: Module] private (proto: Proto[T], context: Context) extends Component[T] {
  def cloneTo(c: Context) = new Instance(proto, c)
  def defRef: String = {
    val definitionContext = context.provenanceContextPath.dropWhile( { p => p.getValue match {
      case Some(i: Instance[_]) => true
      case _ => false
    }
    }).head
    val path = definitionContext.parentContextPath.reverse.dropWhile(p => p.isOrigin).map(_.key).mkString(".")
    //println(context.visualize)
    if(path == "") definitionContext.key else path
  }
}
object Instance {
  def apply[T <: Module](definition: Definition[T]): Instance[T] = {
    val context = Runtime.current.newChild(valName.name, definition.context)
    context.setValue(Instance(definition.proto, context))
  }
  def apply[T <: Module](definition: Definition[T], name: String): Instance[T] = {
    apply(definition)(ValName(name))
  }
}

//class Key[+T](val key: String, protoGen: () => Proto[T]) {
//  def proto = protoGen()
//}
//object Key {
//  def apply[T](key: String, gen: () => T): Key[T] = new Key(key, () => new Proto(Some(gen()), None))
//  def apply[T](key: String, map: Map[String, Any]): Key[T] = new Key(key, () => new Proto(None, Some(map)))
//  def makeName(any: Any*): String = if(any.isEmpty) "" else any.map {
//    case x: Box[_] => x.reference
//    case x => x.toString
//  }.mkString("<", "_", ">")
//}
//
//case class Generator[T] (context: Context, companion: Any) extends Box[T] {
//  val proto = new Proto(None, None)
//  def cloneTo(c: Context) = new Generator[T](c, companion)
//  def declaration: String = s"generator $reference:"
//  def serialize: String = s"generator ${context.key}:"
//
//  def build(key: Key[T])(implicit b: Generator.Builder[T]): b.R = b.build(this, key)
//  def getCompanion: Any = companion
//
//  // Interface used to retrieve a Definition (the generated thing)
//}
//object Generator {
//
//  // Typeclass to return current boxed type (Definition vs Type) when building them from Generator[U]
//  // This isn't right - need to ensure that ChiselGenerator[X] where X is U, but that is for another time.
//  trait Builder[T] {
//    type R
//    def build(g: Generator[T], key: Key[T]): R
//  }
//  implicit def dataExt[T <: Data] = new Builder[T] {
//    type R = Type[T]
//    def build(g: Generator[T], key: Key[T]): Type[T] = {
//      if(!g.context.childrenKeys.contains(key.key)) {
//        val saved = Runtime.jump(g.context.origin)
//        val context = Runtime.priorConstructor(key.key)
//        val evaluated = key.proto
//        Runtime.afterConstructor(context)
//        context.setValue(Type(evaluated, context))
//        Runtime.jumpBack(saved)
//      }
//      val typeToImport = g.context.newContext(key.key).getValue.asInstanceOf[Type[T]]
//      if(Runtime.currentTopName.nonEmpty && !Runtime.top.childrenKeys.contains(g.context.key)) {
//        val newContext = Runtime.top.newChild(key.key, typeToImport.context)
//        if(!newContext.hasLocalValue) {
//          val newValue = Imported(typeToImport.proto, newContext)
//          newContext.setValue(newValue)
//        }
//      }
//      typeToImport
//    }
//  }
//  implicit def moduleExt[T <: Module] = new Builder[T] {
//    type R = Definition[T]
//    def build(g: Generator[T], key: Key[T]): Definition[T] = {
//      if(!g.context.childrenKeys.contains(key.key)) {
//        val saved = Runtime.jump(g.context.origin)
//        val context = Runtime.priorConstructor(key.key)
//        val evaluated = key.proto
//        Runtime.afterConstructor(context)
//        context.setValue(Definition(evaluated, None, context))
//        Runtime.jumpBack(saved)
//      }
//      val moduleToImport = g.context.newContext(key.key).getValue.asInstanceOf[Definition[T]]
//      if(Runtime.currentTopName.nonEmpty && !Runtime.top.childrenKeys.contains(g.context.key)) {
//        val newContext = Runtime.top.newChild(key.key, moduleToImport.context)
//        if(!newContext.hasLocalValue) {
//          val newValue = Imported(moduleToImport.proto, newContext)
//          newContext.setValue(newValue)
//        }
//      }
//      moduleToImport
//    }
//  }
//}
//
//case class Circuit[+T <: Module] (context: Context) extends Box[T] {
//  val proto = new Proto(None, None)
//  def declaration = s"circuit ${context.key}:"
//  def cloneTo(c: Context) = new Circuit(c)
//  def serialize = s"circuit ${context.key}:"
//}
//
//case class Imported[+T] (proto: Proto[T], context: Context) extends Box[T] {
//  def cloneTo(c: Context) = context.provenance.get.getValue.cloneTo(c)
//  def serialize = s"import ${context.provenance.get.targetNoId}:"
//  def declaration = s"import ${context.provenance.get.targetNoId}:"
//}
//
//trait TypeMember[T <: Data] extends Component[T]
//
//case class Type[T <: Data] (proto: Proto[T], context: Context) extends TypeMember[T] {
//  def cloneTo(c: Context) = new Type[T](proto, c)
//  def serialize: String = s"type ${context.key}:"
//  def declaration: String = s"type ${context.key}:"
//}
//object Type {
//  def apply[T <: Data](thing: => T)(implicit valName: ValName): Type[T] = {
//    val context = Runtime.priorConstructor(valName.name)
//    val evaluated = thing
//    Runtime.afterConstructor(context)
//    context.setValue(Type(new Proto(Some(evaluated), None), context))
//  }
//}
//
//case class Field[T <: Data] private (tpe: Type[T], context: Context, aligned: Boolean) extends TypeMember[T] {
//  def cloneTo(c: Context) = new Field[T](tpe, c, aligned) // probably need to change aligned
//  val proto = tpe.proto
//  lazy val modifier = if(aligned) "aligned" else "flipped"
//  def serialize: String = s"$modifier ${context.key}: ${tpe.reference}"
//  def declaration: String = s"$modifier ${context.key}: ${tpe.reference}"
//}
//object Aligned {
//  def apply[T <: Data](tpe: Type[T])(implicit valName: ValName): Field[T] = {
//    val context = Runtime.current.newChild(valName.name, tpe.context)
//    context.setValue(Field(tpe, context, true))
//  }
//}
//object Flipped {
//  def apply[T <: Data](tpe: Type[T])(implicit valName: ValName): Field[T] = {
//    val context = Runtime.current.newChild(valName.name, tpe.context)
//    context.setValue(Field(tpe, context, false))
//  }
//}
//
//case class Port[T <: Data] private (tpe: TypeMember[T], context: Context, outgoing: Boolean, isDefinition: Boolean, isWritable: Boolean) extends Component[T] {
//  val proto = tpe.proto
//  def cloneTo(c: Context) = new Port[T](tpe, c, outgoing, isDefinition, isWritable) // probably need to change aligned
//  lazy val modifier = if(outgoing) "output" else "input "
//  def := (that: Component[T]): Unit = {
//    val connection = Runtime.current.newContext(Runtime.newName())
//    require(isWritable, s"Cannot write to $reference, as it is unwritable from this context.")
//    connection.setValue(new HasSerialize with CloneToContext {
//      def cloneTo(c: Context): CloneToContext = this
//      def serialize: String = s"${reference} := ${that.reference}"
//    })
//  }
//  def defRef: String = {
//    val definitionContext = context.provenanceContextPath.dropWhile( { p => p.getValueOpt match {
//      case Some(i: Port[_]) => true
//      case _ => false
//    }
//    }).head
//    val path = definitionContext.parentContextPath.reverse.dropWhile(p => p.isTemplate).map(_.key).mkString(".")
//    if(path == "") definitionContext.key else path
//  }
//  //def serialize: String = if(isDefinition) s"$modifier ${context.key}: ${defRef}" else ""
//  def serialize: String = s"$modifier ${context.key}: ${defRef}"
//  def declaration: String = s"$modifier ${context.key}: ${defRef}"
//}
//object Incoming {
//  def apply[T <: Data](tpe: Type[T])(implicit valName: ValName): Port[T] = {
//    val context = Runtime.current.newChild(valName.name, tpe.context)
//    context.setValue(Port(tpe, context, false, true, false))
//  }
//}
//object Outgoing {
//  def apply[T <: Data](tpe: Type[T])(implicit valName: ValName): Port[T] = {
//    val context = Runtime.current.newChild(valName.name, tpe.context)
//    context.setValue(Port(tpe, context, true, true, true))
//  }
//}
//