// See LICENSE for license details.

package firrtl
package annotations

import firrtl.ir.Expression
import AnnotationUtils.{toExp, validComponentName, validModuleName}
import TargetToken._

import scala.collection.mutable

/** Refers to something in a FIRRTL [[firrtl.ir.Circuit]]. Used for Annotation targets.
  *
  * Can be in various states of completion/resolved:
  *   - Legal: [[TargetToken]]'s in tokens are in an order that makes sense
  *   - Complete: circuitOpt and moduleOpt are non-empty, and all Instance(_) are followed by OfModule(_)
  *   - Local: tokens does not refer to things through an instance hierarchy (no Instance(_) or OfModule(_) tokens)
  */
sealed trait Target extends Named {

  /** @return Circuit name, if it exists */
  def circuitOpt: Option[String]

  /** @return Module name, if it exists */
  def moduleOpt: Option[String]

  /** @return [[Target]] tokens */
  def tokens: Seq[TargetToken]

  /** @return Returns a new [[GenericTarget]] with new values */
  def modify(circuitOpt: Option[String] = circuitOpt,
             moduleOpt: Option[String] = moduleOpt,
             tokens: Seq[TargetToken] = tokens): GenericTarget = GenericTarget(circuitOpt, moduleOpt, tokens.toVector)

  /** @return Human-readable serialization */
  def serialize: String = {
    val circuitString = "~" + circuitOpt.getOrElse("???")
    val moduleString = "|" + moduleOpt.getOrElse("???")
    val tokensString = tokens.map {
      case Ref(r) => s">$r"
      case Instance(i) => s"/$i"
      case OfModule(o) => s":$o"
      case Field(f) => s".$f"
      case Index(v) => s"[$v]"
      case Clock => s"@clock"
      case Reset => s"@reset"
      case Init => s"@init"
    }.mkString("")
    if(moduleOpt.isEmpty && tokens.isEmpty) {
      circuitString
    } else if(tokens.isEmpty) {
      circuitString + moduleString
    } else {
      circuitString + moduleString + tokensString
    }
  }

  /** Pretty serialization, ideal for error messages. Cannot be deserialized.
    * @return Human-readable serialization
    */
  def prettyPrint(tab: String = ""): String = {
    val circuitString = s"""${tab}circuit ${circuitOpt.getOrElse("???")}:"""
    val moduleString = s"""\n$tab└── module ${moduleOpt.getOrElse("???")}:"""
    var depth = 4
    val tokenString = tokens.map {
      case Ref(r) => val rx = s"""\n$tab${" "*depth}└── $r"""; depth += 4; rx
      case Instance(i) => val ix = s"""\n$tab${" "*depth}└── inst $i """; ix
      case OfModule(o) => val ox = s"of $o:"; depth += 4; ox
      case Field(f) => s".$f"
      case Index(v) => s"[$v]"
      case Clock => s"@clock"
      case Reset => s"@reset"
      case Init => s"@init"
    }.mkString("")

    (moduleOpt.isEmpty, tokens.isEmpty) match {
      case (true, true) => circuitString
      case (_, true) => circuitString + moduleString
      case (_, _) => circuitString + moduleString + tokenString
    }
  }


  /** @return Converts this [[Target]] into a [[GenericTarget]] */
  def toGenericTarget: GenericTarget = GenericTarget(circuitOpt, moduleOpt, tokens.toVector)

  /** @return Converts this [[Target]] into either a [[CircuitName]], [[ModuleName]], or [[ComponentName]] */
  @deprecated("Use Target instead, will be removed in 1.3", "1.2")
  def toNamed: Named = toGenericTarget.toNamed

  /** @return If legal, convert this [[Target]] into a [[CompleteTarget]] */
  def getComplete: Option[CompleteTarget]

  /** @return Converts this [[Target]] into a [[CompleteTarget]] */
  def complete: CompleteTarget = getComplete.get

  /** @return Converts this [[Target]] into a [[CompleteTarget]], or if it can't, return original [[Target]] */
  def tryToComplete: Target = getComplete.getOrElse(this)

  /** Whether the target is directly instantiated in its root module */
  def isLocal: Boolean
}

object Target {

  def apply(circuitOpt: Option[String], moduleOpt: Option[String], reference: Seq[TargetToken]): GenericTarget =
    GenericTarget(circuitOpt, moduleOpt, reference.toVector)

  def unapply(t: Target): Option[(Option[String], Option[String], Seq[TargetToken])] =
    Some((t.circuitOpt, t.moduleOpt, t.tokens))

  case class NamedException(message: String) extends Exception(message)

  implicit def convertCircuitTarget2CircuitName(c: CircuitTarget): CircuitName = c.toNamed
  implicit def convertModuleTarget2ModuleName(c: ModuleTarget): ModuleName = c.toNamed
  implicit def convertIsComponent2ComponentName(c: IsComponent): ComponentName = c.toNamed
  implicit def convertTarget2Named(c: Target): Named = c.toNamed
  implicit def convertCircuitName2CircuitTarget(c: CircuitName): CircuitTarget = c.toTarget
  implicit def convertModuleName2ModuleTarget(c: ModuleName): ModuleTarget = c.toTarget
  implicit def convertComponentName2ReferenceTarget(c: ComponentName): ReferenceTarget = c.toTarget
  implicit def convertNamed2Target(n: Named): CompleteTarget = n.toTarget

  /** Converts [[ComponentName]]'s name into TargetTokens
    * @param name
    * @return
    */
  def toTargetTokens(name: String): Seq[TargetToken] = {
    val tokens = AnnotationUtils.tokenize(name)
    val subComps = mutable.ArrayBuffer[TargetToken]()
    subComps += Ref(tokens.head)
    if(tokens.tail.nonEmpty) {
      tokens.tail.zip(tokens.tail.tail).foreach {
        case (".", value: String) => subComps += Field(value)
        case ("[", value: String) => subComps += Index(value.toInt)
        case other =>
      }
    }
    subComps
  }

  /** Checks if seq only contains [[TargetToken]]'s with select keywords
    * @param seq
    * @param keywords
    * @return
    */
  def isOnly(seq: Seq[TargetToken], keywords:String*): Boolean = {
    seq.map(_.is(keywords:_*)).foldLeft(false)(_ || _) && keywords.nonEmpty
  }

  /** @return [[Target]] from human-readable serialization */
  def deserialize(s: String): Target = {
    val regex = """(?=[~|>/:.\[@])"""
    s.split(regex).foldLeft(GenericTarget(None, None, Vector.empty)) { (t, tokenString) =>
      val value = tokenString.tail
      tokenString(0) match {
        case '~' if t.circuitOpt.isEmpty && t.moduleOpt.isEmpty && t.tokens.isEmpty =>
          if(value == "???") t else t.copy(circuitOpt = Some(value))
        case '|' if t.moduleOpt.isEmpty && t.tokens.isEmpty =>
          if(value == "???") t else t.copy(moduleOpt = Some(value))
        case '/' => t.add(Instance(value))
        case ':' => t.add(OfModule(value))
        case '>' => t.add(Ref(value))
        case '.' => t.add(Field(value))
        case '[' if value.dropRight(1).toInt >= 0 => t.add(Index(value.dropRight(1).toInt))
        case '@' if value == "clock" => t.add(Clock)
        case '@' if value == "init" => t.add(Init)
        case '@' if value == "reset" => t.add(Reset)
        case other => throw NamedException(s"Cannot deserialize Target: $s")
      }
    }.tryToComplete
  }
}

/** Represents incomplete or non-standard [[Target]]s
  * @param circuitOpt Optional circuit name
  * @param moduleOpt Optional module name
  * @param tokens [[TargetToken]]s to represent the target in a circuit and module
  */
case class GenericTarget(circuitOpt: Option[String],
                         moduleOpt: Option[String],
                         tokens: Vector[TargetToken]) extends Target {

  override def toGenericTarget: GenericTarget = this

  override def toNamed: Named = getComplete match {
    case Some(c: IsComponent) if c.isLocal => c.toNamed
    case Some(c: ModuleTarget) => c.toNamed
    case Some(c: CircuitTarget) => c.toNamed
    case other => throw Target.NamedException(s"Cannot convert $this to [[Named]]")
  }

  override def toTarget: CompleteTarget = getComplete.get

  override def getComplete: Option[CompleteTarget] = {
    if(!isComplete) None else {
      val target = this match {
        case GenericTarget(Some(c), None, Vector()) => CircuitTarget(c)
        case GenericTarget(Some(c), Some(m), Vector()) => ModuleTarget(c, m)
        case GenericTarget(Some(c), Some(m), Ref(r) +: component) => ReferenceTarget(c, m, Nil, r, component)
        case GenericTarget(Some(c), Some(m), Instance(i) +: OfModule(o) +: Vector()) => InstanceTarget(c, m, Nil, i, o)
        case GenericTarget(Some(c), Some(m), component) =>
          val path = getPath.getOrElse(Nil)
          (getRef, getInstanceOf) match {
            case (Some((r, comps)), _) => ReferenceTarget(c, m, path, r, comps)
            case (None, Some((i, o)))  => InstanceTarget(c, m, path, i, o)
          }
      }
      Some(target)
    }
  }

  override def isLocal: Boolean = !(getPath.nonEmpty && getPath.get.nonEmpty)

  /** If complete, return this [[GenericTarget]]'s path
    * @return
    */
  def getPath: Option[Seq[(Instance, OfModule)]] = if(isComplete) {
    val allInstOfs = tokens.grouped(2).collect { case Seq(i: Instance, o:OfModule) => (i, o)}.toSeq
    if(tokens.nonEmpty && tokens.last.isInstanceOf[OfModule]) Some(allInstOfs.dropRight(1)) else Some(allInstOfs)
  } else {
    None
  }

  /** If complete and a reference, return the reference and subcomponents
    * @return
    */
  def getRef: Option[(String, Seq[TargetToken])] = if(isComplete) {
    val (optRef, comps) = tokens.foldLeft((None: Option[String], Vector.empty[TargetToken])) {
      case ((None, v), Ref(r)) => (Some(r), v)
      case ((r: Some[String], comps), c) => (r, comps :+ c)
      case ((r, v), other) => (None, v)
    }
    optRef.map(x => (x, comps))
  } else {
    None
  }

  /** If complete and an instance target, return the instance and ofmodule
    * @return
    */
  def getInstanceOf: Option[(String, String)] = if(isComplete) {
    tokens.grouped(2).foldLeft(None: Option[(String, String)]) {
      case (instOf, Seq(i: Instance, o: OfModule)) => Some((i.value, o.value))
      case (instOf, _) => None
    }
  } else {
    None
  }

  /** Requires the last [[TargetToken]] in tokens to be one of the [[TargetToken]] keywords
    * @param default Return value if tokens is empty
    * @param keywords
    */
  private def requireLast(default: Boolean, keywords: String*): Unit = {
    val isOne = if (tokens.isEmpty) default else tokens.last.is(keywords: _*)
    require(isOne, s"${tokens.last} is not one of $keywords")
  }

  /** Appends a target token to tokens, asserts legality
    * @param token
    * @return
    */
  def add(token: TargetToken): GenericTarget = {
    token match {
      case _: Instance  => requireLast(true, "inst", "of")
      case _: OfModule  => requireLast(false, "inst")
      case _: Ref       => requireLast(true, "inst", "of")
      case _: Field     => requireLast(true, "ref", "[]", ".", "init", "clock", "reset")
      case _: Index     => requireLast(true, "ref", "[]", ".", "init", "clock", "reset")
      case Init         => requireLast(true, "ref", "[]", ".", "init", "clock", "reset")
      case Clock        => requireLast(true, "ref", "[]", ".", "init", "clock", "reset")
      case Reset        => requireLast(true, "ref", "[]", ".", "init", "clock", "reset")
    }
    this.copy(tokens = tokens :+ token)
  }

  /** Removes n number of target tokens from the right side of [[tokens]] */
  def remove(n: Int): GenericTarget = this.copy(tokens = tokens.dropRight(n))

  /** Optionally tries to append token to tokens, fails return is not a legal Target */
  def optAdd(token: TargetToken): Option[Target] = {
    try{
      Some(add(token))
    } catch {
      case _: IllegalArgumentException => None
    }
  }

  /** Checks whether the component is legal (incomplete is ok)
    * @return
    */
  def isLegal: Boolean = {
    try {
      var comp: GenericTarget = this.copy(tokens = Vector.empty)
      for(token <- tokens) {
        comp = comp.add(token)
      }
      true
    } catch {
      case _: IllegalArgumentException => false
    }
  }

  /** Checks whether the component is legal and complete, meaning the circuitOpt and moduleOpt are nonEmpty and
    * all Instance(_) are followed by OfModule(_)
    * @return
    */
  def isComplete: Boolean = {
    isLegal && (isCircuitTarget || isModuleTarget || (isComponentTarget && tokens.tails.forall {
      case Instance(_) +: OfModule(_) +: tail => true
      case Instance(_) +: x +: tail => false
      case x +: OfModule(_) +: tail => false
      case _ => true
    } ))
  }


  def isCircuitTarget: Boolean = circuitOpt.nonEmpty && moduleOpt.isEmpty && tokens.isEmpty
  def isModuleTarget: Boolean = circuitOpt.nonEmpty && moduleOpt.nonEmpty && tokens.isEmpty
  def isComponentTarget: Boolean = circuitOpt.nonEmpty && moduleOpt.nonEmpty && tokens.nonEmpty
}

/** Concretely points to a FIRRTL target, no generic selectors
  * IsLegal
  */
trait CompleteTarget extends Target {

  /** @return The circuit of this target */
  def circuit: String

  /** @return The [[CircuitTarget]] of this target's circuit */
  def circuitTarget: CircuitTarget = CircuitTarget(circuitOpt.get)

  def getComplete: Option[CompleteTarget] = Some(this)

  /** Adds another level of instance hierarchy
    * Example: Given root=A and instance=b, transforms (Top, B)/c:C -> (Top, A)/b:B/c:C
    * @param root
    * @param instance
    * @return
    */
  def addHierarchy(root: String, instance: String): IsComponent

  override def toTarget: CompleteTarget = this
}


/** A member of a FIRRTL Circuit (e.g. cannot point to a CircuitTarget)
  * Concrete Subclasses are: [[ModuleTarget]], [[InstanceTarget]], and [[ReferenceTarget]]
  */
trait IsMember extends CompleteTarget {

  /** @return Root module, e.g. top-level module of this target */
  def module: String

  /** @return Returns the instance hierarchy path, if one exists */
  def path: Seq[(Instance, OfModule)]

  /** @return Creates a path, assuming all Instance and OfModules in this [[IsMember]] is used as a path */
  def asPath: Seq[(Instance, OfModule)]

  /** @return Tokens of just this member's path */
  def justPath: Seq[TargetToken]

  /** @return Local tokens of what this member points (not a path) */
  def notPath: Seq[TargetToken]

  /** @return Same target without a path */
  def pathlessTarget: IsMember

  /** @return Member's path target */
  def pathTarget: CompleteTarget

  /** @return Member's top-level module target */
  def moduleTarget: ModuleTarget = ModuleTarget(circuitOpt.get, moduleOpt.get)

  /** @return Member's parent target */
  def targetParent: CompleteTarget

  /** @return List of local Instance Targets refering to each instance/ofModule in this member's path */
  def pathAsTargets: Seq[InstanceTarget] = {
    val targets = mutable.ArrayBuffer[InstanceTarget]()
    path.foldLeft((module, Vector.empty[InstanceTarget])) {
      case ((m, vec), (Instance(i), OfModule(o))) =>
        (o, vec :+ InstanceTarget(circuit, m, Nil, i, o))
    }._2
  }

  /** Resets this target to have a new path
    * @param newPath
    * @return
    */
  def setPathTarget(newPath: IsModule): CompleteTarget
}

/** References a module-like target (e.g. a [[ModuleTarget]] or an [[InstanceTarget]])
  */
trait IsModule extends IsMember {

  /** @return Creates a new Target, appending a ref */
  def ref(value: String): ReferenceTarget

  /** @return Creates a new Target, appending an instance and ofmodule */
  def instOf(instance: String, of: String): InstanceTarget
}

/** A component of a FIRRTL Module (e.g. cannot point to a CircuitTarget or ModuleTarget)
  */
trait IsComponent extends IsMember {

  /** @return The [[ModuleTarget]] of the module that directly contains this component */
  def encapsulatingModule: String = if(path.isEmpty) module else path.last._2.value

  /** Removes n levels of instance hierarchy
    *
    * Example: n=1, transforms (Top, A)/b:B/c:C -> (Top, B)/c:C
    * @param n
    * @return
    */
  def stripHierarchy(n: Int): IsMember

  override def toNamed: ComponentName = {
    if(isLocal){
      val mn = ModuleName(module, CircuitName(circuit))
      Seq(tokens:_*) match {
        case Seq(Ref(name)) => ComponentName(name, mn)
        case Ref(_) :: tail if Target.isOnly(tail, ".", "[]") =>
          val name = tokens.foldLeft(""){
            case ("", Ref(name)) => name
            case (string, Field(value)) => s"$string.$value"
            case (string, Index(value)) => s"$string[$value]"
          }
          ComponentName(name, mn)
        case Seq(Instance(name), OfModule(o)) => ComponentName(name, mn)
      }
    } else {
      throw new Exception(s"Cannot convert $this to [[ComponentName]]")
    }
  }

  override def justPath: Seq[TargetToken] = path.foldLeft(Vector.empty[TargetToken]) {
    case (vec, (i, o)) => vec ++ Seq(i, o)
  }

  override def pathTarget: IsModule = {
    if(path.isEmpty) moduleTarget else {
      val (i, o) = path.last
      InstanceTarget(circuit, module, path.dropRight(1), i.value, o.value)
    }
  }

  override def tokens = justPath ++ notPath

  override def isLocal = path.isEmpty
}


/** Target pointing to a FIRRTL [[firrtl.ir.Circuit]]
  * @param circuit Name of a FIRRTL circuit
  */
case class CircuitTarget(circuit: String) extends CompleteTarget {

  /** Creates a [[ModuleTarget]] of provided name and this circuit
    * @param m
    * @return
    */
  def module(m: String): ModuleTarget = ModuleTarget(circuit, m)

  override def circuitOpt: Option[String] = Some(circuit)

  override def moduleOpt: Option[String] = None

  override def tokens = Nil

  override def isLocal = true

  override def addHierarchy(root: String, instance: String): ReferenceTarget =
    ReferenceTarget(circuit, root, Nil, instance, Nil)

  override def toNamed: CircuitName = CircuitName(circuit)
}

/** Target pointing to a FIRRTL [[firrtl.ir.DefModule]]
  * @param circuit Circuit containing the module
  * @param module Name of the module
  */
case class ModuleTarget(circuit: String, module: String) extends IsModule {

  override def circuitOpt: Option[String] = Some(circuit)

  override def moduleOpt: Option[String] = Some(module)

  override def tokens: Seq[TargetToken] = Nil

  override def targetParent: CircuitTarget = CircuitTarget(circuit)

  override def addHierarchy(root: String, instance: String): InstanceTarget = InstanceTarget(circuit, root, Nil, instance, module)

  override def ref(value: String): ReferenceTarget = ReferenceTarget(circuit, module, Nil, value, Nil)

  override def instOf(instance: String, of: String): InstanceTarget = InstanceTarget(circuit, module, Nil, instance, of)

  override def asPath = Nil

  override def path: Seq[(Instance, OfModule)] = Nil

  override def justPath: Seq[TargetToken] = Nil

  override def notPath: Seq[TargetToken] = Nil

  override def pathlessTarget: ModuleTarget = this

  override def pathTarget: ModuleTarget = this

  override def isLocal = true

  override def setPathTarget(newPath: IsModule): IsModule = newPath

  override def toNamed: ModuleName = ModuleName(module, CircuitName(circuit))
}

/** Target pointing to a declared named component in a [[firrtl.ir.DefModule]]
  * This includes: [[firrtl.ir.Port]], [[firrtl.ir.DefWire]], [[firrtl.ir.DefRegister]], [[firrtl.ir.DefInstance]],
  *   [[firrtl.ir.DefMemory]], [[firrtl.ir.DefNode]]
  * @param circuit Name of the encapsulating circuit
  * @param module Name of the root module of this reference
  * @param path Path through instance/ofModules
  * @param ref Name of component
  * @param component Subcomponent of this reference, e.g. field or index
  */
case class ReferenceTarget(circuit: String,
                           module: String,
                           override val path: Seq[(Instance, OfModule)],
                           ref: String,
                           component: Seq[TargetToken]) extends IsComponent {

  /** @param value Index value of this target
    * @return A new [[ReferenceTarget]] to the specified index of this [[ReferenceTarget]]
    */
  def index(value: Int): ReferenceTarget = ReferenceTarget(circuit, module, path, ref, component :+ Index(value))

  /** @param value Field name of this target
    * @return A new [[ReferenceTarget]] to the specified field of this [[ReferenceTarget]]
    */
  def field(value: String): ReferenceTarget = ReferenceTarget(circuit, module, path, ref, component :+ Field(value))

  /** @return The initialization value of this reference, must be to a [[firrtl.ir.DefRegister]] */
  def init: ReferenceTarget = ReferenceTarget(circuit, module, path, ref, component :+ Init)

  /** @return The reset signal of this reference, must be to a [[firrtl.ir.DefRegister]] */
  def reset: ReferenceTarget = ReferenceTarget(circuit, module, path, ref, component :+ Reset)

  /** @return The clock signal of this reference, must be to a [[firrtl.ir.DefRegister]] */
  def clock: ReferenceTarget = ReferenceTarget(circuit, module, path, ref, component :+ Clock)

  override def circuitOpt: Option[String] = Some(circuit)

  override def moduleOpt: Option[String] = Some(module)

  override def targetParent: CompleteTarget = component match {
    case Nil =>
      if(path.isEmpty) moduleTarget else {
        val (i, o) = path.last
        InstanceTarget(circuit, module, path.dropRight(1), i.value, o.value)
      }
    case other => ReferenceTarget(circuit, module, path, ref, component.dropRight(1))
  }

  override def notPath: Seq[TargetToken] = Ref(ref) +: component

  override def addHierarchy(root: String, instance: String): ReferenceTarget =
    ReferenceTarget(circuit, root, (Instance(instance), OfModule(module)) +: path, ref, component)

  override def stripHierarchy(n: Int): ReferenceTarget = {
    require(path.size >= n, s"Cannot strip $n levels of hierarchy from $this")
    if(n == 0) this else {
      val newModule = path(n - 1)._2.value
      ReferenceTarget(circuit, newModule, path.drop(n), ref, component)
    }
  }

  override def pathlessTarget: ReferenceTarget = ReferenceTarget(circuit, encapsulatingModule, Nil, ref, component)

  override def setPathTarget(newPath: IsModule): ReferenceTarget =
    ReferenceTarget(newPath.circuit, newPath.module, newPath.asPath, ref, component)

  override def asPath: Seq[(Instance, OfModule)] = path
}

/** Points to an instance declaration of a module (termed an ofModule)
  * @param circuit Encapsulating circuit
  * @param module Root module (e.g. the base module of this target)
  * @param path Path through instance/ofModules
  * @param instance Name of the instance
  * @param ofModule Name of the instance's module
  */
case class InstanceTarget(circuit: String,
                          module: String,
                          override val path: Seq[(Instance, OfModule)],
                          instance: String,
                          ofModule: String) extends IsModule with IsComponent {

  /** @return a [[ReferenceTarget]] referring to this declaration of this instance */
  def asReference: ReferenceTarget = ReferenceTarget(circuit, module, path, instance, Nil)

  /** @return a [[ReferenceTarget]] referring to declaration of this ofModule */
  def ofModuleTarget: ModuleTarget = ModuleTarget(circuit, ofModule)

  override def circuitOpt: Option[String] = Some(circuit)

  override def moduleOpt: Option[String] = Some(module)

  override def targetParent: IsModule = {
    if(isLocal) ModuleTarget(circuit, module) else {
      val (newInstance, newOfModule) = path.last
      InstanceTarget(circuit, module, path.dropRight(1), newInstance.value, newOfModule.value)
    }
  }

  override def addHierarchy(root: String, inst: String): InstanceTarget =
    InstanceTarget(circuit, root, (Instance(inst), OfModule(module)) +: path, instance, ofModule)

  override def ref(value: String): ReferenceTarget = ReferenceTarget(circuit, module, asPath, value, Nil)

  override def instOf(inst: String, of: String): InstanceTarget = InstanceTarget(circuit, module, asPath, inst, of)

  override def stripHierarchy(n: Int): IsModule = {
    require(path.size >= n, s"Cannot strip $n levels of hierarchy from $this")
    if(n == 0) this else {
      val newModule = path(n - 1)._2.value
      InstanceTarget(circuit, newModule, path.drop(n), instance, ofModule)
    }
  }

  override def asPath: Seq[(Instance, OfModule)] = path :+ (Instance(instance), OfModule(ofModule))

  override def pathlessTarget: InstanceTarget = InstanceTarget(circuit, encapsulatingModule, Nil, instance, ofModule)

  override def notPath = Seq(Instance(instance), OfModule(ofModule))

  override def setPathTarget(newPath: IsModule): InstanceTarget =
    InstanceTarget(newPath.circuit, newPath.module, newPath.asPath, instance, ofModule)
}


/** Named classes associate an annotation with a component in a Firrtl circuit */
@deprecated("Use Target instead, will be removed in 1.3", "1.2")
sealed trait Named {
  def serialize: String
  def toTarget: CompleteTarget
}

@deprecated("Use Target instead, will be removed in 1.3", "1.2")
final case class CircuitName(name: String) extends Named {
  if(!validModuleName(name)) throw AnnotationException(s"Illegal circuit name: $name")
  def serialize: String = name
  def toTarget: CircuitTarget = CircuitTarget(name)
}

@deprecated("Use Target instead, will be removed in 1.3", "1.2")
final case class ModuleName(name: String, circuit: CircuitName) extends Named {
  if(!validModuleName(name)) throw AnnotationException(s"Illegal module name: $name")
  def serialize: String = circuit.serialize + "." + name
  def toTarget: ModuleTarget = ModuleTarget(circuit.name, name)
}

@deprecated("Use Target instead, will be removed in 1.3", "1.2")
final case class ComponentName(name: String, module: ModuleName) extends Named {
  if(!validComponentName(name)) throw AnnotationException(s"Illegal component name: $name")
  def expr: Expression = toExp(name)
  def serialize: String = module.serialize + "." + name
  def toTarget: ReferenceTarget = {
    Target.toTargetTokens(name).toList match {
      case Ref(r) :: components => ReferenceTarget(module.circuit.name, module.name, Nil, r, components)
      case other => throw Target.NamedException(s"Cannot convert $this into [[ReferenceTarget]]: $other")
    }
  }
}
