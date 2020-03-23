// See LICENSE for license details.

package firrtl

import annotations._
import firrtl.RenameMap.IllegalRenameException
import firrtl.annotations.TargetToken.{Field, Index, Instance, OfModule}

import scala.collection.mutable

object RenameMap {
  def apply(map: collection.Map[Named, Seq[Named]]): RenameMap = {
    val rm = new RenameMap
    rm.addMap(map)
    rm
  }

  def create(map: collection.Map[CompleteTarget, Seq[CompleteTarget]]): RenameMap = {
    val rm = new RenameMap
    rm.recordAll(map)
    rm
  }

  def apply(): RenameMap = new RenameMap

  abstract class RenameTargetException(reason: String) extends Exception(reason)
  case class IllegalRenameException(reason: String) extends RenameTargetException(reason)
  case class CircularRenameException(reason: String) extends RenameTargetException(reason)
}

/** Map old names to new names
  *
  * Transforms that modify names should return a [[RenameMap]] with the [[CircuitState]]
  * These are mutable datastructures for convenience
  * @define noteSelfRename @note Self renames *will* be recorded
  * @define noteDistinct @note Rename to/tos will be made distinct
  */
// TODO This should probably be refactored into immutable and mutable versions
final class RenameMap private (val underlying: mutable.HashMap[CompleteTarget, Seq[CompleteTarget]] = mutable.HashMap[CompleteTarget, Seq[CompleteTarget]](), val chained: Option[RenameMap] = None) {

  /** Chain a [[RenameMap]] with this [[RenameMap]]
    * @param next the map to chain with this map
    * $noteSelfRename
    * $noteDistinct
    */
  def andThen(next: RenameMap): RenameMap = {
    if (next.chained.isEmpty) {
      new RenameMap(next.underlying, chained = Some(this))
    } else {
      new RenameMap(next.underlying, chained = next.chained.map(this.andThen(_)))
    }
  }

  /** Record that the from [[firrtl.annotations.CircuitTarget CircuitTarget]] is renamed to another
    * [[firrtl.annotations.CircuitTarget CircuitTarget]]
    * @param from
    * @param to
    * $noteSelfRename
    * $noteDistinct
    */
  def record(from: CircuitTarget, to: CircuitTarget): Unit = completeRename(from, Seq(to))

  /** Record that the from [[firrtl.annotations.CircuitTarget CircuitTarget]] is renamed to another sequence of
    * [[firrtl.annotations.CircuitTarget CircuitTarget]]s
    * @param from
    * @param tos
    * $noteSelfRename
    * $noteDistinct
    */
  def record(from: CircuitTarget, tos: Seq[CircuitTarget]): Unit = completeRename(from, tos)

  /** Record that the from [[firrtl.annotations.IsMember Member]] is renamed to another [[firrtl.annotations.IsMember
    * IsMember]]
    * @param from
    * @param to
    * $noteSelfRename
    * $noteDistinct
    */
  def record(from: IsMember, to: IsMember): Unit = completeRename(from, Seq(to))

  /** Record that the from [[firrtl.annotations.IsMember IsMember]] is renamed to another sequence of
    * [[firrtl.annotations.IsMember IsMember]]s
    * @param from
    * @param tos
    * $noteSelfRename
    * $noteDistinct
    */
  def record(from: IsMember, tos: Seq[IsMember]): Unit = completeRename(from, tos)

  /** Records that the keys in map are also renamed to their corresponding value seqs. Only
    * ([[firrtl.annotations.CircuitTarget CircuitTarget]] -> Seq[ [[firrtl.annotations.CircuitTarget CircuitTarget]] ])
    * and ([[firrtl.annotations.IsMember IsMember]] -> Seq[ [[firrtl.annotations.IsMember IsMember]] ]) key/value
    * allowed
    * @param map
    * $noteSelfRename
    * $noteDistinct
    */
  def recordAll(map: collection.Map[CompleteTarget, Seq[CompleteTarget]]): Unit =
    map.foreach{
      case (from: IsComponent, tos: Seq[IsMember]) => completeRename(from, tos)
      case (from: IsModule, tos: Seq[IsMember]) => completeRename(from, tos)
      case (from: CircuitTarget, tos: Seq[CircuitTarget]) => completeRename(from, tos)
      case other => Utils.throwInternalError(s"Illegal rename: ${other._1} -> ${other._2}")
    }

  /** Records that a [[firrtl.annotations.CompleteTarget CompleteTarget]] is deleted
    * @param name
    */
  def delete(name: CompleteTarget): Unit = underlying(name) = Seq.empty

  /** Renames a [[firrtl.annotations.CompleteTarget CompleteTarget]]
    * @param t target to rename
    * @return renamed targets
    */
  def apply(t: CompleteTarget): Seq[CompleteTarget] = completeGet(t).getOrElse(Seq(t))

  /** Get renames of a [[firrtl.annotations.CircuitTarget CircuitTarget]]
    * @param key Target referencing the original circuit
    * @return Optionally return sequence of targets that key remaps to
    */
  def get(key: CompleteTarget): Option[Seq[CompleteTarget]] = completeGet(key)

  /** Get renames of a [[firrtl.annotations.CircuitTarget CircuitTarget]]
    * @param key Target referencing the original circuit
    * @return Optionally return sequence of targets that key remaps to
    */
  def get(key: CircuitTarget): Option[Seq[CircuitTarget]] = completeGet(key).map( _.map { case x: CircuitTarget => x } )

  /** Get renames of a [[firrtl.annotations.IsMember IsMember]]
    * @param key Target referencing the original member of the circuit
    * @return Optionally return sequence of targets that key remaps to
    */
  def get(key: IsMember): Option[Seq[IsMember]] = completeGet(key).map { _.map { case x: IsMember => x } }


  /** Create new [[RenameMap]] that merges this and renameMap
    * @param renameMap
    * @return
    */
  def ++ (renameMap: RenameMap): RenameMap = {
    val newChained = if (chained.nonEmpty && renameMap.chained.nonEmpty) {
      Some(chained.get ++ renameMap.chained.get)
    } else {
      chained.map(_.copy())
    }
    new RenameMap(underlying = underlying ++ renameMap.getUnderlying, chained = newChained)
  }

  /** Creates a deep copy of this [[RenameMap]]
    */
  def copy(chained: Option[RenameMap] = chained): RenameMap = {
    val ret = new RenameMap(chained = chained.map(_.copy()))
    ret.recordAll(underlying)
    ret
  }

  /** Returns the underlying map of rename information
    * @return
    */
  def getUnderlying: collection.Map[CompleteTarget, Seq[CompleteTarget]] = underlying

  /** @return Whether this [[RenameMap]] has collected any changes */
  def hasChanges: Boolean = underlying.nonEmpty

  def getReverseRenameMap: RenameMap = {
    val reverseMap = mutable.HashMap[CompleteTarget, Seq[CompleteTarget]]()
    underlying.keysIterator.foreach{ key =>
      apply(key).foreach { v =>
        reverseMap(v) = key +: reverseMap.getOrElse(v, Nil)
      }
    }
    RenameMap.create(reverseMap)
  }

  def keys: Iterator[CompleteTarget] = underlying.keysIterator

  /** Serialize the underlying remapping of keys to new targets
    * @return
    */
  def serialize: String = underlying.map { case (k, v) =>
    k.serialize + "=>" + v.map(_.serialize).mkString(", ")
  }.mkString("\n")

  /** Records which local InstanceTargets will require modification.
    * Used to reduce time to rename nonlocal targets who's path does not require renaming
    */
  private val sensitivity = mutable.HashSet[IsComponent]()

  /** Caches results of recursiveGet. Is cleared any time a new rename target is added
    */
  private val getCache = mutable.HashMap[CompleteTarget, Seq[CompleteTarget]]()

  /** Caches results of referenceGet. Is cleared any time a new rename target is added
    */
  private val traverseTokensCache = mutable.HashMap[ReferenceTarget, Option[Seq[IsComponent]]]()
  private val traverseHierarchyCache = mutable.HashMap[ReferenceTarget, Option[Seq[IsComponent]]]()
  private val traverseLeftCache = mutable.HashMap[InstanceTarget, Option[Seq[IsModule]]]()
  private val traverseRightCache = mutable.HashMap[InstanceTarget, Seq[IsModule]]()

  /** Updates [[sensitivity]]
    * @param from original target
    * @param to new target
    */
  private def recordSensitivity(from: CompleteTarget, to: CompleteTarget): Unit = {
    (from, to) match {
      case (f: IsMember, t: IsMember) =>
        val fromSet = f.pathAsTargets.toSet
        val toSet = t.pathAsTargets
        sensitivity ++= (fromSet -- toSet)
        sensitivity ++= (fromSet.map(_.asReference) -- toSet.map(_.asReference))
      case other =>
    }
  }

  /** Get renames of a [[firrtl.annotations.CompleteTarget CompleteTarget]]
    * @param key Target referencing the original circuit
    * @return Optionally return sequence of targets that key remaps to
    */
  private def completeGet(key: CompleteTarget): Option[Seq[CompleteTarget]] = {
    if (chained.nonEmpty) {
      val chainedRet = chained.get.completeGet(key).getOrElse(Seq(key))
      if (chainedRet.isEmpty) {
        Some(chainedRet)
      } else {
        val hereRet = (chainedRet.flatMap { target =>
          hereCompleteGet(target).getOrElse(Seq(target))
        }).distinct
        if (hereRet.size == 1 && hereRet.head == key) { None } else { Some(hereRet) }
      }
    } else {
      hereCompleteGet(key)
    }
  }

  private def hereCompleteGet(key: CompleteTarget): Option[Seq[CompleteTarget]] = {
    val errors = mutable.ArrayBuffer[String]()
    val ret = if(hasChanges) {
      val ret = recursiveGet(errors)(key)
      if(errors.nonEmpty) { throw IllegalRenameException(errors.mkString("\n")) }
      if(ret.size == 1 && ret.head == key) { None } else { Some(ret) }
    } else { None }
    ret
  }


  /** Checks for renames of only the component portion of a [[ReferenceTarget]]
    * Recursively checks parent [[ReferenceTarget]]s until a match is found
    * Parents with longer paths/components are checked first. longer paths take
    * priority over longer components.
    *
    * For example, the order that targets are checked for ~Top|Top/a:A/b:B/c:C>f.g is:
    * ~Top|Top/a:A/b:B/c:C>f.g
    * ~Top|Top/a:A/b:B/c:C>f
    * ~Top|A/b:B/c:C>f.g
    * ~Top|A/b:B/c:C>f
    * ~Top|B/c:C>f.g
    * ~Top|B/c:C>f
    * ~Top|C>f.g
    * ~Top|C>f
    *
    * @param errors Used to record illegal renames
    * @param key Target to rename
    * @return Renamed targets if a match is found, otherwise None
    */
  private def referenceGet(errors: mutable.ArrayBuffer[String])(key: ReferenceTarget): Option[Seq[IsComponent]] = {
    def traverseTokens(key: ReferenceTarget): Option[Seq[IsComponent]] = traverseTokensCache.getOrElseUpdate(key, {
      if (underlying.contains(key)) {
        Some(underlying(key).flatMap {
          case comp: IsComponent => Some(comp)
          case other =>
            errors += s"reference ${key.targetParent} cannot be renamed to a non-component ${other}"
            None
        })
      } else {
        key match {
          case t: ReferenceTarget if t.component.nonEmpty =>
            val last = t.component.last
            val parent = t.copy(component = t.component.dropRight(1))
            traverseTokens(parent).map(_.flatMap { x =>
              (x, last) match {
                case (t2: InstanceTarget, Field(f)) => Some(t2.ref(f))
                case (t2: ReferenceTarget, Field(f)) => Some(t2.field(f))
                case (t2: ReferenceTarget, Index(i)) => Some(t2.index(i))
                case other =>
                  errors += s"Illegal rename: ${key.targetParent} cannot be renamed to ${other._1} - must rename $key directly"
                  None
              }
            })
          case t: ReferenceTarget => None
        }
      }
    })

    def traverseHierarchy(key: ReferenceTarget): Option[Seq[IsComponent]] = traverseHierarchyCache.getOrElseUpdate(key, {
      val tokenRenamed = traverseTokens(key)
      if (tokenRenamed.nonEmpty) {
        tokenRenamed
      } else {
        key match {
          case t: ReferenceTarget if t.isLocal => None
          case t: ReferenceTarget =>
            val encapsulatingInstance = t.path.head._1.value
            val stripped = t.stripHierarchy(1)
            traverseHierarchy(stripped).map(_.map {
              _.addHierarchy(t.module, encapsulatingInstance)
            })
        }
      }
    })

    traverseHierarchy(key)
  }

  /** Checks for renames of only the path portion of an [[InstanceTarget]]
    * Recursively checks parent [[IsModule]]s until a match is found
    * First checks all parent paths from longest to shortest. Then
    * recursively checks paths leading to the encapsulating module.
    * Stops on the first match found.
    *
    * For example, the order that targets are checked for ~Top|Top/a:A/b:B/c:C is:
    * ~Top|Top/a:A/b:B/c:C
    * ~Top|A/b:B/c:C
    * ~Top|B/c:C
    * ~Top|Top/a:A/b:B
    * ~Top|A/b:B
    * ~Top|Top/a:A
    *
    * @param errors Used to record illegal renames
    * @param key Target to rename
    * @return Renamed targets, contains only the original target if none are found
    */
  private def instanceGet(errors: mutable.ArrayBuffer[String])(key: InstanceTarget): Seq[IsModule] = {
    def traverseLeft(key: InstanceTarget): Option[Seq[IsModule]] = traverseLeftCache.getOrElseUpdate(key, {
      val getOpt = underlying.get(key)

      if (getOpt.nonEmpty) {
        getOpt.map(_.flatMap {
          case isMod: IsModule => Some(isMod)
          case other =>
            errors += s"IsModule: $key cannot be renamed to non-IsModule $other"
            None
        })
      } else {
        key match {
          case t: InstanceTarget if t.isLocal => None
          case t: InstanceTarget =>
            val (Instance(outerInst), OfModule(outerMod)) = t.path.head
            val stripped = t.copy(path = t.path.tail, module = outerMod)
            traverseLeft(stripped).map(_.map {
              case absolute if absolute.path.nonEmpty && absolute.circuit == absolute.path.head._2.value => absolute
              case relative => relative.addHierarchy(t.module, outerInst)
            })
        }
      }
    })

    def traverseRight(key: InstanceTarget): Seq[IsModule] = traverseRightCache.getOrElseUpdate(key, {
      val findLeft = traverseLeft(key)
      if (findLeft.nonEmpty) {
        findLeft.get
      } else {
        key match {
          case t: InstanceTarget if t.isLocal => Seq(key)
          case t: InstanceTarget =>
            val (Instance(i), OfModule(m)) = t.path.last
            val parent = t.copy(path = t.path.dropRight(1), instance = i, ofModule = m)
            traverseRight(parent).map(_.instOf(t.instance, t.ofModule))
        }
      }
    })

    traverseRight(key)
  }

  private def circuitGet(errors: mutable.ArrayBuffer[String])(key: CircuitTarget): Seq[CircuitTarget] = {
    underlying.get(key).map(_.flatMap {
      case c: CircuitTarget => Some(c)
      case other =>
        errors += s"Illegal rename: $key cannot be renamed to non-circuit target: $other"
        None
    }).getOrElse(Seq(key))
  }

  private def moduleGet(errors: mutable.ArrayBuffer[String])(key: ModuleTarget): Seq[IsModule] = {
    underlying.get(key).map(_.flatMap {
      case mod: IsModule => Some(mod)
      case other =>
        errors += s"Illegal rename: $key cannot be renamed to non-module target: $other"
        None
    }).getOrElse(Seq(key))
  }

  /** Recursively renames a target so the returned targets are complete renamed
    * @param errors Used to record illegal renames
    * @param key Target to rename
    * @return Renamed targets
    */
  private def recursiveGet(errors: mutable.ArrayBuffer[String])(key: CompleteTarget): Seq[CompleteTarget] = {
    if(getCache.contains(key)) {
      getCache(key)
    } else {
      val getter = recursiveGet(errors)(_)

      // rename just the first level e.g. just rename component/path portion for ReferenceTargets
      val topRename = key match {
        case t: CircuitTarget => Seq(t)
        case t: ModuleTarget => Seq(t)
        case t: InstanceTarget => instanceGet(errors)(t)
        case ref: ReferenceTarget if ref.isLocal => referenceGet(errors)(ref).getOrElse(Seq(ref))
        case ref @ ReferenceTarget(c, m, p, r, t) =>
          val (Instance(inst), OfModule(ofMod)) = p.last
          referenceGet(errors)(ref).getOrElse {
            val parent = InstanceTarget(c, m, p.dropRight(1), inst, ofMod)
            instanceGet(errors)(parent).map(ref.setPathTarget(_))
          }
      }

      // rename the next level up
      val midRename = topRename.flatMap {
        case t: CircuitTarget => Seq(t)
        case t: ModuleTarget => moduleGet(errors)(t)
        case t: IsComponent =>
          // rename all modules on the path
          val renamedPath = t.asPath.reverse.foldLeft((Option.empty[IsModule], Seq.empty[(Instance, OfModule)])) {
            case (absolute@ (Some(_), _), _) => absolute
            case ((None, children), pair) =>
              val pathMod = ModuleTarget(t.circuit, pair._2.value)
              moduleGet(errors)(pathMod) match {
                case Seq(absolute: IsModule) if absolute.circuit == t.circuit && absolute.module == t.circuit =>
                  val withChildren = children.foldLeft(absolute) {
                    case (target, (inst, ofMod)) => target.instOf(inst.value, ofMod.value)
                  }
                  (Some(withChildren), children)
                case Seq(isMod: ModuleTarget) if isMod.circuit == t.circuit =>
                  (None, pair.copy(_2 = OfModule(isMod.module)) +: children)
                case Seq(isMod: InstanceTarget) if isMod.circuit == t.circuit =>
                  (None, pair +: children)
                case other =>
                  val error = s"ofModule ${pathMod} cannot be renamed to $other " +
                    "- an ofModule can only be renamed to a single IsModule with the same circuit"
                  errors += error
                  (None, pair +: children)
            }
          }

          renamedPath match {
            case (Some(absolute), _) =>
              t match {
                case ref: ReferenceTarget => Seq(ref.copy(circuit = absolute.circuit, module = absolute.module, path = absolute.asPath))
                case inst: InstanceTarget => Seq(absolute)
              }
            case (_, children) =>
              // rename the root module and set the new path
              moduleGet(errors)(ModuleTarget(t.circuit, t.module)).map { mod =>
                val newPath = mod.asPath ++ children

                t match {
                  case ref: ReferenceTarget => ref.copy(circuit = mod.circuit, module = mod.module, path = newPath)
                  case inst: InstanceTarget =>
                    val (Instance(newInst), OfModule(newOfMod)) = newPath.last
                    inst.copy(circuit = mod.circuit,
                      module = mod.module,
                      path = newPath.dropRight(1),
                      instance = newInst,
                      ofModule = newOfMod)
                }
              }
          }
      }

      // rename the last level
      val botRename = midRename.flatMap {
        case t: CircuitTarget => circuitGet(errors)(t)
        case t: ModuleTarget =>
          circuitGet(errors)(CircuitTarget(t.circuit)).map {
            case CircuitTarget(c) => t.copy(circuit = c)
          }
        case t: IsComponent =>
          circuitGet(errors)(CircuitTarget(t.circuit)).map {
            case CircuitTarget(c) =>
              t match {
                case ref: ReferenceTarget => ref.copy(circuit = c)
                case inst: InstanceTarget => inst.copy(circuit = c)
              }
          }
      }

      // Cache result
      getCache(key) = botRename
      botRename
    }
  }

  /** Fully rename `from` to `tos`
    * @param from
    * @param tos
    */
  private def completeRename(from: CompleteTarget, tos: Seq[CompleteTarget]): Unit = {
    tos.foreach{recordSensitivity(from, _)}
    val existing = underlying.getOrElse(from, Vector.empty)
    val updated = (existing ++ tos).distinct
    underlying(from) = updated
    getCache.clear()
    traverseTokensCache.clear()
    traverseHierarchyCache.clear()
    traverseLeftCache.clear()
    traverseRightCache.clear()
  }

  /* DEPRECATED ACCESSOR/SETTOR METHODS WITH [[firrtl.ir.Named Named]] */

  def rename(from: Named, to: Named): Unit = rename(from, Seq(to))

  def rename(from: Named, tos: Seq[Named]): Unit = recordAll(Map(from.toTarget -> tos.map(_.toTarget)))

  def rename(from: ComponentName, to: ComponentName): Unit = record(from, to)

  def rename(from: ComponentName, tos: Seq[ComponentName]): Unit = record(from, tos.map(_.toTarget))

  def delete(name: CircuitName): Unit = underlying(name) = Seq.empty

  def delete(name: ModuleName): Unit = underlying(name) = Seq.empty

  def delete(name: ComponentName): Unit = underlying(name) = Seq.empty

  def addMap(map: collection.Map[Named, Seq[Named]]): Unit =
    recordAll(map.map { case (key, values) => (Target.convertNamed2Target(key), values.map(Target.convertNamed2Target)) })

  def get(key: CircuitName): Option[Seq[CircuitName]] = {
    get(Target.convertCircuitName2CircuitTarget(key)).map(_.collect{ case c: CircuitTarget => c.toNamed })
  }

  def get(key: ModuleName): Option[Seq[ModuleName]] = {
    get(Target.convertModuleName2ModuleTarget(key)).map(_.collect{ case m: ModuleTarget => m.toNamed })
  }

  def get(key: ComponentName): Option[Seq[ComponentName]] = {
    get(Target.convertComponentName2ReferenceTarget(key)).map(_.collect{ case c: IsComponent => c.toNamed })
  }

  def get(key: Named): Option[Seq[Named]] = key match {
    case t: CompleteTarget => get(t)
    case other => get(key.toTarget).map(_.collect{ case c: IsComponent => c.toNamed })
  }


  // Mutable helpers - APIs that set these are deprecated!
  private var circuitName: String = ""
  private var moduleName: String = ""

  /** Sets mutable state to record current module we are visiting
    * @param module
    */
  def setModule(module: String): Unit = moduleName = module

  /** Sets mutable state to record current circuit we are visiting
    * @param circuit
    */
  def setCircuit(circuit: String): Unit = circuitName = circuit

  /** Records how a reference maps to a new reference
    * @param from
    * @param to
    */
  def rename(from: String, to: String): Unit = rename(from, Seq(to))

  /** Records how a reference maps to a new reference
    * The reference's root module and circuit are determined by whomever called setModule or setCircuit last
    * @param from
    * @param tos
    */
  def rename(from: String, tos: Seq[String]): Unit = {
    val mn = ModuleName(moduleName, CircuitName(circuitName))
    val fromName = ComponentName(from, mn).toTarget
    val tosName = tos map { to => ComponentName(to, mn).toTarget }
    record(fromName, tosName)
  }

  /** Records named reference is deleted
    * The reference's root module and circuit are determined by whomever called setModule or setCircuit last
    * @param name
    */
  def delete(name: String): Unit = {
    Target(Some(circuitName), Some(moduleName), AnnotationUtils.toSubComponents(name)).getComplete match {
      case Some(t: CircuitTarget) => delete(t)
      case Some(m: IsMember) => delete(m)
      case other =>
    }
  }

  /** Records that references in names are all deleted
    * The reference's root module and circuit are determined by whomever called setModule or setCircuit last
    * @param names
    */
  def delete(names: Seq[String]): Unit = names.foreach(delete(_))
}
