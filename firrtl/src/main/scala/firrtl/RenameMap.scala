// SPDX-License-Identifier: Apache-2.0

package firrtl

import annotations._
import firrtl.RenameMap.IllegalRenameException
import firrtl.annotations.TargetToken.{Field, Index, Instance, OfModule}
import firrtl.renamemap._

import scala.collection.mutable
import scala.annotation.nowarn

object RenameMap {
  def apply(map: collection.Map[Named, Seq[Named]]): RenameMap = MutableRenameMap.fromNamed(map)

  def create(map: collection.Map[CompleteTarget, Seq[CompleteTarget]]): RenameMap = MutableRenameMap(map)

  /** Initialize a new RenameMap */
  def apply(): RenameMap = new MutableRenameMap

  abstract class RenameTargetException(reason: String) extends Exception(reason)
  case class IllegalRenameException(reason: String) extends RenameTargetException(reason)
  case class CircularRenameException(reason: String) extends RenameTargetException(reason)
}

/** Map old names to new names
  *
  * These are mutable datastructures for convenience
  * @define noteSelfRename @note Self renames *will* be recorded
  * @define noteDistinct @note Rename to/tos will be made distinct
  */
// TODO This should probably be refactored into immutable and mutable versions
sealed trait RenameMap {

  protected def _underlying: mutable.HashMap[CompleteTarget, Seq[CompleteTarget]]

  protected def _chained: Option[RenameMap]

  // This is a private internal API for transforms where the .distinct operation is very expensive
  // (eg. LowerTypes). The onus is on the user of this API to be very careful and not inject
  // duplicates. This is a bad, hacky API that no one should use
  protected def doDistinct: Boolean

  /** Chain a [[RenameMap]] with this [[RenameMap]]
    * @param next the map to chain with this map
    * $noteSelfRename
    * $noteDistinct
    */
  def andThen(next: RenameMap): RenameMap = {
    if (next._chained.isEmpty) {
      new MutableRenameMap(next._underlying, Some(this))
    } else {
      new MutableRenameMap(next._underlying, next._chained.map(this.andThen(_)))
    }
  }

  protected def _recordAll(map: collection.Map[CompleteTarget, Seq[CompleteTarget]]): Unit =
    map.foreach {
      case (from: IsComponent, tos: Seq[_]) => completeRename(from, tos)
      case (from: IsModule, tos: Seq[_]) => completeRename(from, tos)
      case (from: CircuitTarget, tos: Seq[_]) => completeRename(from, tos)
      case other => Utils.throwInternalError(s"Illegal rename: ${other._1} -> ${other._2}")
    }

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
  def get(key: CircuitTarget): Option[Seq[CircuitTarget]] = completeGet(key).map(_.map { case x: CircuitTarget => x })

  /** Get renames of a [[firrtl.annotations.IsMember IsMember]]
    * @param key Target referencing the original member of the circuit
    * @return Optionally return sequence of targets that key remaps to
    */
  def get(key: IsMember): Option[Seq[IsMember]] = completeGet(key).map { _.map { case x: IsMember => x } }

  /** Create new [[RenameMap]] that merges this and renameMap
    * @param renameMap
    * @return
    */
  def ++(renameMap: RenameMap): RenameMap = {
    val newChained = if (_chained.nonEmpty && renameMap._chained.nonEmpty) {
      Some(_chained.get ++ renameMap._chained.get)
    } else {
      _chained.map(_._copy())
    }
    new MutableRenameMap(_underlying ++ renameMap._underlying, newChained)
  }

  private def _copy(chained: Option[RenameMap] = _chained): RenameMap = {
    val ret = new MutableRenameMap(_chained = _chained.map(_._copy()))
    ret.recordAll(_underlying)
    ret
  }

  /** @return Whether this [[RenameMap]] has collected any changes */
  def hasChanges: Boolean = _underlying.nonEmpty

  /** Visualize the [[RenameMap]]
    */
  def serialize: String = _underlying.map {
    case (k, v) =>
      k.serialize + "=>" + v.map(_.serialize).mkString(", ")
  }.mkString("\n")

  /** Records which local InstanceTargets will require modification.
    * Used to reduce time to rename nonlocal targets who's path does not require renaming
    */
  private val sensitivity = mutable.HashSet[IsComponent]()

  /** Caches results of referenceGet. Is cleared any time a new rename target is added
    */
  private val traverseTokensCache = mutable.HashMap[ReferenceTarget, Option[Seq[IsComponent]]]()
  private val traverseHierarchyCache = mutable.HashMap[ReferenceTarget, Option[Seq[IsComponent]]]()
  private val traverseLeftCache = mutable.HashMap[InstanceTarget, Option[Seq[IsModule]]]()
  private val traverseRightCache = mutable.HashMap[InstanceTarget, Option[Seq[IsModule]]]()

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
    if (_chained.nonEmpty) {
      val chainedRet = _chained.get.completeGet(key).getOrElse(Seq(key))
      if (chainedRet.isEmpty) {
        Some(chainedRet)
      } else {
        val hereRet = (chainedRet.flatMap { target =>
          hereCompleteGet(target).getOrElse(Seq(target))
        }).distinct
        if (hereRet.size == 1 && hereRet.head == key) { None }
        else { Some(hereRet) }
      }
    } else {
      hereCompleteGet(key)
    }
  }

  private def hereCompleteGet(key: CompleteTarget): Option[Seq[CompleteTarget]] = {
    val errors = mutable.ArrayBuffer[String]()
    val ret = if (hasChanges) {
      val ret = recursiveGet(errors)(key)
      if (errors.nonEmpty) { throw IllegalRenameException(errors.mkString("\n")) }
      if (ret.size == 1 && ret.head == key) { None }
      else { Some(ret) }
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
    def traverseTokens(key: ReferenceTarget): Option[Seq[IsComponent]] = traverseTokensCache.getOrElseUpdate(
      key, {
        if (_underlying.contains(key)) {
          Some(_underlying(key).flatMap {
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
      }
    )

    def traverseHierarchy(key: ReferenceTarget): Option[Seq[IsComponent]] = traverseHierarchyCache.getOrElseUpdate(
      key, {
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
      }
    )

    traverseHierarchy(key)
  }

  /** Checks for renames of only the path portion of an [[InstanceTarget]]
    * Recursively checks parent [[IsModule]]s until a match is found. First
    * checks all parent paths from longest to shortest. Then recursively checks
    * paths leading to the encapsulating module.  Stops on the first match
    * found. When a match is found the parent instances that were stripped off
    * are added back unless the child is renamed to an absolute instance target
    * (target.module == target.circuit).
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
    * @return Renamed targets if a match is found, otherwise None
    */
  private def instanceGet(errors: mutable.ArrayBuffer[String])(key: InstanceTarget): Option[Seq[IsModule]] = {
    def traverseLeft(key: InstanceTarget): Option[Seq[IsModule]] = traverseLeftCache.getOrElseUpdate(
      key, {
        val getOpt = _underlying.get(key)

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
                case absolute if absolute.circuit == absolute.module => absolute
                case relative                                        => relative.addHierarchy(t.module, outerInst)
              })
          }
        }
      }
    )

    def traverseRight(key: InstanceTarget): Option[Seq[IsModule]] = traverseRightCache.getOrElseUpdate(
      key, {
        val findLeft = traverseLeft(key)
        if (findLeft.isDefined) {
          findLeft
        } else {
          key match {
            case t: InstanceTarget if t.isLocal => None
            case t: InstanceTarget =>
              val (Instance(i), OfModule(m)) = t.path.last
              val parent = t.copy(path = t.path.dropRight(1), instance = i, ofModule = m)
              traverseRight(parent).map(_.map(_.instOf(t.instance, t.ofModule)))
          }
        }
      }
    )

    traverseRight(key)
  }

  private def circuitGet(errors: mutable.ArrayBuffer[String])(key: CircuitTarget): Seq[CircuitTarget] = {
    _underlying
      .get(key)
      .map(_.flatMap {
        case c: CircuitTarget => Some(c)
        case other =>
          errors += s"Illegal rename: $key cannot be renamed to non-circuit target: $other"
          None
      })
      .getOrElse(Seq(key))
  }

  private def moduleGet(errors: mutable.ArrayBuffer[String])(key: ModuleTarget): Option[Seq[IsModule]] = {
    _underlying
      .get(key)
      .map(_.flatMap {
        case mod: IsModule => Some(mod)
        case other =>
          errors += s"Illegal rename: $key cannot be renamed to non-module target: $other"
          None
      })
  }

  // the possible results returned by ofModuleGet
  private sealed trait OfModuleRenameResult

  // an OfModule was renamed to an absolute module (t.module == t.circuit) and parent OfModules were not renamed
  private case class AbsoluteOfModule(isMod: IsModule) extends OfModuleRenameResult

  // all OfModules were renamed to relative modules, and their paths are concatenated together
  private case class RenamedOfModules(children: Seq[(Instance, OfModule)]) extends OfModuleRenameResult

  // an OfModule was deleted, thus the entire target was deleted
  private case object DeletedOfModule extends OfModuleRenameResult

  // no renamed of OfModules were found
  private case object NoOfModuleRenames extends OfModuleRenameResult

  /** Checks for renames of [[OfModule]]s in the path of and [[IsComponent]]
    * from right to left.  Renamed [[OfModule]]s must all have the same circuit
    * name and cannot be renamed to more than one target. [[OfModule]]s that
    * are renamed to relative targets are inlined into the path of the original
    * target. If it is renamed to an absolute target, then it becomes the
    * parent path of the original target and renaming stops.
    *
    * Examples:
    *
    * RenameMap(~Top|A -> ~Top|Foo/bar:Bar):
    * ofModuleGet(~Top|Top/a:A/b:B) == RenamedOfModules(a:Foo, bar:Bar, b:B)
    *
    * RenameMap(~Top|B -> ~Top|Top/foo:Foo/bar:Bar, ~Top|A -> ~Top|C):
    * ofModuleGet(~Top|Top/a:A/b:B) == AbsoluteOfModule(~Top|Top/foo:Foo/bar:Bar/b:B)
    *
    * RenameMap(~Top|B -> deleted, ~Top|A -> ~Top|C):
    * ofModuleGet(~Top|Top/a:A/b:B) == DeletedOfModule
    *
    * RenameMap.empty
    * ofModuleGet(~Top|Top/a:A/b:B) == NoOfModuleRenames
    *
    * @param errors Used to record illegal renames
    * @param key Target to rename
    * @return rename results (see examples)
    */
  private def ofModuleGet(errors: mutable.ArrayBuffer[String])(key: IsComponent): OfModuleRenameResult = {
    val circuit = key.circuit
    def renameOfModules(
      path:          Seq[(Instance, OfModule)],
      foundRename:   Boolean,
      newCircuitOpt: Option[String],
      children:      Seq[(Instance, OfModule)]
    ): OfModuleRenameResult = {
      if (path.isEmpty && foundRename) {
        RenamedOfModules(children)
      } else if (path.isEmpty) {
        NoOfModuleRenames
      } else {
        val pair = path.head
        val pathMod = ModuleTarget(circuit, pair._2.value)
        moduleGet(errors)(pathMod) match {
          case None => renameOfModules(path.tail, foundRename, newCircuitOpt, pair +: children)
          case Some(rename) =>
            if (newCircuitOpt.isDefined && rename.exists(_.circuit != newCircuitOpt.get)) {
              val error = s"ofModule ${pathMod} of target ${key.serialize} cannot be renamed to $rename " +
                s"- renamed ofModules must have the same circuit name, expected circuit ${newCircuitOpt.get}"
              errors += error
            }
            rename match {
              case Seq(absolute: IsModule) if absolute.module == absolute.circuit =>
                val withChildren = children.foldLeft(absolute) {
                  case (target, (inst, ofMod)) => target.instOf(inst.value, ofMod.value)
                }
                AbsoluteOfModule(withChildren)
              case Seq(isMod: ModuleTarget) =>
                val newPair = pair.copy(_2 = OfModule(isMod.module))
                renameOfModules(path.tail, true, Some(isMod.circuit), newPair +: children)
              case Seq(isMod: InstanceTarget) =>
                val newPair = pair.copy(_2 = OfModule(isMod.module))
                renameOfModules(path.tail, true, Some(isMod.circuit), newPair +: (isMod.asPath ++ children))
              case Nil =>
                DeletedOfModule
              case other =>
                val error = s"ofModule ${pathMod} of target ${key.serialize} cannot be renamed to $other " +
                  "- an ofModule can only be deleted or renamed to a single IsModule"
                errors += error
                renameOfModules(path.tail, foundRename, newCircuitOpt, pair +: children)
            }
        }
      }
    }
    renameOfModules(key.asPath.reverse, false, None, Nil)
  }

  /** Recursively renames a target so the returned targets are complete renamed
    * @param errors Used to record illegal renames
    * @param key Target to rename
    * @return Renamed targets
    */
  private def recursiveGet(errors: mutable.ArrayBuffer[String])(key: CompleteTarget): Seq[CompleteTarget] = {
    // rename just the component portion; path/ref/component for ReferenceTargets or path/instance for InstanceTargets
    val componentRename = key match {
      case t:   CircuitTarget                  => None
      case t:   ModuleTarget                   => None
      case t:   InstanceTarget                 => instanceGet(errors)(t)
      case ref: ReferenceTarget if ref.isLocal => referenceGet(errors)(ref)
      case ref @ ReferenceTarget(c, m, p, r, t) =>
        val (Instance(inst), OfModule(ofMod)) = p.last
        val refGet = referenceGet(errors)(ref)
        if (refGet.isDefined) {
          refGet
        } else {
          val parent = InstanceTarget(c, m, p.dropRight(1), inst, ofMod)
          instanceGet(errors)(parent).map(_.map(ref.setPathTarget(_)))
        }
    }

    // if no component rename was found, look for Module renames; root module/OfModules in path
    val moduleRename = if (componentRename.isDefined) {
      componentRename
    } else {
      key match {
        case t: CircuitTarget => None
        case t: ModuleTarget => moduleGet(errors)(t)
        case t: IsComponent =>
          ofModuleGet(errors)(t) match {
            case AbsoluteOfModule(absolute) =>
              t match {
                case ref: ReferenceTarget =>
                  Some(Seq(ref.copy(circuit = absolute.circuit, module = absolute.module, path = absolute.asPath)))
                case inst: InstanceTarget => Some(Seq(absolute))
              }
            case RenamedOfModules(children) =>
              // rename the root module and set the new path
              val modTarget = ModuleTarget(t.circuit, t.module)
              val result = moduleGet(errors)(modTarget).getOrElse(Seq(modTarget)).map { mod =>
                val newPath = mod.asPath ++ children

                t match {
                  case ref:  ReferenceTarget => ref.copy(circuit = mod.circuit, module = mod.module, path = newPath)
                  case inst: InstanceTarget =>
                    val (Instance(newInst), OfModule(newOfMod)) = newPath.last
                    inst.copy(
                      circuit = mod.circuit,
                      module = mod.module,
                      path = newPath.dropRight(1),
                      instance = newInst,
                      ofModule = newOfMod
                    )
                }
              }
              Some(result)
            case DeletedOfModule => Some(Nil)
            case NoOfModuleRenames =>
              val modTarget = ModuleTarget(t.circuit, t.module)
              val children = t.asPath
              moduleGet(errors)(modTarget).map(_.map { mod =>
                val newPath = mod.asPath ++ children

                t match {
                  case ref:  ReferenceTarget => ref.copy(circuit = mod.circuit, module = mod.module, path = newPath)
                  case inst: InstanceTarget =>
                    val (Instance(newInst), OfModule(newOfMod)) = newPath.last
                    inst.copy(
                      circuit = mod.circuit,
                      module = mod.module,
                      path = newPath.dropRight(1),
                      instance = newInst,
                      ofModule = newOfMod
                    )
                }
              })
          }
      }
    }

    // if no module renames were found, look for circuit renames;
    val circuitRename = if (moduleRename.isDefined) {
      moduleRename.get
    } else {
      key match {
        case t: CircuitTarget => circuitGet(errors)(t)
        case t: ModuleTarget =>
          circuitGet(errors)(CircuitTarget(t.circuit)).map {
            case CircuitTarget(c) => t.copy(circuit = c)
          }
        case t: IsComponent =>
          circuitGet(errors)(CircuitTarget(t.circuit)).map {
            case CircuitTarget(c) =>
              t match {
                case ref:  ReferenceTarget => ref.copy(circuit = c)
                case inst: InstanceTarget  => inst.copy(circuit = c)
              }
          }
      }
    }

    circuitRename
  }

  /** Fully rename `from` to `tos`
    * @param from
    * @param tos
    */
  protected def completeRename(from: CompleteTarget, tos: Seq[CompleteTarget]): Unit = {
    tos.foreach { recordSensitivity(from, _) }
    val existing = _underlying.getOrElse(from, Vector.empty)
    val updated = {
      val all = (existing ++ tos)
      if (doDistinct) all.distinct else all
    }
    _underlying(from) = updated
    traverseTokensCache.clear()
    traverseHierarchyCache.clear()
    traverseLeftCache.clear()
    traverseRightCache.clear()
  }

  def get(key: CircuitName): Option[Seq[CircuitName]] = {
    get(Target.convertCircuitName2CircuitTarget(key)).map(_.collect { case c: CircuitTarget => c.toNamed })
  }

  def get(key: ModuleName): Option[Seq[ModuleName]] = {
    get(Target.convertModuleName2ModuleTarget(key)).map(_.collect { case m: ModuleTarget => m.toNamed })
  }

  def get(key: ComponentName): Option[Seq[ComponentName]] = {
    get(Target.convertComponentName2ReferenceTarget(key)).map(_.collect { case c: IsComponent => c.toNamed })
  }

  def get(key: Named): Option[Seq[Named]] = key match {
    case t: CompleteTarget => get(t)
    case other => get(key.toTarget).map(_.collect { case c: IsComponent => c.toNamed })
  }

}

// This must be in same file as RenameMap because RenameMap is sealed
package object renamemap {
  object MutableRenameMap {
    def fromNamed(map: collection.Map[Named, Seq[Named]]): MutableRenameMap = {
      val rm = new MutableRenameMap
      rm.addMap(map)
      rm
    }

    def apply(map: collection.Map[CompleteTarget, Seq[CompleteTarget]]): MutableRenameMap = {
      val rm = new MutableRenameMap
      rm.recordAll(map)
      rm
    }

    /** Initialize a new RenameMap */
    def apply(): MutableRenameMap = new MutableRenameMap

    // This is a private internal API for transforms where the .distinct operation is very expensive
    // (eg. LowerTypes). The onus is on the user of this API to be very careful and not inject
    // duplicates. This is a bad, hacky API that no one should use
    private[firrtl] def noDistinct(): MutableRenameMap = new MutableRenameMap(doDistinct = false)
  }

  final class MutableRenameMap private[firrtl] (
    protected val _underlying: mutable.HashMap[CompleteTarget, Seq[CompleteTarget]] =
      mutable.HashMap[CompleteTarget, Seq[CompleteTarget]](),
    protected val _chained:   Option[RenameMap] = None,
    protected val doDistinct: Boolean = true)
      extends RenameMap {

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
      super._recordAll(map)

    /** Records that a [[firrtl.annotations.CompleteTarget CompleteTarget]] is deleted
      * @param name
      */
    def delete(name: CompleteTarget): Unit = _underlying(name) = Seq.empty

    def rename(from: Named, to: Named): Unit = rename(from, Seq(to))

    def rename(from: Named, tos: Seq[Named]): Unit = recordAll(Map(from.toTarget -> tos.map(_.toTarget)))

    def rename(from: ComponentName, to: ComponentName): Unit = record(from, to)

    def rename(from: ComponentName, tos: Seq[ComponentName]): Unit = record(from, tos.map(_.toTarget))

    def delete(name: CircuitName): Unit = _underlying(name) = Seq.empty

    def delete(name: ModuleName): Unit = _underlying(name) = Seq.empty

    def delete(name: ComponentName): Unit = _underlying(name) = Seq.empty

    def addMap(map: collection.Map[Named, Seq[Named]]): Unit =
      recordAll(map.map {
        case (key, values) => (Target.convertNamed2Target(key), values.map(Target.convertNamed2Target))
      })
  }
}
