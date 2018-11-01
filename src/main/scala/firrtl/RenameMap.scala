// See LICENSE for license details.

package firrtl

import annotations._
import firrtl.RenameMap.{CircularRenameException, IllegalRenameException}
import firrtl.annotations.TargetToken.{Field, Index}

import scala.collection.mutable

object RenameMap {
  @deprecated("Use create with CompleteTarget instead, this will be removed in 1.3", "1.2")
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
  */
// TODO This should probably be refactored into immutable and mutable versions
final class RenameMap private () {

  /** Record that the from [[CircuitTarget]] is renamed to another [[CircuitTarget]]
    * @param from
    * @param to
    */
  def record(from: CircuitTarget, to: CircuitTarget): Unit = completeRename(from, Seq(to))

  /** Record that the from [[CircuitTarget]] is renamed to another sequence of [[CircuitTarget]]s
    * @param from
    * @param tos
    */
  def record(from: CircuitTarget, tos: Seq[CircuitTarget]): Unit = completeRename(from, tos)

  /** Record that the from [[IsMember]] is renamed to another [[IsMember]]
    * @param from
    * @param to
    */
  def record(from: IsMember, to: IsMember): Unit = completeRename(from, Seq(to))

  /** Record that the from [[IsMember]] is renamed to another sequence of [[IsMember]]s
    * @param from
    * @param tos
    */
  def record(from: IsMember, tos: Seq[IsMember]): Unit = completeRename(from, tos)

  /** Records that the keys in map are also renamed to their corresponding value seqs.
    * Only ([[CircuitTarget]] -> Seq[ [[CircuitTarget]] ]) and ([[IsMember]] -> Seq[ [[IsMember]] ]) key/value allowed
    * @param map
    */
  def recordAll(map: collection.Map[CompleteTarget, Seq[CompleteTarget]]): Unit =
    map.foreach{
      case (from: IsComponent, tos: Seq[IsMember]) => completeRename(from, tos)
      case (from: IsModule, tos: Seq[IsMember]) => completeRename(from, tos)
      case (from: CircuitTarget, tos: Seq[CircuitTarget]) => completeRename(from, tos)
      case other => Utils.throwInternalError(s"Illegal rename: ${other._1} -> ${other._2}")
    }

  /** Records that a [[CompleteTarget]] is deleted
    * @param name
    */
  def delete(name: CompleteTarget): Unit = underlying(name) = Seq.empty

  /** Renames a [[CompleteTarget]]
    * @param t target to rename
    * @return renamed targets
    */
  def apply(t: CompleteTarget): Seq[CompleteTarget] = completeGet(t).getOrElse(Seq(t))

  /** Get renames of a [[CircuitTarget]]
    * @param key Target referencing the original circuit
    * @return Optionally return sequence of targets that key remaps to
    */
  def get(key: CompleteTarget): Option[Seq[CompleteTarget]] = completeGet(key)

  /** Get renames of a [[CircuitTarget]]
    * @param key Target referencing the original circuit
    * @return Optionally return sequence of targets that key remaps to
    */
  def get(key: CircuitTarget): Option[Seq[CircuitTarget]] = completeGet(key).map( _.map { case x: CircuitTarget => x } )

  /** Get renames of a [[IsMember]]
    * @param key Target referencing the original member of the circuit
    * @return Optionally return sequence of targets that key remaps to
    */
  def get(key: IsMember): Option[Seq[IsMember]] = completeGet(key).map { _.map { case x: IsMember => x } }


  /** Create new [[RenameMap]] that merges this and renameMap
    * @param renameMap
    * @return
    */
  def ++ (renameMap: RenameMap): RenameMap = RenameMap(underlying ++ renameMap.getUnderlying)

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

  /** Maps old names to new names. New names could still require renaming parts of their name
    * Old names must refer to existing names in the old circuit
    */
  private val underlying = mutable.HashMap[CompleteTarget, Seq[CompleteTarget]]()

  /** Records which local InstanceTargets will require modification.
    * Used to reduce time to rename nonlocal targets who's path does not require renaming
    */
  private val sensitivity = mutable.HashSet[IsComponent]()

  /** Caches results of recursiveGet. Is cleared any time a new rename target is added
    */
  private val getCache = mutable.HashMap[CompleteTarget, Seq[CompleteTarget]]()

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

  /** Get renames of a [[CompleteTarget]]
    * @param key Target referencing the original circuit
    * @return Optionally return sequence of targets that key remaps to
    */
  private def completeGet(key: CompleteTarget): Option[Seq[CompleteTarget]] = {
    val errors = mutable.ArrayBuffer[String]()
    val ret = if(hasChanges) {
      val ret = recursiveGet(mutable.LinkedHashSet.empty[CompleteTarget], errors)(key)
      if(errors.nonEmpty) { throw IllegalRenameException(errors.mkString("\n")) }
      if(ret.size == 1 && ret.head == key) { None } else { Some(ret) }
    } else { None }
    ret
  }

  // scalastyle:off
  // This function requires a large cyclomatic complexity, and is best naturally expressed as a large function
  /** Recursively renames a target so the returned targets are complete renamed
    * @param set Used to detect circular renames
    * @param errors Used to record illegal renames
    * @param key Target to rename
    * @return Renamed targets
    */
  private def recursiveGet(set: mutable.LinkedHashSet[CompleteTarget],
                           errors: mutable.ArrayBuffer[String]
                          )(key: CompleteTarget): Seq[CompleteTarget] = {
    if(getCache.contains(key)) {
      getCache(key)
    } else {
      // First, check if whole key is remapped
      // Note that remapped could hold stale parent targets that require renaming
      val remapped = underlying.getOrElse(key, Seq(key))

      // If we've seen this key before in recursive calls to parentTargets, then we know a circular renaming
      // mapping has occurred, and no legal name exists
      if(set.contains(key) && !key.isInstanceOf[CircuitTarget]) {
        throw CircularRenameException(s"Illegal rename: circular renaming is illegal - ${set.mkString(" -> ")}")
      }

      // Add key to set to detect circular renaming
      set += key

      // Curry recursiveGet for cleaner syntax below
      val getter = recursiveGet(set, errors)(_)

      // For each remapped key, call recursiveGet on their parentTargets
      val ret = remapped.flatMap {

        // If t is a CircuitTarget, return it because it has no parent target
        case t: CircuitTarget => Seq(t)

        // If t is a ModuleTarget, try to rename parent target, then update t's parent
        case t: ModuleTarget => getter(t.targetParent).map {
          case CircuitTarget(c) => ModuleTarget(c, t.module)
        }

        /** If t is an InstanceTarget (has a path) but has no references:
          * 1) Check whether the instance has been renamed (asReference)
          * 2) Check whether the ofModule of the instance has been renamed (only 1:1 renaming is ok)
          */
        case t: InstanceTarget =>
          getter(t.asReference).map {
            case t2:InstanceTarget => t2
            case t2@ReferenceTarget(c, m, p, r, Nil) =>
              val t3 = InstanceTarget(c, m, p, r, t.ofModule)
              val ofModuleTarget = t3.ofModuleTarget
              getter(ofModuleTarget) match {
                case Seq(ModuleTarget(newCircuit, newOf)) if newCircuit == t3.circuit => t3.copy(ofModule = newOf)
                case other =>
                  errors += s"Illegal rename: ofModule of $t is renamed to $other - must rename $t directly."
                  t
              }
            case other =>
              errors += s"Illegal rename: $t has new instance reference $other"
              t
          }

        /** If t is a ReferenceTarget:
          * 1) Check parentTarget to tokens
          * 2) Check ReferenceTarget with one layer stripped from its path hierarchy (i.e. a new root module)
          */
        case t: ReferenceTarget =>
          val ret: Seq[CompleteTarget] = if(t.component.nonEmpty) {
            val last = t.component.last
            getter(t.targetParent).map{ x =>
              (x, last) match {
                case (t2: ReferenceTarget, Field(f)) => t2.field(f)
                case (t2: ReferenceTarget, Index(i)) => t2.index(i)
                case other =>
                  errors += s"Illegal rename: ${t.targetParent} cannot be renamed to ${other._1} - must rename $t directly"
                  t
              }
            }
          } else {
            val pathTargets = sensitivity.empty ++ (t.pathAsTargets ++ t.pathAsTargets.map(_.asReference))
            if(t.pathAsTargets.nonEmpty && sensitivity.intersect(pathTargets).isEmpty) Seq(t) else {
              getter(t.pathTarget).map {
                case newPath: IsModule => t.setPathTarget(newPath)
                case other =>
                  errors += s"Illegal rename: path ${t.pathTarget} of $t cannot be renamed to $other - must rename $t directly"
                  t
              }
            }
          }
          ret.flatMap {
            case y: IsComponent if !y.isLocal =>
              val encapsulatingInstance = y.path.head._1.value
              getter(y.stripHierarchy(1)).map {
                _.addHierarchy(y.moduleOpt.get, encapsulatingInstance)
              }
            case other => Seq(other)
          }
      }

      // Remove key from set as visiting the same key twice is ok, as long as its not during the same recursive call
      set -= key

      // Cache result
      getCache(key) = ret

      // Return result
      ret

    }
  }
  // scalastyle:on

  /** Fully renames from to tos
    * @param from
    * @param tos
    */
  private def completeRename(from: CompleteTarget, tos: Seq[CompleteTarget]): Unit = {
    def check(from: CompleteTarget, to: CompleteTarget)(t: CompleteTarget): Unit = {
      require(from != t, s"Cannot record $from to $to, as it is a circular constraint")
      t match {
        case _: CircuitTarget =>
        case other: IsMember => check(from, to)(other.targetParent)
      }
    }
    tos.foreach { to => if(from != to) check(from, to)(to) }
    (from, tos) match {
      case (x, Seq(y)) if x == y =>
      case _ =>
        tos.foreach{recordSensitivity(from, _)}
        val existing = underlying.getOrElse(from, Vector.empty)
        val updated = existing ++ tos
        underlying(from) = updated
        getCache.clear()
    }
  }

  /* DEPRECATED ACCESSOR/SETTOR METHODS WITH [[Named]] */

  @deprecated("Use record with CircuitTarget instead, this will be removed in 1.3", "1.2")
  def rename(from: Named, to: Named): Unit = rename(from, Seq(to))

  @deprecated("Use record with IsMember instead, this will be removed in 1.3", "1.2")
  def rename(from: Named, tos: Seq[Named]): Unit = recordAll(Map(from.toTarget -> tos.map(_.toTarget)))

  @deprecated("Use record with IsMember instead, this will be removed in 1.3", "1.2")
  def rename(from: ComponentName, to: ComponentName): Unit = record(from, to)

  @deprecated("Use record with IsMember instead, this will be removed in 1.3", "1.2")
  def rename(from: ComponentName, tos: Seq[ComponentName]): Unit = record(from, tos.map(_.toTarget))

  @deprecated("Use delete with CircuitTarget instead, this will be removed in 1.3", "1.2")
  def delete(name: CircuitName): Unit = underlying(name) = Seq.empty

  @deprecated("Use delete with IsMember instead, this will be removed in 1.3", "1.2")
  def delete(name: ModuleName): Unit = underlying(name) = Seq.empty

  @deprecated("Use delete with IsMember instead, this will be removed in 1.3", "1.2")
  def delete(name: ComponentName): Unit = underlying(name) = Seq.empty

  @deprecated("Use recordAll with CompleteTarget instead, this will be removed in 1.3", "1.2")
  def addMap(map: collection.Map[Named, Seq[Named]]): Unit =
    recordAll(map.map { case (key, values) => (Target.convertNamed2Target(key), values.map(Target.convertNamed2Target)) })

  @deprecated("Use get with CircuitTarget instead, this will be removed in 1.3", "1.2")
  def get(key: CircuitName): Option[Seq[CircuitName]] = {
    get(Target.convertCircuitName2CircuitTarget(key)).map(_.collect{ case c: CircuitTarget => c.toNamed })
  }

  @deprecated("Use get with IsMember instead, this will be removed in 1.3", "1.2")
  def get(key: ModuleName): Option[Seq[ModuleName]] = {
    get(Target.convertModuleName2ModuleTarget(key)).map(_.collect{ case m: ModuleTarget => m.toNamed })
  }

  @deprecated("Use get with IsMember instead, this will be removed in 1.3", "1.2")
  def get(key: ComponentName): Option[Seq[ComponentName]] = {
    get(Target.convertComponentName2ReferenceTarget(key)).map(_.collect{ case c: IsComponent => c.toNamed })
  }

  @deprecated("Use get with IsMember instead, this will be removed in 1.3", "1.2")
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
  @deprecated("Use typesafe rename defs instead, this will be removed in 1.3", "1.2")
  def setModule(module: String): Unit = moduleName = module

  /** Sets mutable state to record current circuit we are visiting
    * @param circuit
    */
  @deprecated("Use typesafe rename defs instead, this will be removed in 1.3", "1.2")
  def setCircuit(circuit: String): Unit = circuitName = circuit

  /** Records how a reference maps to a new reference
    * @param from
    * @param to
    */
  @deprecated("Use typesafe rename defs instead, this will be removed in 1.3", "1.2")
  def rename(from: String, to: String): Unit = rename(from, Seq(to))

  /** Records how a reference maps to a new reference
    * The reference's root module and circuit are determined by whomever called setModule or setCircuit last
    * @param from
    * @param tos
    */
  @deprecated("Use typesafe rename defs instead, this will be removed in 1.3", "1.2")
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
  @deprecated("Use typesafe rename defs instead, this will be removed in 1.3", "1.2")
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
  @deprecated("Use typesafe rename defs instead, this will be removed in 1.3", "1.2")
  def delete(names: Seq[String]): Unit = names.foreach(delete(_))
}


