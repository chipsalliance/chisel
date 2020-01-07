// See LICENSE for license details.

package chisel3.incremental

import java.io._

import _root_.firrtl.annotations.NoTargetAnnotation
import _root_.firrtl.AnnotationSeq
import _root_.firrtl.options.{HasShellOptions, ShellOption, Unserializable}
import chisel3.Cacheable
import chisel3.experimental.BaseModule
import com.twitter.chill.MeatLocker

import scala.collection.mutable

case class Link(originalModule: Long, wrapperModule: Long, parentModule: Long)


/** A stash contains all caches available to a running Chisel elaboration
  * Every elaboration can only have one Stash
  *
  * After elaboration, a Stash can export its state as a new Cache
  *
  * @param caches
  * @param useLatest
  */
case class Stash(caches: Map[String, Cache],
                 useLatest: Boolean
                ) extends NoTargetAnnotation with Unserializable {

  val builderTags: mutable.HashMap[Tag, Long] = mutable.HashMap.empty
  val builderModules: mutable.HashMap[Long, BaseModule] = mutable.HashMap.empty
  val usedTags: mutable.HashSet[Tag] = mutable.HashSet.empty

  /*
  def store[T <: BaseModule](tag: ItemTag[T], module: T): Unit = {
    builderTags(tag) = module._id
    builderModules(module._id) = module
  }
   */

  def store[T <: BaseModule](module: T): Unit = {
    module match {
      case c: Cacheable[_] => builderTags(c.tag) = module._id
      case _ =>
    }
    builderModules(module._id) = module
  }

  /** If cache package is specified, look in that cache first
    * Otherwise, first look in builder modules, then in all caches sorted by name
    * @param tag
    * @param cachePackageOpt
    * @tparam T
    * @return
    */
  def retrieve[T <: BaseModule](tag: ItemTag[T], cachePackageOpt: Option[String] = None): Option[T] = {
    val retrieved = if(cachePackageOpt.nonEmpty) {
      val cachePackage = cachePackageOpt.get
      require(caches.contains(cachePackage))
      caches(cachePackage).retrieve(tag)
    } else {
      if(builderTags.contains(tag)) {
        builderModules.get(builderTags(tag)).asInstanceOf[Option[T]]
      } else {
        val matchingCaches = caches.collect {
          case (_, cache) if cache.contains(tag) => cache
        }

        if(matchingCaches.isEmpty) {
          None
        } else if(useLatest) {
          matchingCaches.toSeq.minBy { c: Cache => c.packge }.retrieve(tag)
        } else {
          require(matchingCaches.size == 1, "More than one cache matched item!")
          matchingCaches.head.retrieve(tag)
        }
      }
    }

    if(retrieved.nonEmpty) usedTags += tag.asInstanceOf[ItemTag[BaseModule]]

    retrieved
  }

  def exportAsCache(packge: String, backingDirectory: Option[String], isFat: Boolean): Cache = {
    val (tagMap, moduleMap) = if(isFat) {
      usedTags.foldLeft((builderTags.toMap, builderModules.toMap)) {
        case ((tags, modules), tag) =>
          val module = retrieve[BaseModule](tag.asInstanceOf[ItemTag[BaseModule]]).get
          (tags + (tag -> module._id), modules + (module._id -> module))
      }
    } else {
      (builderTags.toMap, builderModules.toMap)
    }
    Cache(packge, tagMap, moduleMap, dynamicLoading = false, backingDirectory)
  }

  /** Get modules from this elaboration, using id
    *
    * @param id
    * @tparam T
    * @return
    */
  def get[T <: BaseModule](id: Long): Option[T] = builderModules.get(id).asInstanceOf[Option[T]]

  def apply(id: Long): BaseModule = {
    require(builderModules.contains(id), s"Module with id $id is not contained in stash!")
    builderModules(id)
  }

  /*
  // Maps current child -> parents relationships in design
  private val parentMap = mutable.HashMap[Long, Seq[Long]]()
  def getParents(moduleId: Long): Seq[Long] = {
    parentMap.getOrElse(moduleId, Nil)
  }
  def setParent(child: Long, parent: Long) = {
    val parents = getParents(child)
    assert(child != parent)
    parentMap(child) = parent +: parents
  }

  // Tracks which parent is the "active" parent, if a child has been
  //   imported multiple times
  private val activeParentMap = mutable.HashMap[Long, Long]()
  def setActiveParent(id: Long, parent: Long): Unit = {
    assert(id != parent)
    require(!activeParentMap.contains(id) || activeParentMap(id) == parent, s"Cannot set active parent $parent for $id, as it already has parent ${activeParentMap(id)}")
    activeParentMap(id) = parent
  }
  def clearActiveParent(id: Long): Unit = {
    activeParentMap.remove(id)
  }
  def getActiveParent(id: Long): Option[Long] = {
    activeParentMap.get(id)
  }

  // Maps modules with their imported versions
  private val linkedModuleMap = mutable.HashMap[Long, Set[Long]]()
  def setLink(original: Long, imported: Long): Unit = {
    linkedModuleMap(original) = linkedModuleMap.getOrElse(original, Set.empty) + imported
  }
  def clearLink(original: Long, imported: Long): Unit = {
    linkedModuleMap(original) = linkedModuleMap.getOrElse(original, Set.empty) - imported
  }
  def getLinks(original: Long): Set[Long] = {
    linkedModuleMap.getOrElse(original, Set.empty)
  }

  // Tracks which linked module is the "active" linked module, if a module has been
  //   imported multiple times
  private val activeLinkMap = mutable.HashMap[Long, Long]()
  def setActiveLink(original: Long, imported: Long): Unit = {
    assert(original != imported)
    require(!activeLinkMap.contains(original) || activeLinkMap(original) == imported, s"Cannot set active link $imported for $original, as it already has imported ${activeLinkMap(original)}")
    activeLinkMap(original) = imported
    //println("Set", activeLinkMap)
  }
  def clearActiveLink(original: Long): Unit = {
    //println("Remove", activeLinkMap)
    activeLinkMap.remove(original)
  }
  def getActiveLink(original: Long): Option[Long] = {
    //println("Get", activeLinkMap)
    activeLinkMap.get(original)
  }
  def printStatus() = {
    val stackElts = Thread.currentThread().getStackTrace()
      .reverse  // so stack frame numbers are deterministic across calls
      .dropRight(2)  // discard Thread.getStackTrace and updateBundleStack
    //println("CurrentModule:", Builder.currentModule, Builder.currentModule.map(_._id))
    println("Module Stash:")
    moduleStash.foreach(x => println(s"    $x"))
    println("Active Parents:")
    activeParentMap.foreach(x => println(s"    $x"))
    println("Active Links:")
    activeLinkMap.foreach(x => println(s"    $x"))
    println("Parents:")
    parentMap.foreach(x => println(s"    $x"))
    println("Links:")
    linkedModuleMap.foreach(x => println(s"    $x"))
    println("StackTrace:")
    stackElts.foreach(x => println(s"    $x"))
  }
   */
}

case class StashOptions(useLatest: Boolean) extends NoTargetAnnotation
case class ExportCache(packge: String, backingDirectory: Option[String], isFat: Boolean) extends NoTargetAnnotation

object Stash extends HasShellOptions {
  val options = Seq(
    new ShellOption[String](
      longOption = "export-cache",
      toAnnotationSeq = (a: String) => {
        a.split("::") match {
          case Array(packge, directory) => Seq(ExportCache(packge/*Package.deserialize(packge)*/, Some(directory), isFat=false))
        }
      },
      helpText = "Package name, then the relative or absolute path to the directory to export elaborated modules.",
      helpValueName = Some("<packageName>::<path>") ) ,
    new ShellOption[String](
      longOption = "export-fat-cache",
      toAnnotationSeq = (a: String) => {
        a.split("::") match {
          case Array(packge, directory) => Seq(ExportCache(packge/*Package.deserialize(packge)*/, Some(directory), isFat=false))
        }
      },
      helpText = "Package name, then the relative or absolute path to the directory to export elaborated modules.",
      helpValueName = Some("<packageName>::<path>") ) ,
    new ShellOption[String](
        longOption = "stash-behavior",
        toAnnotationSeq = { (a: String) =>
          a match {
            case "alphabetical" => Seq(StashOptions(useLatest = true))
            case "strict" => Seq(StashOptions(useLatest = false))
          }
        },
        helpText = "Use alphabetically-first cache if tag matches in >1 cache",
        helpValueName = Some("<packageName>::<path>") )
  )

  def getListOfFiles(dir: String): List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  def store[X](file: File, item: X): Unit = {
    val oos = new ObjectOutputStream(new FileOutputStream(file))
    oos.writeObject(MeatLocker(item))
    oos.close
  }

  def store[X](directory: String, filename: String, item: X): Unit = {
    val dir = new File(directory)
    dir.mkdirs()
    val file = new File(dir.getAbsolutePath + "/" + filename)
    println(dir.getAbsolutePath)
    store(file, item)
  }

  def load[X](file: File): Option[X] = {
    require(file.exists())
    try {
      val ois = new ObjectInputStream(new FileInputStream(file.getAbsolutePath))
      val obj = ois.readObject.asInstanceOf[MeatLocker[X]]
      ois.close()
      Some(obj.get)
    } catch { case e: Exception => None }
  }

  def load[X](directory: String, filename: String): Option[X] = {
    val dir = new File(directory)
    val file = new File(dir.getAbsolutePath + "/" + filename)
    load(file)
  }

  // Stash of either elaborated to currently elaborating modules
  // Additionally, has accessors/setters

}

