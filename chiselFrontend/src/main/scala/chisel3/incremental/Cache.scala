// See LICENSE for license details.

package chisel3.incremental

import chisel3.experimental.BaseModule
import chisel3.RawModule
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.{HasShellOptions, ShellOption, Unserializable}

import scala.collection.mutable


/** Contains previously elaborated modules and their tags from a single Chisel elaboration
  * Is optionally backed from the filesystem
  * Can be set to dynamically load tags/modules from backing directory
  *
  * Module id's are guaranteed to be unique within the cache
  * Package name is set externally, when using a cache (e.g. creating the Cache object)
  * Used to enable interoperability between caches (more than one cache can be used
  *   in a future Chisel elaboration)
  *
  * @param packge
  * @param tags
  * @param modules
  * @param dynamicLoading
  * @param backingDirectory
  */
case class Cache(packge: String,
                 tags: Map[Tag, Long],
                 modules: Map[Long, BaseModule],
                 dynamicLoading: Boolean,
                 backingDirectory: Option[String]
                ) extends NoTargetAnnotation with Unserializable {

  // In the future, make these garbage collectable
  def dynamicTags: mutable.HashMap[Tag, Long] = mutable.HashMap.empty
  def dynamicModules: mutable.HashMap[Long, BaseModule] = mutable.HashMap.empty

  def retrieve[T <: BaseModule](tag: ItemTag[T]): Option[T] = {
    if(tags.contains(tag)) Some(modules(tags(tag)).asInstanceOf[T])
    else if(dynamicTags.contains(tag)) Some(dynamicModules(dynamicTags(tag)).asInstanceOf[T])
    else if(dynamicLoading) dynamicallyLoad(tag)
    else None
  }

  def contains[T <: BaseModule](tag: ItemTag[T]): Boolean = {
    tags.contains(tag) || dynamicTags.contains(tag)
  }

  def dynamicallyLoad[T <: BaseModule](tag: ItemTag[T]): Option[T] = {
    if(dynamicLoading && backingDirectory.nonEmpty) {
      tag.load(backingDirectory.get) match {
        case Some(module) =>
          dynamicTags(tag) = module._id
          dynamicModules(module._id) = module
          Some(module)
        case None => None
      }
    } else None
  }

  def writeTo(directory: String): Unit = {
    tags.foreach { case (tag: ItemTag[_], id) => tag.store(directory, modules(id)) }
    dynamicTags.foreach { case (tag: ItemTag[_], id) => tag.store(directory, modules(id)) }
  }
}

object Cache extends HasShellOptions {
  val options = Seq(
    new ShellOption[String](
      longOption = "with-cache",
      toAnnotationSeq = (a: String) => {
        a.split("::") match {
          case Array(packge, directory) => Seq(Cache.load(directory, packge/*Package.deserialize(packge)*/))
        }
      },
      helpText = "The relative or absolute path to the directory containing the cached Chisel elaborations to consume.",
      helpValueName = Some("<packageName>::<path>") ) ,
    new ShellOption[String](
      longOption = "with-dynamic-cache",
      toAnnotationSeq = (a: String) => {
        a.split("::") match {
          case Array(packge, directory) => Seq(Cache.load(directory, packge/*Package.deserialize(packge)*/, dynamicLoading=true))
        }
      },
      helpText = "The relative or absolute path to the directory containing the cached Chisel elaborations to consume.",
      helpValueName = Some("<packageName>::<path>") )
  )

  def load(directory: String, packge: String, dynamicLoading: Boolean = false): Cache = {
    if(dynamicLoading) {
      Cache(packge, Map.empty, Map.empty, dynamicLoading=true, Some(directory))
    } else {
      val files = Stash.getListOfFiles(directory);
      val modules = files.collect {
        case file if file.getName.split('.').last == "tag" =>
          val tag = Stash.load[Tag](file).get
          (tag, tag.untypedLoad(directory))
      }

      val invalidTags = modules.collect { case (tag, None) => tag }
      require(invalidTags.isEmpty, s"Cannot load cache $packge from $directory: invalid tags $invalidTags")

      val (tagMap, moduleMap) = modules.foldLeft((Map.empty[Tag, Long], Map.empty[Long, BaseModule])) {
        case ((tags, modules), (tag, Some(module: BaseModule))) =>
          (tags + (tag -> module._id), (modules + (module._id -> module)))
      }

      Cache(packge, tagMap, moduleMap, dynamicLoading, Some(directory))
    }
  }
}
