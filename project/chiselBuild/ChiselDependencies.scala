package chiselBuild

import sbt._

// If you make changes here, you may need to update the SubprojectGenerator.
object ChiselDependencies {

  // The basic chisel dependencies.
  val basicDependencies = collection.immutable.Map[String, Seq[String]](
    "chisel3" -> Seq("firrtl"),
    "chisel-iotesters" -> Seq("firrtl", "firrtl-interpreter", "chisel3"),
    "firrtl" -> Seq(),
    "firrtl-interpreter" -> Seq("firrtl")
  )

  // The following are the default development versions of chisel libraries,
  //  not the "release" versions.
  val defaultVersions = collection.immutable.Map[String, String](
    "chisel3" -> "3.1-SNAPSHOT",
    "firrtl" -> "1.1-SNAPSHOT",
    "firrtl-interpreter" -> "1.1-SNAPSHOT",
    "chisel-iotesters" -> "1.2-SNAPSHOT"
  )

  // Set up the versions Map, overriding the default if -DXVersion="" is supplied on the command line (via JAVA_OPTS).
  val versions = collection.mutable.Map[String, String](
    (
      for ((name, version) <- defaultVersions)
        yield (name, sys.props.getOrElse(name + "Version", version))
    ).toSeq: _*
  )

  // The unmanaged classPath - jars found here will automatically satisfy libraryDependencies.
  var unmanagedClasspath: Option[String] = None

  case class PackageVersion(packageName: String, version: String) {
    implicit def toStringTuple: Tuple2[String, String] = {
      (packageName, version)
    }
  }

  /** Set one or more of the BIG4 versions.
    *
    * @param package_versions package name and version
    * @return map of prior name -> version.
    */
  def setVersions(package_versions: Seq[PackageVersion]): collection.immutable.Map[String, String] = {
    // Return the old settings.
    val ret = collection.immutable.Map[String, String](versions.toSeq: _*)
    for (pv <- package_versions) {
      versions(pv.packageName) = pv.version
    }
    ret
  }

  // Give a module/project name, return the ModuleID
  private def nameToModuleID(name: String): ModuleID = {
    "edu.berkeley.cs" %% name % versions(name)
  }

  case class PackageProject(packageName: String, base: Option[File] = None, settings: Option[Seq[Def.Setting[_]]] = None)

  lazy val subProjectsSetting = settingKey[Seq[PackageProject]]("Subprojects to build")

  type PackageProjectsMap = scala.collection.immutable.Map[String, ProjectReference]

  var packageProjectsMap: PackageProjectsMap = Map.empty

  // Chisel projects as library dependencies.
  def chiselLibraryDependencies(names: Seq[String]): Seq[ModuleID] = {
    def unmanaged(dep: String): Boolean = {
      unmanagedClasspath match {
        case None => false
        case Some(classpath: String) => classpath.contains(s"$dep.jar")
      }
    }
    val result = names.filterNot(dep => packageProjectsMap.contains(dep) || unmanaged(dep)).map(nameToModuleID(_))
    result
  }

  /** For a given chisel project, return a sequence of project references,
    *  suitable for use as an argument to dependsOn().
    * @param names - the package names to check for dependencies. The default is all packages being built.
    * @return - a sequence of project references, suitable for passing directly to the dependsOn() method.
    */
  def chiselProjectDependencies(names: Seq[String] = packageProjectsMap.keys.toSeq): Seq[ClasspathDep[ProjectReference]] = {
    val result = names.filter(dep => packageProjectsMap.contains(dep)).map { dep: String => classpathDependency(packageProjectsMap(dep)) }
    result
  }
}
