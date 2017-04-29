package chiselBuild

import sbt._

object ChiselDependencies {

  // The basic chisel dependencies.
  val basicDependencies = collection.immutable.Map[String, Seq[String]](
    "chisel3" -> Seq("firrtl"),
    "chisel_testers" -> Seq("firrtl", "firrtl_interpreter", "chisel3"),
    "firrtl" -> Seq(),
    "firrtl_interpreter" -> Seq("firrtl")
  )

  // The following are the default development versions of chisel libraries,
  //  not the "release" versions.
  val defaultVersions = collection.immutable.Map[String, String](
    "chisel3" -> "3.1-SNAPSHOT",
    "firrtl" -> "1.1-SNAPSHOT",
    "firrtl_interpreter" -> "1.1-SNAPSHOT",
    "chisel_testers" -> "1.2-SNAPSHOT"
  )

  val versions = collection.mutable.Map[String, String](defaultVersions.toSeq: _*)

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
  // Provide a managed dependency on X if -DXVersion="" is supplied on the command line (via JAVA_OPTS).
  private def nameToModuleID(name: String): ModuleID = {
    "edu.berkeley.cs" %% name % sys.props.getOrElse(name + "Version", versions(name))
  }

  case class PackageProject(packageName: String, base: Option[File] = None, settings: Option[Seq[Def.Setting[_]]] = None)

  lazy val subProjectsSetting = settingKey[Seq[PackageProject]]("Subprojects to build")

  var packageProjects = scala.collection.mutable.Map[String, ProjectReference]()

  // Chisel projects as library dependencies.
  def chiselLibraryDependencies(name: String): Seq[ModuleID] = {
    def unmanaged(dep: String): Boolean = {
      unmanagedClasspath match {
        case None => false
        case Some(classpath: String) => classpath.contains(s"$dep.jar")
      }
    }
    val result = basicDependencies(name).filterNot(dep => packageProjects.contains(dep) || unmanaged(dep)).map(nameToModuleID(_))
    result
  }

  // For a given chisel project, return a sequence of project references,
  //  suitable for use as an argument to dependsOn().
  def chiselProjectDependencies(name: String): Seq[ClasspathDep[ProjectReference]] = {
    val result = basicDependencies(name).filter(dep => packageProjects.contains(dep)).map { dep: String => classpathDependency(packageProjects(dep)) }
    result
  }
}
