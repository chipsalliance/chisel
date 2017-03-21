import sbt._
import Keys._

object Dependencies {
  // The basic chisel dependencies.
  val chiselDependencies = collection.immutable.HashMap[String, Seq[String]](
    "chisel" -> Seq("firrtl"),
    "chisel-testers" -> Seq("firrtl", "firrtl-interpreter"),
    "firrtl" -> Seq(),
    "firrtl-interpreter" -> Seq("firrtl")
  )

  // A map from name (string) to project build definition.
  // These will be used to construct the project dependsOn() dependencies.
//  lazy val subProjects = collection.immutable.HashMap[String, ProjectReference](
//    "chisel" -> ChiselBuild.chisel,
//    "chisel-testers" -> ChiselBuild.chisel_testers,
//    "firrtl" -> ChiselBuild.firrtl,
//    "firrtl-interpreter" -> ChiselBuild.firrtl_interpreter
//  )

  // For a given chisel project, return a sequence of project references,
  //  suitable for use as an argument to dependsOn().
  // By default, we assume we're fetching the dependencies as libraries,
  //  not as dependent sbt projects, so this returns an empty sequence.
  def chiselProjectDependencies(name: String): Seq[ClasspathDep[ProjectReference]] = {
//    (chiselDependencies(name) map {p: String => classpathDependency(subProjects(p))})
    Seq()
  }

  // The following are the default development versions of chisel libraries,
  //  not the "release" versions.
  val chiselDefaultVersions = Map(
    "chisel" -> "3.1-SNAPSHOT",
    "chisel3" -> "3.1-SNAPSHOT",
    "firrtl" -> "1.1-SNAPSHOT",
    "firrtl-interpreter" -> "1.1-SNAPSHOT",
    "chisel-testers" -> "1.2-SNAPSHOT"
    )

  // Give a module/project name, return the ModuleID
  // Provide a managed dependency on X if -DXVersion="" is supplied on the command line (via JAVA_OPTS).
  private def nameToModuleID(name: String): ModuleID = {
    "edu.berkeley.cs" %% name % sys.props.getOrElse(name + "Version", chiselDefaultVersions(name))
  }

  // Chisel projects as library dependencies.
  // The optional argument is a classpath to check.
  // Libraries appearing on that classpath will be skipped.
  def chiselLibraryDependencies(name: String, optionClasspath: Option[String] = None): Seq[ModuleID] = {
    if (false) {
      Seq()
    } else {
      optionClasspath match {
	case None =>
	  chiselDependencies(name) map { nameToModuleID(_) }
	case Some(classpath: String) =>
	  chiselDependencies(name).collect {
	    // If we have an unmanaged jar file on the classpath, assume we're to use that,
	    case dep: String if !classpath.contains(s"$dep.jar") => nameToModuleID(dep)
	  }
      }
    }
  }

}
