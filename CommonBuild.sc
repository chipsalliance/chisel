import ammonite.ops._
import mill._
import mill.scalalib._

def scalacOptionsVersion(scalaVersion: String): Seq[String] = {
  Seq() ++ {
    // If we're building with Scala > 2.11, enable the compile option
    //  switch to support our anonymous Bundle definitions:
    //  https://github.com/scala/bug/issues/10047
    if (scalaVersion.startsWith("2.11.")) {
      Seq()
    } else {
      Seq(
        "-Xsource:2.11",
        "-Ywarn-unused:imports",
        "-Ywarn-unused:locals"
      )
    }
  }
}

def javacOptionsVersion(scalaVersion: String): Seq[String] = {
  Seq() ++ {
    // Scala 2.12 requires Java 8. We continue to generate
    //  Java 7 compatible code for Scala 2.11
    //  for compatibility with old clients.
    if (scalaVersion.startsWith("2.11.")) {
      Seq("-source", "1.7", "-target", "1.7")
    } else {
      Seq("-source", "1.8", "-target", "1.8")
    }
  }
}

// Define our own BuildInfo since mill doesn't currently have one.
trait BuildInfo extends ScalaModule { outer =>

  def buildInfoObjectName: String = "BuildInfo"

  def buildInfoMembers: T[Map[String, String]] = T {
    Map.empty[String, String]
  }

  private def generateBuildInfo(outputPath: Path, members: Map[String, String]) = {
    val outputFile = outputPath / "BuildInfo.scala"
    val packageName = members.getOrElse("buildInfoPackage", "")
    val packageDef = if (packageName != "") {
      s"package ${packageName}"
    } else {
      ""
    }
    val internalMembers =
      members
        .map {
          case (name, value) => s"""  val ${name}: String = "${value}""""
        }
        .mkString("\n")
    write(outputFile,
      s"""
         |${packageDef}
         |case object ${buildInfoObjectName}{
         |$internalMembers
         |  override val toString: String = {
         |    "buildInfoPackage: %s, version: %s, scalaVersion: %s" format (
         |        buildInfoPackage, version, scalaVersion
         |    )
         |  }
         |}
       """.stripMargin)
    outputPath
  }

  override def generatedSources = T {
    super.generatedSources() :+ PathRef(generateBuildInfo(T.ctx().dest, buildInfoMembers()))
  }
}

// Define some file filters to exclude unwanted files from created jars.
type JarFileFilter = (Path, RelPath) => Boolean
// Exclude any `.DS_Store` files
val noDS_StoreFiles: JarFileFilter = (p: Path, relPath: RelPath) => {
  relPath.last != ".DS_Store"
}

// Exclude non-source files - accept all resource files, but only *.{java,scala} from source paths
val onlySourceFiles: JarFileFilter = (p: Path, relPath: RelPath) => {
  p.last == "resources" || (relPath.ext == "scala" || relPath.ext == "java")
}

// Apply a sequence of file filters - only accept files which satisfy all filters.
// We expect this to be curried, the resulting file filter passed to createJar()
def forallFilters(fileFilters: Seq[JarFileFilter])(p: Path, relPath: RelPath): Boolean = {
  fileFilters.forall(f => f(p, relPath))
}
