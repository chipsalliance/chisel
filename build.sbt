// See LICENSE for license details.

name := "chisel3"

enablePlugins(SiteScaladocPlugin)

enablePlugins(GhpagesPlugin)

val defaultVersions = Map("firrtl" -> "1.1-SNAPSHOT")

def chiselVersion(proj: String): String = {
  sys.props.getOrElse(proj + "Version", defaultVersions(proj))
}

val chiselDeps = chisel.dependencies(Seq(
    ("edu.berkeley.cs" %% "firrtl" % chiselVersion("firrtl"), "firrtl")
))

val dependentProjects = chiselDeps.projects

lazy val commonSettings = ChiselProjectDependenciesPlugin.chiselProjectSettings ++ Seq (
  version := "3.1-SNAPSHOT",
  git.remoteRepo := "git@github.com:freechipsproject/chisel3.git",
  autoAPIMappings := true,
  scalacOptions ++= Seq("-deprecation", "-feature"),
  // Use the root project's unmanaged base for all sub-projects.
  unmanagedBase := (unmanagedBase in root).value,
  // Use the root project's classpath for all sub-projects.
  fullClasspath := (fullClasspath in Compile in root).value,

  scalaVersion := "2.11.11",
  crossScalaVersions := Seq("2.11.11", "2.12.3"),
  libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value,
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
  // Use the root project's unmanaged base for all sub-projects.
  unmanagedBase := (unmanagedBase in root).value,

  libraryDependencies ++= Seq(
    "org.scalatest" %% "scalatest" % "3.0.1" % "test",
    "org.scalacheck" %% "scalacheck" % "1.13.4" % "test",
    "com.github.scopt" %% "scopt" % "3.6.0"
  ) ++ chiselDeps.libraries,


  // Tests from other projects may still run concurrently.
  parallelExecution in Test := true,

  // Since we want to examine the classpath to determine if a dependency on firrtl is required,
  //  this has to be a Task setting.
  //  Fortunately, allDependencies is a Task Setting, so we can modify that.
//  allDependencies := allDependencies.value ++ chiselLibraryDependencies(dependentProjects)

  pomExtra := pomExtra.value ++
    <scm>
      <url>https://github.com/freechipsproject/chisel3.git</url>
      <connection>scm:git:github.com/freechipsproject/chisel3.git</connection>
    </scm>
    <developers>
      <developer>
        <id>jackbackrack</id>
        <name>Jonathan Bachrach</name>
        <url>http://www.eecs.berkeley.edu/~jrb/</url>
      </developer>
    </developers>,

  publishTo := {
    val v = version.value
    val nexus = "https://oss.sonatype.org/"
    if (v.trim.endsWith("SNAPSHOT")) {
      Some("snapshots" at nexus + "content/repositories/snapshots")
    }
    else {
      Some("releases" at nexus + "service/local/staging/deploy/maven2")
    }
  },

  resolvers ++= Seq(
    Resolver.sonatypeRepo("snapshots"),
    Resolver.sonatypeRepo("releases")
  ),

  // Tests from other projects may still run concurrently.
  parallelExecution in Test := true
)

lazy val coreMacros = (project in file("coreMacros")).
  settings(commonSettings: _*).
  settings(publishArtifact := false)

lazy val chiselFrontend = (project in file("chiselFrontend")).
  settings(commonSettings: _*).
  settings(publishArtifact := false).
  dependsOn(coreMacros).
  dependsOn(dependentProjects.map(classpathDependency(_)):_*)

// There will always be a root project.
lazy val root = RootProject(file("."))

lazy val chisel3 = (project in file(".")).
  enablePlugins(BuildInfoPlugin).
  enablePlugins(ScalaUnidocPlugin).
  settings(ChiselProjectDependenciesPlugin.chiselBuildInfoSettings: _*).
  settings(commonSettings: _*).
  // Prevent separate JARs from being generated for coreMacros and chiselFrontend.
  dependsOn(coreMacros % "compile-internal;test-internal").
  dependsOn(chiselFrontend % "compile-internal;test-internal").
  // The following is required until sbt-scoverage correctly deals with inDependencies
  // Unfortunately, it also revives publishing of the subproject jars. Disable until the latter is resolved (again).
  //aggregate(coreMacros, chiselFrontend).
  settings(
    scalacOptions in Test ++= Seq("-language:reflectiveCalls"),
    scalacOptions in Compile in doc ++= Seq(
      "-diagrams",
      "-diagrams-max-classes", "25",
      "-doc-version", version.value,
      "-doc-title", name.value,
      "-doc-root-content", baseDirectory.value+"/root-doc.txt"
    ),
    aggregate in doc := false,
    // Include macro classes, resources, and sources main JAR.
    mappings in (Compile, packageBin) ++= (mappings in (coreMacros, Compile, packageBin)).value,
    mappings in (Compile, packageSrc) ++= (mappings in (coreMacros, Compile, packageSrc)).value,
    mappings in (Compile, packageBin) ++= (mappings in (chiselFrontend, Compile, packageBin)).value,
    mappings in (Compile, packageSrc) ++= (mappings in (chiselFrontend, Compile, packageSrc)).value,
    // Export the packaged JAR so projects that depend directly on Chisel project (rather than the
    // published artifact) also see the stuff in coreMacros and chiselFrontend.
    exportJars := true
  ).
  dependsOn(dependentProjects.map(classpathDependency(_)):_*)
