// See LICENSE for license details.
import Dependencies._


site.settings

site.includeScaladoc()

ghpages.settings

import UnidocKeys._

lazy val customUnidocSettings = unidocSettings ++ Seq (
  doc in Compile := (doc in ScalaUnidoc).value,
  target in unidoc in ScalaUnidoc := crossTarget.value / "api"
)

val defaultVersions = Map("firrtl" -> "1.1-SNAPSHOT")

lazy val commonSettings = Seq (
  organization := "edu.berkeley.cs",
  version := "3.1-SNAPSHOT",
  git.remoteRepo := "git@github.com:ucb-bar/chisel3.git",
  autoAPIMappings := true,
  scalaVersion := "2.11.7",
  scalacOptions := Seq("-deprecation", "-feature"),
  // Use the root project's unmanaged base for all sub-projects.
  unmanagedBase := (unmanagedBase in root).value,
  // Use the root project's classpath for all sub-projects.
  fullClasspath := (fullClasspath in Compile in root).value,
  libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value,
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)
)

lazy val chiselSettings = Seq (
  name := "chisel3",

  publishMavenStyle := true,
  publishArtifact in Test := false,
  pomIncludeRepository := { x => false },
  pomExtra := <url>http://chisel.eecs.berkeley.edu/</url>
    <licenses>
      <license>
        <name>BSD-style</name>
        <url>http://www.opensource.org/licenses/bsd-license.php</url>
        <distribution>repo</distribution>
      </license>
    </licenses>
    <scm>
      <url>https://github.com/ucb-bar/chisel3.git</url>
      <connection>scm:git:github.com/ucb-bar/chisel3.git</connection>
    </scm>
    <developers>
      <developer>
        <id>jackbackrack</id>
        <name>Jonathan Bachrach</name>
        <url>http://www.eecs.berkeley.edu/~jrb/</url>
      </developer>
    </developers>,

  publishTo <<= version { v: String =>
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

  libraryDependencies ++= Seq(
    "org.scalatest" %% "scalatest" % "2.2.5" % "test",
    "org.scalacheck" %% "scalacheck" % "1.12.4" % "test",
    "com.github.scopt" %% "scopt" % "3.4.0"
  ),

  // Tests from other projects may still run concurrently.
  parallelExecution in Test := true,

  javacOptions ++= Seq("-target", "1.7")
  //  Hopefully we get these options back in Chisel3
  //  scalacOptions in (Compile, doc) <++= (baseDirectory in LocalProject("chisel"), version) map { (bd, v) =>
  //    Seq("-diagrams", "-diagrams-max-classes", "25", "-sourcepath", bd.getAbsolutePath, "-doc-source-url",
  //        "https://github.com/ucb-bar/chisel/tree/master/€{FILE_PATH}.scala")
  //  }
)

lazy val coreMacros = (project in file("coreMacros")).
  settings(commonSettings: _*).
  settings(publishArtifact := false)

lazy val chiselFrontend = (project in file("chiselFrontend")).
  settings(commonSettings: _*).
  settings(publishArtifact := false).
  dependsOn(coreMacros).
  dependsOn((Dependencies.chiselProjectDependencies("chisel")):_*)


// This will always be the root project, even if we are a sub-project.
lazy val root = RootProject(file("."))

lazy val chisel = (project in file(".")).
  enablePlugins(BuildInfoPlugin).
  settings(
    buildInfoPackage := name.value,
    buildInfoOptions += BuildInfoOption.BuildTime,
    buildInfoUsePackageAsPath := true,
    buildInfoKeys := Seq[BuildInfoKey](buildInfoPackage, version, scalaVersion, sbtVersion)
  ).
  settings(commonSettings: _*).
  settings(customUnidocSettings: _*).
  settings(chiselSettings: _*).
  // Prevent separate JARs from being generated for coreMacros and chiselFrontend.
  dependsOn(coreMacros % "compile-internal;test-internal").
  dependsOn(chiselFrontend % "compile-internal;test-internal").
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
    mappings in (Compile, packageBin) <++= mappings in (coreMacros, Compile, packageBin),
    mappings in (Compile, packageSrc) <++= mappings in (coreMacros, Compile, packageSrc),
    mappings in (Compile, packageBin) <++= mappings in (chiselFrontend, Compile, packageBin),
    mappings in (Compile, packageSrc) <++= mappings in (chiselFrontend, Compile, packageSrc),
    // Export the packaged JAR so projects that depend directly on Chisel project (rather than the
    // published artifact) also see the stuff in coreMacros and chiselFrontend.
    exportJars := true
  ).
  dependsOn((Dependencies.chiselProjectDependencies("chisel")):_*)

