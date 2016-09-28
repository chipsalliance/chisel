// See LICENSE for license details.

site.settings

site.includeScaladoc()

ghpages.settings

import UnidocKeys._

lazy val customUnidocSettings = unidocSettings ++ Seq (
  doc in Compile := (doc in ScalaUnidoc).value,
  target in unidoc in ScalaUnidoc := crossTarget.value / "api"
)

lazy val commonSettings = Seq (
  organization := "edu.berkeley.cs",
  version := "3.1-SNAPSHOT",
  git.remoteRepo := "git@github.com:ucb-bar/chisel3.git",
  autoAPIMappings := true,
  scalaVersion := "2.11.7"
)

lazy val chiselSettings = Seq (
  name := "Chisel3",

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
    "Sonatype Snapshots" at "http://oss.sonatype.org/content/repositories/snapshots",
    "Sonatype Releases" at "http://oss.sonatype.org/content/repositories/releases"
  ),

  /* Bumping "com.novocode" % "junit-interface" % "0.11", causes DelayTest testSeqReadBundle to fail
   *  in subtly disturbing ways on Linux (but not on Mac):
   *  - some fields in the generated .h file are re-named,
   *  - an additional field is added
   *  - the generated .cpp file has additional differences:
   *    - different temps in clock_lo
   *    - missing assignments
   *    - change of assignment order
   *    - use of "Tx" vs. "Tx.values"
   */
  libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.5" % "test",
  libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value,
  libraryDependencies += "org.scalacheck" %% "scalacheck" % "1.12.4" % "test",

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
  settings(
    libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value
  )

lazy val chiselFrontend = (project in file("chiselFrontend")).
  settings(commonSettings: _*).
  settings(
    libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value
  ).
  dependsOn(coreMacros)

lazy val chisel = (project in file(".")).
  settings(commonSettings: _*).
  settings(customUnidocSettings: _*).
  settings(chiselSettings: _*).
  dependsOn(coreMacros).
  dependsOn(chiselFrontend).
  settings(
    aggregate in doc := false,
    // Include macro classes, resources, and sources main jar.
    mappings in (Compile, packageBin) <++= mappings in (coreMacros, Compile, packageBin),
    mappings in (Compile, packageSrc) <++= mappings in (coreMacros, Compile, packageSrc),
    mappings in (Compile, packageBin) <++= mappings in (chiselFrontend, Compile, packageBin),
    mappings in (Compile, packageSrc) <++= mappings in (chiselFrontend, Compile, packageSrc)
  ).
  aggregate(coreMacros, chiselFrontend)
