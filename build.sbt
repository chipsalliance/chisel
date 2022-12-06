// See LICENSE for license details.

import com.typesafe.tools.mima.core._

enablePlugins(SiteScaladocPlugin)

def scalacOptionsVersion(scalaVersion: String): Seq[String] = {
  Seq() ++ {
    // If we're building with Scala > 2.11, enable the compile option
    //  switch to support our anonymous Bundle definitions:
    //  https://github.com/scala/bug/issues/10047
    CrossVersion.partialVersion(scalaVersion) match {
      case Some((2, scalaMajor: Long)) if scalaMajor < 12 => Seq()
      case _ => Seq("-Xsource:2.11")
    }
  }
}

def javacOptionsVersion(scalaVersion: String): Seq[String] = {
  Seq() ++ {
    // Scala 2.12 requires Java 8. We continue to generate
    //  Java 7 compatible code for Scala 2.11
    //  for compatibility with old clients.
    CrossVersion.partialVersion(scalaVersion) match {
      case Some((2, scalaMajor: Long)) if scalaMajor < 12 =>
        Seq("-source", "1.7", "-target", "1.7")
      case _ =>
        Seq("-source", "1.8", "-target", "1.8")
    }
  }
}

val defaultVersions = Map("firrtl" -> "1.3-SNAPSHOT")

lazy val commonSettings = Seq (
  resolvers ++= Seq(
    Resolver.sonatypeRepo("snapshots"),
    Resolver.sonatypeRepo("releases")
  ),
  organization := "edu.berkeley.cs",
  version := "3.3-SNAPSHOT",
  autoAPIMappings := true,
  scalaVersion := "2.12.11",
  crossScalaVersions := Seq("2.12.11", "2.11.12"),
  scalacOptions := Seq("-deprecation", "-feature") ++ scalacOptionsVersion(scalaVersion.value),
  libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value,
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.1" cross CrossVersion.full),
  (scalastyleConfig in Test) := (baseDirectory in root).value / "scalastyle-test-config.xml",
  // Use the root project's unmanaged base for all sub-projects.
  unmanagedBase := (unmanagedBase in root).value,
  // Since we want to examine the classpath to determine if a dependency on firrtl is required,
  //  this has to be a Task setting.
  //  Fortunately, allDependencies is a Task Setting, so we can modify that.
  allDependencies := {
    allDependencies.value ++ Seq("firrtl").collect {
      // If we have an unmanaged jar file on the classpath, assume we're to use that,
      case dep: String if !(unmanagedClasspath in Compile).value.toString.contains(s"$dep.jar") =>
        //  otherwise let sbt fetch the appropriate version.
        "edu.berkeley.cs" %% dep % sys.props.getOrElse(dep + "Version", defaultVersions(dep))
    }
  }
)

<<<<<<< HEAD
lazy val publishSettings = Seq (
=======
lazy val publishSettings = Seq(
  versionScheme := Some("pvp"),
>>>>>>> 9f080d51 (Add versionScheme (PVP) to SBT publish settings (#2871))
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
  }
)

lazy val chiselSettings = Seq (
  name := "chisel3",

// sbt 1.2.6 fails with `Symbol 'term org.junit' is missing from the classpath`
// when compiling tests under 2.11.12
// An explicit dependency on junit seems to alleviate this.
  libraryDependencies ++= Seq(
    "junit" % "junit" % "4.13" % "test",
    "org.scalatest" %% "scalatest" % "3.1.2" % "test",
    "org.scalatestplus" %% "scalacheck-1-14" % "3.1.1.1" % "test",
    "com.github.scopt" %% "scopt" % "3.7.1"
  ),
  javacOptions ++= javacOptionsVersion(scalaVersion.value)
) ++ (
  // Tests from other projects may still run concurrently
  //  if we're not running with -DminimalResources.
  // Another option would be to experiment with:
  //  concurrentRestrictions in Global += Tags.limit(Tags.Test, 1),
  sys.props.contains("minimalResources") match {
    case true  => Seq( Test / parallelExecution := false )
    case false => Seq( fork := true,
                       Test / testForkedParallel := true )
  }
)

lazy val macros = (project in file("macros")).
  settings(name := "chisel3-macros").
  settings(commonSettings: _*).
  settings(publishSettings: _*).
  settings(mimaPreviousArtifacts := Set("edu.berkeley.cs" %% "chisel3-macros" % "3.3.3"))

lazy val core = (project in file("core")).
  settings(commonSettings: _*).
  settings(publishSettings: _*).
  settings(
    name := "chisel3-core",
    scalacOptions := scalacOptions.value ++ Seq(
      "-deprecation",
      "-explaintypes",
      "-feature",
      "-language:reflectiveCalls",
      "-unchecked",
      "-Xcheckinit",
      "-Xlint:infer-any"
//      "-Xlint:missing-interpolator"
    ),
    mimaPreviousArtifacts := Set("edu.berkeley.cs" %% "chisel3-core" % "3.3.3"),
    mimaBinaryIssueFilters ++= Seq(
      // Modified package private methods (https://github.com/lightbend/mima/issues/53)
      ProblemFilters.exclude[IncompatibleResultTypeProblem]("chisel3.RawModule.generateComponent"),
      ProblemFilters.exclude[IncompatibleResultTypeProblem]("chisel3.BlackBox.generateComponent"),
      ProblemFilters.exclude[IncompatibleResultTypeProblem]("chisel3.internal.LegacyModule.generateComponent"),
      ProblemFilters.exclude[IncompatibleResultTypeProblem]("chisel3.experimental.BaseModule.generateComponent"),
      ProblemFilters.exclude[IncompatibleResultTypeProblem]("chisel3.experimental.ExtModule.generateComponent"),
      // Scala 2.11 only issue, new concrete methods in traits require recompilation of implementing classes
      // Not a problem because HasId is package private so all implementers are in chisel3 itself
      // Note there is no problem for user subtypes of Record because setRef is implemented by Data
      ProblemFilters.exclude[ReversedMissingMethodProblem]("chisel3.internal.HasId._parent_="),
      // Not a problem because generateComponent is package private and unimplemented in BaseModule
      // Users cannot practically extend BaseModule (they'd need to implement package private methods which they cannot)
      ProblemFilters.exclude[ReversedMissingMethodProblem]("chisel3.experimental.BaseModule.generateComponent")
    )
  ).
  dependsOn(macros)

// This will always be the root project, even if we are a sub-project.
lazy val root = RootProject(file("."))

lazy val chisel = (project in file(".")).
  enablePlugins(BuildInfoPlugin).
  enablePlugins(ScalaUnidocPlugin).
  settings(
    buildInfoPackage := name.value,
    buildInfoUsePackageAsPath := true,
    buildInfoKeys := Seq[BuildInfoKey](buildInfoPackage, version, scalaVersion, sbtVersion)
  ).
  settings(commonSettings: _*).
  settings(chiselSettings: _*).
  settings(publishSettings: _*).
  dependsOn(macros).
  dependsOn(core).
  aggregate(macros, core).
  settings(
    mimaPreviousArtifacts := Set("edu.berkeley.cs" %% "chisel3" % "3.3.3"),
    scalacOptions in Test ++= Seq("-language:reflectiveCalls"),
    scalacOptions in Compile in doc ++= Seq(
      "-diagrams",
      "-groups",
      "-skip-packages", "chisel3.internal",
      "-diagrams-max-classes", "25",
      "-doc-version", version.value,
      "-doc-title", name.value,
      "-doc-root-content", baseDirectory.value+"/root-doc.txt",
      "-sourcepath", (baseDirectory in ThisBuild).value.toString,
      "-doc-source-url",
      {
        val branch =
          if (version.value.endsWith("-SNAPSHOT")) {
            "master"
          } else {
            s"v${version.value}"
          }
        s"https://github.com/chipsalliance/chisel3/tree/$branch€{FILE_PATH_EXT}#L€{FILE_LINE}"
      }
    )
  )
