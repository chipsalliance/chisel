// See LICENSE for license details.

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

val defaultVersions = Map(
  "firrtl" -> "edu.berkeley.cs" %% "firrtl" % "1.5-SNAPSHOT",
  "treadle" -> "edu.berkeley.cs" %% "treadle" % "1.5-SNAPSHOT"
)

lazy val commonSettings = Seq (
  resolvers ++= Seq(
    Resolver.sonatypeRepo("snapshots"),
    Resolver.sonatypeRepo("releases")
  ),
  organization := "edu.berkeley.cs",
  version := "3.5-SNAPSHOT",
  autoAPIMappings := true,
  scalaVersion := "2.12.12",
  crossScalaVersions := Seq("2.12.12", "2.11.12"),
  scalacOptions := Seq("-deprecation", "-feature") ++ scalacOptionsVersion(scalaVersion.value),
  libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value,
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.1" cross CrossVersion.full),
)

lazy val publishSettings = Seq (
  publishMavenStyle := true,
  publishArtifact in Test := false,
  pomIncludeRepository := { x => false },
  pomExtra := <url>http://chisel.eecs.berkeley.edu/</url>
    <licenses>
      <license>
        <name>apache-v2</name>
        <url>https://opensource.org/licenses/Apache-2.0</url>
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
    "junit" % "junit" % "4.13.1" % "test",
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

autoCompilerPlugins := true

// Plugin must be fully cross-versioned (published for Scala minor version)
// The plugin only works in Scala 2.12+
lazy val pluginScalaVersions = Seq(
  "2.11.12", // Only to support chisel3 cross building for 2.11, plugin does nothing in 2.11
  // scalamacros paradise version used is not published for 2.12.0 and 2.12.1
  "2.12.2",
  "2.12.3",
  "2.12.4",
  "2.12.5",
  "2.12.6",
  "2.12.7",
  "2.12.8",
  "2.12.9",
  "2.12.10",
  "2.12.11",
  "2.12.12",
)

lazy val plugin = (project in file("plugin")).
  settings(name := "chisel3-plugin").
  settings(commonSettings: _*).
  settings(publishSettings: _*).
  settings(
    libraryDependencies += "org.scala-lang" % "scala-compiler" % scalaVersion.value,
    scalacOptions += "-Xfatal-warnings",
    crossScalaVersions := pluginScalaVersions,
    // Must be published for Scala minor version
    crossVersion := CrossVersion.full,
    crossTarget := {
      // workaround for https://github.com/sbt/sbt/issues/5097
      target.value / s"scala-${scalaVersion.value}"
    },
    // Only publish for Scala 2.12
    publish / skip := !scalaVersion.value.startsWith("2.12")
  ).
  settings(
    mimaPreviousArtifacts := {
      // Not published for 2.11, do not try to check binary compatibility with a 2.11 artifact
      if (scalaVersion.value.startsWith("2.11")) Set()
      else Set()
    }
  )

lazy val usePluginSettings = Seq(
  scalacOptions in Compile ++= {
    val jar = (plugin / Compile / Keys.`package`).value
    val addPlugin = "-Xplugin:" + jar.getAbsolutePath
    // add plugin timestamp to compiler options to trigger recompile of
    // main after editing the plugin. (Otherwise a 'clean' is needed.)
    val dummy = "-Jdummy=" + jar.lastModified
    Seq(addPlugin, dummy)
  }
)

lazy val macros = (project in file("macros")).
  settings(name := "chisel3-macros").
  settings(commonSettings: _*).
  settings(publishSettings: _*).
  settings(mimaPreviousArtifacts := Set())

lazy val firrtlRef = ProjectRef(workspaceDirectory / "firrtl", "firrtl")

lazy val core = (project in file("core")).
  sourceDependency(firrtlRef, defaultVersions("firrtl")).
  settings(commonSettings: _*).
  enablePlugins(BuildInfoPlugin).
  settings(
    buildInfoPackage := "chisel3",
    buildInfoUsePackageAsPath := true,
    buildInfoKeys := Seq[BuildInfoKey](buildInfoPackage, version, scalaVersion, sbtVersion)
  ).
  settings(publishSettings: _*).
  settings(mimaPreviousArtifacts := Set()).
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
//      , "-Xlint:missing-interpolator"
    )
  ).
  dependsOn(macros)

// This will always be the root project, even if we are a sub-project.
lazy val root = RootProject(file("."))

lazy val chisel = (project in file(".")).
  enablePlugins(ScalaUnidocPlugin).
  settings(commonSettings: _*).
  settings(chiselSettings: _*).
  settings(publishSettings: _*).
  settings(usePluginSettings: _*).
  dependsOn(macros).
  dependsOn(core).
  aggregate(macros, core, plugin).
  settings(
    mimaPreviousArtifacts := Set(),
    libraryDependencies += defaultVersions("treadle") % "test",
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
        s"https://github.com/freechipsproject/chisel3/tree/$branch/€{FILE_PATH}.scala"
      }
    )
  )

lazy val docs = project       // new documentation project
  .in(file("docs-target")) // important: it must not be docs/
  .dependsOn(chisel)
  .enablePlugins(MdocPlugin)
  .settings(usePluginSettings: _*)
  .settings(commonSettings)
  .settings(
    scalacOptions += "-language:reflectiveCalls",
    mdocIn := file("docs/src"),
    mdocOut := file("docs/generated"),
    mdocExtraArguments := Seq("--cwd", "docs"),
    mdocVariables := Map(
      "BUILD_DIR" -> "docs-target" // build dir for mdoc programs to dump temp files
    )
  )

addCommandAlias("com", "all compile")
addCommandAlias("lint", "; compile:scalafix --check ; test:scalafix --check")
addCommandAlias("fix", "all compile:scalafix test:scalafix")
