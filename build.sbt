// See LICENSE for license details.

enablePlugins(SiteScaladocPlugin)

addCommandAlias("fmt", "; scalafmtAll ; scalafmtSbt")
addCommandAlias("fmtCheck", "; scalafmtCheckAll ; scalafmtSbtCheck")

lazy val firtoolVersion = settingKey[Option[String]]("Determine the version of firtool on the PATH")
ThisBuild / firtoolVersion := {
  import scala.sys.process._
  val Version = """^CIRCT firtool-(\S+)$""".r
  try {
    val lines = Process(Seq("firtool", "--version")).lineStream
    lines.collectFirst { case Version(v) => v }
  } catch {
    case e: java.io.IOException => None
  }
}

// Previous versions are read from project/previous-versions.txt
// If this file is empty or does not exist, no binary compatibility checking will be done
// Add waivers to the directory defined by key `mimaFiltersDirectory` in files named: <since version>.backwards.excludes
//   eg. unipublish/src/main/mima-filters/5.0.0.backwards.excludes
val previousVersions = settingKey[Set[String]]("Previous versions for binary compatibility checking")
ThisBuild / previousVersions := {
  val file = new java.io.File("project", "previous-versions.txt")
  if (file.isFile) {
    scala.io.Source.fromFile(file).getLines.toSet
  } else {
    Set()
  }
}

val emitVersion = taskKey[Unit]("Write the version to version.txt")
emitVersion := {
  IO.write(new java.io.File("version.txt"), version.value)
}

lazy val minimalSettings = Seq(
  organization := "org.chipsalliance",
  scalacOptions := Seq("-deprecation", "-feature"),
  scalaVersion := "2.13.12"
)

lazy val commonSettings = minimalSettings ++ Seq(
  resolvers ++= Resolver.sonatypeOssRepos("snapshots"),
  resolvers ++= Resolver.sonatypeOssRepos("releases"),
  autoAPIMappings := true,
  libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value,
  // Macros paradise is integrated into 2.13 but requires a scalacOption
  scalacOptions += "-Ymacro-annotations"
)

lazy val fatalWarningsSettings = Seq(
  scalacOptions ++= {
    if (sys.props.contains("disableFatalWarnings")) {
      Nil
    } else {
      "-Werror" :: Nil
    }
  }
)

lazy val warningSuppression = Seq(
  scalacOptions += "-Wconf:" + Seq(
    "msg=APIs in chisel3.internal:s",
    "msg=Importing from firrtl:s",
    "msg=migration to the MLIR:s",
    "msg=method hasDefiniteSize in trait IterableOnceOps is deprecated:s", // replacement `knownSize` is not in 2.12
    "msg=object JavaConverters in package collection is deprecated:s",
    "msg=undefined in comment for method cf in class PrintableHelper:s",
    // This is deprecated for external users but not internal use
    "cat=deprecation&origin=firrtl\\.options\\.internal\\.WriteableCircuitAnnotation:s",
    "cat=deprecation&origin=chisel3\\.util\\.experimental\\.BoringUtils.*:s"
  ).mkString(",")
)

// This should only be mixed in by projects that are published
// See 'unipublish' project below
lazy val publishSettings = Seq(
  versionScheme := Some("semver-spec"),
  publishMavenStyle := true,
  Test / publishArtifact := false,
  pomIncludeRepository := { x => false },
  homepage := Some(url("https://www.chisel-lang.org")),
  organizationHomepage := Some(url("https://www.chipsalliance.org")),
  licenses := List(License.Apache2),
  developers := List(
    Developer("jackkoenig", "Jack Koenig", "jack.koenig3@gmail.com", url("https://github.com/jackkoenig")),
    Developer("azidar", "Adam Izraelevitz", "azidar@gmail.com", url("https://github.com/azidar")),
    Developer("seldridge", "Schuyler Eldridge", "schuyler.eldridge@gmail.com", url("https://github.com/seldridge"))
  ),
  sonatypeCredentialHost := "s01.oss.sonatype.org",
  sonatypeRepository := "https://s01.oss.sonatype.org/service/local",
  // We are just using 'publish / skip' as a hook to run checks required for publishing,
  // but that are not necessarily required for local development or running testing in CI
  publish / skip := {
    // Check that SBT Dynver can properly derive a version which requires unshallow clone
    val v = version.value
    if (dynverGitDescribeOutput.value.hasNoTags) {
      sys.error(s"Failed to derive version from git tags. Maybe run `git fetch --unshallow`? Version: $v")
    }
    // Check that firtool exists on the PATH so Chisel can use the version it was tested against
    // in error messages
    if (firtoolVersion.value.isEmpty) {
      sys.error(s"Failed to determine firtool version. Make sure firtool is found on the PATH.")
    }
    (publish / skip).value
  },
  publishTo := {
    val v = version.value
    val nexus = "https://s01.oss.sonatype.org/"
    if (v.trim.endsWith("SNAPSHOT")) {
      Some("snapshots".at(nexus + "content/repositories/snapshots"))
    } else {
      Some("releases".at(nexus + "service/local/staging/deploy/maven2"))
    }
  }
)

// FIRRTL SETTINGS

lazy val firrtlSettings = Seq(
  name := "firrtl",
  addCompilerPlugin(scalafixSemanticdb),
  scalacOptions := Seq(
    "-deprecation",
    "-unchecked",
    "-language:reflectiveCalls",
    "-language:existentials",
    "-language:implicitConversions",
    "-Yrangepos" // required by SemanticDB compiler plugin
  ),
  // Always target Java8 for maximum compatibility
  javacOptions ++= Seq("-source", "1.8", "-target", "1.8"),
  libraryDependencies ++= Seq(
    "org.scala-lang" % "scala-reflect" % scalaVersion.value,
    "org.scalatest" %% "scalatest" % "3.2.14" % "test",
    "org.scalatestplus" %% "scalacheck-1-16" % "3.2.14.0" % "test",
    "com.github.scopt" %% "scopt" % "4.1.0",
    "net.jcazevedo" %% "moultingyaml" % "0.4.2",
    "org.json4s" %% "json4s-native" % "4.0.6",
    "org.apache.commons" % "commons-text" % "1.10.0",
    "io.github.alexarchambault" %% "data-class" % "0.2.6",
    "com.lihaoyi" %% "os-lib" % "0.9.1"
  ),
  scalacOptions += "-Ymacro-annotations",
  // starting with scala 2.13 the parallel collections are separate from the standard library
  libraryDependencies += "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
)

lazy val assemblySettings = Seq(
  assembly / assemblyJarName := "firrtl.jar",
  assembly / test := {},
  assembly / assemblyOutputPath := file("./utils/bin/firrtl.jar")
)

lazy val testAssemblySettings = Seq(
  Test / assembly / test := {}, // Ditto above
  Test / assembly / assemblyMergeStrategy := {
    case PathList("firrtlTests", xs @ _*) => MergeStrategy.discard
    case x =>
      val oldStrategy = (Test / assembly / assemblyMergeStrategy).value
      oldStrategy(x)
  },
  Test / assembly / assemblyJarName := s"firrtl-test.jar",
  Test / assembly / assemblyOutputPath := file("./utils/bin/" + (Test / assembly / assemblyJarName).value)
)

lazy val svsim = (project in file("svsim"))
  .settings(minimalSettings)
  .settings(
    // Published as part of unipublish
    publish / skip := true,
    libraryDependencies ++= Seq(
      "org.scalatest" %% "scalatest" % "3.2.16" % "test",
      "org.scalatestplus" %% "scalacheck-1-16" % "3.2.14.0" % "test"
    )
  )

lazy val firrtl = (project in file("firrtl"))
  .enablePlugins(ScalaUnidocPlugin)
  .settings(
    fork := true,
    Test / testForkedParallel := true
  )
  .settings(commonSettings)
  .settings(firrtlSettings)
  .settings(assemblySettings)
  .settings(inConfig(Test)(baseAssemblySettings))
  .settings(testAssemblySettings)
  .settings(
    // Published as part of unipublish
    publish / skip := true
  )
  .enablePlugins(BuildInfoPlugin)
  .settings(
    buildInfoPackage := name.value,
    buildInfoUsePackageAsPath := true,
    buildInfoKeys := Seq[BuildInfoKey](buildInfoPackage, version, scalaVersion, sbtVersion)
  )
  .settings(warningSuppression: _*)
  .settings(fatalWarningsSettings: _*)

lazy val chiselSettings = Seq(
  name := "chisel",
  libraryDependencies ++= Seq(
    "org.scalatest" %% "scalatest" % "3.2.16" % "test",
    "org.scalatestplus" %% "scalacheck-1-16" % "3.2.14.0" % "test",
    "com.lihaoyi" %% "upickle" % "3.1.0"
  )
) ++ (
  // Tests from other projects may still run concurrently
  //  if we're not running with -DminimalResources.
  // Another option would be to experiment with:
  //  concurrentRestrictions in Global += Tags.limit(Tags.Test, 1),
  sys.props.contains("minimalResources") match {
    case true  => Seq(Test / parallelExecution := false)
    case false => Seq(fork := true, Test / testForkedParallel := true)
  }
)

autoCompilerPlugins := true
autoAPIMappings := true

// Plugin must be fully cross-versioned (published for Scala minor version)
lazy val pluginScalaVersions = Seq(
  "2.13.0",
  "2.13.1",
  "2.13.2",
  "2.13.3",
  "2.13.4",
  "2.13.5",
  "2.13.6",
  "2.13.7",
  "2.13.8",
  "2.13.9",
  "2.13.10",
  "2.13.11",
  "2.13.12"
)

lazy val plugin = (project in file("plugin"))
  .settings(name := "chisel-plugin")
  .settings(commonSettings: _*)
  .settings(publishSettings: _*)
  .settings(
    libraryDependencies += "org.scala-lang" % "scala-compiler" % scalaVersion.value,
    crossScalaVersions := pluginScalaVersions,
    // Must be published for Scala minor version
    crossVersion := CrossVersion.full,
    crossTarget := {
      // workaround for https://github.com/sbt/sbt/issues/5097
      target.value / s"scala-${scalaVersion.value}"
    }
  )
  .settings(fatalWarningsSettings: _*)
  .settings(
    mimaPreviousArtifacts := previousVersions.value.map { version =>
      (organization.value % name.value % version).cross(CrossVersion.full)
    }
  )

lazy val usePluginSettings = Seq(
  Compile / scalacOptions ++= {
    val jar = (plugin / Compile / Keys.`package`).value
    val addPlugin = "-Xplugin:" + jar.getAbsolutePath
    // add plugin timestamp to compiler options to trigger recompile of
    // main after editing the plugin. (Otherwise a 'clean' is needed.)
    val dummy = "-Jdummy=" + jar.lastModified
    Seq(addPlugin, dummy)
  }
)

lazy val macros = (project in file("macros"))
  .settings(name := "chisel-macros")
  .settings(commonSettings: _*)
  .settings(
    // Published as part of unipublish
    publish / skip := true
  )

lazy val core = (project in file("core"))
  .settings(commonSettings: _*)
  .enablePlugins(BuildInfoPlugin)
  .settings(
    buildInfoPackage := "chisel3",
    buildInfoUsePackageAsPath := true,
    buildInfoKeys := Seq[BuildInfoKey](buildInfoPackage, version, scalaVersion, sbtVersion, firtoolVersion)
  )
  .settings(
    // Published as part of unipublish
    publish / skip := true
  )
  .settings(warningSuppression: _*)
  .settings(fatalWarningsSettings: _*)
  .settings(
    name := "chisel-core",
    libraryDependencies ++= Seq(
      "com.lihaoyi" %% "upickle" % "3.1.0",
      "com.lihaoyi" %% "os-lib" % "0.9.1"
    ),
    scalacOptions := scalacOptions.value ++ Seq(
      "-explaintypes",
      "-feature",
      "-language:reflectiveCalls",
      "-unchecked",
      "-Xcheckinit",
      "-Xlint:infer-any"
//      , "-Xlint:missing-interpolator"
    )
  )
  .dependsOn(macros)
  .dependsOn(firrtl)

// This will always be the root project, even if we are a sub-project.
lazy val root = RootProject(file("."))

lazy val chisel = (project in file("."))
  .settings(commonSettings: _*)
  .settings(chiselSettings: _*)
  .settings(
    // Published as part of unipublish
    publish / skip := true
  )
  .settings(usePluginSettings: _*)
  .dependsOn(macros)
  .dependsOn(core)
  .dependsOn(firrtl)
  .dependsOn(svsim)
  .aggregate(macros, core, plugin, firrtl, svsim)
  .settings(
    // Suppress Scala 3 behavior requiring explicit types on implicit definitions
    // Note this must come before the -Wconf is warningSuppression
    Test / scalacOptions += "-Wconf:cat=other-implicit-type:s"
  )
  .settings(warningSuppression: _*)
  .settings(fatalWarningsSettings: _*)
  .settings(
    Test / scalacOptions ++= Seq("-language:reflectiveCalls")
  )

def addUnipublishDeps(proj: Project)(deps: Project*): Project = {
  def inTestScope(module: ModuleID): Boolean = module.configurations.exists(_ == "test")
  deps.foldLeft(proj) {
    case (p, dep) =>
      p.settings(
        libraryDependencies ++= (dep / libraryDependencies).value.filterNot(inTestScope),
        Compile / packageBin / mappings ++= (dep / Compile / packageBin / mappings).value,
        Compile / packageSrc / mappings ++= (dep / Compile / packageSrc / mappings).value
      )
  }
}

// This is a pseudo-project that unifies all compilation units (excluding the plugin) into a single artifact
// It should be used for all publishing and MiMa binary compatibility checking
lazy val unipublish =
  addUnipublishDeps(project in file("unipublish"))(
    firrtl,
    svsim,
    macros,
    core,
    chisel
  )
    .aggregate(plugin) // Also publish the plugin when publishing this project
    .settings(name := (chisel / name).value)
    .enablePlugins(ScalaUnidocPlugin)
    .settings(
      // Plugin isn't part of Chisel's public API, exclude from ScalaDoc
      ScalaUnidoc / unidoc / unidocProjectFilter := inAnyProject -- inProjects(plugin)
    )
    .settings(commonSettings: _*)
    .settings(publishSettings: _*)
    .settings(usePluginSettings: _*)
    .settings(warningSuppression: _*)
    .settings(fatalWarningsSettings: _*)
    .settings(
      mimaPreviousArtifacts := previousVersions.value.map { version =>
        organization.value %% name.value % version
      },
      // This is a pseudo-project with no class files, use the package jar instead
      mimaCurrentClassfiles := (Compile / packageBin).value,
      // Forward doc command to unidoc
      Compile / doc := (ScalaUnidoc / doc).value,
      // Include unidoc as the ScalaDoc for publishing
      Compile / packageDoc / mappings := (ScalaUnidoc / packageDoc / mappings).value,
      Compile / doc / scalacOptions ++= Seq(
        "-diagrams",
        "-groups",
        "-skip-packages",
        "chisel3.internal",
        "-diagrams-max-classes",
        "25",
        "-doc-version",
        version.value,
        "-doc-title",
        name.value,
        "-doc-root-content",
        baseDirectory.value + "/root-doc.txt",
        "-sourcepath",
        (ThisBuild / baseDirectory).value.toString,
        "-doc-source-url", {
          val branch =
            if (version.value.endsWith("-SNAPSHOT")) {
              "master"
            } else {
              s"v${version.value}"
            }
          s"https://github.com/chipsalliance/chisel/tree/$branch€{FILE_PATH_EXT}#L€{FILE_LINE}"
        },
        "-language:implicitConversions"
      ) ++
        // Suppress compiler plugin for source files in core
        // We don't need this in regular compile because we just don't add the chisel-plugin to core's scalacOptions
        // This works around an issue where unidoc uses the exact same arguments for all source files.
        // This is probably fundamental to how ScalaDoc works so there may be no solution other than this workaround.
        // See https://github.com/sbt/sbt-unidoc/issues/107
        (core / Compile / sources).value.map("-P:chiselplugin:INTERNALskipFile:" + _)
        ++ Seq("-implicits")
    )

// End-to-end tests that check the functionality of the emitted design with simulation
lazy val integrationTests = (project in file("integration-tests"))
  .dependsOn(chisel % "compile->compile;test->test")
  .dependsOn(firrtl) // SBT doesn't seem to be propagating transitive library dependencies...
  .dependsOn(standardLibrary)
  .settings(commonSettings: _*)
  .settings(warningSuppression: _*)
  .settings(fatalWarningsSettings: _*)
  .settings(chiselSettings: _*)
  .settings(usePluginSettings: _*)

// the chisel standard library
lazy val standardLibrary = (project in file("stdlib"))
  .dependsOn(chisel)
  .settings(commonSettings: _*)
  .settings(chiselSettings: _*)
  .settings(usePluginSettings: _*)

lazy val docs = project // new documentation project
  .in(file("docs-target")) // important: it must not be docs/
  .dependsOn(chisel)
  .enablePlugins(MdocPlugin)
  .settings(usePluginSettings: _*)
  .settings(commonSettings)
  .settings(
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-language:implicitConversions",
      "-Wconf:msg=firrtl:s,cat=other-implicit-type:s"
    ),
    mdocIn := file("docs/src"),
    mdocOut := file("docs/generated"),
    // None of our links are hygienic because they're primarily used on the website with .html
    mdocExtraArguments := Seq("--cwd", "docs", "--no-link-hygiene"),
    mdocVariables := Map(
      "BUILD_DIR" -> "docs-target" // build dir for mdoc programs to dump temp files
    )
  )
  .settings(fatalWarningsSettings: _*)
