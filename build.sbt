// See LICENSE for license details.

enablePlugins(SiteScaladocPlugin)

lazy val commonSettings = Seq(
  resolvers ++= Resolver.sonatypeOssRepos("snapshots"),
  resolvers ++= Resolver.sonatypeOssRepos("releases"),
  organization := "edu.berkeley.cs",
  version := "3.6-SNAPSHOT",
  autoAPIMappings := true,
  scalaVersion := "2.13.10",
  crossScalaVersions := Seq("2.13.10", "2.12.17"),
  scalacOptions := Seq("-deprecation", "-feature"),
  libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value,
  // Macros paradise is integrated into 2.13 but requires a scalacOption
  scalacOptions ++= {
    CrossVersion.partialVersion(scalaVersion.value) match {
      case Some((2, n)) if n >= 13 => "-Ymacro-annotations" :: Nil
      case _                       => Nil
    }
  },
  libraryDependencies ++= {
    CrossVersion.partialVersion(scalaVersion.value) match {
      case Some((2, n)) if n >= 13 => Nil
      case _                       => compilerPlugin(("org.scalamacros" % "paradise" % "2.1.1").cross(CrossVersion.full)) :: Nil
    }
  }
)

lazy val fatalWarningsSettings = Seq(
  scalacOptions ++= {
    CrossVersion.partialVersion(scalaVersion.value) match {
      case Some((2, n)) if n >= 13 =>
          if (sys.props.contains("disableFatalWarnings")) {
            Nil
          } else {
            "-Werror" :: Nil
          }

      case _                       => Nil
    }
  }
)

lazy val warningSuppression = Seq(
  scalacOptions += "-Wconf:" + Seq(
    "msg=APIs in chisel3.internal:s",
    "msg=Importing from firrtl:s",
    "msg=migration to the MLIR:s",
    "msg=method hasDefiniteSize in trait IterableOnceOps is deprecated:s",  // replacement `knownSize` is not in 2.12
    "msg=object JavaConverters in package collection is deprecated:s",
    "msg=undefined in comment for method cf in class PrintableHelper:s"
  ).mkString(",")
)

lazy val publishSettings = Seq(
  versionScheme := Some("pvp"),
  publishMavenStyle := true,
  Test / publishArtifact := false,
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
      Some("snapshots".at(nexus + "content/repositories/snapshots"))
    } else {
      Some("releases".at(nexus + "service/local/staging/deploy/maven2"))
    }
  }
)

// FIRRTL SETTINGS

lazy val isAtLeastScala213 = Def.setting {
  import Ordering.Implicits._
  CrossVersion.partialVersion(scalaVersion.value).exists(_ >= (2, 13))
}

lazy val firrtlSettings = Seq(
  name := "firrtl",
  version := "1.6-SNAPSHOT",
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
    "org.scalatestplus" %% "scalacheck-1-15" % "3.2.11.0" % "test",
    "com.github.scopt" %% "scopt" % "3.7.1",
    "net.jcazevedo" %% "moultingyaml" % "0.4.2",
    "org.json4s" %% "json4s-native" % "4.0.6",
    "org.apache.commons" % "commons-text" % "1.10.0",
    "io.github.alexarchambault" %% "data-class" % "0.2.5",
    "com.lihaoyi" %% "os-lib" % "0.8.1"
  ),
  // macros for the data-class library
  libraryDependencies ++= {
    if (isAtLeastScala213.value) Nil
    else Seq(compilerPlugin(("org.scalamacros" % "paradise" % "2.1.1").cross(CrossVersion.full)))
  },
  scalacOptions ++= {
    if (isAtLeastScala213.value) Seq("-Ymacro-annotations")
    else Nil
  },
  // starting with scala 2.13 the parallel collections are separate from the standard library
  libraryDependencies ++= {
    CrossVersion.partialVersion(scalaVersion.value) match {
      case Some((2, major)) if major <= 12 => Seq()
      case _                               => Seq("org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4")
    }
  },
  resolvers ++= Seq(
    Resolver.sonatypeRepo("snapshots"),
    Resolver.sonatypeRepo("releases")
  )
)

lazy val mimaSettings = Seq(
  mimaPreviousArtifacts := Set()
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
  .settings(publishSettings)
  .enablePlugins(BuildInfoPlugin)
  .settings(
    buildInfoPackage := name.value,
    buildInfoUsePackageAsPath := true,
    buildInfoKeys := Seq[BuildInfoKey](buildInfoPackage, version, scalaVersion, sbtVersion)
  )
  .settings(mimaSettings)
  .settings(warningSuppression: _*)
  .settings(fatalWarningsSettings: _*)

lazy val chiselSettings = Seq(
  name := "chisel3",
  libraryDependencies ++= Seq(
    "org.scalatest" %% "scalatest" % "3.2.15" % "test",
    "org.scalatestplus" %% "scalacheck-1-14" % "3.2.2.0" % "test",
    "com.lihaoyi" %% "upickle" % "2.0.0"
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
// The plugin only works in Scala 2.12+
lazy val pluginScalaVersions = Seq(
  // scalamacros paradise version used is not published for 2.12.0 and 2.12.1
  "2.12.2",
  "2.12.3",
  // 2.12.4 is broken in newer versions of Zinc: https://github.com/sbt/sbt/issues/6838
  "2.12.5",
  "2.12.6",
  "2.12.7",
  "2.12.8",
  "2.12.9",
  "2.12.10",
  "2.12.11",
  "2.12.12",
  "2.12.13",
  "2.12.14",
  "2.12.15",
  "2.12.16",
  "2.12.17",
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
  "2.13.10"
)

lazy val plugin = (project in file("plugin"))
  .settings(name := "chisel3-plugin")
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
    mimaPreviousArtifacts := {
      Set()
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
  .settings(name := "chisel3-macros")
  .settings(commonSettings: _*)
  .settings(publishSettings: _*)
  .settings(mimaPreviousArtifacts := Set())

lazy val core = (project in file("core"))
  .settings(commonSettings: _*)
  .enablePlugins(BuildInfoPlugin)
  .settings(
    buildInfoPackage := "chisel3",
    buildInfoUsePackageAsPath := true,
    buildInfoKeys := Seq[BuildInfoKey](buildInfoPackage, version, scalaVersion, sbtVersion)
  )
  .settings(publishSettings: _*)
  .settings(mimaPreviousArtifacts := Set())
  .settings(warningSuppression: _*)
  .settings(fatalWarningsSettings: _*)
  .settings(
    name := "chisel3-core",
    libraryDependencies ++= Seq(
      "com.lihaoyi" %% "upickle" % "2.0.0",
      "com.lihaoyi" %% "os-lib" % "0.8.1"
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
  .enablePlugins(ScalaUnidocPlugin)
  .settings(commonSettings: _*)
  .settings(chiselSettings: _*)
  .settings(publishSettings: _*)
  .settings(usePluginSettings: _*)
  .dependsOn(macros)
  .dependsOn(core)
  .dependsOn(firrtl)
  .aggregate(macros, core, plugin, firrtl)
  .settings(warningSuppression: _*)
  .settings(
    mimaPreviousArtifacts := Set(),
    Test / scalacOptions ++= Seq("-language:reflectiveCalls"),
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
        s"https://github.com/chipsalliance/chisel3/tree/$branch€{FILE_PATH_EXT}#L€{FILE_LINE}"
      },
      "-language:implicitConversions"
    ) ++
      // Suppress compiler plugin for source files in core
      // We don't need this in regular compile because we just don't add the chisel3-plugin to core's scalacOptions
      // This works around an issue where unidoc uses the exact same arguments for all source files.
      // This is probably fundamental to how ScalaDoc works so there may be no solution other than this workaround.
      // See https://github.com/sbt/sbt-unidoc/issues/107
      (core / Compile / sources).value.map("-P:chiselplugin:INTERNALskipFile:" + _)
      ++ {
        CrossVersion.partialVersion(scalaVersion.value) match {
          case Some((2, n)) if n >= 13 => "-implicits" :: Nil
          case _                       => Nil
        }
      }
  )

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
      "-Wconf:msg=firrtl:s"
    ),
    mdocIn := file("docs/src"),
    mdocOut := file("docs/generated"),
    // None of our links are hygienic because they're primarily used on the website with .html
    mdocExtraArguments := Seq("--cwd", "docs", "--no-link-hygiene"),
    mdocVariables := Map(
      "BUILD_DIR" -> "docs-target" // build dir for mdoc programs to dump temp files
    )
  )
