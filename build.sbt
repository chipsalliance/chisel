// SPDX-License-Identifier: Apache-2.0

enablePlugins(SiteScaladocPlugin)

lazy val commonSettings = Seq(
  organization := "edu.berkeley.cs",
  scalaVersion := "2.12.13",
  crossScalaVersions := Seq("2.13.4", "2.12.13")
)

lazy val isAtLeastScala213 = Def.setting {
  import Ordering.Implicits._
  CrossVersion.partialVersion(scalaVersion.value).exists(_ >= (2, 13))
}

lazy val firrtlSettings = Seq(
  name := "firrtl",
  version := "1.5-SNAPSHOT",
  addCompilerPlugin(scalafixSemanticdb),
  scalacOptions := Seq(
    "-deprecation",
    "-unchecked",
    "-language:reflectiveCalls",
    "-language:existentials",
    "-language:implicitConversions",
    "-Yrangepos",          // required by SemanticDB compiler plugin
  ),
  // Always target Java8 for maximum compatibility
  javacOptions ++= Seq("-source", "1.8", "-target", "1.8"),
  libraryDependencies ++= Seq(
    "org.scala-lang" % "scala-reflect" % scalaVersion.value,
    "org.scalatest" %% "scalatest" % "3.2.8" % "test",
    "org.scalatestplus" %% "scalacheck-1-14" % "3.1.3.0" % "test",
    "com.github.scopt" %% "scopt" % "3.7.1",
    "net.jcazevedo" %% "moultingyaml" % "0.4.2",
    "org.json4s" %% "json4s-native" % "3.6.11",
    "org.apache.commons" % "commons-text" % "1.8",
    "io.github.alexarchambault" %% "data-class" % "0.2.5",
  ),
  // macros for the data-class library
  libraryDependencies ++= {
    if (isAtLeastScala213.value) Nil
    else Seq(compilerPlugin("org.scalamacros" % "paradise" % "2.1.1" cross CrossVersion.full))
  },
  scalacOptions ++= {
    if (isAtLeastScala213.value) Seq("-Ymacro-annotations")
    else Nil
  },
  // starting with scala 2.13 the parallel collections are separate from the standard library
  libraryDependencies ++= {
    CrossVersion.partialVersion(scalaVersion.value) match {
      case Some((2, major)) if major <= 12 => Seq()
      case _ => Seq("org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.2")
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

lazy val protobufSettings = Seq(
  sourceDirectory in ProtobufConfig := baseDirectory.value / "src" / "main" / "proto",
  protobufRunProtoc in ProtobufConfig := (args =>
    com.github.os72.protocjar.Protoc.runProtoc("-v351" +: args.toArray)
  )
)

lazy val assemblySettings = Seq(
  assemblyJarName in assembly := "firrtl.jar",
  test in assembly := {},
  assemblyOutputPath in assembly := file("./utils/bin/firrtl.jar")
)


lazy val testAssemblySettings = Seq(
  test in (Test, assembly) := {}, // Ditto above
  assemblyMergeStrategy in (Test, assembly) := {
    case PathList("firrtlTests", xs @ _*) => MergeStrategy.discard
    case x =>
      val oldStrategy = (assemblyMergeStrategy in (Test, assembly)).value
      oldStrategy(x)
  },
  assemblyJarName in (Test, assembly) := s"firrtl-test.jar",
  assemblyOutputPath in (Test, assembly) := file("./utils/bin/" + (Test / assembly / assemblyJarName).value)
)

lazy val antlrSettings = Seq(
  antlr4GenVisitor in Antlr4 := true,
  antlr4GenListener in Antlr4 := false,
  antlr4PackageName in Antlr4 := Option("firrtl.antlr"),
  antlr4Version in Antlr4 := "4.9.2",
  javaSource in Antlr4 := (sourceManaged in Compile).value
)

lazy val publishSettings = Seq(
  publishMavenStyle := true,
  publishArtifact in Test := false,
  pomIncludeRepository := { x => false },
  // scm is set by sbt-ci-release
  pomExtra := <url>http://chisel.eecs.berkeley.edu/</url>
    <licenses>
      <license>
        <name>apache_v2</name>
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
    } else {
      Some("releases" at nexus + "service/local/staging/deploy/maven2")
    }
  }
)


lazy val docSettings = Seq(
  doc in Compile := (doc in ScalaUnidoc).value,
  autoAPIMappings := true,
  scalacOptions in Compile in doc ++= Seq(
    "-Xfatal-warnings",
    "-feature",
    "-diagrams",
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
      s"https://github.com/chipsalliance/firrtl/tree/$branch€{FILE_PATH_EXT}#L€{FILE_LINE}"
    }
  )
)

lazy val firrtl = (project in file("."))
  .enablePlugins(ProtobufPlugin)
  .enablePlugins(ScalaUnidocPlugin)
  .enablePlugins(Antlr4Plugin)
  .settings(
    fork := true,
    Test / testForkedParallel := true
  )
  .settings(commonSettings)
  .settings(firrtlSettings)
  .settings(protobufSettings)
  .settings(antlrSettings)
  .settings(assemblySettings)
  .settings(inConfig(Test)(baseAssemblySettings))
  .settings(testAssemblySettings)
  .settings(publishSettings)
  .settings(docSettings)
  .enablePlugins(BuildInfoPlugin)
  .settings(
    buildInfoPackage := name.value,
    buildInfoUsePackageAsPath := true,
    buildInfoKeys := Seq[BuildInfoKey](buildInfoPackage, version, scalaVersion, sbtVersion)
  )
  .settings(mimaSettings)

lazy val benchmark = (project in file("benchmark"))
  .dependsOn(firrtl)
  .settings(commonSettings)
  .settings(
    assemblyJarName in assembly := "firrtl-benchmark.jar",
    test in assembly := {},
    assemblyOutputPath in assembly := file("./utils/bin/firrtl-benchmark.jar")
  )

val JQF_VERSION = "1.5"

lazy val jqf = (project in file("jqf"))
  .settings(commonSettings)
  .settings(
    libraryDependencies ++= Seq(
      "edu.berkeley.cs.jqf" % "jqf-fuzz" % JQF_VERSION,
      "edu.berkeley.cs.jqf" % "jqf-instrument" % JQF_VERSION,
      "com.github.scopt" %% "scopt" % "3.7.1",
    )
  )


lazy val jqfFuzz = sbt.inputKey[Unit]("input task that runs the firrtl.jqf.JQFFuzz main method")
lazy val jqfRepro = sbt.inputKey[Unit]("input task that runs the firrtl.jqf.JQFRepro main method")

lazy val testClassAndMethodParser = {
  import sbt.complete.DefaultParsers._
  val spaces = SpaceClass.+.string
  val testClassName = token(Space) ~> token(charClass(c => isScalaIDChar(c) || (c == '.')).+.string, "<test class name>")
  val testMethod = spaces ~> token(charClass(isScalaIDChar).+.string, "<test method name>")
  val rest = spaces.? ~> token(any.*.string, "<other args>")
  (testClassName ~ testMethod ~ rest).map {
    case ((a, b), c) => (a, b, c)
  }
}

lazy val fuzzer = (project in file("fuzzer"))
  .dependsOn(firrtl)
  .settings(commonSettings)
  .settings(
    libraryDependencies ++= Seq(
      "com.pholser" % "junit-quickcheck-core" % "0.8",
      "com.pholser" % "junit-quickcheck-generators" % "0.8",
      "edu.berkeley.cs.jqf" % "jqf-fuzz" % JQF_VERSION,
      "org.scalacheck" %% "scalacheck" % "1.14.3" % Test
    ),

    jqfFuzz := (Def.inputTaskDyn {
      val (testClassName, testMethod, otherArgs) = testClassAndMethodParser.parsed
      val outputDir = target.in(Compile).value / "JQF" / testClassName / testMethod
      val classpath = (Compile / fullClasspathAsJars).toTask.value.files.mkString(":")
      (jqf/runMain).in(Compile).toTask(
        s" firrtl.jqf.JQFFuzz " +
        s"--testClassName $testClassName " +
        s"--testMethod $testMethod " +
        s"--classpath $classpath " +
        s"--outputDirectory $outputDir " +
        otherArgs)
    }).evaluated,

    jqfRepro := (Def.inputTaskDyn {
      val (testClassName, testMethod, otherArgs) = testClassAndMethodParser.parsed
      val classpath = (Compile / fullClasspathAsJars).toTask.value.files.mkString(":")
      (jqf/runMain).in(Compile).toTask(
        s" firrtl.jqf.JQFRepro " +
        s"--testClassName $testClassName " +
        s"--testMethod $testMethod " +
        s"--classpath $classpath " +
        otherArgs)
    }).evaluated,
  )
