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

val defaultVersions = Map("firrtl" -> "1.2-SNAPSHOT")

lazy val commonSettings = Seq (
  resolvers ++= Seq(
    Resolver.sonatypeRepo("snapshots"),
    Resolver.sonatypeRepo("releases")
  ),
  organization := "edu.berkeley.cs",
  version := "3.2-SNAPSHOT",
  autoAPIMappings := true,
  scalaVersion := "2.12.6",
  crossScalaVersions := Seq("2.12.6", "2.11.12"),
  scalacOptions := Seq("-deprecation", "-feature") ++ scalacOptionsVersion(scalaVersion.value),
  libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value,
  addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full),
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

lazy val publishSettings = Seq (
  publishMavenStyle := true,
  publishArtifact in Test := false,
  pomIncludeRepository := { x => false },
  // Don't add 'scm' elements if we have a git.remoteRepo definition,
  //  but since we don't (with the removal of ghpages), add them in below.
  pomExtra := <url>http://chisel.eecs.berkeley.edu/</url>
    <licenses>
      <license>
        <name>BSD-style</name>
        <url>http://www.opensource.org/licenses/bsd-license.php</url>
        <distribution>repo</distribution>
      </license>
    </licenses>
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
  }
)

lazy val chiselSettings = Seq (
  name := "chisel3",

// sbt 1.2.6 fails with `Symbol 'term org.junit' is missing from the classpath`
// when compiling tests under 2.11.12
// An explicit dependency on junit seems to alleviate this.
  libraryDependencies ++= Seq(
    "junit" % "junit" % "4.12" % "test",
    "org.scalatest" %% "scalatest" % "3.0.5" % "test",
    "org.scalacheck" %% "scalacheck" % "1.14.0" % "test",
    "com.github.scopt" %% "scopt" % "3.7.0"
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

lazy val coreMacros = (project in file("coreMacros")).
  settings(commonSettings: _*).
  // Prevent separate JARs from being generated for coreMacros.
  settings(skip in publish := true)

lazy val chiselFrontend = (project in file("chiselFrontend")).
  settings(commonSettings: _*).
  // Prevent separate JARs from being generated for chiselFrontend.
  settings(skip in publish := true).
  settings(
    scalacOptions := scalacOptions.value ++ Seq(
      "-deprecation",
      "-explaintypes",
      "-feature",
      "-language:reflectiveCalls",
      "-unchecked",
      "-Xcheckinit",
      "-Xlint:infer-any"
//      "-Xlint:missing-interpolator"
    )
  ).
  dependsOn(coreMacros)

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
  dependsOn(coreMacros % "compile-internal;test-internal").
  dependsOn(chiselFrontend % "compile-internal;test-internal").
  // We used to have to disable aggregation in general in order to suppress
  //  creation of subproject JARs (coreMacros and chiselFrontend) during publishing.
  // This had the unfortunate side-effect of suppressing coverage tests and scaladoc generation in subprojects.
  // The "skip in publish := true" setting in subproject settings seems to be
  //   sufficient to suppress subproject JAR creation, so we can restore
  //   general aggregation, and thus get coverage tests and scaladoc for subprojects.
  aggregate(coreMacros, chiselFrontend).
  settings(
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
    ),
    // Include macro classes, resources, and sources main JAR since we don't create subproject JARs.
    mappings in (Compile, packageBin) ++= (mappings in (coreMacros, Compile, packageBin)).value,
    mappings in (Compile, packageSrc) ++= (mappings in (coreMacros, Compile, packageSrc)).value,
    mappings in (Compile, packageBin) ++= (mappings in (chiselFrontend, Compile, packageBin)).value,
    mappings in (Compile, packageSrc) ++= (mappings in (chiselFrontend, Compile, packageSrc)).value,
    // Export the packaged JAR so projects that depend directly on Chisel project (rather than the
    // published artifact) also see the stuff in coreMacros and chiselFrontend.
    exportJars := true
  )
