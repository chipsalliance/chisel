// See LICENSE for license details.

// sbt-site - sbt-ghpages

site.settings

site.includeScaladoc()

ghpages.settings

git.remoteRepo := "git@github.com:ucb-bar/firrtl.git"

// Firrtl code

organization := "edu.berkeley.cs"

name := "firrtl"

version := "1.1-SNAPSHOT"

scalaVersion := "2.11.7"

libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value

libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.1.0"

libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.1.2"

libraryDependencies += "org.scalatest" % "scalatest_2.11" % "2.2.6" % "test"

libraryDependencies += "org.scalacheck" %% "scalacheck" % "1.12.5" % "test"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.4.0"

libraryDependencies += "net.jcazevedo" %% "moultingyaml" % "0.2"

// Assembly

assemblyJarName in assembly := "firrtl.jar"

test in assembly := {} // Should there be tests?

assemblyOutputPath in assembly := file("./utils/bin/firrtl.jar")

// ANTLRv4

antlr4Settings

antlr4GenVisitor in Antlr4 := true // default = false

antlr4GenListener in Antlr4 := false // default = true

antlr4PackageName in Antlr4 := Option("firrtl.antlr")

// ScalaDoc

import UnidocKeys._

lazy val customUnidocSettings = unidocSettings ++ Seq (
  doc in Compile := (doc in ScalaUnidoc).value,
  target in unidoc in ScalaUnidoc := crossTarget.value / "api"
)

autoAPIMappings := true

scalacOptions in Compile in doc ++= Seq(
  "-diagrams",
  "-diagrams-max-classes", "25",
  "-doc-version", version.value,
  "-doc-title", name.value,
  "-doc-root-content", baseDirectory.value+"/root-doc.txt"
)
