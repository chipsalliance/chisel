organization := "edu.berkeley.cs"

version := "3.0"

name := "Chisel"

scalaVersion := "2.11.6"

libraryDependencies ++= Seq("org.scala-lang" % "scala-reflect" % scalaVersion.value,
                            "org.scalatest" % "scalatest_2.11" % "2.2.4" % "test",
                            "org.scalacheck" %% "scalacheck" % "1.12.4" % "test")

site.settings

site.includeScaladoc()

ghpages.settings

git.remoteRepo := "git@github.com:ucb-bar/chisel3.git"
