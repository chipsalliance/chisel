organization := "edu.berkeley.cs"

version := "3.0"

name := "Chisel"

scalaVersion := "2.11.6"

crossScalaVersions := Seq("2.10.4", "2.11.6")

libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value

site.settings

site.includeScaladoc()

ghpages.settings

git.remoteRepo := "git@github.com:ucb-bar/chisel3.git"
