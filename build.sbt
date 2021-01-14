import Dependencies._

ThisBuild / scalaVersion     := "2.12.12"
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "com.sifive"
ThisBuild / organizationName := "SiFive"

lazy val root = (project in file("."))
  .settings(
    name := "chisel-circt",
    libraryDependencies += scalaTest % Test,
    libraryDependencies += chisel3
  )

// See https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html for instructions on how to publish to Sonatype.
