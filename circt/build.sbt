import Dependencies._

ThisBuild / scalaVersion         := "2.12.15"
ThisBuild / crossScalaVersions   := Seq("2.13.6", scalaVersion.value)
ThisBuild / organization         := "com.sifive"
ThisBuild / organizationName     := "SiFive"
ThisBuild / organizationHomepage := Some(url("https://www.sifive.com/"))
ThisBuild / description          := "Infrastructure to compile Chisel projects using MLIR-based infrastructure (CIRCT)"
ThisBuild / homepage             := Some(url("https://github.com/sifive/chisel-circt"))
ThisBuild / licenses             := List("Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0"))
ThisBuild / developers           := List(
  Developer(
    "seldridge",
    "Schuyler Eldridge",
    "schuyler.eldridge@gmail.com",
    url("https://www.seldridge.dev")
  )
)

lazy val root = (project in file("."))
  .settings(
    name := "chisel-circt",
    libraryDependencies += scalaTest % Test,
    libraryDependencies += chisel3,
    addCompilerPlugin(chiselCompilerPlugin cross CrossVersion.full)
  )
