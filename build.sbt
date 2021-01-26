import Dependencies._

ThisBuild / scalaVersion     := "2.12.12"
ThisBuild / organization     := "com.sifive"
ThisBuild / organizationName := "SiFive"
ThisBuild / homepage         := Some(url("https://github.com/sifive/chisel-circt"))
ThisBuild / licenses         := List("Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0"))
ThisBuild / developers       := List(
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
    libraryDependencies += chisel3
  )
