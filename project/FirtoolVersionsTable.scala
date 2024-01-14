// SPDX-License-Identifier: Apache-2.0

import sbt.util.Logger

import Version.SemanticVersion

object FirtoolVersionsTable extends App {
  // Some versions predate the BuildInfo.firtoolVersion API and thus have to be hardcoded
  def knownVersions: Seq[(SemanticVersion, String)] =
    Seq(
      "v5.0.0-RC1" -> "1.38.0",
      "v5.0.0-M2" -> "1.36.0",
      "v5.0.0-M1" -> "1.31.0",
      "v3.6.0" -> "1.37.0",
      "v3.6.0-RC3" -> "1.37.0",
      "v3.6.0-RC2" -> "1.31.0",
      "v3.6.0-RC1" -> "1.30.0"
    ).map { case (cv, fv) => SemanticVersion.parse(cv) -> fv }

  // Minimum version to record in the table
  // Earlier versions don't use firtool
  // This needs to be a def to avoid null pointer exception
  def min = SemanticVersion.parse("v3.6.0-RC1")

  def lookupFirtoolVersion(chiselVersion: SemanticVersion): String = {
    val version = chiselVersion.serialize
    // echo "println(chisel3.BuildInfo.firtoolVersion.get)" | scala-cli -S 2.13 --dep org.chipsalliance::chisel:6.0.0-RC1 -
    val cmd = "println(chisel3.BuildInfo.firtoolVersion.get)"
    // --server=false makes it a little slower but avoids hangs in CI
    val proc = os
      .proc("scala-cli", "--server=false", "-S", "2.13", "--dep", s"org.chipsalliance::chisel:$version", "-")
      .call(stdin = cmd, stdout = os.Pipe, stderr = os.Pipe)
    proc.out.trim
  }

  def firtoolGithubLink(version: String): String = s"https://github.com/llvm/circt/releases/tag/firtool-$version"

  def generateTable(logger: Logger): String = {
    val releases = Releases.releases(logger)

    val filtered = releases.filter(_ >= min)
    val unknown = {
      val isKnownVersion = knownVersions.map(_._1).toSet
      filtered.filterNot(isKnownVersion)
    }

    val lookedUp = unknown.map(v => v -> lookupFirtoolVersion(v))
    val allVersions = (lookedUp ++ knownVersions).sortBy(_._1).reverse // descending

    val header = Vector("| Chisel Version | Firtool Version |", "| --- | --- |")
    val table = (header ++ allVersions.map {
      case (sv, fv) => s"| ${sv.serialize} | [$fv](${firtoolGithubLink(fv)}) |"
    }).mkString("\n")
    table
  }
}
