// SPDX-License-Identifier: Apache-2.0

package chisel3.testing

import firrtl.options.StageUtils.dramaticMessage
import java.io.{ByteArrayOutputStream, IOException, PrintWriter}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.Console.{withErr, withOut}
import scala.io.Source
import scala.util.control.NoStackTrace
import scala.sys.process._

object FileCheck {

  object Exceptions {

    /** Indicates that `FileCheck` was not found. */
    class NotFound private[FileCheck] (message: String)
        extends RuntimeException(
          dramaticMessage(
            header = Some("FileCheck was not found! Did you forget to install it?"),
            body = message
          )
        )
        with NoStackTrace

    /** Indicates that `FileCheck` failed. */
    class NonZeroExitCode private[FileCheck] (binary: String, exitCode: Int, message: String)
        extends RuntimeException(
          dramaticMessage(
            header = Some(s"FileCheck returned a non-zero exit code."),
            body = s"""|Exit Code: $exitCode
                       |
                       |Command:
                       |  $binary
                       |
                       |Command output:
                       |---
                       |$message
                       |---""".stripMargin
          )
        )
        with NoStackTrace

  }

}

trait FileCheck {

  /** Helpers to run `FileCheck` on a string input. */
  implicit class StringHelpers(input: String) {

    /** Run `FileCheck` on a string with some options.
      *
      * {{{
      * import chisel3.testing.FileCheck
      * import chisel3.testing.scalatest.TestingDirectory
      * import org.scalatest.flatspec.AnyFlatSpec
      * import org.scalatest.matchers.should.Matchers
      *
      * class Foo extends AnyFlatSpec with Matchers with FileCheck with TestingDirectory {
      *
      *   behavior of ("Bar")
      *
      *   it should "work" in {
      *     "Hello world!".fileCheck()(
      *       """|CHECK:      Hello
      *          |CHECK-SAME: world
      *          |""".stripMargin
      *     )
      *   }
      *
      * }
      *
      * }}}
      *
      * @param fileCheckArgs arguments to pass directly to FileCheck
      * @param check a string of checks to pass to `FileCheck`
      * @param testingDirectory an implementation of [[HasTestingDirectory]]
      * that controls where intermediary files are written.
      *
      * @note See [FileCheck
      * Documentation](https://llvm.org/docs/CommandGuide/FileCheck.html) for
      * more information.
      */
    def fileCheck(fileCheckArgs: String*)(
      check: String
    )(implicit testingDirectory: HasTestingDirectory): Unit = {
      // Filecheck needs to have the thing to check in a file.
      //
      // TODO: This could be made ephemeral or use a named pipe?
      val dir = testingDirectory.getDirectory.toAbsolutePath.normalize()
      Files.createDirectories(dir)
      val tempDir = Files.createTempDirectory(dir, "")
      val checkFile = tempDir.resolve("check")
      val inputFile = tempDir.resolve("input")
      Files.write(checkFile, check.getBytes(StandardCharsets.UTF_8))
      Files.write(inputFile, input.getBytes(StandardCharsets.UTF_8))

      val command = Seq("FileCheck", checkFile.toString) ++ fileCheckArgs
      val stdoutStream, stderrStream = new java.io.ByteArrayOutputStream
      val stdoutWriter = new PrintWriter(stdoutStream)
      val stderrWriter = new PrintWriter(stderrStream)
      val processIO = new ProcessIO(
        // Feed the input file to FileCheck's stdin.
        in = { stdin =>
          try Files.copy(inputFile, stdin)
          finally stdin.close()
        },
        out = { stdout =>
          try Source.fromInputStream(stdout).getLines().foreach(stdoutWriter.println)
          finally stdout.close()
        },
        err = { stderr =>
          try Source.fromInputStream(stderr).getLines().foreach(stderrWriter.println)
          finally stderr.close()
        }
      )
      val exitCode =
        try {
          Process(command).run(processIO).exitValue()
        } catch {
          case a: IOException if a.getMessage.startsWith("Cannot run program") =>
            throw new FileCheck.Exceptions.NotFound(a.getMessage)
        }
      stdoutWriter.close()
      stderrWriter.close()

      if (exitCode == 0) {
        deleteRecursively(tempDir)
      } else {
        throw new FileCheck.Exceptions.NonZeroExitCode(
          s"cat $inputFile | ${command.mkString(" ")}",
          exitCode,
          stderrStream.toString
        )
      }
    }

    /** Recursively delete a file or directory and its contents. */
    private def deleteRecursively(path: Path): Unit = {
      if (Files.isDirectory(path)) {
        val stream = Files.newDirectoryStream(path)
        try {
          val it = stream.iterator()
          while (it.hasNext) deleteRecursively(it.next())
        } finally stream.close()
      }
      Files.deleteIfExists(path)
    }

  }

}
