// SPDX-License-Identifier: Apache-2.0

package circt.stage.phases

import _root_.logger.{Logger => FLogger}
import scribe.{Logger => SLogger, LogRecord, Level}
import scribe.handler.LogHandler

/** Wrap our logger up as a scribe logger
  *
  * This is private and located here instead of in package logger because we should probably just
  * replace our logger with scribe. We probably shouldn't commit to a public interop API.
  *
  * @note this interop is extremely limited, scribe formatting features are ignored
  */
private[phases] object loggerToScribe {
  def apply(logger: FLogger): SLogger = {
    val handler = new LogHandler {
      def log(record: LogRecord): Unit = {
        val msg = record.logOutput.plainText
        val level = record.level
        if (level >= Level.Error) logger.error(msg)
        else if (level >= Level.Warn) logger.warn(msg)
        else if (level >= Level.Info) logger.info(msg)
        else if (level >= Level.Debug) logger.debug(msg)
        else logger.trace(msg)
      }
    }
    SLogger("Chisel")
      .withHandler(handler)
  }

}
