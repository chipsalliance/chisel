#ifndef CIRCT_SUPPORT_VERSION_H
#define CIRCT_SUPPORT_VERSION_H

#include <string>

namespace circt {
const char *getCirctVersion();
const char *getCirctVersionComment();

/// A generic bug report message for CIRCT-related projects
constexpr const char *circtBugReportMsg =
    "PLEASE submit a bug report to https://github.com/llvm/circt and include "
    "the crash backtrace.\n";
} // namespace circt

#endif // CIRCT_SUPPORT_VERSION_H
