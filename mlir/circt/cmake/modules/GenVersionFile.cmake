# This file borrows ideas from LLVM's GenerateVersionFromVCS.cmake and VersionFromVCS.cmake
# Input variables:
#   IN_FILE             - An absolute path of Version.cpp.in
#   OUT_FILE            - An absolute path of Version.cpp
#   RELEASE_PATTERN     - A pattern to search release tags
#   DRY_RUN             - If true, make the version unknown.
#   SOURCE_ROOT         - Path to root directory of source

set(GIT_DESCRIBE_DEFAULT "unknown git version")
if (DRY_RUN)
  set(GIT_DESCRIBE_OUTPUT "${GIT_DESCRIBE_DEFAULT}")
else ()
  message(STATUS "Generating ${OUT_FILE} from ${IN_FILE} by `git describe --dirty --tags --match ${RELEASE_PATTERN}`")
  find_package(Git QUIET)
  if (Git_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --dirty --tags --match ${RELEASE_PATTERN}
      WORKING_DIRECTORY "${SOURCE_ROOT}"
      RESULT_VARIABLE GIT_OUTPUT_RESULT
      OUTPUT_VARIABLE GIT_DESCRIBE_OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT ${GIT_OUTPUT_RESULT} EQUAL 0)
      message(WARNING "git describe failed, set version to ${GIT_DESCRIBE_DEFAULT}")
      set(GIT_DESCRIBE_OUTPUT "${GIT_DESCRIBE_DEFAULT}")
    endif ()
  else ()
    message(WARNING "Git not found: ${GIT_EXECUTABLE}, set version to ${GIT_DESCRIBE_DEFAULT}")
    set(GIT_DESCRIBE_OUTPUT "${GIT_DESCRIBE_DEFAULT}")
  endif ()
endif()

# This command will prepend CMAKE_CURRENT_{SOURCE,BINARY}_DIR if <input> or <output> is relative,
# that's why I need IN_FILE and OUT_FILE to be absolute path.
configure_file("${IN_FILE}" "${OUT_FILE}.tmp")

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OUT_FILE}.tmp" "${OUT_FILE}")
file(REMOVE "${OUT_FILE}.tmp")
