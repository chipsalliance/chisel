import os
import platform
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = 'CHISEL'
config.test_format = lit.formats.ShTest()
if 'CHISEL_FIRTOOL_PATH' in os.environ:
    config.environment['CHISEL_FIRTOOL_PATH'] = os.environ['CHISEL_FIRTOOL_PATH']
config.suffixes = [".sc"]
config.substitutions = [
    ('%SCALAVERSION', config.scala_version),
    ('%RUNCLASSPATH', ':'.join(config.run_classpath)),
    ('%SCALAPLUGINJARS', ':'.join(config.scala_plugin_jars)),
    ('%JAVAHOME', config.java_home),
    ('%JAVALIBRARYPATH', ':'.join(config.java_library_path))
]
config.test_source_root = os.path.dirname(__file__)
