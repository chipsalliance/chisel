import platform
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = 'CHISEL'
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".sc"]
config.test_source_root = os.path.dirname(__file__)
