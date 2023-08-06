import glob, os

dir_path = os.path.dirname(os.path.realpath(__file__))
for pyfile in glob.glob(os.path.join(dir_path, "**", "*.py")):
  # remove dir from pyfile
  config.excludes.add(os.path.basename(pyfile))
