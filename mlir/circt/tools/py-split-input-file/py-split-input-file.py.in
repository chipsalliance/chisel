#!/usr/bin/env python3
import sys
"""
A simple python script to split an input file input multiple files based on
a split token, and execute each file separately, catching any exceptions
which might have occured.
"""

if len(sys.argv) != 2:
  print("Usage: ./py-split-input-file.py <input_file>")
  sys.exit(1)
input_file = sys.argv[1]

# Read input file into separate strings, splitting at # -----
splits = []
with open(input_file, "r") as f:
  current_split = []

  def push_current_split():
    splits.append("\n".join(current_split))
    current_split.clear()

  for line in f:
    if line.startswith("# -----"):
      push_current_split()
    else:
      current_split.append(line)
  push_current_split()

# Execute each split.
for split in splits:
  try:
    exec(split)
  except Exception as e:
    ex_name = type(e).__name__
    print(f"{ex_name}: {e}")
