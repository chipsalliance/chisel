# SPDX-License-Identifier: Apache-2.0

"""
Utilities for running a subprocess and collecting runtime and memory use information.
"""

from typing import NamedTuple, List
from enum import Enum
import sys
import subprocess
import re

# All times in seconds, sizes in kibibytes (KiB)
JobResourceUse = NamedTuple('JobResourceUse', [('user_time', float),
                                               ('system_time', float),
                                               ('wall_clock_time', float),
                                               ('maxrss', int)])


class JobFailedError(Exception):
    pass


def monitor_job(args: List[str], timeout=None) -> JobResourceUse:
    """Run a job with resource monitoring, returns resource usage"""
    platform = get_platform()
    cmd = time(platform) + args
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                            timeout=timeout)
    if result.returncode != 0 :
        msg = "[stdout]\n{}\n[stderr]\n{}".format(result.stdout.decode('utf-8'),
                                                  result.stderr.decode('utf-8'))
        raise JobFailedError(msg)
    stderr = result.stderr.decode('utf-8')

    user_time =       extract_user_time(platform, stderr)
    system_time =     extract_system_time(platform, stderr)
    wall_clock_time = extract_wall_clock_time(platform, stderr)
    maxrss =          extract_maxrss(platform, stderr)
    return JobResourceUse(user_time, system_time, wall_clock_time, maxrss)


Platform = Enum('Platform', 'linux macos')


def get_platform() -> Platform:
    p = sys.platform
    if p == 'darwin':
        return Platform.macos
    elif p.startswith('linux'):
        return Platform.linux
    else:
        raise Exception('Unsupported platform: ' + p)


def time(platform):
    return { Platform.macos: ['/usr/bin/time', '-l'],
             Platform.linux: ['/usr/bin/time', '-v'] }[platform]


def extract_maxrss(platform, output):
    """Returns maxrss in kbytes"""
    regex = {
        Platform.macos: '(\d+)\s+maximum resident set size',
        Platform.linux: 'Maximum resident set size[^:]*:\s+(\d+)'
    }[platform]

    m = re.search(regex, output, re.MULTILINE)
    if m :
        return {
            # /usr/bin/time -l on MacOS returns size in Bytes
            Platform.macos: int(m.group(1)) // 1024,
            # /usr/bin/time -v on Linux returns size in kibibytes
            # https://stackoverflow.com/questions/61392725/kilobytes-or-kibibytes-in-gnu-time
            Platform.linux: int(m.group(1))
        }[platform]
    else :
        raise Exception('Max set size not found!')


def extract_user_time(platform, output):
    res = None
    regex = {
        Platform.macos: '(\d+\.\d+)\s+user',
        Platform.linux: 'User time \(seconds\): (\d+\.\d+)'
    }[platform]
    m = re.search(regex, output, re.MULTILINE)
    if m :
        return float(m.group(1))
    raise Exception('User time not found!')


def extract_system_time(platform, output):
    res = None
    regex = {
        Platform.macos: '(\d+\.\d+)\s+sys',
        Platform.linux: 'System time \(seconds\): (\d+\.\d+)'
    }[platform]
    m = re.search(regex, output, re.MULTILINE)
    if m :
        return float(m.group(1))
    raise Exception('System time not found!')


def extract_wall_clock_time(platform, output):
    res = None
    regex = {
        Platform.macos: '(\d+\.\d+)\s+real',
        Platform.linux: 'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)'
    }[platform]
    m = re.search(regex, output, re.MULTILINE)
    if m :
        text = m.group(1)
        if platform == Platform.macos:
            return float(text)
        if platform == Platform.linux:
            parts = text.split(':')
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[0])
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
    raise Exception('Wall clock time not found!')

