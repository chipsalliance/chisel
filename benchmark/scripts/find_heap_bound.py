#!/usr/bin/env python3
# See LICENSE for license details.

import re
import argparse
from typing import NamedTuple
from subprocess import TimeoutExpired
import logging

from monitor_job import monitor_job, JobFailedError

BaseHeapSize = NamedTuple('JavaHeapSize', [('value', int), ('suffix', str)])
class HeapSize(BaseHeapSize):
    K_FACTOR = 1024
    M_FACTOR = 1024*1024
    G_FACTOR = 1024*1024*1024

    def toBytes(self) -> int:
        return {
            "": 1,
            "K": self.K_FACTOR,
            "M": self.M_FACTOR,
            "G": self.G_FACTOR
        }[self.suffix] * self.value

    def round_to(self, target):
        """Round to positive multiple of target, only if we have the same suffix"""
        if self.suffix == target.suffix:
            me = self.toBytes()
            tar = target.toBytes()
            res = tar * round(me / tar)
            if res == 0:
                res = tar
            return HeapSize.from_bytes(res)
        else:
            return self

    def __truediv__(self, div):
        b = self.toBytes()
        res = int(b / div)
        return HeapSize.from_bytes(res)

    def __mul__(self, m):
        b = self.toBytes()
        res = int(b * m)
        return HeapSize.from_bytes(res)

    def __add__(self, rhs):
        return HeapSize.from_bytes(self.toBytes() + rhs.toBytes())

    def __sub__(self, rhs):
        return HeapSize.from_bytes(self.toBytes() - rhs.toBytes())

    @classmethod
    def from_str(cls, s: str):
        regex = '(\d+)([kKmMgG])?'
        m = re.match(regex, s)
        if m:
            suffix = m.group(2)
            if suffix is None:
                return HeapSize(int(m.group(1)), "")
            else:
                return HeapSize(int(m.group(1)), suffix.upper())
        else:
            msg = "Invalid Heap Size '{}'! Format should be: '{}'".format(s, regex)
            raise Exception(msg)

    @classmethod
    def from_bytes(cls, b: int):
        if b % cls.G_FACTOR == 0:
            return HeapSize(round(b / cls.G_FACTOR), "G")
        if b % cls.M_FACTOR == 0:
            return HeapSize(round(b / cls.M_FACTOR), "M")
        if b % cls.K_FACTOR == 0:
            return HeapSize(round(b / cls.K_FACTOR), "K")
        return HeapSize(round(b), "")


    def __str__(self):
        return str(self.value) + self.suffix


def parseargs():
    parser = argparse.ArgumentParser(
        prog="find_heap_bound.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", "-v", action="count", default=0,
                        help="Increase verbosity level (cumulative)")
    parser.add_argument("args", type=str, nargs="+",
                        help="Arguments to JVM, include classpath and main")
    parser.add_argument("--start-size", type=str, default="4G",
                        help="Starting heap size")
    parser.add_argument("--min-step", type=str, default="100M",
                        help="Minimum heap size step")
    parser.add_argument("--java", type=str, default="java",
                        help="Java executable to use")
    parser.add_argument("--timeout-factor", type=float, default=4.0,
                        help="Multiple of wallclock time of first successful run "
                             "that counts as a timeout, runs over this time count as a fail")
    return parser.parse_args()


def get_logger(args):
    logger = logging.getLogger("find_heap_bound")
    if args.verbose == 1:
        #logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    elif args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    return logger


def mk_cmd(java, heap, args):
    return [java, "-Xmx{}".format(heap)] + args


def job_failed_msg(e):
    if isinstance(e, JobFailedError):
        if "java.lang.OutOfMemoryError" in str(e):
            return "Job failed, out of memory"
        else:
            return "Unexpected job failure\n{}".format(e)
    elif isinstance(e, TimeoutExpired):
        return "Job timed out at {} seconds".format(e.timeout)
    else:
        raise e


def main():
    args = parseargs()
    logger = get_logger(args)

    results = []

    min_step = HeapSize.from_str(args.min_step)
    step = None
    seen = set()
    timeout = None # Set by first successful run
    cur = HeapSize.from_str(args.start_size)
    while cur not in seen:
        seen.add(cur)
        try:
            cmd = mk_cmd(args.java, cur, args.args)
            logger.info("Running {}".format(" ".join(cmd)))
            stats = monitor_job(cmd, timeout=timeout)
            logger.debug(stats)
            if timeout is None:
                timeout = stats.wall_clock_time * args.timeout_factor
                logger.debug("Timeout set to {} s".format(timeout))
            results.append((cur, stats))
            if step is None:
                step = (cur / 2).round_to(min_step)
            else:
                step = (step / 2).round_to(min_step)
            cur = (cur - step).round_to(min_step)
        except (JobFailedError, TimeoutExpired) as e:
            logger.debug(job_failed_msg(e))
            results.append((cur, None))
            if step is None:
                # Don't set step because we don't want to keep decreasing it
                # when we haven't had a passing run yet
                amt = (cur * 2).round_to(min_step)
            else:
                step = (step / 2).round_to(min_step)
                amt = step
            cur = (cur + step).round_to(min_step)
        logger.debug("Next = {}, step = {}".format(cur, step))

    sorted_results = sorted(results, key=lambda tup: tup[0].toBytes(), reverse=True)

    table = [["Xmx", "Max RSS (MiB)", "Wall Clock (s)", "User Time (s)", "System Time (s)"]]
    for heap, resources in sorted_results:
        line = [str(heap)]
        if resources is None:
            line.extend(["-"]*4)
        else:
            line.append(str(resources.maxrss // 1024))
            line.append(str(resources.wall_clock_time))
            line.append(str(resources.user_time))
            line.append(str(resources.system_time))
        table.append(line)

    csv = "\n".join([",".join(row) for row in table])
    print(csv)


if __name__ == "__main__":
    main()
