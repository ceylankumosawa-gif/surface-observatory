"""Record one research command's wall/CPU/peak-RSS without external packages."""
import json
import resource
import subprocess
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv)<4 or sys.argv[2]!='--':
        raise SystemExit('Usage: research_timer.py NEW_RESOURCE_JSON -- COMMAND [ARGS]')
    destination=Path(sys.argv[1])
    if destination.exists():raise FileExistsError(destination)
    started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=subprocess.run(sys.argv[3:],check=False)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    report={'elapsed_seconds':time.monotonic()-started,'user_cpu_seconds':after.ru_utime-before.ru_utime,
        'system_cpu_seconds':after.ru_stime-before.ru_stime,'peak_rss_bytes':after.ru_maxrss*1024,
        'exit_code':result.returncode,'method':'resource.getrusage(RUSAGE_CHILDREN), Linux KiB peak RSS',
        'maximum_child_rss_not_sum_of_concurrent_processes':True}
    with destination.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
