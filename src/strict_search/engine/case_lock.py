"""OS-released per-case locks for overlapping resumable batch runners."""
from contextlib import contextmanager
import os, time


@contextmanager
def locked(folder, name, timeout=600):
    folder.mkdir(parents=True, exist_ok=True)
    f=(folder/(name+'.lock')).open('a+b')
    f.seek(0,2)
    if f.tell()==0:f.write(b'0');f.flush()
    start=time.monotonic()
    if os.name=='nt':
        import msvcrt
        while True:
            try:
                f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:
                if time.monotonic()-start>timeout:f.close();raise TimeoutError(name)
                time.sleep(.5)
        try:yield
        finally:
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1);f.close()
    else:
        import fcntl
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN);f.close()
