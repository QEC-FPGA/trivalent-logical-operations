"""Shared, read-only dependencies and explicit output location for this study."""
from pathlib import Path
import sys, json, hashlib, os
from functools import lru_cache
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parent
PACKAGE = SOURCE.parent
ROOT = Path(os.environ.get('MOVE_W_OUTPUT', str(PACKAGE/'output'))).resolve()
PREVIOUS = SOURCE
FROZEN = SOURCE
sys.path.insert(0, str(SOURCE))
import generate_memory as memory
memory.ROOT = ROOT
base = memory.p.old.base
compiler = memory.p.old.compiler

# Exact binary representation only: no circuit or syndrome convention changes.
# The original Python per-qubit sum dominated the large-d validation runtime.
_original_mask = memory.p.mask
def packed_mask(p):
    xs, zs = p.to_numpy(bit_packed=True)
    return int.from_bytes(xs.tobytes(), 'little') | (int.from_bytes(zs.tobytes(), 'little') << len(p))
memory.p.mask = packed_mask
# These functions return values which callers only read or append by copy.
# Reuse identical static geometries and native extraction circuits instead of
# rebuilding the d-round preparation for every position/channel comparison.
memory.p.ORIGINAL_INITIAL_CODE = lru_cache(maxsize=128)(memory.p.ORIGINAL_INITIAL_CODE)
memory.p.projected_code = lru_cache(maxsize=256)(memory.p.projected_code)
memory.extraction = lru_cache(maxsize=512)(memory.extraction)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def coordinate(q,d):
    x,y=divmod(q,4*d+5)
    return ('a',x,y) if y<2*d+3 else ('q',x,y-(2*d+3))
def qubit(c,d):
    kind,x,y=c;return (4*d+5)*x+y+(2*d+3 if kind=='q' else 0)
