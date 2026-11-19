"""V37 paths, helpers and target-year calendar; earlier experiment folders are imported read-only."""
from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, hashlib, importlib.util
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
DS = ROOT / 'dataset_one'
RAW = DS / '07_eraa_preprocessed/01_extracted_raw'
DASH = RAW / 'ERAA_2025_Dashboard_RawData/ERAA_2025_DashboardRawData1/Dashboard Raw Data'
V2 = DS / '13_central_adequacy_v2'
V27 = DS / '39_public_validation_system_comparison_v27/system'
V32 = DS / '44_followup_corrected_system_v32'
V32B = V32 / 'stage_b'
V36 = DS / '49_use_value_v36'
PY = '/home/cxw/anaconda3/envs/pv/bin/python'
YEARS = [2028, 2030, 2033, 2035]
YI = {2028: 0, 2030: 1, 2033: 2, 2035: 3}
TRAIN = [23, 11, 9]                      # V32 A5 frozen training weather (gas role 'train')
POWER_V36 = 25.96676631231336            # V36 Central total boost power (MW)
V36_CENTRAL_TOTAL_MW = None              # filled lazily from the V36 resource table


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False, default=float) + '\n'); tmp.replace(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as f: return {k: f[k].copy() for k in f.files}


def save_arrays(path, values):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **values)
    tmp.replace(path)


def load_module(path, name):
    """Import a frozen script from an earlier (read-only) folder without writing bytecode there."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def v2_module():
    sys.path.insert(0, str(V2 / 'scripts'))
    import prepare_central_inputs_v2 as m
    return m


def calendar(year):
    """ERAA 8760-hour calendar without 29 February, identical to V2 calendar_for_year."""
    return v2_module().calendar_for_year(year)


def available_gib():
    return next(int(x.split()[1]) / 1024 ** 2 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))


def child_env():
    v = os.environ.copy()
    for k in list(v):
        if k.startswith(('V28_', 'V31_', 'V32_', 'V36_')): v.pop(k)
    v.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
             PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', V32_SOLVER_MEMORY_MB='12288', V36_STAGE_METHODS='1,2')
    return v


def gas_role(weather):
    return 'train' if weather in TRAIN else 'holdout'


def input_path(year, weather):
    if year == 2035: return V32B / 'inputs' / f'2035_WS{weather:02d}.npz'
    return HERE / 'inputs/system' / f'{year}_WS{weather:02d}.npz'


def thermal_path(year):
    if year == 2035: return V36 / 'inputs/thermal_split_2035.npz'
    return HERE / 'inputs/system' / f'thermal_split_{year}.npz'
