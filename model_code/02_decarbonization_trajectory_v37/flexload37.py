"""Post-hoc sensitivity S-flex (amendment_post_01): ERAA additional dispatchable consumption in the LP.

ERAA 2025 'Additional dispatchable consumption.csv' lists electrolysers and power-to-heat per market node and
target year with a maximum capacity and a weighted-average activation price. Each (zone, type) becomes one
consumption column per hour: 0 <= c <= MAX CAPACITY, entering the nodal balance with coefficient -1 (like
pumping or battery charging) and the stage-2 objective with cost -activation price, so it consumes only when the
nodal marginal cost is below its price (an ERAA-style demand bid). No energy or hydrogen-offtake limits, as in ERAA.
The frozen V36 model file is not modified: columns are appended to the built problem and matrices re-assembled.
"""
from pathlib import Path
import numpy as np, pandas as pd
from common37 import RAW

SRC = RAW / 'ERAA_2025_DemandData/Demand data/Additional dispatchable consumption.csv'


def units(year, zones):
    f = pd.read_csv(SRC); f = f[f.TARGET_YEAR.eq(year) & f.MARKET_NODE.isin(list(zones)) & f['MAX CAPACITY (MW)'].gt(0)]
    zl = list(zones)
    return dict(zone=np.array([zl.index(z) for z in f.MARKET_NODE], dtype=int), type=f.PEMMDB_PLANT_TYPE.to_numpy(dtype='U20'),
                cap=f['MAX CAPACITY (MW)'].to_numpy(float), price=f['Weighted Average Activation Price'].to_numpy(float),
                eff=f['Weighted Average Electrolyser Efficiency'].to_numpy(float) / 100.)


def patch(model, year):
    """Return a build_problem replacement that appends the dispatchable-consumption block."""
    original = model.build_problem

    def build(d, *a, **kw):
        p = original(d, *a, **kw); b = p['builder']; T, Z = d['demand'].shape
        u = units(year, d['zones']); K = len(u['cap'])
        n_before = b.n
        ids = b.block('flex_consumption', (T, K), upper=np.broadcast_to(u['cap'], (T, K)), cost=np.broadcast_to(-u['price'], (T, K)))
        rows = (np.arange(T)[:, None] * Z + u['zone'][None, :] + p['balance_start']).ravel()
        b.er.append(rows); b.ec.append(ids.ravel()); b.ev.append(np.full(ids.size, -1.))
        eq, rhs, ub, budget = b.matrix()
        # Keep the original bounds (they may carry post-assembly edits such as boost-hour masks) and append the block.
        bounds = np.vstack([p['bounds'], np.column_stack([b.lo[-1], b.hi[-1]])]); cost = np.concatenate([p['cost'], b.cost[-1]])
        # Identity checks: original rows, columns and right-hand sides are unchanged.
        assert eq.shape == (p['eq'].shape[0], b.n) and (eq[:, :n_before] - p['eq']).nnz == 0 and np.array_equal(rhs, p['rhs'])
        assert eq[:, n_before:].nnz == ids.size
        if p['ub'] is not None: assert (ub[:, :n_before] - p['ub']).nnz == 0 and ub[:, n_before:].nnz == 0 and np.array_equal(budget, p['budget'])
        p.update(eq=eq, rhs=rhs, ub=ub, budget=budget, bounds=bounds, cost=cost, flex_units=u)
        return p
    model.build_problem = build
    return original
