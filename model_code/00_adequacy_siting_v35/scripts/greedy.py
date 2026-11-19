"""No-foresight boost rule for the review comparison with the annual LP.

Each zone reacts only to the current hour. In every hour with system shortfall
in the paired baseline it runs its boost engines at the largest output allowed
by power and gas above a protective floor. The floor is the smallest stock that
keeps the base commitment feasible for the rest of the year without boosting
and still reaches the registered 50% year-end stock. It depends only on the
digester production profile, not on future shortfalls. No stock is held back
for later shortfall hours.
"""
from __future__ import annotations

import numpy as np


def protective_floor(prod, base_fuel, S, q, kappa, eta):
    T = len(prod)
    cmax = min(2 * q, max(eta * q - base_fuel * eta, 0) / kappa) if kappa > 0 else 2 * q
    f = np.zeros(T)
    f[-1] = 0.5 * S
    for t in range(T - 1, 0, -1):
        surplus = prod[t] - base_fuel
        if surplus >= 0:
            gain = min(surplus / (1 + kappa / eta), cmax, 2 * q)
            f[t - 1] = max(0., f[t] - gain)
        else:
            f[t - 1] = f[t] - surplus
    return f, cmax


def zone_profile(prod, base_fuel, P, S, q, allowed, kappa=0.03, eta=0.4):
    T = len(prod)
    boost = np.zeros(T)
    inv_path = np.zeros(T)
    if q <= 0 or P <= 0:
        return boost, inv_path, dict(max_floor=0., min_slack=0.)
    f, cmax = protective_floor(prod, base_fuel, S, q, kappa, eta)
    if f.max() > S + 1e-6:
        raise ValueError('Base commitment infeasible for this buffer')
    inv = 0.5 * S
    min_slack = np.inf
    for t in range(T):
        surplus = prod[t] - base_fuel
        if allowed[t]:
            # Largest boost that still leaves at least f[t] in stock at the end of the hour;
            # refilling below the floor costs compression fuel (1 + kappa/eta per MWh stored).
            if inv >= f[t]:
                fuel = surplus + (inv - f[t])
            else:
                fuel = surplus - (f[t] - inv) * (1 + kappa / eta)
            boost[t] = min(P, max(0., eta * fuel))
        net = surplus - boost[t] / eta
        if net >= 0:
            inv += min(net / (1 + kappa / eta), cmax, 2 * q, S - inv)
        else:
            inv += net
        min_slack = min(min_slack, inv - f[t])
        inv_path[t] = inv
        if inv < f[t] - 1e-7 or inv > S + 1e-7:
            raise RuntimeError(f'Inventory rule violated at hour {t}')
    return boost, inv_path, dict(max_floor=float(f.max()), min_slack=float(min_slack),
                                 end_stock=float(inv), cmax=float(cmax))


def profile(d, P, S, allowed):
    """Return the (T, Z) boost profile and per-zone diagnostics."""
    eta, beta = float(d['eta_bio']), float(d['beta'])
    kappa = float(d.get('compression_kappa', 0.03))
    q = np.asarray(d['q'], float)
    prod = np.asarray(d['q_hourly'], float)
    T, Z = prod.shape
    out = np.zeros((T, Z))
    inv = np.zeros((T, Z))
    diag = []
    for z in range(Z):
        b, s, info = zone_profile(prod[:, z], beta * q[z], float(P[z]), float(S[z]), float(q[z]),
                                  allowed, kappa, eta)
        out[:, z] = b
        inv[:, z] = s
        diag.append(info)
    return out, inv, diag
