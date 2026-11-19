"""Reproduce a 168-hour Pinerolo dispatch pair with NumPy and SciPy."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack


def run(path):
    with np.load(path) as data:
        def matrix(k):
            return csr_matrix((data[k+"_data"],data[k+"_indices"],data[k+"_indptr"]),shape=data[k+"_shape"])
        eq,ub=matrix("eq"),matrix("ub")
        eens=np.zeros_like(data["cost"])
        eens[data["uns"]]=1
        sol={}
        for case in ("base","site"):
            rhs,budget,bounds=(data[case+"_"+k] for k in ("rhs","budget","bounds"))
            kwargs=dict(A_eq=eq,b_eq=rhs,bounds=bounds,method="highs-ds")
            s1=linprog(eens,A_ub=ub,b_ub=budget,**kwargs)
            if not s1.success:raise ValueError(s1.message)
            s2=linprog(data["cost"],A_ub=vstack([ub,csr_matrix(eens[None,:])]),
                       b_ub=np.r_[budget,s1.fun+1e-4],**kwargs)
            if not s2.success:raise ValueError(s2.message)
            assert np.max(np.abs(eq@s2.x-rhs))<2e-4 and np.max(ub@s2.x-budget)<2e-4
            sol[case]=s2.x
        change=sol["base"]-sol["site"]
        D=data["fuel"]@change/float(data["raw"])
        mu=data["emissions"]@change/float(data["electricity"])
        assert abs(D-data["expected_D"])<1e-5 and abs(mu-data["expected_mu"])<1e-5
        print(json.dumps({"case":path.stem,"hours":168,"D":D,"avoided_tCO2e_per_MWh_electricity":mu,"verified":True}))


if __name__=="__main__":
    for name in ("pinerolo","region"):
        run(Path(__file__).with_name(name+"_2035_ws17_april.npz"))
