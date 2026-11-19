"""Summarize the archived regional responses and the new paired dispatch experiments."""
from itertools import permutations, product
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
OUT=HERE/"source_data"
OLD=OUT/"reference"
K=.98*.96
params=json.loads((OLD/"nonpower_parameters.json").read_text())
g=params["GWP100"]
v=params["gas_upstream_inventory_g_per_MJ"]
G=params["gas_combustion_t_per_MWh"]+.0036*(v["CO2"]+v["CH4"]*g["fossil_CH4"]+v["N2O"]*g["N2O"])
M=params["methane_mass_t_per_MWh"]*g["bio_CH4"]
HEAT=.488*G/.9
DIESEL=params["diesel_combustion_t_per_MWh"]+params["diesel_upstream_t_per_MWh"]
SITE=pd.read_csv(OUT/"facility_inputs.csv").set_index("key").value
YEARS=[2028,2030,2033,2035]


def save(df,name):
    df.to_csv(OUT/(name+".csv"),index=False)
    return df


def annual():
    b=pd.read_csv(OLD/"boundary_base_by_weather.csv")
    a=b[["year","weather","B_MWh","D","avoided_domestic","MEF_domestic",
         "extra_H2_reported_MWh_per_MWh","avoided_external_import_MWhe_per_MWh"]].copy()
    a=a.rename(columns={"B_MWh":"archive_input_MWh","D":"D_per_simulated_raw",
                        "avoided_domestic":"a","MEF_domestic":"mu"})
    a["physically_available_raw_MWh"]=a.archive_input_MWh/K
    a["D_per_available_raw_fixed_numerator"]=K*a.D_per_simulated_raw
    a["site_net"]=a.a-.025*M
    a["grid_net"]=K*(G-.03*a.mu)-.025*M
    a["h_zero"]=-a.site_net/HEAT
    a["h_match"]=(a.grid_net-a.site_net)/HEAT
    a["q_zero_heat"]=(a.a+K*.03*a.mu)/(K*G)
    assert len(a)==144 and not a.duplicated(["year","weather"]).any()
    save(a,"annual_equal_raw")
    margins=[]
    for ls,lg,bound,compression,h in product((.01,.025,.042),(.01,.025,.042),
                                            ("domestic","extended_LHV","extended_HHV"),
                                            ("domestic","symmetric"),(0,.25,.5,.75,1)):
        ext=a.a*0
        if bound!="domestic":
            hv=33.33 if bound.endswith("LHV") else 39.41
            ext=a.extra_H2_reported_MWh_per_MWh*12/hv+a.avoided_external_import_MWhe_per_MWh*.5
        mu=(a.a+ext)/.4 if compression=="symmetric" else a.mu
        site=a.a+h*HEAT+ext-ls*M
        grid_q1=K*(G-.03*mu)-lg*M
        qstar=(site+lg*M+K*.03*mu)/(K*G)
        t=a[["year","weather"]].assign(site_leak=ls,grid_leak=lg,credits=bound,compression=compression,h=h,
                site_net=site,grid_at_q1=grid_q1,grid_minus_site_q1=grid_q1-site,qstar=qstar)
        assert np.allclose(grid_q1-site,K*G*(1-qstar))
        margins.append(t)
    save(pd.concat(margins).groupby(["year","site_leak","grid_leak","credits","compression","h"],
         as_index=False).agg(n=("weather","size"),qstar=("qstar","mean"),qstar_min=("qstar","min"),
         qstar_max=("qstar","max"),grid_minus_site_q1=("grid_minus_site_q1","mean"),
         n_grid_wins_q1=("grid_minus_site_q1",lambda x:int((x>0).sum()))),"threshold_summary")
    sums=[]
    for yr,t in a.groupby("year"):
        for waste in [-.01,0,.004,.01]:
            sums.append(dict(year=yr,common_t_per_raw=waste,mean=t.site_net.mean()+waste,
                             n_negative=int((t.site_net+waste<0).sum()),n=len(t)))
    save(pd.DataFrame(sums),"common_credit")
    return a


def strategies(a):
    countries=pd.read_csv(OLD/"country_cases.csv").groupby(["country","year"]).avoided.mean()
    resources=pd.read_csv(OLD/"resource.csv").query("scenario=='Central'").set_index(["country_code","year"])
    yeargrid=np.arange(2028,2036)
    mus=a.groupby("year").mu.mean()
    rows=[]
    for c,waste in product(countries.index.get_level_values(0).unique(),[-.01,0,.004,.01]):
        R=np.interp(yeargrid,YEARS,resources.loc[c].loc[YEARS].deliverable_TWh)/K
        site=np.interp(yeargrid,YEARS,countries[c].loc[YEARS])-.025*M+waste
        grid=K*(G-.03*np.interp(yeargrid,YEARS,mus.loc[YEARS]))-.025*M+waste
        still=np.logical_and.accumulate(site>0)
        for y,q,s,t,h in zip(yeargrid,R,site,grid,still):
            rows.append(dict(country=c,year=y,common_t_per_raw=waste,raw_TWh=q,still_chp=int(h),
                             keep_Mt=q*s,grid_Mt=q*t,switch_Mt=q*(s if h else t)))
    save(pd.DataFrame(rows),"strategy_country_year")
    base=pd.DataFrame(rows).query("common_t_per_raw==0")
    scenarios=[]
    for eligible,q,start_year in product((.25,.50,.75,1.),(.50,.75,1.),(2028,2030,2032)):
        # An equal share within every country; the baseline quantity remains
        # in continuous power before start_year and at all ineligible sites.
        frac=(base.year>=start_year)*eligible
        alternative_grid=base.grid_Mt-base.raw_TWh*K*(1-q)*G
        total=((1-frac)*base.keep_Mt+frac*alternative_grid).sum()
        scenarios.append(dict(eligible_share=eligible,fossil_share=q,start_year=start_year,
                     abatement_Mt=total,increment_over_keep_Mt=total-base.keep_Mt.sum()))
    save(pd.DataFrame(scenarios),"access_and_timing")


def weights():
    res=pd.read_csv(OLD/"resource.csv").query("year==2035")
    # Use individual national response slopes for this explicitly linear diagnostic.
    c=pd.read_csv(OLD/"country_cases.csv").query("year==2035").pivot(index="country",columns="weather",values="D")
    rows=[]
    for name,t in res.groupby("scenario"):
        x=t.set_index("country_code")
        r=x.fleet_fuel_TWh if name=="fleet" else x.deliverable_TWh
        w=(r/r.sum()).reindex(c.index)
        assert w.notna().all() and np.isclose(w.sum(),1)
        for weather,value in c.mul(w,axis=0).sum().items():
            rows.append(dict(composition=name,weather=weather,linear_D=value))
    save(pd.DataFrame(rows),"national_weights")
    fossil=pd.read_csv(OLD/"country_cases.csv")
    cols=[n for n in fossil.columns if n.startswith("fossil_gen_reduction_in_")]
    shares=fossil[cols].div(.4*fossil.B,axis=0)
    shares.columns=[col.removeprefix("fossil_gen_reduction_in_").removesuffix("_MWh") for col in cols]
    assert np.allclose(shares.sum(axis=1),fossil.s_f)
    mat=pd.concat([fossil[["country","year","weather"]],shares],axis=1).melt(
        id_vars=["country","year","weather"],var_name="responding_country",value_name="signed_fossil_MWhe_per_added_MWhe")
    save(mat,"signed_country_response")
    v=fossil[["year","weather","country","B","D","s_f","share_reduced_external_import","share_reduced_gas"]].copy()
    v=pd.concat([v,shares.add_prefix("fossil_share_in_")],axis=1)
    v["net_regional_response"]=shares.sum(axis=1)
    save(v[v.country.eq("GR")],"greece_signed")


def archived_imports():
    rows=[]
    for y,w in product(YEARS,range(1,37)):
        file=ROOT/f"dataset_one/50_decarbonization_trajectory_v37/results/F0_{y}_R0_WS{w:02}.npz"
        with np.load(file) as f:
            rows.append(dict(year=y,weather=w,imports_TWh=f["external_import"].sum()/1e6))
    save(pd.DataFrame(rows),"archive_imports")


def gather(pattern,nfiles,keys):
    files=sorted((HERE/"experiments"/pattern).glob("*.csv"))
    if len(files)!=nfiles:
        raise RuntimeError(f"{pattern}: expected {nfiles} complete files, found {len(files)}")
    t=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    assert not t.duplicated(keys).any() and t.residual.max()<=2e-4
    return t


def windows():
    ws=gather("windows",32,["year","weather","month","availability","price","intervention"])
    save(ws,"window_pairs")
    r=ws.query("intervention=='region_raw'")
    assert len(r)==128
    summary=r.groupby(["year","availability","price"],as_index=False).agg(
        n_windows=("D","size"),D=("D","mean"),D_min=("D","min"),D_max=("D","max"),
        import_share=("import_share","mean"),baseline_import_MWh=("baseline_import_MWh","mean"),
        price_eur_MWh=("price_eur_MWh","mean"))
    save(summary,"import_windows")
    wide=ws.query("availability=='all_hours' and price=='fixed'").pivot(
        index=["year","weather","month"],columns="intervention",values="D")
    out=[]
    for actual,ref in [("region_raw","region_legacy_dose"),("SI_actual","SI_floor"),("HR_actual","HR_floor")]:
        for (y,weather,month),t in wide[[actual,ref]].dropna().iterrows():
            out.append(dict(year=y,weather=weather,month=month,comparison=actual,actual_D=t[actual],
                            reference_D=t[ref],difference=t[actual]-t[ref]))
    save(pd.DataFrame(out),"dose_checks")


def shapley():
    f=gather("factorial",64,["background","vre","cost","weather","month","trade"])
    save(f,"factorial_windows")
    rows=[]
    for (trade,weather,month),part in f.groupby(["trade","weather","month"]):
        lookup=dict(zip(part[["background","vre","cost"]].itertuples(index=False,name=None),part.D))
        assert len(lookup)==8
        effects=np.zeros(3)
        for order in permutations(range(3)):
            state=[2028]*3
            for j in order:
                before=lookup[tuple(state)]
                state[j]=2035
                effects[j]+=(lookup[tuple(state)]-before)/6
        delta=lookup[(2035,)*3]-lookup[(2028,)*3]
        assert np.isclose(effects.sum(),delta)
        rows.append(dict(weather=weather,month=month,trade=trade,first=lookup[(2028,)*3],
                         last=lookup[(2035,)*3],delta=delta,background=effects[0],vre=effects[1],cost=effects[2]))
    save(pd.DataFrame(rows),"factorial_effects")


def facilities():
    f=gather("site_year",848,["year","weather","site","block"])
    assert all((t.start.to_numpy()==np.arange(0,8760,168)).all() and t.hours.sum()==8760
               for _,t in f.sort_values("start").groupby(["year","weather","site"]))
    save(f,"site_blocks")
    a=f.groupby(["year","weather","site"],as_index=False)[["hours","raw_MWh","electricity_MWh",
                        "avoided_t","fossil_MWh","import_MWh"]].sum()
    a["mu"]=a.avoided_t/a.electricity_MWh
    a["D"]=a.fossil_MWh/a.raw_MWh
    save(a,"site_year")
    monthly=gather("site_month",24,["year","weather","site","block"])
    save(monthly,"site_month_blocks")
    cal=monthly.groupby(["year","weather","site"],as_index=False)[["hours","raw_MWh","electricity_MWh","avoided_t","fossil_MWh"]].sum()
    assert cal.hours.eq(8760).all()
    cal["D_month"]=cal.fossil_MWh/cal.raw_MWh
    cal["mu_month"]=cal.avoided_t/cal.electricity_MWh
    audit=cal.merge(a[["year","weather","site","D","mu"]],on=["year","weather","site"],validate="one_to_one")
    audit["diff_D"]=audit.D_month-audit.D
    audit["diff_mu"]=audit.mu_month-audit.mu
    save(audit,"site_partition_check")
    it=a.query("site=='Pinerolo'")
    rows=[]
    for row in it.itertuples():
        for gas_mode,gas in [("adjusted_Q1",SITE["it_grid_q1"]),("adjusted_median",SITE["it_grid_adjusted"]),
                             ("adjusted_Q3",SITE["it_grid_q3"])]:
            for heat_delivery,slip in product((.85,1.),(0,.032,-.032)):
                heat=SITE["it_chp_heat"]*heat_delivery
                site=row.avoided_t/row.raw_MWh+heat*G/.9
                q=(site+slip*M)/(gas*G)
                # These differences cancel any common leakage or upstream waste credit.
                rows.append(dict(year=row.year,weather=row.weather,gas_basis=gas_mode,biomethane_per_raw=gas,
                       heat_delivery_fraction=heat_delivery,leak_grid_minus_site=slip,mu=row.mu,
                       CHP_heat_and_power_gross_credit=site,qstar=q,grid_minus_chp_at_q1=gas*G*(1-q),
                       additional_useful_heat_at_q1=(gas*G*(1-q))/(G/.9)))
    save(pd.DataFrame(rows),"pinerolo_thresholds")
    envelope=[]
    for rec in it.itertuples():
        for electric_key,gas_key in product(("it_electric_q1","it_chp_electric","it_electric_q3"),
                                            ("it_grid_q1","it_grid_adjusted","it_grid_q3")):
            e,y=SITE[electric_key],SITE[gas_key]
            site_credit=e*rec.mu+SITE["it_chp_heat"]*SITE["it_dh_delivery"]*G/.9
            envelope.append(dict(year=rec.year,weather=rec.weather,electric_basis=electric_key,electric_yield=e,
                  gas_basis=gas_key,gas_yield=y,qstar=site_credit/(y*G),grid_minus_chp_at_q1=y*G-site_credit))
    save(pd.DataFrame(envelope),"pinerolo_yield_envelope")
    basis=[]
    for rec in it.itertuples():
        for input_factor,product_factor in product((1.,.90),repeat=2):
            electricity=SITE["it_chp_electric"]/input_factor
            heat=SITE["it_chp_heat"]*SITE["it_dh_delivery"]/input_factor
            gas=SITE["it_grid_adjusted"]*product_factor/input_factor
            site_credit=electricity*rec.mu+heat*G/.9
            basis.append(dict(year=rec.year,weather=rec.weather,input_reported_to_lhv=input_factor,
                    gas_reported_to_lhv=product_factor,electric_per_raw_LHV=electricity,
                    useful_heat_per_raw_LHV=heat,gas_LHV_per_raw_LHV=gas,
                    qstar=site_credit/(gas*G),grid_minus_chp_at_q1=gas*G-site_credit))
    save(pd.DataFrame(basis),"pinerolo_energy_basis")
    rows=[]
    for row in a.query("site=='Ruhleben'").itertuples():
        for methane,basis,alloc,h,slip in product((.55,SITE["de_ch4_backcalc"],SITE["de_raw_ch4_assumed"],.65),
                 (1.,SITE["de_hhv_to_lhv"]),(0.,1.),(0.,.5),(0.,.032,-.032)):
            raw=SITE["de_raw_volume"]*methane*SITE["ch4_lhv"]/1000
            product_MWh=SITE["de_grid"]*basis
            # Same calorific convention for the natural gas withdrawal and product.
            charged=alloc*(SITE["de_full_electric"]*row.mu+SITE["de_full_gas"]*basis*G+
                           SITE["de_full_fuel"]*DIESEL)/raw
            production=.4*row.mu+h*HEAT
            q=(production+charged+slip*M)/(product_MWh/raw*G)
            rows.append(dict(year=row.year,weather=row.weather,raw_ch4=methane,
                reported_to_lhv=basis,whole_plant_aux_attributed=alloc,h=h,leak_grid_minus_site=slip,
                mu=row.mu,raw_MWh=raw,gas_per_raw=product_MWh/raw,gas_per_wet_t=product_MWh/SITE["de_feed"],
                qstar=q,grid_minus_chp_at_q1=(1-q)*G*product_MWh/raw))
    save(pd.DataFrame(rows),"ruhleben_thresholds")


def physical_checks():
    s=SITE
    gas=.2359
    potential=s["it_heat_potential"]*s["it_dh_delivery"]*gas/.90
    rows=[]
    for label,mu in [("Italian mix 2022",(.3089+.0736)),("Italian mix 2030",(.120+.024)),
                     ("Italian NGCC 2022",(.3916+.082))]:
        h=(s["it_grid_adjusted"]*gas-s["it_chp_electric"]*mu)/potential
        rows.append(dict(comparator=label,electricity_t_per_MWh=mu,gas_t_per_MWh=gas,h_parity=h))
    save(pd.DataFrame(rows),"pinerolo_published_check")
    grid=s["de_grid"]
    rows=[("reported gas per tonne accepted",grid/s["de_feed"],"MWh reported/t wet"),
          ("reported raw gas per tonne accepted",s["de_raw_volume"]/s["de_feed"],"m3 reported/t wet"),
          ("reported net all-energy balance",grid-s["de_full_electric"]-s["de_full_gas"]-s["de_full_fuel"],"MWh/year"),
          ("electricity per reported product",s["de_full_electric"]/grid,"MWh/MWh"),
          ("natural gas per reported product",s["de_full_gas"]/grid,"MWh/MWh"),
          ("other fuel per reported product",s["de_full_fuel"]/grid,"MWh/MWh")]
    save(pd.DataFrame(rows,columns=["quantity","value","unit"]),"ruhleben_published_check")


def main():
    a=annual()
    strategies(a)
    weights()
    archived_imports()
    windows()
    shapley()
    physical_checks()
    matched=gather("matched",32,["year","weather","month","site"])
    assert len(matched)==64 and np.isclose(matched.power_MW.max(),matched.power_MW.min())
    save(matched,"matched_sites")
    facilities()


if __name__=="__main__":
    main()
