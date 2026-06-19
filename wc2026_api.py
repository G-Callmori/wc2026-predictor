"""
FastAPI backend per il Predittore Mondiali 2026.
Importa le funzioni da WorldCupPredictor2026.py senza eseguire main().
Terminale: python3 WorldCupPredictor2026.py --predict  (invariato)
API:       python3 wc2026_api.py
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from WorldCupPredictor2026 import (
    scarica_risultati, calcola_elo, costruisci_features,
    allena, allena_gol,
    scarica_cartellini_statsbomb, costruisci_features_disciplina, allena_cartellini,
    scarica_stats_arbitri_club, _mergia_ref_stats,
    analizza_wc2026,
    predici_partita, predici_gol_poisson, predici_cartellini_poisson,
    _norm_probs, _cerca_team_wc,
)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Caricando dati e allenando modelli...")

    df_raw = scarica_risultati()
    df, elo_ratings = calcola_elo(df_raw)
    feat_df, team_hist, h2h_hist = costruisci_features(df)
    model, le = allena(feat_df)
    model_gh, model_ga = allena_gol(feat_df)

    disc_hist, ref_stats, model_yh, model_ya = {}, {}, None, None
    has_cards = False
    try:
        df_cards = scarica_cartellini_statsbomb()
        feat_disc, disc_hist, ref_stats = costruisci_features_disciplina(df_cards, elo_ratings)
        model_yh, model_ya = allena_cartellini(feat_disc)
        has_cards = True
    except Exception as e:
        print(f"⚠ Cartellini non disponibili: {e}")

    try:
        ref_stats_club = scarica_stats_arbitri_club()
        ref_stats = _mergia_ref_stats(ref_stats, ref_stats_club)
    except Exception as e:
        print(f"⚠ Stats club non disponibili: {e}")

    wc_info = analizza_wc2026(df)

    _state.update({
        "elo_ratings": elo_ratings,
        "team_hist": team_hist,
        "h2h_hist": h2h_hist,
        "model": model,
        "le": le,
        "model_gh": model_gh,
        "model_ga": model_ga,
        "disc_hist": disc_hist,
        "ref_stats": ref_stats,
        "model_yh": model_yh,
        "model_ya": model_ya,
        "has_cards": has_cards,
        "wc_info": wc_info,
    })
    print(f"✓ Pronto — {len(elo_ratings)} squadre caricate.")
    yield
    _state.clear()


app = FastAPI(title="WC 2026 Predictor API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    home: str
    away: str
    oh: Optional[float] = None
    od: Optional[float] = None
    oa: Optional[float] = None
    referee: Optional[str] = None
    is_knockout: bool = False
    neutral: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "teams": len(_state.get("elo_ratings", {}))}


@app.get("/teams")
def get_teams():
    teams = sorted(_state.get("elo_ratings", {}).keys())
    return {"teams": teams}


@app.get("/referees")
def get_referees():
    ref_stats = _state.get("ref_stats") or {}
    refs = [
        {
            "name": k,
            "matches": v["matches"],
            "avg_yellow": round(v["total_yellow"] / v["matches"], 2),
        }
        for k, v in ref_stats.items()
        if v.get("matches", 0) > 0
    ]
    refs.sort(key=lambda x: -x["matches"])
    return {"referees": refs}


@app.post("/predict")
def predict(req: PredictRequest):
    if not _state:
        raise HTTPException(503, "Modelli non ancora pronti")

    s = _state

    if req.oh and req.od and req.oa:
        ph, pd_, pa = _norm_probs([req.oh, req.od, req.oa])
        has_odds = True
    else:
        ph = pd_ = pa = -1.0
        has_odds = False

    wc_info = s["wc_info"]
    if wc_info:
        h_key = _cerca_team_wc(req.home, wc_info)
        a_key = _cerca_team_wc(req.away, wc_info)
        hs  = wc_info["statuses"].get(h_key, "in_gioco")
        as_ = wc_info["statuses"].get(a_key, "in_gioco")
    else:
        hs = as_ = "in_gioco"

    try:
        probs, pred, fh, fa, elo_h, elo_a, nota, w = predici_partita(
            req.home, req.away, s["model"], s["le"], s["elo_ratings"], s["team_hist"],
            ph=ph, pd_=pd_, pa=pa, neutral=req.neutral,
            home_status=hs, away_status=as_,
            h2h_hist=s["h2h_hist"],
        )
    except Exception as e:
        raise HTTPException(400, f"Errore previsione: {e}")

    gol = predici_gol_poisson(
        req.home, req.away, s["model_gh"], s["model_ga"],
        s["elo_ratings"], s["team_hist"],
        neutral=req.neutral, home_status=hs, away_status=as_,
        h2h_hist=s["h2h_hist"],
    )

    cards = None
    if s["has_cards"] and s["model_yh"]:
        cards = predici_cartellini_poisson(
            req.home, req.away, s["model_yh"], s["model_ya"],
            s["elo_ratings"], s["disc_hist"],
            is_knockout=req.is_knockout, home_status=hs, away_status=as_,
            referee=req.referee, ref_stats=s["ref_stats"],
        )

    def f(x, n=3):
        return round(float(x), n)

    top_scores = [
        [int(s[0][0]), int(s[0][1]), f(s[1])]
        for s in gol.get("top_scores", [])
    ]

    return {
        "home": req.home,
        "away": req.away,
        "prediction": str(pred),
        "probs": {
            "home": f(probs[0]),
            "draw": f(probs[1]),
            "away": f(probs[2]),
        },
        "elo_home": int(round(float(elo_h))),
        "elo_away": int(round(float(elo_a))),
        "blend_weight": f(w, 2),
        "nota_motivazione": str(nota or ""),
        "home_status": hs,
        "away_status": as_,
        "goals": {
            "home_expected": f(gol["lam_h"], 2),
            "away_expected": f(gol["lam_a"], 2),
            "over_2_5": f(gol["over25"]),
            "btts": f(gol["btts"]),
            "top_scores": top_scores,
        },
        "cards": {
            "home_team_avg": f(cards["lam_yh_team"], 2),
            "away_team_avg": f(cards["lam_ya_team"], 2),
            "home_final": f(cards["lam_yh"], 2),
            "away_final": f(cards["lam_ya"], 2),
            "total": f(cards["lam_tot"], 2),
            "ref_factor": f(cards["ref_factor"], 3),
            "ref_name": str(cards.get("ref_key") or ""),
            "ref_matches": int(cards["ref_info"].get("matches", 0)),
            "ref_avg_yellow": f(cards.get("ref_total_avg", 0), 2),
            "over_2_5": f(cards["over_2_5"]),
            "over_3_5": f(cards["over_3_5"]),
            "over_4_5": f(cards["over_4_5"]),
        } if cards else None,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("wc2026_api:app", host="0.0.0.0", port=port, reload=False)
