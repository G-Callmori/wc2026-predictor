"""
=============================================================
  PREDITTORE MONDIALI 2026
  Dati:  risultati internazionali storici (martj42/GitHub)
         + partite WC 2026 già giocate (aggiornamento automatico)
  Quote: the-odds-api.com (API key gratuita richiesta)
  Model: Elo + Forma recente + XGBoost
         + aggiustamento motivazione (già qualificata/eliminata)
=============================================================
  Uso:
    python WorldCupPredictor2026.py --predict     # menu squadre WC
    python WorldCupPredictor2026.py --live        # quote live API
    python WorldCupPredictor2026.py --standings   # classifiche gironi
=============================================================
"""

import argparse
import warnings
import os
import sys
import math
from collections import deque
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    _COLOR = True
except ImportError:
    _COLOR = False

def col(text, color="green"):
    if not _COLOR:
        return text
    palette = {
        "green": Fore.GREEN, "red": Fore.RED, "yellow": Fore.YELLOW,
        "cyan": Fore.CYAN, "white": Fore.WHITE, "bold": Style.BRIGHT,
        "magenta": Fore.MAGENTA, "blue": Fore.BLUE,
    }
    return f"{palette.get(color,'')}{text}{Style.RESET_ALL}"

# ─── CONFIGURAZIONE ───────────────────────────────────────────

RESULTS_URL   = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "14c2f8bbb9052287d5d50e496c0194af")

ELO_K = {
    "FIFA World Cup":         60,
    "Confederations Cup":     50,
    "Copa América":           50,
    "UEFA Euro":              50,
    "Africa Cup of Nations":  50,
    "Asian Cup":              45,
    "Gold Cup":               40,
    "Olympic Games":          40,
    "Nations League":         35,
    "Friendly":               20,
}
ELO_DEFAULT_K  = 35
ELO_INITIAL    = 1500
ELO_HOME_BONUS = 100

# Peso quote: dinamico in base al divario Elo.
# Squadre simili → mercato più informativo → peso quote alto.
# Divario grande → modello più affidabile → peso quote basso.
def _blend_weight(elo_h, elo_a):
    diff = abs(elo_h - elo_a)
    w = 0.65 - (diff / 900)
    return round(max(0.30, min(0.65, w)), 2)
ANNO_DA           = 2000
GRUPPI_LETTERE    = "ABCDEFGHIJKL"

STATSBOMB_BASE    = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CARTELLINI_CACHE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".card_cache.csv")
STATSBOMB_COMPS   = [
    (43,  3,   "WC 2018"),
    (55,  43,  "Euro 2020"),
    (43,  106, "WC 2022"),
    (223, 282, "Copa America 2024"),
    (55,  282, "Euro 2024"),
]
DISC_FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff",
    "h_avg_yellow", "h_avg_red",
    "a_avg_yellow", "a_avg_red",
    "diff_avg_yellow", "is_knockout",
]
REF_PRIOR_YELLOW  = 3.0  # prior Bayesiano: media gialli/partita nei tornei
REF_PRIOR_MATCHES = 5    # pseudo-count per smoothing

# Dati arbitri da campionati club (football-data.co.uk — no auth)
FDCO_BASE         = "https://www.football-data.co.uk/mmz4281"
FDCO_LEAGUES      = ["E0", "SP1", "I1", "D1", "F1"]          # Prem, LaLiga, SerieA, Bundesliga, Ligue1
FDCO_SEASONS      = ["2021", "2122", "2223", "2324", "2425"]  # 2020-21 → 2024-25
REF_STATS_CACHE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ref_club_cache.csv")


# ─── 1. DATI STORICI ──────────────────────────────────────────

def scarica_risultati():
    print(col("📥 Scaricando risultati internazionali...", "cyan"))
    r = requests.get(RESULTS_URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df["date"]       = pd.to_datetime(df["date"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"]    = df["neutral"].astype(bool)
    df = df.sort_values("date").reset_index(drop=True)

    df_filt = df[df["date"].dt.year >= ANNO_DA].reset_index(drop=True)
    n_wc = df_filt[
        df_filt["tournament"].str.contains("FIFA World Cup", case=False, na=False) &
        (df_filt["date"].dt.year == 2026)
    ].shape[0]
    print(col(f"  ✓ {len(df_filt):,} partite caricate "
              f"({df_filt['date'].dt.year.min()}–{df_filt['date'].dt.year.max()})  "
              f"[di cui {n_wc} Mondiali 2026]", "green"))
    return df_filt


# ─── 2. ELO RATING DINAMICO ───────────────────────────────────

def _k_factor(tournament):
    t = tournament.lower()
    for nome, k in ELO_K.items():
        if nome.lower() in t:
            return k
    return ELO_DEFAULT_K

def _tournament_weight(tournament):
    """Peso importanza torneo per la rolling window di forma (Hvattum & Arntzen 2010)."""
    t = tournament.lower()
    if "world cup" in t:                                         return 1.5
    if any(x in t for x in ("euro", "copa", "african", "asian cup", "gold cup")): return 1.3
    if any(x in t for x in ("nations league", "olympic")):      return 1.1
    if "friendly" in t:                                          return 0.7
    return 1.0

def _expected(elo_a, elo_b):
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

def calcola_elo(df):
    print(col("⚙️  Calcolando rating Elo...", "cyan"))
    ratings = {}
    elo_home_pre, elo_away_pre = [], []

    for _, row in df.iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            elo_home_pre.append(ratings.get(row["home_team"], ELO_INITIAL))
            elo_away_pre.append(ratings.get(row["away_team"], ELO_INITIAL))
            continue

        home, away = row["home_team"], row["away_team"]
        neutral    = bool(row["neutral"])
        tournament = str(row.get("tournament", ""))
        gh, ga     = float(row["home_score"]), float(row["away_score"])

        elo_h = ratings.get(home, ELO_INITIAL)
        elo_a = ratings.get(away, ELO_INITIAL)
        elo_home_pre.append(elo_h)
        elo_away_pre.append(elo_a)

        elo_h_adj = elo_h + (0 if neutral else ELO_HOME_BONUS)
        exp_h = _expected(elo_h_adj, elo_a)

        if gh > ga:    act_h = 1.0
        elif gh < ga:  act_h = 0.0
        else:          act_h = 0.5

        gd = abs(gh - ga)
        if gd <= 1:    gd_mult = 1.0
        elif gd == 2:  gd_mult = 1.5
        else:          gd_mult = (11.0 + gd) / 8.0

        k = _k_factor(tournament)
        ratings[home] = elo_h + k * gd_mult * (act_h - exp_h)
        ratings[away] = elo_a + k * gd_mult * ((1 - act_h) - (1 - exp_h))

    df = df.copy()
    df["elo_home_pre"] = elo_home_pre
    df["elo_away_pre"] = elo_away_pre

    print(col(f"  ✓ Elo calcolato per {len(ratings)} nazionali", "green"))
    top = sorted(ratings.items(), key=lambda x: -x[1])[:12]
    print(col("\n  Top 12 per Elo aggiornato ai Mondiali 2026:", "yellow"))
    for i, (team, elo) in enumerate(top, 1):
        bar = "█" * int((elo - 1400) / 10)
        print(col(f"    {i:2}. {team:<28} {elo:6.0f}  {bar}", "white"))

    return df, ratings


# ─── 3. FEATURE BUILDING (O(n) finestra mobile) ───────────────

FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff",
    "h_win_rate", "h_draw_rate", "h_gol_fatti", "h_gol_subiti", "h_elo_avv",
    "a_win_rate", "a_draw_rate", "a_gol_fatti", "a_gol_subiti", "a_elo_avv",
    "diff_win_rate", "diff_gol_fatti", "diff_gol_subiti",
    "neutral",
    "h2h_home_winrate", "h2h_goal_diff",       # testa a testa storico
    "h_tournament_weight", "a_tournament_weight",  # qualità media dei recenti match
]

_EMPTY_FORM = {
    "win_rate": 0.33, "draw_rate": 0.33,
    "gol_fatti": 1.2, "gol_subiti": 1.2, "elo_avv": 1500,
    "tournament_weight": 1.0,
}

def _stats(history):
    """Media ponderata per importanza torneo (WC > Europei > Amichevoli)."""
    if not history:
        return _EMPTY_FORM
    total_w = sum(h.get("w", 1.0) for h in history)
    if total_w == 0:
        return _EMPTY_FORM
    wins  = sum(h.get("w", 1.0) for h in history if h["gf"] > h["gs"])
    draws = sum(h.get("w", 1.0) for h in history if h["gf"] == h["gs"])
    return {
        "win_rate":          wins / total_w,
        "draw_rate":         draws / total_w,
        "gol_fatti":         sum(h["gf"] * h.get("w", 1.0) for h in history) / total_w,
        "gol_subiti":        sum(h["gs"] * h.get("w", 1.0) for h in history) / total_w,
        "elo_avv":           sum(h["elo_opp"] * h.get("w", 1.0) for h in history) / total_w,
        "tournament_weight": total_w / len(history),
    }


def _h2h_stats(home, away, h2h_hist):
    """
    Storico testa a testa tra due squadre (ultimi 8 incontri).
    Ritorna win_rate e differenza reti dalla prospettiva di home.
    """
    key  = tuple(sorted([home, away]))
    hist = h2h_hist.get(key, []) if h2h_hist else []
    if not hist:
        return {"h2h_home_winrate": 0.33, "h2h_goal_diff": 0.0}
    wins_home = 0
    total_gd  = 0.0
    for h in hist:
        gf = h["gf"] if h["home_name"] == home else h["ga"]
        ga = h["ga"] if h["home_name"] == home else h["gf"]
        if gf > ga:
            wins_home += 1
        total_gd += gf - ga
    return {
        "h2h_home_winrate": wins_home / len(hist),
        "h2h_goal_diff":    total_gd  / len(hist),
    }

def costruisci_features(df):
    print(col("\n⚙️  Costruendo features (forma, H2H, peso torneo)...", "cyan"))
    N = 30
    team_hist = {}
    h2h_hist  = {}   # {(team_a, team_b) sorted: deque di sfide dirette}
    righe = []

    def _aggiorna(team, gf, gs, elo_opp, w):
        if team not in team_hist:
            team_hist[team] = deque(maxlen=N)
        team_hist[team].append({"gf": gf, "gs": gs, "elo_opp": elo_opp, "w": w})

    def _aggiorna_h2h(home, away, gh, ga):
        key = tuple(sorted([home, away]))
        h2h_hist.setdefault(key, deque(maxlen=30)).append(
            {"home_name": home, "gf": gh, "ga": ga}
        )

    for i, row in df.iterrows():
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue
        home, away = row["home_team"], row["away_team"]
        gh, ga     = float(row["home_score"]), float(row["away_score"])
        elo_h      = row["elo_home_pre"]
        elo_a      = row["elo_away_pre"]
        w          = _tournament_weight(str(row.get("tournament", "")))

        if i >= 30:
            fh  = _stats(team_hist.get(home))
            fa  = _stats(team_hist.get(away))
            h2h = _h2h_stats(home, away, h2h_hist)
            if gh > ga:    ris = "H"
            elif gh < ga:  ris = "A"
            else:          ris = "D"
            righe.append({
                "elo_home":             elo_h, "elo_away":   elo_a, "elo_diff":  elo_h - elo_a,
                "h_win_rate":           fh["win_rate"],   "h_draw_rate":  fh["draw_rate"],
                "h_gol_fatti":          fh["gol_fatti"],  "h_gol_subiti": fh["gol_subiti"],
                "h_elo_avv":            fh["elo_avv"],
                "a_win_rate":           fa["win_rate"],   "a_draw_rate":  fa["draw_rate"],
                "a_gol_fatti":          fa["gol_fatti"],  "a_gol_subiti": fa["gol_subiti"],
                "a_elo_avv":            fa["elo_avv"],
                "diff_win_rate":        fh["win_rate"]   - fa["win_rate"],
                "diff_gol_fatti":       fh["gol_fatti"]  - fa["gol_fatti"],
                "diff_gol_subiti":      fh["gol_subiti"] - fa["gol_subiti"],
                "neutral":              int(row["neutral"]),
                "h2h_home_winrate":     h2h["h2h_home_winrate"],
                "h2h_goal_diff":        h2h["h2h_goal_diff"],
                "h_tournament_weight":  fh["tournament_weight"],
                "a_tournament_weight":  fa["tournament_weight"],
                "risultato":            ris,
                "gol_home":             gh,
                "gol_away":             ga,
            })

        _aggiorna(home, gh, ga, elo_a, w)
        _aggiorna(away, ga, gh, elo_h, w)
        _aggiorna_h2h(home, away, gh, ga)

    feat_df = pd.DataFrame(righe)
    print(col(f"  ✓ {len(feat_df):,} esempi con features", "green"))
    return feat_df, team_hist, h2h_hist


# ─── 4. MODELLO XGBOOST ───────────────────────────────────────

def allena(feat_df):
    print(col("\n🤖 Allenando XGBoost...", "cyan"))
    le = LabelEncoder()
    y  = le.fit_transform(feat_df["risultato"])
    X  = feat_df[FEATURE_COLS]
    split   = int(len(feat_df) * 0.80)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split], y[split:]

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="mlogloss", random_state=42, verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    print(col(f"\n📊 Test set ({len(y_te):,} partite recenti):", "bold"))
    print(col(f"   Accuratezza: {acc*100:.1f}%", "yellow"))
    print()
    print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0))
    baseline = (y_te == le.transform(["H"])[0]).mean()
    print(col(f"   Baseline (sempre 'casa'): {baseline*100:.1f}%  →  Miglioramento: +{(acc-baseline)*100:.1f}%", "green"))
    return model, le


# ─── 4b. MODELLO GOL (Regressori XGBoost + Poisson) ──────────

def allena_gol(feat_df):
    from sklearn.metrics import mean_absolute_error
    print(col("\n⚽ Allenando regressori gol...", "cyan"))

    X     = feat_df[FEATURE_COLS]
    y_gh  = feat_df["gol_home"]
    y_ga  = feat_df["gol_away"]
    split = int(len(feat_df) * 0.80)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]

    params = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=42, verbosity=0)

    model_gh = xgb.XGBRegressor(**params)
    model_gh.fit(X_tr, y_gh.iloc[:split],
                 eval_set=[(X_te, y_gh.iloc[split:])], verbose=False)

    model_ga = xgb.XGBRegressor(**params)
    model_ga.fit(X_tr, y_ga.iloc[:split],
                 eval_set=[(X_te, y_ga.iloc[split:])], verbose=False)

    mae_h = mean_absolute_error(y_gh.iloc[split:], model_gh.predict(X_te))
    mae_a = mean_absolute_error(y_ga.iloc[split:], model_ga.predict(X_te))
    print(col(f"  ✓ MAE gol casa:       {mae_h:.2f}  (media storica: {y_gh.mean():.2f})", "green"))
    print(col(f"  ✓ MAE gol trasferta:  {mae_a:.2f}  (media storica: {y_ga.mean():.2f})", "green"))
    return model_gh, model_ga


def _poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def predici_gol_poisson(home, away, model_gh, model_ga,
                         elo_ratings, team_hist, neutral=True,
                         home_status="in_gioco", away_status="in_gioco",
                         h2h_hist=None):
    """Predice gol attesi (XGBoost) e distribuzione Poisson dei risultati esatti."""
    elo_h = _cerca_elo(home, elo_ratings)
    elo_a = _cerca_elo(away, elo_ratings)
    fh    = _stats(team_hist.get(home))
    fa    = _stats(team_hist.get(away))
    h2h   = _h2h_stats(home, away, h2h_hist)

    feat = {
        "elo_home":             elo_h, "elo_away":   elo_a, "elo_diff":  elo_h - elo_a,
        "h_win_rate":           fh["win_rate"],   "h_draw_rate":  fh["draw_rate"],
        "h_gol_fatti":          fh["gol_fatti"],  "h_gol_subiti": fh["gol_subiti"], "h_elo_avv": fh["elo_avv"],
        "a_win_rate":           fa["win_rate"],   "a_draw_rate":  fa["draw_rate"],
        "a_gol_fatti":          fa["gol_fatti"],  "a_gol_subiti": fa["gol_subiti"], "a_elo_avv": fa["elo_avv"],
        "diff_win_rate":        fh["win_rate"]   - fa["win_rate"],
        "diff_gol_fatti":       fh["gol_fatti"]  - fa["gol_fatti"],
        "diff_gol_subiti":      fh["gol_subiti"] - fa["gol_subiti"],
        "neutral":              int(neutral),
        "h2h_home_winrate":     h2h["h2h_home_winrate"],
        "h2h_goal_diff":        h2h["h2h_goal_diff"],
        "h_tournament_weight":  fh.get("tournament_weight", 1.0),
        "a_tournament_weight":  fa.get("tournament_weight", 1.0),
    }
    X = pd.DataFrame([feat])[FEATURE_COLS]

    lam_h = float(np.clip(model_gh.predict(X)[0], 0.1, 6.0))
    lam_a = float(np.clip(model_ga.predict(X)[0], 0.1, 6.0))

    # Aggiustamento motivazione sui gol attesi
    if home_status == "qualificata" and away_status == "in_gioco":
        lam_h *= 0.88
    elif away_status == "qualificata" and home_status == "in_gioco":
        lam_a *= 0.88
    elif home_status == "eliminata" and away_status == "in_gioco":
        lam_h *= 0.82
    elif away_status == "eliminata" and home_status == "in_gioco":
        lam_a *= 0.82

    # Matrice Poisson dei risultati esatti (0-0 fino a 6-6)
    MAX = 7
    score_probs = {}
    for gh in range(MAX):
        for ga in range(MAX):
            p = _poisson_pmf(gh, lam_h) * _poisson_pmf(ga, lam_a)
            score_probs[(gh, ga)] = p

    # Normalizza
    tot = sum(score_probs.values())
    score_probs = {k: v / tot for k, v in score_probs.items()}

    # Top 6 risultati più probabili
    top_scores = sorted(score_probs.items(), key=lambda x: -x[1])[:8]

    # Over 2.5, BTTS
    over25 = sum(p for (gh, ga), p in score_probs.items() if gh + ga > 2)
    btts   = sum(p for (gh, ga), p in score_probs.items() if gh > 0 and ga > 0)

    return {
        "lam_h":      lam_h,
        "lam_a":      lam_a,
        "top_scores": top_scores,
        "over25":     over25,
        "btts":       btts,
    }


# ─── 4c. CARTELLINI (StatsBomb Open Data) ────────────────────

def _sb_get(url, retries=2):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries:
                raise


def scarica_cartellini_statsbomb():
    """
    Scarica cartellini + nome arbitro da WC 2018/22, Euro 2020/24,
    Copa America 2024 (e WC 2026 se già disponibile su StatsBomb).
    Cache locale 7 giorni; invalida se manca la colonna 'referee'.
    """
    import time as _time

    if os.path.exists(CARTELLINI_CACHE):
        age_days = (_time.time() - os.path.getmtime(CARTELLINI_CACHE)) / 86400
        if age_days < 7:
            df = pd.read_csv(CARTELLINI_CACHE, parse_dates=["date"])
            if "referee" in df.columns:
                print(col(f"  ✓ Cartellini da cache: {len(df)} partite", "green"))
                return df

    # Costruisce lista competizioni + cerca WC 2026 dinamicamente
    comps_to_dl = list(STATSBOMB_COMPS)
    try:
        all_comps = _sb_get(f"{STATSBOMB_BASE}/competitions.json")
        wc26 = next(
            (c for c in all_comps
             if c["competition_id"] == 43 and "2026" in str(c.get("season_name", ""))),
            None
        )
        if wc26:
            comps_to_dl.append((43, wc26["season_id"], "WC 2026"))
            print(col(f"  ✓ StatsBomb: WC 2026 disponibile (season {wc26['season_id']})", "green"))
    except Exception:
        pass

    print(col("\n🟨 Scaricando dati cartellini (StatsBomb)...", "cyan"))
    righe = []

    for comp_id, season_id, label in comps_to_dl:
        try:
            matches = _sb_get(f"{STATSBOMB_BASE}/matches/{comp_id}/{season_id}.json")
            n_ok = 0
            for m in matches:
                home      = m["home_team"]["home_team_name"]
                away      = m["away_team"]["away_team_name"]
                date_str  = m["match_date"]
                stage     = m.get("competition_stage", {}).get("name", "")
                is_ko     = int(stage not in ("Group Stage", "Group", ""))
                match_id  = m["match_id"]
                ref_obj   = m.get("referee") or {}
                referee   = ref_obj.get("name", "") if isinstance(ref_obj, dict) else ""

                try:
                    events = _sb_get(f"{STATSBOMB_BASE}/events/{match_id}.json")
                except Exception:
                    continue

                cards = [e for e in events if e.get("type", {}).get("name") == "Bad Behaviour"]
                yh = rh = ya = ra = 0
                for c in cards:
                    team_n = c.get("team", {}).get("name", "")
                    tipo   = c.get("bad_behaviour", {}).get("card", {}).get("name", "")
                    ih     = team_n == home
                    if tipo in ("Yellow Card", "Second Yellow"):
                        if ih: yh += 1
                        else:  ya += 1
                    elif tipo == "Red Card":
                        if ih: rh += 1
                        else:  ra += 1

                righe.append({
                    "date":        date_str,
                    "home_team":   home,
                    "away_team":   away,
                    "yellow_home": yh,
                    "red_home":    rh,
                    "yellow_away": ya,
                    "red_away":    ra,
                    "is_knockout": is_ko,
                    "referee":     referee,
                    "competition": label,
                })
                n_ok += 1

            print(col(f"  ✓ {label}: {n_ok}/{len(matches)} partite", "green"))

        except Exception as e:
            print(col(f"  ✗ {label}: {e}", "red"))

    df = pd.DataFrame(righe)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(CARTELLINI_CACHE, index=False)
    print(col(f"\n  ✓ Totale: {len(df)} partite salvate in cache", "green"))
    return df


def _cerca_disc(team, disc_hist):
    if not disc_hist or team in disc_hist:
        return disc_hist.get(team, [])
    tl = team.lower()
    for k, v in disc_hist.items():
        if k.lower() == tl:
            return v
    for k, v in disc_hist.items():
        if tl in k.lower() or k.lower() in tl:
            return v
    return []


def _disc_avg(team, disc_hist):
    h = _cerca_disc(team, disc_hist)
    if not h:
        return {"y": 2.2, "r": 0.14}
    return {"y": sum(x["y"] for x in h) / len(h),
            "r": sum(x["r"] for x in h) / len(h)}


def _cerca_arbitro(nome, ref_stats):
    """Fuzzy match nome arbitro nelle statistiche storiche."""
    if not nome or not ref_stats:
        return ""
    if nome in ref_stats:
        return nome
    nl = nome.lower().strip()
    for k in ref_stats:
        if k.lower() == nl:
            return k
    for k in ref_stats:
        if nl in k.lower() or k.lower() in nl:
            return k
    # Match per solo cognome
    parts = nl.split()
    if parts:
        last = parts[-1]
        for k in ref_stats:
            if last in k.lower().split():
                return k
    return ""


def _ref_avg_yellow(referee, ref_stats):
    """Media gialli per arbitro con smoothing Bayesiano."""
    if not referee or not ref_stats:
        return REF_PRIOR_YELLOW
    key = _cerca_arbitro(referee, ref_stats)
    if not key:
        return REF_PRIOR_YELLOW
    s = ref_stats[key]
    return (s["total_yellow"] + REF_PRIOR_YELLOW * REF_PRIOR_MATCHES) / \
           (s["matches"] + REF_PRIOR_MATCHES)


def costruisci_features_disciplina(df_cards, elo_ratings, N=30):
    """
    Rolling window O(n) per features disciplina.
    Traccia anche le statistiche per arbitro (senza leakage futuro).
    Ritorna (feat_df, disc_hist, ref_stats).
    """
    disc_hist   = {}
    ref_running = {}  # {referee: [tot_yellow per partita]}
    righe = []

    def _upd_team(team, yellow, red):
        if team not in disc_hist:
            disc_hist[team] = deque(maxlen=N)
        disc_hist[team].append({"y": yellow, "r": red})

    for i, row in df_cards.iterrows():
        home, away = row["home_team"], row["away_team"]
        yh, rh     = int(row["yellow_home"]), int(row["red_home"])
        ya, ra     = int(row["yellow_away"]), int(row["red_away"])
        ref_raw    = row.get("referee", "")
        referee    = str(ref_raw).strip() \
                     if ref_raw and not (isinstance(ref_raw, float) and math.isnan(ref_raw)) \
                     else ""
        elo_h      = _cerca_elo(home, elo_ratings)
        elo_a      = _cerca_elo(away, elo_ratings)

        if i >= 8:
            fh = _disc_avg(home, disc_hist)
            fa = _disc_avg(away, disc_hist)
            righe.append({
                "elo_home":        elo_h,
                "elo_away":        elo_a,
                "elo_diff":        elo_h - elo_a,
                "h_avg_yellow":    fh["y"],
                "h_avg_red":       fh["r"],
                "a_avg_yellow":    fa["y"],
                "a_avg_red":       fa["r"],
                "diff_avg_yellow": fh["y"] - fa["y"],
                "is_knockout":     int(row["is_knockout"]),
                "yellow_home":     yh,
                "red_home":        rh,
                "yellow_away":     ya,
                "red_away":        ra,
            })

        _upd_team(home, yh, rh)
        _upd_team(away, ya, ra)
        if referee:
            ref_running.setdefault(referee, []).append(yh + ya)

    ref_stats = {
        r: {"total_yellow": sum(v), "matches": len(v)}
        for r, v in ref_running.items()
    }

    return pd.DataFrame(righe), disc_hist, ref_stats


def allena_cartellini(feat_disc):
    from sklearn.metrics import mean_absolute_error
    if len(feat_disc) < 40:
        raise ValueError(f"Dati insufficienti per cartellini ({len(feat_disc)} esempi)")

    print(col("\n🟨 Allenando regressori cartellini...", "cyan"))
    X     = feat_disc[DISC_FEATURE_COLS]
    split = max(20, int(len(feat_disc) * 0.80))

    params = dict(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.5, reg_lambda=1.5,
        random_state=42, verbosity=0,
    )

    models = {}
    for target in ("yellow_home", "yellow_away"):
        y = feat_disc[target]
        m = xgb.XGBRegressor(**params)
        m.fit(X.iloc[:split], y.iloc[:split],
              eval_set=[(X.iloc[split:], y.iloc[split:])], verbose=False)
        mae = mean_absolute_error(y.iloc[split:], m.predict(X.iloc[split:]))
        print(col(f"  ✓ {target}: MAE {mae:.2f}  (media: {y.mean():.2f} gialli/partita)", "green"))
        models[target] = m

    return models["yellow_home"], models["yellow_away"]


def predici_cartellini_poisson(home, away, model_yh, model_ya,
                                elo_ratings, disc_hist,
                                is_knockout=False,
                                home_status="in_gioco", away_status="in_gioco",
                                referee=None, ref_stats=None):
    """
    Predice gialli via Poisson.
    Il modello usa solo feature di squadra; il fattore arbitro viene moltiplicato
    post-predizione: lam_finale = lam_squadra × ref_factor.
    """
    elo_h = _cerca_elo(home, elo_ratings)
    elo_a = _cerca_elo(away, elo_ratings)
    fh    = _disc_avg(home, disc_hist)
    fa    = _disc_avg(away, disc_hist)

    feat = {
        "elo_home":        elo_h,
        "elo_away":        elo_a,
        "elo_diff":        elo_h - elo_a,
        "h_avg_yellow":    fh["y"],
        "h_avg_red":       fh["r"],
        "a_avg_yellow":    fa["y"],
        "a_avg_red":       fa["r"],
        "diff_avg_yellow": fh["y"] - fa["y"],
        "is_knockout":     int(is_knockout),
    }
    X = pd.DataFrame([feat])[DISC_FEATURE_COLS]

    lam_yh_team = float(np.clip(model_yh.predict(X)[0], 0.3, 7.0))
    lam_ya_team = float(np.clip(model_ya.predict(X)[0], 0.3, 7.0))

    if home_status in ("qualificata", "eliminata"):
        lam_yh_team *= 0.85
    if away_status in ("qualificata", "eliminata"):
        lam_ya_team *= 0.85

    # Fattore arbitro: rapporto tra media arbitro e media globale
    ref_key  = _cerca_arbitro(referee or "", ref_stats or {})
    ref_info = ref_stats.get(ref_key, {}) if ref_stats and ref_key else {}

    if ref_stats:
        valori = [s["total_yellow"] / s["matches"]
                  for s in ref_stats.values() if s["matches"] > 0]
        global_avg = float(np.mean(valori)) if valori else REF_PRIOR_YELLOW
    else:
        global_avg = REF_PRIOR_YELLOW

    if ref_info and ref_info.get("matches", 0) > 0:
        ref_total_avg = (ref_info["total_yellow"] + REF_PRIOR_YELLOW * REF_PRIOR_MATCHES) / \
                        (ref_info["matches"] + REF_PRIOR_MATCHES)
    else:
        ref_total_avg = global_avg

    ref_factor = float(np.clip(ref_total_avg / global_avg, 0.3, 2.5))

    lam_yh  = lam_yh_team * ref_factor
    lam_ya  = lam_ya_team * ref_factor
    lam_tot = lam_yh + lam_ya

    def _p_over(lam, k):
        return 1.0 - sum(_poisson_pmf(i, lam) for i in range(int(k) + 1))

    return {
        "lam_yh_team": lam_yh_team,
        "lam_ya_team": lam_ya_team,
        "lam_yh":      lam_yh,
        "lam_ya":      lam_ya,
        "lam_tot":     lam_tot,
        "ref_factor":  ref_factor,
        "ref_total_avg": ref_total_avg,
        "global_avg":  global_avg,
        "over_2_5":    _p_over(lam_tot, 2),
        "over_3_5":    _p_over(lam_tot, 3),
        "over_4_5":    _p_over(lam_tot, 4),
        "ref_key":     ref_key,
        "ref_info":    ref_info,
    }


# ─── 5. ANALISI WC 2026 ───────────────────────────────────────

def analizza_wc2026(df):
    """
    Estrae le partite WC 2026 dal dataset, ricostruisce gironi
    e calcola classifiche. Restituisce None se nessuna partita trovata.
    """
    wc_tutti = df[
        (df["date"] >= "2026-06-11") &
        df["tournament"].str.contains("FIFA World Cup", case=False, na=False)
    ].copy()

    wc_giocate = wc_tutti.dropna(subset=["home_score", "away_score"]).copy()

    if len(wc_tutti) == 0:
        return None

    print(col(f"\n⚽ Mondiali 2026: "
              f"{len(wc_giocate)} partite giocate  |  "
              f"{len(wc_tutti) - len(wc_giocate)} ancora in programma", "cyan"))

    # Ricostruisce gironi dalla struttura del calendario
    adj = {}
    for _, row in wc_tutti.iterrows():
        h, a = row["home_team"], row["away_team"]
        adj.setdefault(h, set()).add(a)
        adj.setdefault(a, set()).add(h)

    visited = set()
    gironi = []
    for team in sorted(adj):
        if team in visited:
            continue
        g = set()
        queue = [team]
        while queue:
            t = queue.pop(0)
            if t in visited:
                continue
            visited.add(t)
            g.add(t)
            for nb in adj[t]:
                if nb not in visited:
                    queue.append(nb)
        gironi.append(sorted(g))
    gironi.sort(key=lambda g: g[0])
    team_girone = {t: i for i, g in enumerate(gironi) for t in g}

    # Classifica da risultati giocati
    standings = {}
    for _, row in wc_giocate.iterrows():
        h, a   = row["home_team"], row["away_team"]
        gh, ga = int(row["home_score"]), int(row["away_score"])
        for team, gf, gs in [(h, gh, ga), (a, ga, gh)]:
            s = standings.setdefault(team, {"P":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"GD":0,"Pts":0})
            s["P"] += 1; s["GF"] += gf; s["GA"] += gs; s["GD"] += gf - gs
            if gf > gs:    s["W"] += 1; s["Pts"] += 3
            elif gf == gs: s["D"] += 1; s["Pts"] += 1
            else:          s["L"] += 1

    # Stato qualificazione
    matches_per_team = max(len(g) - 1 for g in gironi) if gironi else 3
    statuses = {}

    for girone in gironi:
        sorted_g = sorted(girone, key=lambda t: (
            standings.get(t, {"Pts":0})["Pts"],
            standings.get(t, {"GD":0})["GD"],
            standings.get(t, {"GF":0})["GF"],
        ), reverse=True)

        for rank, team in enumerate(sorted_g):
            s       = standings.get(team, {"P":0,"Pts":0,"GD":0})
            played  = s["P"]
            remaining = matches_per_team - played
            pts     = s["Pts"]
            max_pts = pts + remaining * 3

            pts_2nd = standings.get(sorted_g[1], {"Pts":0})["Pts"] if len(sorted_g) > 1 else 0
            pts_3rd = standings.get(sorted_g[2], {"Pts":0})["Pts"] if len(sorted_g) > 2 else 0
            max_3rd = pts_3rd + (matches_per_team - standings.get(sorted_g[2], {"P":0})["P"]) * 3 if len(sorted_g) > 2 else 0

            if remaining == 0:
                statuses[team] = "qualificata" if rank <= 1 else ("in_gioco" if rank == 2 else "eliminata")
            elif max_pts < pts_2nd:
                statuses[team] = "eliminata"
            elif rank <= 1 and pts > max_3rd:
                statuses[team] = "qualificata"
            else:
                statuses[team] = "in_gioco"

    return {
        "gironi":     gironi,
        "team_girone": team_girone,
        "standings":  standings,
        "statuses":   statuses,
        "wc_tutti":   wc_tutti,
        "n_giocate":  len(wc_giocate),
        "n_programma": len(wc_tutti) - len(wc_giocate),
    }


def stampa_standings(wc_info, elo_ratings):
    gironi    = wc_info["gironi"]
    standings = wc_info["standings"]
    statuses  = wc_info["statuses"]

    print(col("\n" + "═"*68, "cyan"))
    print(col("  CLASSIFICHE MONDIALI 2026  (dati aggiornati dal dataset)", "bold"))
    print(col("  ✓ qualificata   ?  in gioco   ✗ eliminata", "white"))
    print(col("═"*68, "cyan"))

    for i, girone in enumerate(gironi):
        lettera = GRUPPI_LETTERE[i] if i < len(GRUPPI_LETTERE) else str(i+1)
        print(col(f"\n  GRUPPO {lettera}", "yellow"))
        print(col(f"  {'Squadra':<28} {'Pt':>3} {'G':>2} {'V':>2} {'N':>2} {'P':>2} {'GF':>3} {'GS':>3} {'DR':>4}  Elo", "white"))
        print(col(f"  {'─'*66}", "white"))

        sorted_g = sorted(girone, key=lambda t: (
            standings.get(t, {"Pts":0})["Pts"],
            standings.get(t, {"GD":0})["GD"],
            standings.get(t, {"GF":0})["GF"],
        ), reverse=True)

        for team in sorted_g:
            s      = standings.get(team, {"P":0,"W":0,"D":0,"L":0,"GF":0,"GA":0,"GD":0,"Pts":0})
            elo    = elo_ratings.get(team, ELO_INITIAL)
            status = statuses.get(team, "in_gioco")
            badge  = col("✓", "green") if status == "qualificata" else (col("✗", "red") if status == "eliminata" else col("?", "yellow"))
            print(col(f"  {badge} {team:<27} {s['Pts']:>3} {s['P']:>2} {s['W']:>2} {s['D']:>2} {s['L']:>2} "
                      f"{s['GF']:>3} {s['GA']:>3} {s['GD']:>+4}  {elo:.0f}", "white"))


def _cerca_team_wc(nome, wc_info):
    """Cerca il nome esatto della squadra nel dataset WC (fuzzy)."""
    if wc_info is None:
        return nome
    nome_l = nome.lower()
    all_teams = [t for g in wc_info["gironi"] for t in g]
    for t in all_teams:
        if t.lower() == nome_l:
            return t
    for t in all_teams:
        if nome_l in t.lower() or t.lower() in nome_l:
            return t
    return nome


def seleziona_squadra_wc(wc_info, elo_ratings, label="Squadra"):
    """Menu numerato di tutte le squadre WC con stato qualificazione ed Elo."""
    gironi    = wc_info["gironi"]
    standings = wc_info["standings"]
    statuses  = wc_info["statuses"]

    print(col(f"\n  Scegli {label}  (✓=qualificata  ?=in gioco  ✗=eliminata):", "cyan"))

    idx = 1
    num_team = {}

    for i, girone in enumerate(gironi):
        lettera = GRUPPI_LETTERE[i] if i < len(GRUPPI_LETTERE) else str(i+1)
        print(col(f"\n  ── Gruppo {lettera} ──────────────────────────────────────", "yellow"))

        sorted_g = sorted(girone, key=lambda t: (
            standings.get(t, {"Pts":0})["Pts"],
            standings.get(t, {"GD":0})["GD"],
        ), reverse=True)

        for team in sorted_g:
            status = statuses.get(team, "in_gioco")
            elo    = elo_ratings.get(team, ELO_INITIAL)
            pts    = standings.get(team, {"Pts":0})["Pts"]
            played = standings.get(team, {"P":0})["P"]
            badge  = col("✓", "green") if status == "qualificata" else (col("✗", "red") if status == "eliminata" else col("?", "yellow"))
            print(col(f"  {idx:3}. {badge} {team:<28}  Elo {elo:.0f}  {pts}pt/{played}g", "white"))
            num_team[idx] = team
            idx += 1

    print()
    while True:
        try:
            n = int(input(col(f"  Numero {label}: ", "cyan")).strip())
            if n in num_team:
                return num_team[n]
        except ValueError:
            pass
        print(col("  Numero non valido, riprova.", "red"))


# ─── 6. AGGIUSTAMENTO MOTIVAZIONE ────────────────────────────

def aggiusta_motivazione(probs, home_status, away_status):
    """
    Riduce leggermente le probabilità di vittoria per una squadra
    già qualificata che affronta una ancora in gioco.
    Aumenta la probabilità di pareggio se entrambe sono qualificate.
    """
    h, d, a = probs[0], probs[1], probs[2]
    nota = ""

    if home_status == "qualificata" and away_status == "in_gioco":
        h -= 0.07; a += 0.045; d += 0.025
        nota = "⚠ Casa già qualificata → trasferta più motivata (+7%)"
    elif away_status == "qualificata" and home_status == "in_gioco":
        a -= 0.07; h += 0.045; d += 0.025
        nota = "⚠ Trasferta già qualificata → casa più motivata (+7%)"
    elif home_status == "qualificata" and away_status == "qualificata":
        h -= 0.04; a -= 0.04; d += 0.08
        nota = "⚠ Entrambe qualificate → pareggio più probabile (+8%)"
    elif home_status == "eliminata" and away_status == "in_gioco":
        h -= 0.09; a += 0.06; d += 0.03
        nota = "⚠ Casa eliminata → trasferta molto favorita (+9%)"
    elif away_status == "eliminata" and home_status == "in_gioco":
        a -= 0.09; h += 0.06; d += 0.03
        nota = "⚠ Trasferta eliminata → casa molto favorita (+9%)"

    h = max(h, 0.02); d = max(d, 0.02); a = max(a, 0.02)
    tot = h + d + a
    return np.array([h/tot, d/tot, a/tot]), nota


# ─── 7. ODDS API ─────────────────────────────────────────────

def lista_sport_calcio(api_key):
    r = requests.get(f"{ODDS_API_BASE}/sports", params={"apiKey": api_key}, timeout=10)
    r.raise_for_status()
    return [s for s in r.json() if "soccer" in s.get("key", "").lower()]

def fetch_quote(api_key, sport_key="soccer_fifa_world_cup"):
    params = {"apiKey": api_key, "regions": "eu", "markets": "h2h",
              "oddsFormat": "decimal", "dateFormat": "iso"}
    r = requests.get(f"{ODDS_API_BASE}/sports/{sport_key}/odds", params=params, timeout=15)
    if r.status_code == 401:
        print(col("  ✗ API key non valida.", "red")); return []
    if r.status_code == 422:
        print(col(f"  ✗ Sport '{sport_key}' non trovato.", "red")); return []
    r.raise_for_status()
    rem = r.headers.get("x-requests-remaining", "?")
    used = r.headers.get("x-requests-used", "?")
    print(col(f"  ✓ Richieste API: usate {used}  |  rimanenti {rem}", "green"))
    return r.json()

def _norm_probs(raw_odds):
    probs = [1.0 / o for o in raw_odds]
    tot   = sum(probs)
    return [p / tot for p in probs]

def parse_partita_api(entry):
    home = entry["home_team"]
    away = entry["away_team"]
    time = entry.get("commence_time", "")[:16].replace("T", " ")
    for book in entry.get("bookmakers", []):
        h2h = next((m for m in book.get("markets", []) if m["key"] == "h2h"), None)
        if not h2h:
            continue
        om = {o["name"]: o["price"] for o in h2h["outcomes"]}
        oh, oa, od = om.get(home), om.get(away), om.get("Draw")
        if oh and oa and od:
            ph, pd_, pa = _norm_probs([oh, od, oa])
            return {"home_team": home, "away_team": away, "time": time,
                    "oh": oh, "od": od, "oa": oa,
                    "ph": ph, "pd": pd_, "pa": pa, "bookmaker": book["title"]}
    return None


# ─── 8. PREVISIONE ────────────────────────────────────────────

def _cerca_elo(team, ratings):
    tl = team.lower()
    if team in ratings: return ratings[team]
    for k, v in ratings.items():
        if k.lower() == tl: return v
    for k, v in ratings.items():
        if tl in k.lower() or k.lower() in tl: return v
    return ELO_INITIAL

def predici_partita(home, away, model, le, elo_ratings, team_hist,
                    ph=-1, pd_=-1, pa=-1, neutral=True,
                    home_status="in_gioco", away_status="in_gioco",
                    h2h_hist=None):
    elo_h = _cerca_elo(home, elo_ratings)
    elo_a = _cerca_elo(away, elo_ratings)
    fh    = _stats(team_hist.get(home))
    fa    = _stats(team_hist.get(away))
    h2h   = _h2h_stats(home, away, h2h_hist)

    feat = {
        "elo_home":             elo_h, "elo_away":   elo_a, "elo_diff":  elo_h - elo_a,
        "h_win_rate":           fh["win_rate"],   "h_draw_rate":  fh["draw_rate"],
        "h_gol_fatti":          fh["gol_fatti"],  "h_gol_subiti": fh["gol_subiti"], "h_elo_avv": fh["elo_avv"],
        "a_win_rate":           fa["win_rate"],   "a_draw_rate":  fa["draw_rate"],
        "a_gol_fatti":          fa["gol_fatti"],  "a_gol_subiti": fa["gol_subiti"], "a_elo_avv": fa["elo_avv"],
        "diff_win_rate":        fh["win_rate"]   - fa["win_rate"],
        "diff_gol_fatti":       fh["gol_fatti"]  - fa["gol_fatti"],
        "diff_gol_subiti":      fh["gol_subiti"] - fa["gol_subiti"],
        "neutral":              int(neutral),
        "h2h_home_winrate":     h2h["h2h_home_winrate"],
        "h2h_goal_diff":        h2h["h2h_goal_diff"],
        "h_tournament_weight":  fh.get("tournament_weight", 1.0),
        "a_tournament_weight":  fa.get("tournament_weight", 1.0),
    }

    X     = pd.DataFrame([feat])[FEATURE_COLS]
    raw_p = model.predict_proba(X)[0]
    cls   = list(le.classes_)
    model_probs = np.array([raw_p[cls.index("H")], raw_p[cls.index("D")], raw_p[cls.index("A")]])

    has_odds = (ph > 0 and pd_ > 0 and pa > 0)
    w = _blend_weight(elo_h, elo_a)
    if has_odds:
        odds_probs = np.array([ph, pd_, pa])
        blended    = w * odds_probs + (1 - w) * model_probs
        blended   /= blended.sum()
    else:
        blended = model_probs
        w = 0.0

    blended, nota = aggiusta_motivazione(blended, home_status, away_status)
    pred_label = ["H", "D", "A"][int(np.argmax(blended))]
    return blended, pred_label, fh, fa, elo_h, elo_a, nota, w


# ─── 9. OUTPUT ────────────────────────────────────────────────

def stampa_previsione(home, away, probs, pred, fh, fa, elo_h, elo_a,
                      oh=None, od=None, oa=None, has_odds=False,
                      nota_motivazione="",
                      home_status="in_gioco", away_status="in_gioco",
                      gol_info=None, card_info=None, w_blend=0.0):
    W = 54
    prob_H, prob_D, prob_A = probs[0], probs[1], probs[2]
    etiq = {"H": f"Vince {home}", "D": "Pareggio", "A": f"Vince {away}"}

    def badge(s):
        return col("✓", "green") if s=="qualificata" else (col("✗","red") if s=="eliminata" else col("?","yellow"))

    print(col(f"\n  ┌{'─'*W}┐", "cyan"))
    print(col(f"  │  {home:^{W-2}}│", "bold"))
    print(col(f"  │  {'vs':^{W-2}}│", "white"))
    print(col(f"  │  {away:^{W-2}}│", "bold"))
    print(col(f"  ├{'─'*W}┤", "cyan"))
    print(col(f"  │  Elo: {home} {elo_h:.0f}  vs  {away} {elo_a:.0f}  (diff: {elo_h-elo_a:+.0f})", "white"))
    stato_riga = f"  │  Stato: {badge(home_status)} {home}  /  {badge(away_status)} {away}"
    print(stato_riga)
    print(col(f"  ├{'─'*W}┤", "cyan"))

    for lbl, prob in [("H", prob_H), ("D", prob_D), ("A", prob_A)]:
        label   = etiq[lbl]
        barra   = "█" * int(prob * 32)
        colore  = "green" if lbl == pred else "white"
        marker  = " ◀" if lbl == pred else ""
        print(col(f"  │  {label:<30} {prob*100:5.1f}%  {barra}{marker}", colore))

    if has_odds and oh and od and oa:
        print(col(f"  ├{'─'*W}┤", "cyan"))
        print(col(f"  │  Quote: @{oh:.2f} / @{od:.2f} / @{oa:.2f}  "
                  f"→ blend dinamico: modello {(1-w_blend)*100:.0f}% + mercato {w_blend*100:.0f}%", "yellow"))

    if nota_motivazione:
        print(col(f"  ├{'─'*W}┤", "cyan"))
        print(col(f"  │  {nota_motivazione}", "magenta"))

    # Sezione gol
    if gol_info:
        lam_h      = gol_info["lam_h"]
        lam_a      = gol_info["lam_a"]
        top_scores = gol_info["top_scores"]
        over25     = gol_info["over25"]
        btts       = gol_info["btts"]
        best_score = top_scores[0][0]

        print(col(f"  ├{'─'*W}┤", "cyan"))
        print(col(f"  │  ⚽ Gol attesi:  {home} {lam_h:.1f}  –  {lam_a:.1f}  {away}", "yellow"))
        print(col(f"  │  Risultato più probabile:  {best_score[0]}-{best_score[1]}  "
                  f"({top_scores[0][1]*100:.1f}%)", "yellow"))
        print(col(f"  │", "cyan"))
        print(col(f"  │  Top risultati esatti:", "white"))
        for (gh, ga), prob in top_scores[:6]:
            barra  = "█" * int(prob * 120)
            lbl_ris = "V" if gh > ga else ("X" if gh == ga else "2")
            clr    = "green" if (gh, ga) == best_score else "white"
            print(col(f"  │    {gh}-{ga} [{lbl_ris}]   {prob*100:5.1f}%  {barra}", clr))
        print(col(f"  │", "cyan"))
        print(col(f"  │  Over 2.5: {over25*100:.1f}%   |   "
                  f"BTTS (entrambe a segno): {btts*100:.1f}%", "white"))

    # Sezione cartellini
    if card_info:
        ref_key    = card_info.get("ref_key", "")
        ref_info   = card_info.get("ref_info", {})
        ref_factor = card_info.get("ref_factor", 1.0)
        ref_pct    = (ref_factor - 1.0) * 100
        ref_sign   = "▲" if ref_pct >= 0 else "▼"

        print(col(f"  ├{'─'*W}┤", "cyan"))
        if ref_key and ref_info and ref_info.get("matches", 0) > 0:
            print(col(f"  │  Arbitro: {ref_key}  "
                      f"({ref_info['matches']}p, "
                      f"{card_info['ref_total_avg']:.1f} gialli/p → "
                      f"fattore {ref_factor:.2f}x {ref_sign}{abs(ref_pct):.0f}%)", "magenta"))
        elif ref_key:
            print(col(f"  │  Arbitro: {ref_key}  "
                      f"(no storico → fattore {ref_factor:.2f}x)", "magenta"))
        else:
            print(col(f"  │  Arbitro: n.d.  (fattore neutro 1.00x)", "white"))

        lam_yh_team = card_info["lam_yh_team"]
        lam_ya_team = card_info["lam_ya_team"]
        lam_yh      = card_info["lam_yh"]
        lam_ya      = card_info["lam_ya"]
        print(col(f"  │  🟨 {home}: {lam_yh_team:.1f} × {ref_factor:.2f} = {lam_yh:.1f} gialli", "yellow"))
        print(col(f"  │  🟨 {away}: {lam_ya_team:.1f} × {ref_factor:.2f} = {lam_ya:.1f} gialli", "yellow"))
        print(col(f"  │  Totale: {card_info['lam_tot']:.1f}"
                  f"  |  Over 2.5: {card_info['over_2_5']*100:.0f}%"
                  f"  Over 3.5: {card_info['over_3_5']*100:.0f}%"
                  f"  Over 4.5: {card_info['over_4_5']*100:.0f}%", "white"))

    print(col(f"  ├{'─'*W}┤", "cyan"))
    print(col(f"  │  ▶  Previsione: {etiq[pred]}", "yellow"))
    print(col(f"  └{'─'*W}┘", "cyan"))

    print(col(f"\n  Forma {home} (ult.30): {fh['win_rate']*100:.0f}%V  "
              f"{fh['gol_fatti']:.1f} gol/p segnati  {fh['gol_subiti']:.1f} subiti", "white"))
    print(col(f"  Forma {away} (ult.30): {fa['win_rate']*100:.0f}%V  "
              f"{fa['gol_fatti']:.1f} gol/p segnati  {fa['gol_subiti']:.1f} subiti\n", "white"))


# ─── 10. MAIN ─────────────────────────────────────────────────

def scarica_stats_arbitri_club():
    """
    Scarica cartellini per arbitro dai 5 grandi campionati europei
    (football-data.co.uk, partite ufficiali — nessuna amichevole).
    Cache locale 7 giorni.
    """
    import time as _time

    if os.path.exists(REF_STATS_CACHE):
        age_days = (_time.time() - os.path.getmtime(REF_STATS_CACHE)) / 86400
        if age_days < 7:
            df = pd.read_csv(REF_STATS_CACHE)
            stats = {row["referee"]: {"total_yellow": int(row["total_yellow"]),
                                      "matches": int(row["matches"])}
                     for _, row in df.iterrows()}
            print(col(f"  ✓ Stats arbitri club da cache: {len(stats)} arbitri", "green"))
            return stats

    print(col("\n👤 Scaricando stats arbitri club (football-data.co.uk)...", "cyan"))
    ref_acc = {}
    tot_matches = 0

    for season in FDCO_SEASONS:
        for league in FDCO_LEAGUES:
            url = f"{FDCO_BASE}/{season}/{league}.csv"
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                df = pd.read_csv(StringIO(r.text), on_bad_lines="skip")
                if "Referee" not in df.columns:
                    continue
                for _, row in df.iterrows():
                    ref = str(row.get("Referee", "")).strip()
                    if not ref or ref.lower() == "nan":
                        continue
                    hy = pd.to_numeric(row.get("HY", 0), errors="coerce")
                    ay = pd.to_numeric(row.get("AY", 0), errors="coerce")
                    if pd.isna(hy) or pd.isna(ay):
                        continue
                    ref_acc.setdefault(ref, []).append(int(hy) + int(ay))
                    tot_matches += 1
            except Exception:
                pass

    stats = {r: {"total_yellow": sum(v), "matches": len(v)}
             for r, v in ref_acc.items() if len(v) >= 5}

    pd.DataFrame([{"referee": r, "total_yellow": s["total_yellow"], "matches": s["matches"]}
                  for r, s in stats.items()]).to_csv(REF_STATS_CACHE, index=False)
    print(col(f"  ✓ {len(stats)} arbitri  |  {tot_matches:,} partite di club scaricate", "green"))
    return stats


def _mergia_ref_stats(stats_intl, stats_club):
    """
    Unisce statistiche da tornei internazionali (StatsBomb) e club
    (football-data.co.uk). Tenta il merge per cognome + iniziale quando
    i nomi differiscono (es. 'M Oliver' → 'Michael Oliver').
    """
    merged = {k: dict(v) for k, v in stats_intl.items()}

    for ref_c, s_c in stats_club.items():
        if ref_c in merged:
            merged[ref_c]["total_yellow"] += s_c["total_yellow"]
            merged[ref_c]["matches"]       += s_c["matches"]
            continue

        parts_c = ref_c.lower().split()
        last_c  = parts_c[-1] if parts_c else ""
        init_c  = parts_c[0][0] if parts_c and parts_c[0] else ""

        best = None
        for ref_i in merged:
            parts_i = ref_i.lower().split()
            last_i  = parts_i[-1] if parts_i else ""
            init_i  = parts_i[0][0] if parts_i and parts_i[0] else ""
            if last_c == last_i and init_c == init_i:
                best = ref_i
                break

        if best:
            merged[best]["total_yellow"] += s_c["total_yellow"]
            merged[best]["matches"]       += s_c["matches"]
        else:
            merged[ref_c] = dict(s_c)

    return merged


def _mostra_arbitri(ref_stats):
    """Stampa lista arbitri con statistiche, ordinata per partite dirette."""
    if not ref_stats:
        print(col("  Nessun dato arbitri disponibile.", "yellow"))
        return
    print(col("\n  Arbitri nel database (ordinati per partite):", "cyan"))
    for ref, s in sorted(ref_stats.items(), key=lambda x: -x[1]["matches"]):
        avg = s["total_yellow"] / s["matches"]
        print(col(f"  {ref:<36} {s['matches']:>3}p   {avg:.1f} gialli/p", "white"))
    print()


# ─── 11. ANALISI SCOMMESSE ─────────────────────────────────────

def _kelly(prob, quota):
    """Frazione Kelly ottimale (0 se EV negativo)."""
    edge = prob * quota - 1.0
    if edge <= 0 or quota <= 1:
        return 0.0
    return edge / (quota - 1.0)


def _stampa_distribuzione_bet(profits, zero_line):
    """ASCII histogram della distribuzione profitti."""
    bins = 12
    counts, edges = np.histogram(profits, bins=bins)
    max_c = max(counts)
    bar_w = 22
    print(col("\n  Distribuzione profitti:", "cyan"))
    for c, e in zip(counts, edges):
        bar = "█" * int(c / max_c * bar_w) if max_c else ""
        color = "green" if e >= 0 else "red"
        print(col(f"  {e:>+8.2f}€  {bar}", color))
    print()


_MERCATI_VALIDI = (
    "1","X","2","1X","X2","12",
    "GG","NG",
    "O1.5","O2.5","O3.5","O4.5",
    "U1.5","U2.5","U3.5",
    "CO2.5","CO3.5","CO4.5",
    "CU2.5","CU3.5","CU4.5",
)

def _p_over_pois_term(lam: float, k: float) -> float:
    import math
    k_int = int(k)
    return 1.0 - sum(
        math.exp(-lam) * (lam ** i) / math.factorial(i)
        for i in range(k_int + 1)
    )

def _prob_per_sel(sel: str, ph: float, pd: float, pa: float,
                  lam_gol: float, btts: float, lam_cards: float) -> float:
    s = sel.upper().replace(" ", "")
    if s == "1":  return ph
    if s == "X":  return pd
    if s == "2":  return pa
    if s == "1X": return min(ph + pd, 1.0)
    if s == "X2": return min(pd + pa, 1.0)
    if s == "12": return min(ph + pa, 1.0)
    if s == "GG": return btts
    if s == "NG": return max(1.0 - btts, 0.0)
    if s.startswith("O") and not s.startswith("OC"):
        return _p_over_pois_term(lam_gol, float(s[1:]))
    if s.startswith("U") and not s.startswith("UC"):
        return max(1.0 - _p_over_pois_term(lam_gol, float(s[1:])), 0.0)
    if s.startswith("CO"):
        return _p_over_pois_term(lam_cards, float(s[2:]))
    if s.startswith("CU"):
        return max(1.0 - _p_over_pois_term(lam_cards, float(s[2:])), 0.0)
    raise ValueError(f"Mercato non riconosciuto: '{sel}'")


def analizza_scommessa(model, le, elo_ratings, team_hist, h2h_hist,
                       wc_info, model_gh=None, model_ga=None,
                       model_yh=None, model_ya=None,
                       disc_hist=None, ref_stats=None,
                       n_sim=100_000):
    """Modalità --bet: input interattivo + EV + Monte Carlo."""
    from itertools import combinations as _combs

    print(col("\n" + "═" * 62, "cyan"))
    print(col("  🎰  ANALISI SCOMMESSE  —  EV + Monte Carlo", "bold"))
    print(col("═" * 62, "cyan"))

    _merc_str = "1/X/2/1X/X2/12/GG/NG/O2.5/U2.5/CO3.5/CU3.5 …"

    # ── input partite ───────────────────────────────────────────
    try:
        n_partite = int(input(col("\n  Quante partite nella scommessa? ", "cyan")).strip())
        if not 1 <= n_partite <= 12:
            raise ValueError
    except ValueError:
        print(col("  Numero non valido (1-12).", "red")); return

    partite = []
    for i in range(n_partite):
        print(col(f"\n  ── Partita {i + 1} ──────────────────────────────────", "cyan"))
        home = input(col("  Squadra casa:       ", "cyan")).strip()
        away = input(col("  Squadra trasferta:  ", "cyan")).strip()
        if not home or not away:
            print(col("  Nomi mancanti.", "red")); return

        try:
            probs_raw, _, fh, fa, elo_h, elo_a, nota, w = predici_partita(
                home, away, model, le, elo_ratings, team_hist,
                neutral=True, h2h_hist=h2h_hist,
            )
        except Exception as e:
            print(col(f"  Errore previsione: {e}", "red")); return

        ph  = float(probs_raw[0])
        pd_ = float(probs_raw[1])
        pa  = float(probs_raw[2])

        print(col(f"  Elo: {round(elo_h)} — {round(elo_a)}", "white"))
        print(col(f"  Esito →  1:{ph*100:.0f}%  X:{pd_*100:.0f}%  2:{pa*100:.0f}%", "white"))

        # Gol Poisson
        lam_gol = 0.0; btts = 0.0
        if model_gh is not None:
            try:
                gol = predici_gol_poisson(
                    home, away, model_gh, model_ga,
                    elo_ratings, team_hist, neutral=True, h2h_hist=h2h_hist,
                )
                lam_gol = float(gol["lam_h"]) + float(gol["lam_a"])
                btts    = float(gol["btts"])
                print(col(
                    f"  Gol  →  λ={lam_gol:.2f}  "
                    f"O2.5:{_p_over_pois_term(lam_gol,2)*100:.0f}%  "
                    f"GG:{btts*100:.0f}%",
                    "white",
                ))
            except Exception:
                pass

        # Cartellini Poisson
        lam_cards = 0.0
        if model_yh is not None:
            try:
                cards = predici_cartellini_poisson(
                    home, away, model_yh, model_ya,
                    elo_ratings, disc_hist or {},
                    ref_stats=ref_stats,
                )
                lam_cards = float(cards["lam_tot"])
                print(col(
                    f"  Card →  λ={lam_cards:.2f}  "
                    f"CO3.5:{_p_over_pois_term(lam_cards,3)*100:.0f}%  "
                    f"CO4.5:{_p_over_pois_term(lam_cards,4)*100:.0f}%",
                    "white",
                ))
            except Exception:
                pass

        if nota:
            print(col(f"  ⚠ {nota}", "yellow"))

        sel = input(col(f"  Mercato ({_merc_str}):  ", "cyan")).strip().upper().replace(" ", "")
        try:
            prob_sel = _prob_per_sel(sel, ph, pd_, pa, lam_gol, btts, lam_cards)
        except ValueError as e:
            print(col(f"  {e}  — uso '1'.", "yellow"))
            sel = "1"; prob_sel = ph

        try:
            quota = float(input(col("  Quota bookmaker:    ", "cyan")).strip().replace(",", "."))
            if quota <= 1.0:
                raise ValueError
        except ValueError:
            print(col("  Quota non valida.", "red")); return

        prob_imp  = 1.0 / quota
        edge_pct  = (prob_sel - prob_imp) * 100
        kelly_f   = _kelly(prob_sel, quota)
        edge_col  = "green" if edge_pct > 0 else "red"

        print(col(
            f"  Modello {prob_sel*100:.1f}%  |  Impl. {prob_imp*100:.1f}%  |  "
            f"Edge {edge_pct:+.1f}%  |  Kelly {kelly_f*100:.1f}%",
            edge_col,
        ))

        partite.append({
            "label":    f"{home} vs {away} [{sel}@{quota:.2f}]",
            "home":     home, "away": away,
            "selezione": sel, "quota": quota,
            "prob":     prob_sel, "prob_imp": prob_imp,
            "edge":     edge_pct, "kelly": kelly_f,
        })

    # ── tipo scommessa ──────────────────────────────────────────
    M = len(partite)
    print(col(f"\n  ── Tipo scommessa {'─'*38}", "cyan"))
    print(col("  1) Multipla (tutte le selezioni)", "white"))
    if M > 1:
        print(col("  2) Sistema (N su M — tutte le combinazioni di N)", "white"))
    tipo_s = input(col("  Scelta (1/2): ", "cyan")).strip()

    tipo  = "sistema" if tipo_s == "2" and M > 1 else "multipla"
    n_min = M

    if tipo == "sistema":
        try:
            n_min = int(input(col(f"  Minimo vincenti su {M} (es. {M-1}): ", "cyan")).strip())
            if not 1 <= n_min <= M:
                raise ValueError
        except ValueError:
            print(col("  Valore non valido, uso 1.", "yellow")); n_min = 1

    try:
        stake = float(input(col("  Puntata per combinazione €: ", "cyan")).strip().replace(",", "."))
    except ValueError:
        stake = 10.0

    # ── calcolo ─────────────────────────────────────────────────
    probs_arr = np.array([p["prob"] for p in partite])
    quote_arr = np.array([p["quota"] for p in partite])

    combos     = list(_combs(range(M), n_min))
    n_combos   = len(combos)
    total_stake = stake * n_combos

    # EV analitico
    ev_analitico = 0.0
    for combo in combos:
        p_c = np.prod(probs_arr[list(combo)])
        q_c = np.prod(quote_arr[list(combo)])
        ev_analitico += p_c * stake * q_c
    ev_analitico -= total_stake
    roi_analitico = ev_analitico / total_stake * 100

    # Monte Carlo
    rng = np.random.default_rng()
    draws = rng.random((n_sim, M))
    wins_m = draws < probs_arr  # (n_sim, M)

    payouts = np.zeros(n_sim)
    for combo in combos:
        idx = list(combo)
        combo_wins = wins_m[:, idx].all(axis=1)
        q_c = float(np.prod(quote_arr[idx]))
        payouts += combo_wins * stake * q_c

    profits = payouts - total_stake
    ev_mc   = float(profits.mean())
    p_prof  = float((profits > 0).mean() * 100)
    p5, p25, p75, p95 = (float(x) for x in np.percentile(profits, [5, 25, 75, 95]))

    # ── stampa risultati ─────────────────────────────────────────
    W = 60
    print(col(f"\n  ╔{'═'*W}╗", "cyan"))
    label_tipo = f"Multipla {M}/{M}" if tipo == "multipla" else f"Sistema {n_min}/{M}"
    print(col(f"  ║  {label_tipo}  —  {n_combos} comb. × {stake:.2f}€  =  {total_stake:.2f}€ totali", "bold"))
    print(col(f"  ╠{'═'*W}╣", "cyan"))

    # Riepilogo partite
    for p in partite:
        edge_c = "green" if p["edge"] > 0 else "red"
        print(col(
            f"  ║  {p['label']:<38}  "
            f"Edge {p['edge']:+.1f}%  Kelly {p['kelly']*100:.1f}%",
            edge_c,
        ))

    print(col(f"  ╠{'═'*W}╣", "cyan"))

    # EV
    ev_col = "green" if ev_analitico > 0 else "red"
    print(col(f"  ║  EV analitico: {ev_analitico:>+8.2f}€   ROI: {roi_analitico:>+6.1f}%", ev_col))
    p_prof_col = "green" if p_prof > 40 else "yellow" if p_prof > 20 else "red"
    print(col(f"  ║  P(profitto):  {p_prof:>7.1f}%   EV Monte Carlo: {ev_mc:>+.2f}€", p_prof_col))
    print(col(f"  ║  Perc.  5°: {p5:>+7.2f}€   25°: {p25:>+7.2f}€   75°: {p75:>+7.2f}€   95°: {p95:>+7.2f}€", "white"))

    # Kelly sul sistema/multipla complessiva
    p_win_tot = np.prod(probs_arr) if tipo == "multipla" else float((profits > 0).mean())
    q_eff     = total_stake / stake if n_combos == 1 else (ev_analitico + total_stake) / (p_win_tot * total_stake + 1e-9)
    kelly_sis = _kelly(p_win_tot, q_eff) if tipo == "multipla" else 0.0
    if tipo == "multipla" and kelly_sis > 0:
        print(col(f"  ║  Kelly sull'intera multipla: {kelly_sis*100:.1f}%", "white"))

    print(col(f"  ╚{'═'*W}╝", "cyan"))

    _stampa_distribuzione_bet(profits, 0)

    # Consiglio
    if ev_analitico > 0:
        print(col("  ✓ EV positivo: il modello vede valore in questa scommessa.", "green"))
    else:
        print(col("  ✗ EV negativo: il bookmaker ha un vantaggio strutturale su questa scommessa.", "red"))
    print()


def main():
    parser = argparse.ArgumentParser(description="Predittore Mondiali 2026")
    parser.add_argument("--predict",     action="store_true", help="Menu squadre WC per previsione")
    parser.add_argument("--live",        action="store_true", help="Quote live API")
    parser.add_argument("--standings",   action="store_true", help="Mostra classifiche gironi")
    parser.add_argument("--bet",         action="store_true", help="Analisi scommesse: EV + Monte Carlo")
    parser.add_argument("--sport-key",   default="soccer_fifa_world_cup")
    parser.add_argument("--list-sports", action="store_true")
    args = parser.parse_args()

    print(col("=" * 62, "cyan"))
    print(col("  PREDITTORE MONDIALI 2026", "bold"))
    print(col("  Elo + Forma + XGBoost + Quote + Gol + Cartellini", "white"))
    print(col("=" * 62, "cyan"))

    df_raw                        = scarica_risultati()
    df, elo_ratings               = calcola_elo(df_raw)
    feat_df, team_hist, h2h_hist  = costruisci_features(df)
    model, le           = allena(feat_df)
    model_gh, model_ga  = allena_gol(feat_df)

    try:
        df_cards                        = scarica_cartellini_statsbomb()
        feat_disc, disc_hist, ref_stats = costruisci_features_disciplina(df_cards, elo_ratings)
        model_yh, model_ya              = allena_cartellini(feat_disc)
        _has_cards                      = True
    except Exception as _e:
        print(col(f"\n  ⚠ Modello cartellini non disponibile: {_e}", "yellow"))
        disc_hist, ref_stats, model_yh, model_ya = {}, {}, None, None
        _has_cards = False

    # Arricchisce ref_stats con tutta la carriera club (esclude amichevoli)
    try:
        ref_stats_club = scarica_stats_arbitri_club()
        ref_stats      = _mergia_ref_stats(ref_stats, ref_stats_club)
        print(col(f"  ✓ Database arbitri: {len(ref_stats)} arbitri (tornei + club)", "green"))
    except Exception as _e:
        print(col(f"  ⚠ Stats club non disponibili: {_e}", "yellow"))

    wc_info             = analizza_wc2026(df)

    # ── --standings ────────────────────────────────────────────
    if args.standings:
        if wc_info:
            stampa_standings(wc_info, elo_ratings)
        else:
            print(col("  Nessun dato WC 2026 disponibile nel dataset.", "yellow"))
        return

    # ── --bet ──────────────────────────────────────────────────
    if args.bet:
        while True:
            analizza_scommessa(
                model, le, elo_ratings, team_hist, h2h_hist, wc_info,
                model_gh=model_gh, model_ga=model_ga,
                model_yh=model_yh, model_ya=model_ya,
                disc_hist=disc_hist, ref_stats=ref_stats,
            )
            if input(col("  Analizzare un'altra scommessa? (s/n): ", "cyan")).strip().lower() != "s":
                break
        return

    # ── --list-sports ──────────────────────────────────────────
    if args.list_sports:
        api_key = ODDS_API_KEY or input(col("  API key: ", "cyan")).strip()
        sports  = lista_sport_calcio(api_key)
        for s in sports:
            stato = "✓ attivo" if s.get("active") else "  inattivo"
            print(col(f"    {stato}  {s['key']:<45} {s.get('title','')}", "white"))
        return

    # ── --live ─────────────────────────────────────────────────
    if args.live:
        api_key = ODDS_API_KEY or input(col("  API key (the-odds-api.com): ", "cyan")).strip()
        if not api_key:
            print(col("  API key mancante.", "red")); sys.exit(1)

        print(col(f"\n📡 Scaricando quote per '{args.sport_key}'...", "cyan"))
        partite_api = fetch_quote(api_key, args.sport_key)

        if not partite_api:
            print(col("  Nessuna partita trovata. Usa --list-sports per vedere le chiavi.", "yellow"))
            return

        print(col(f"\n  {len(partite_api)} partita/e in programma:\n", "bold"))
        for entry in partite_api:
            parsed = parse_partita_api(entry)
            if parsed is None:
                continue
            home, away = parsed["home_team"], parsed["away_team"]

            # Stato qualificazione per aggiustamento motivazione
            if wc_info:
                h_key = _cerca_team_wc(home, wc_info)
                a_key = _cerca_team_wc(away, wc_info)
                hs = wc_info["statuses"].get(h_key, "in_gioco")
                as_ = wc_info["statuses"].get(a_key, "in_gioco")
            else:
                hs = as_ = "in_gioco"

            print(col(f"  ⏰ {parsed['time']} UTC  |  {parsed['bookmaker']}", "yellow"))
            probs, pred, fh, fa, elo_h, elo_a, nota, w = predici_partita(
                home, away, model, le, elo_ratings, team_hist,
                ph=parsed["ph"], pd_=parsed["pd"], pa=parsed["pa"],
                neutral=True, home_status=hs, away_status=as_,
                h2h_hist=h2h_hist,
            )
            gol_info = predici_gol_poisson(
                home, away, model_gh, model_ga, elo_ratings, team_hist,
                neutral=True, home_status=hs, away_status=as_,
                h2h_hist=h2h_hist,
            )
            card_info = predici_cartellini_poisson(
                home, away, model_yh, model_ya,
                elo_ratings, disc_hist,
                is_knockout=False, home_status=hs, away_status=as_,
                referee=None, ref_stats=ref_stats,
            ) if _has_cards else None
            stampa_previsione(
                home, away, probs, pred, fh, fa, elo_h, elo_a,
                oh=parsed["oh"], od=parsed["od"], oa=parsed["oa"],
                has_odds=True, nota_motivazione=nota,
                home_status=hs, away_status=as_,
                gol_info=gol_info, card_info=card_info, w_blend=w,
            )
        return

    # ── --predict ──────────────────────────────────────────────
    if args.predict:
        if wc_info:
            print()
            stampa_standings(wc_info, elo_ratings)

        while True:
            print(col("\n" + "─"*62, "cyan"))

            if wc_info:
                home = seleziona_squadra_wc(wc_info, elo_ratings, "Squadra 1")
                away = seleziona_squadra_wc(wc_info, elo_ratings, "Squadra 2")
                hs   = wc_info["statuses"].get(home, "in_gioco")
                as_  = wc_info["statuses"].get(away, "in_gioco")
            else:
                print(col("  Nomi in inglese (es: Brazil, France, Argentina...)", "white"))
                home = input(col("  Squadra 1: ", "cyan")).strip()
                away = input(col("  Squadra 2: ", "cyan")).strip()
                hs = as_ = "in_gioco"

            if not home or not away:
                break

            print(col("\n  Quote decimali bookmaker (facoltative — INVIO per saltare):", "white"))
            oh_s = input(col(f"  Quota vittoria {home} (es: 2.10): ", "cyan")).strip()
            od_s = input(col("  Quota pareggio: ", "cyan")).strip()
            oa_s = input(col(f"  Quota vittoria {away}: ", "cyan")).strip()

            try:
                oh  = float(oh_s.replace(",", ".")) if oh_s else None
                od  = float(od_s.replace(",", ".")) if od_s else None
                oa  = float(oa_s.replace(",", ".")) if oa_s else None
                if oh and od and oa:
                    ph, pd_, pa = _norm_probs([oh, od, oa])
                    has_odds = True
                else:
                    ph = pd_ = pa = -1.0; has_odds = False; oh = od = oa = None
            except ValueError:
                ph = pd_ = pa = -1.0; has_odds = False; oh = od = oa = None

            # Input arbitro
            arbitro = None
            if _has_cards and ref_stats:
                print(col("\n  Arbitro (INVIO=salta  'list'=vedi tutti):", "white"))
                arb_inp = input(col("  Nome arbitro: ", "cyan")).strip()
                if arb_inp.lower() == "list":
                    _mostra_arbitri(ref_stats)
                    arb_inp = input(col("  Nome arbitro: ", "cyan")).strip()
                arbitro = arb_inp if arb_inp else None

            probs, pred, fh, fa, elo_h, elo_a, nota, w = predici_partita(
                home, away, model, le, elo_ratings, team_hist,
                ph=ph, pd_=pd_, pa=pa, neutral=True,
                home_status=hs, away_status=as_,
                h2h_hist=h2h_hist,
            )
            gol_info = predici_gol_poisson(
                home, away, model_gh, model_ga, elo_ratings, team_hist,
                neutral=True, home_status=hs, away_status=as_,
                h2h_hist=h2h_hist,
            )
            card_info = predici_cartellini_poisson(
                home, away, model_yh, model_ya,
                elo_ratings, disc_hist,
                is_knockout=False, home_status=hs, away_status=as_,
                referee=arbitro, ref_stats=ref_stats,
            ) if _has_cards else None
            stampa_previsione(
                home, away, probs, pred, fh, fa, elo_h, elo_a,
                oh=oh, od=od, oa=oa, has_odds=has_odds,
                nota_motivazione=nota, home_status=hs, away_status=as_,
                gol_info=gol_info, card_info=card_info, w_blend=w,
            )

            if input(col("  Altra previsione? (s/n): ", "cyan")).strip().lower() != "s":
                break
        return

    # ── Default ────────────────────────────────────────────────
    if wc_info:
        stampa_standings(wc_info, elo_ratings)
    print(col("\n💡 Comandi:", "yellow"))
    print(col("   --predict     Menu squadre WC → previsione con quote", "white"))
    print(col("   --live        Quote live API (partite in programma)", "white"))
    print(col("   --standings   Classifiche gironi aggiornate", "white"))
    print(col("   --bet         Analisi scommesse: EV + Monte Carlo", "white"))
    print(col("   --list-sports Lista sport disponibili nell'API\n", "white"))


if __name__ == "__main__":
    main()
