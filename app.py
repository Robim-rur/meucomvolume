import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz

st.set_page_config(layout="wide")
st.title("Scanner – Setup Roberson (Diário + Semanal + Estatística)")

# =========================================================
# LISTA FIXA DE ATIVOS
# =========================================================

ativos_scan = sorted(set([
"RRRP3.SA","ALOS3.SA","ALPA4.SA","ABEV3.SA","ARZZ3.SA","ASAI3.SA","AZUL4.SA","B3SA3.SA","BBAS3.SA","BBDC3.SA",
"BBDC4.SA","BBSE3.SA","BEEF3.SA","BPAC11.SA","BRAP4.SA","BRFS3.SA","BRKM5.SA","CCRO3.SA","CMIG4.SA","CMIN3.SA",
"COGN3.SA","CPFE3.SA","CPLE6.SA","CRFB3.SA","CSAN3.SA","CSNA3.SA","CYRE3.SA","DXCO3.SA","EGIE3.SA","ELET3.SA",
"ELET6.SA","EMBR3.SA","ENEV3.SA","ENGI11.SA","EQTL3.SA","EZTC3.SA","FLRY3.SA","GGBR4.SA","GOAU4.SA","GOLL4.SA",
"HAPV3.SA","HYPE3.SA","ITSA4.SA","ITUB4.SA","JBSS3.SA","KLBN11.SA","LREN3.SA","LWSA3.SA","MGLU3.SA","MRFG3.SA",
"MRVE3.SA","MULT3.SA","NTCO3.SA","PETR3.SA","PETR4.SA","PRIO3.SA","RADL3.SA","RAIL3.SA","RAIZ4.SA","RENT3.SA",
"RECV3.SA","SANB11.SA","SBSP3.SA","SLCE3.SA","SMTO3.SA","SUZB3.SA","TAEE11.SA","TIMS3.SA","TTEN3.SA","TOTS3.SA","TRPL4.SA",
"UGPA3.SA","USIM5.SA","VALE3.SA","VIVT3.SA","VIVA3.SA","WEGE3.SA","YDUQ3.SA","AURE3.SA","BHIA3.SA","CASH3.SA",
"CVCB3.SA","DIRR3.SA","ENAT3.SA","GMAT3.SA","IFCM3.SA","INTB3.SA","JHSF3.SA","KEPL3.SA","MOVI3.SA","ORVR3.SA",
"PETZ3.SA","PLAS3.SA","POMO4.SA","POSI3.SA","RANI3.SA","RAPT4.SA","STBP3.SA","TEND3.SA","TUPY3.SA",
"BRSR6.SA","CXSE3.SA","AAPL34.SA","AMZO34.SA","GOGL34.SA","MSFT34.SA","TSLA34.SA","META34.SA","NFLX34.SA",
"NVDC34.SA","MELI34.SA","BABA34.SA","DISB34.SA","PYPL34.SA","JNJB34.SA","PGCO34.SA","KOCH34.SA","VISA34.SA",
"WMTB34.SA","NIKE34.SA","ADBE34.SA","AVGO34.SA","CSCO34.SA","COST34.SA","CVSH34.SA","GECO34.SA","GSGI34.SA",
"HDCO34.SA","INTC34.SA","JPMC34.SA","MAEL34.SA","MCDP34.SA","MDLZ34.SA","MRCK34.SA","ORCL34.SA","PEP334.SA",
"PFIZ34.SA","PMIC34.SA","QCOM34.SA","SBUX34.SA","TGTB34.SA","TMOS34.SA","TXN34.SA","UNHH34.SA","UPSB34.SA",
"VZUA34.SA","ABTT34.SA","AMGN34.SA","AXPB34.SA","BAOO34.SA","CATP34.SA","HONB34.SA","BOVA11.SA","IVVB11.SA",
"SMAL11.SA","HASH11.SA","GOLD11.SA","GARE11.SA","HGLG11.SA","XPLG11.SA","VILG11.SA","BRCO11.SA","BTLG11.SA",
"XPML11.SA","VISC11.SA","HSML11.SA","MALL11.SA","KNRI11.SA","JSRE11.SA","PVBI11.SA","HGRE11.SA","MXRF11.SA",
"KNCR11.SA","KNIP11.SA","CPTS11.SA","IRDM11.SA","DIVO11.SA","NDIV11.SA","SPUB11.SA"
]))

ETFS_INDICE = {"BOVA11.SA","IVVB11.SA","SMAL11.SA","HASH11.SA","DIVO11.SA","NDIV11.SA","SPUB11.SA"}

# =========================================================
# Funções
# =========================================================

def ajustar_colunas(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def stochastic_kd(df, k_period=14, d_period=3, smooth=3):
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    k = k.rolling(smooth).mean()
    d = k.rolling(d_period).mean()
    return k, d

def dmi_adx_tradingview(df, period=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_rma = rma(tr, period)
    plus_dm_rma = rma(pd.Series(plus_dm, index=df.index), period)
    minus_dm_rma = rma(pd.Series(minus_dm, index=df.index), period)

    plus_di = 100 * plus_dm_rma / tr_rma
    minus_di = 100 * minus_dm_rma / tr_rma

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = rma(dx, period)

    return plus_di, minus_di, adx

def preparar_semanal(df):

    semanal = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    return semanal

def indice_candle_fechado():

    tz = pytz.timezone("America/Sao_Paulo")
    agora = datetime.now(tz).time()

    if agora >= time(19,15):
        return -1
    else:
        return -2

def classe_ativo(ticker):

    if ticker.endswith("34.SA"):
        return "BDR"

    if ticker in ETFS_INDICE:
        return "ETF"

    return "AÇÃO"

def parametros_trade(classe):

    if classe == "AÇÃO":
        return -0.05, 0.08

    if classe == "BDR":
        return -0.04, 0.06

    return -0.03, 0.05


# =========================================================
# Estatística
# =========================================================

def calcular_estatistica(df, semanal, lookback=1000, max_forward=60):

    ganhos = 0
    perdas = 0
    total = 0

    start = max(0, len(df) - lookback)

    for i in range(start, len(df)-max_forward):

        row = df.iloc[i]

        if not (
            row["Close"] > row["EMA69"] and
            row["K"] > row["D"] and
            row["DIp"] > row["DIm"] and
            row["Vol_MA20"] > row["Vol_MA50"]
        ):
            continue

        data = df.index[i]

        if data not in semanal.index:
            continue

        row_w = semanal.loc[data]

        if not (
            row_w["K"] > row_w["D"] and
            row_w["DIp"] > row_w["DIm"]
        ):
            continue

        classe = classe_ativo("X")  # dummy para pegar parâmetros depois

        total += 1

        entrada = row["Close"]

        stop_p, gain_p = parametros_trade(classe_ativo("X"))

        stop = entrada * (1 + stop_p)
        gain = entrada * (1 + gain_p)

        saiu = False

        for j in range(i+1, i+1+max_forward):

            hi = df.iloc[j]["High"]
            lo = df.iloc[j]["Low"]

            if lo <= stop:
                perdas += 1
                saiu = True
                break

            if hi >= gain:
                ganhos += 1
                saiu = True
                break

        if not saiu:
            total -= 1

    return total, ganhos, perdas


# =========================================================
# Scanner
# =========================================================

if st.button("Rodar Scanner"):

    resultados = []
    progress = st.progress(0.0)

    idx = indice_candle_fechado()

    for i, ticker in enumerate(ativos_scan):

        try:

            df = yf.download(ticker, period="1200d", interval="1d", progress=False)

            if df.empty:
                continue

            df = ajustar_colunas(df)

            if len(df) < 300:
                continue

            df["EMA69"] = ema(df["Close"], 69)

            k, d = stochastic_kd(df)
            df["K"] = k
            df["D"] = d

            di_p, di_m, adx = dmi_adx_tradingview(df)
            df["DIp"] = di_p
            df["DIm"] = di_m
            df["ADX"] = adx

            df["Vol_MA20"] = df["Volume"].rolling(20).mean()
            df["Vol_MA50"] = df["Volume"].rolling(50).mean()

            row = df.iloc[idx]

            if not (
                row["Close"] > row["EMA69"] and
                row["K"] > row["D"] and
                row["DIp"] > row["DIm"] and
                row["Vol_MA20"] > row["Vol_MA50"]
            ):
                continue

            semanal = preparar_semanal(df)

            kw, dw = stochastic_kd(semanal)
            semanal["K"] = kw
            semanal["D"] = dw

            di_pw, di_mw, _ = dmi_adx_tradingview(semanal)
            semanal["DIp"] = di_pw
            semanal["DIm"] = di_mw

            semanal = semanal.dropna()

            if df.index[idx] not in semanal.index:
                continue

            row_w = semanal.loc[df.index[idx]]

            if not (row_w["K"] > row_w["D"] and row_w["DIp"] > row_w["DIm"]):
                continue

            classe = classe_ativo(ticker)

            total, ganhos, perdas = calcular_estatistica(df, semanal)

            prob = round((ganhos / total) * 100, 2) if total >= 1 else np.nan

            resultados.append({
                "Ativo": ticker,
                "Classe": classe,
                "Ocorrências": total,
                "Gains antes do stop": ganhos,
                "Stops antes do gain": perdas,
                "Probabilidade (%)": prob,
                "Close": round(float(row["Close"]),2),
                "K(D)": round(float(row["K"]),2),
                "D(D)": round(float(row["D"]),2),
                "K(W)": round(float(row_w["K"]),2),
                "D(W)": round(float(row_w["D"]),2)
            })

        except:
            pass

        progress.progress((i+1)/len(ativos_scan))

    st.subheader("Ativos aprovados + Estatística")

    if len(resultados) == 0:
        st.warning("Nenhum ativo passou no setup.")
    else:
        df_res = pd.DataFrame(resultados).sort_values(
            ["Probabilidade (%)","Ocorrências"], ascending=False
        )
        st.dataframe(df_res, use_container_width=True)
