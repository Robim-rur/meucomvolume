import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz

st.set_page_config(layout="wide")
st.title("Scanner – Setup Roberson (Diário + Semanal / TradingView)")

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
"RECV3.SA","SANB11.SA","SBSP3.SA","SLCE3.SA","SMTO3.SA","SUZB3.SA","TAEE11.SA","TIMS3.SA", "TTEN3.SA","TOTS3.SA","TRPL4.SA",
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

# =========================================================
# Funções auxiliares
# =========================================================

def ajustar_colunas(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in df.columns:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    return df

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def stochastic_kd(df, k_period=14, d_period=3, smooth=3):

    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()

    k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    k_smooth = k.rolling(smooth).mean()
    d = k_smooth.rolling(d_period).mean()

    return k_smooth, d

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

    if agora >= time(18, 30):
        return -1
    else:
        return -2


# =========================================================
# Scanner
# =========================================================

if st.button("Rodar Scanner"):

    resultados = []
    progress = st.progress(0.0)

    idx = indice_candle_fechado()

    for i, ticker in enumerate(ativos_scan):

        try:

            df = yf.download(
                ticker,
                period="450d",
                interval="1d",
                progress=False,
                auto_adjust=False
            )

            if df.empty:
                progress.progress((i + 1) / len(ativos_scan))
                continue

            df = ajustar_colunas(df)

            if len(df) < 120:
                progress.progress((i + 1) / len(ativos_scan))
                continue

            # =========================
            # Indicadores diários
            # =========================

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

            if len(df.dropna()) < abs(idx):
                progress.progress((i + 1) / len(ativos_scan))
                continue

            row = df.iloc[idx]

            cond_ema   = row["Close"] > row["EMA69"]
            cond_stoch = row["K"] > row["D"]
            cond_dmi   = row["DIp"] > row["DIm"]
            cond_vol   = row["Vol_MA20"] > row["Vol_MA50"]

            if not (cond_ema and cond_stoch and cond_dmi and cond_vol):
                progress.progress((i + 1) / len(ativos_scan))
                continue

            # =========================
            # Semanal
            # =========================

            semanal = preparar_semanal(df)
            semanal = ajustar_colunas(semanal)

            di_pw, di_mw, _ = dmi_adx_tradingview(semanal)
            semanal["DIp"] = di_pw
            semanal["DIm"] = di_mw

            if len(semanal.dropna()) < abs(idx):
                progress.progress((i + 1) / len(ativos_scan))
                continue

            row_w = semanal.iloc[idx]

            if not (row_w["DIp"] > row_w["DIm"]):
                progress.progress((i + 1) / len(ativos_scan))
                continue

            resultados.append({
                "Ativo": ticker,
                "Data": df.index[idx].date(),
                "Close": round(float(row["Close"]), 2),
                "K": round(float(row["K"]), 2),
                "D": round(float(row["D"]), 2),
                "DI+ (D)": round(float(row["DIp"]), 2),
                "DI- (D)": round(float(row["DIm"]), 2),
                "ADX (D)": round(float(row["ADX"]), 2),
                "Vol MA20": round(float(row["Vol_MA20"]), 0),
                "Vol MA50": round(float(row["Vol_MA50"]), 0),
                "Vol MA20 > MA50": cond_vol
            })

        except Exception:
            pass

        progress.progress((i + 1) / len(ativos_scan))

    st.subheader("Ativos aprovados no setup")

    if len(resultados) == 0:
        st.warning("Nenhum ativo passou em todos os filtros.")
    else:
        df_res = pd.DataFrame(resultados).sort_values("Ativo")
        st.dataframe(df_res, use_container_width=True)
