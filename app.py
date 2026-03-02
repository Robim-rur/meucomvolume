import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Scanner – Setup Roberson (Diário + Confirmação Semanal)")

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
"RECV3.SA","SANB11.SA","SBSP3.SA","SLCE3.SA","SMTO3.SA","SUZB3.SA","TAEE11.SA","TIMS3.SA","TOTS3.SA","TRPL4.SA",
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
# Funções
# =========================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def stochastic_kd(df, k_period=14, d_period=3, smooth=3):

    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()

    k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    k_smooth = k.rolling(smooth).mean()
    d = k_smooth.rolling(d_period).mean()

    return k_smooth, d

def dmi_adx(df, period=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = low.shift() - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(period).mean()

    return plus_di, minus_di, adx

def preparar_semanal(df):

    semanal = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    return semanal


# =========================================================
# Scanner
# =========================================================

if st.button("Rodar Scanner"):

    resultados = []
    progress = st.progress(0.0)

    for i, ticker in enumerate(ativos_scan):

        try:
            df = yf.download(
                ticker,
                period="420d",
                interval="1d",
                progress=False
            )

            if df.empty or len(df) < 120:
                continue

            df["EMA69"] = ema(df["Close"], 69)

            k, d = stochastic_kd(df, 14, 3, 3)
            df["K"] = k
            df["D"] = d

            di_p, di_m, adx = dmi_adx(df, 14)
            df["DIp"] = di_p
            df["DIm"] = di_m
            df["ADX"] = adx

            df_ind = df.dropna(subset=["EMA69","K","D","DIp","DIm","ADX"])

            if df_ind.empty:
                continue

            last = df_ind.iloc[-1]

            cond_ema   = last["Close"] > last["EMA69"]
            cond_stoch = last["K"] > last["D"]
            cond_dmi   = last["DIp"] > last["DIm"]

            if not (cond_ema and cond_stoch and cond_dmi):
                continue

            # -------------------------
            # Semanal
            # -------------------------

            semanal = preparar_semanal(df)

            di_p_w, di_m_w, _ = dmi_adx(semanal, 14)

            semanal["DIp"] = di_p_w
            semanal["DIm"] = di_m_w

            semanal_ind = semanal.dropna(subset=["DIp","DIm"])

            if semanal_ind.empty:
                continue

            last_w = semanal_ind.iloc[-1]

            if not (last_w["DIp"] > last_w["DIm"]):
                continue

            resultados.append({
                "Ativo": ticker,
                "Close": round(last["Close"], 2),
                "K": round(last["K"], 2),
                "D": round(last["D"], 2),
                "DI+ (D)": round(last["DIp"], 2),
                "DI- (D)": round(last["DIm"], 2),
                "ADX (D)": round(last["ADX"], 2)
            })

        except:
            pass

        progress.progress((i + 1) / len(ativos_scan))

    st.subheader("Ativos aprovados no setup")

    if len(resultados) == 0:
        st.warning("Nenhum ativo passou em todos os filtros.")
    else:
        df_res = pd.DataFrame(resultados).sort_values("Ativo")
        st.dataframe(df_res, use_container_width=True)
