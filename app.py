import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("Diagnóstico – Setup Roberson (TradingView)")

# ativos que você confirmou que passaram no gráfico
ativos_teste = [
    "BPAC11.SA",
    "GOLD11.SA",
    "MXRF11.SA",
    "RANI3.SA",
    "SLCE3.SA",
    "VALE3.SA"
]

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


if st.button("Rodar diagnóstico"):

    linhas = []

    for ticker in ativos_teste:

        try:

            df = yf.download(
                ticker,
                period="450d",
                interval="1d",
                progress=False,
                auto_adjust=False
            )

            if df.empty:
                continue

            df = ajustar_colunas(df)

            if len(df) < 120:
                continue

            # ===============================
            # Indicadores diários
            # ===============================

            df["EMA69"] = ema(df["Close"], 69)

            k, d = stochastic_kd(df)
            df["K"] = k
            df["D"] = d

            di_p, di_m, adx = dmi_adx_tradingview(df)
            df["DIp"] = di_p
            df["DIm"] = di_m
            df["ADX"] = adx

            # ===============================
            # Volume
            # ===============================

            df["Vol_MA20"] = df["Volume"].rolling(20).mean()
            df["Vol_MA50"] = df["Volume"].rolling(50).mean()

            # candle diário fechado
            row = df.iloc[-2]

            # ===============================
            # Semanal
            # ===============================

            semanal = preparar_semanal(df)
            semanal = ajustar_colunas(semanal)

            di_pw, di_mw, _ = dmi_adx_tradingview(semanal)
            semanal["DIp"] = di_pw
            semanal["DIm"] = di_mw

            row_w = semanal.iloc[-2]

            linhas.append({

                "Ativo": ticker,
                "Data diário": df.index[-2].date(),

                "Close > EMA69": row["Close"] > row["EMA69"],
                "K > D": row["K"] > row["D"],
                "DI+ > DI- (D)": row["DIp"] > row["DIm"],
                "DI+ > DI- (W)": row_w["DIp"] > row_w["DIm"],

                # >>> NOVO FILTRO DE VOLUME
                "Vol MA20 > MA50": row["Vol_MA20"] > row["Vol_MA50"],

                # ===============================
                # Valores
                # ===============================

                "Close": round(float(row["Close"]), 2),
                "EMA69": round(float(row["EMA69"]), 2),

                "K": round(float(row["K"]), 2),
                "D": round(float(row["D"]), 2),

                "DI+ D": round(float(row["DIp"]), 2),
                "DI- D": round(float(row["DIm"]), 2),

                "DI+ W": round(float(row_w["DIp"]), 2),
                "DI- W": round(float(row_w["DIm"]), 2),

                # >>> valores do volume
                "Volume": int(row["Volume"]),
                "Vol MA20": round(float(row["Vol_MA20"]), 0),
                "Vol MA50": round(float(row["Vol_MA50"]), 0),
            })

        except Exception as e:
            linhas.append({
                "Ativo": ticker,
                "Erro": str(e)
            })

    df_diag = pd.DataFrame(linhas)
    st.dataframe(df_diag, use_container_width=True)
