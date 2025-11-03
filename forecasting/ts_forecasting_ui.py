from pathlib import Path
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st

# ---------- Path setup ----------
THIS = Path(__file__).resolve()
FORECASTING_DIR = THIS.parent
PROJECT_ROOT = THIS.parents[1]
for p in (str(FORECASTING_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from forecasting.live_infer import LiveForecaster


def _nice_badge(text, color="#4f8df5"):
    st.markdown(
        f"<span style='display:inline-block;background:{color};padding:4px 8px;border-radius:8px;"
        f"font-size:12px;color:white;font-weight:600;margin-right:6px'>{text}</span>",
        unsafe_allow_html=True
    )


def _reset_session_state():
    """Clear counters/buffer pointers when user switches data or model."""
    st.session_state["live_last_len"] = 0
    st.session_state["live_key"] = None


def ts_forecasting_streamlit():
    # (Removed the outer panel wrapper that created the blank bar)
    st.subheader("📈 Time-Series Forecasting (Live)")

    # --- Layout: left controls, right display
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("### Controls")

        # Checkpoint selector (default to RL)
        ckpt_choice = st.selectbox(
            "Checkpoint",
            ["checkpoint_rl.pt", "checkpoint_stae.pt", "checkpoint_min.pt"],
            index=0,
            help="RL is recommended for best predictions."
        )
        ckpt_path = str(FORECASTING_DIR / ckpt_choice)

        # CSV path where your device appends readings
        csv_path = st.text_input(
            "Live readings CSV path",
            value="live_readings.csv",
            help="Append one new row per reading; columns must match the model."
        )

        # Refresh logic tuned for slow sampling (e.g., every 10 minutes)
        st.write("**Update policy**")
        auto_refresh = st.toggle(
            "Auto-refresh",
            value=False,
            help="Enable if you want the page to refresh on a timer."
        )
        # For 10-minute sampling, 600 seconds is sensible. Wider range for flexibility.
        if auto_refresh:
            refresh_secs = st.slider(
                "Refresh every (sec)",
                min_value=60, max_value=3600, value=600, step=60,
                help="Match your device interval (e.g., 10 min = 600 s)."
            )
        else:
            refresh_secs = 0  # ignored

        # Manual refresh button (no timer needed)
        if st.button("🔄 Refresh now"):
            st.experimental_rerun()

        @st.cache_resource(show_spinner=False)
        def _get_forecaster(path: str):
            return LiveForecaster(path)

        # Build/restore model; if it fails, stop early
        try:
            forecaster = _get_forecaster(ckpt_path)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            return

        # Display expected columns from checkpoint's meta
        _nice_badge("Expected columns")
        st.code(", ".join(forecaster.cols), language="text")

        # Variable selector for charting
        var = st.selectbox("Variable to chart", options=forecaster.cols, index=0)

    with right:
        st.markdown("### Live Forecast")
        placeholder_panel = st.container()

        # Reset session if user changed CSV or checkpoint
        live_key_now = f"{csv_path}::{ckpt_path}"
        if st.session_state.get("live_key") != live_key_now:
            _reset_session_state()
            st.session_state["live_key"] = live_key_now

        # Read CSV once (low_memory=False for mixed dtypes)
        try:
            df_live = pd.read_csv(csv_path, low_memory=False)
        except FileNotFoundError:
            st.warning("CSV not found yet. Waiting for the first file to appear…")
            return
        except Exception as e:
            st.error(f"Could not open CSV: {e}")
            return

        # Ensure expected columns exist and order them (extra columns are ignored)
        for c in forecaster.cols:
            if c not in df_live.columns:
                df_live[c] = np.nan
        df_live = df_live[forecaster.cols]

        # Only process new rows since last render
        last_seen = st.session_state.get("live_last_len", 0)
        if last_seen < 0 or last_seen > len(df_live):
            # In case the CSV was rotated/rewritten
            last_seen = 0

        latest_pred = None
        if len(df_live) > last_seen:
            # Iterate only over new rows
            new_rows = df_live.iloc[last_seen:]
            for _, row in new_rows.iterrows():
                latest_pred = forecaster.update_with_reading(row.to_dict())
            # Update pointer
            st.session_state["live_last_len"] = len(df_live)

        # Render
        with placeholder_panel:
            if latest_pred is not None:
                # Compact KPI row (first three variables)
                keys = list(latest_pred.keys())
                kcols = st.columns(min(3, len(keys)))
                for i, k in enumerate(keys[: len(kcols)]):
                    kcols[i].metric(label=k, value=f"{latest_pred[k]:.3f}")

                # Full row
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.write("**Latest forecast (full row)**")
                st.dataframe(pd.DataFrame([latest_pred]), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Chart: last window inputs + forecast point
                st.markdown("<div class='panel'>", unsafe_allow_html=True)
                st.write(f"**{var} — last {forecaster.window} inputs & next-step forecast**")
                hist = forecaster.buffer[var].tail(forecaster.window).astype(float).tolist()
                y_next = float(latest_pred[var])
                idx = list(range(len(hist))) + [len(hist)]
                vals = hist + [y_next]
                df_chart = pd.DataFrame({"t": idx, var: vals})
                st.line_chart(df_chart.set_index("t"))
                st.caption("Solid line shows the window history; last point is the model forecast.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                if len(df_live) < forecaster.window:
                    st.info(
                        f"Waiting for enough rows to fill the model window (need {forecaster.window}, have {len(df_live)})."
                    )
                else:
                    st.info("No new rows since last update.")

        # Auto refresh (timer) — sensible for slow sampling
        if auto_refresh and refresh_secs > 0:
            time.sleep(refresh_secs)
            st.experimental_rerun()
