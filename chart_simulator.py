# [V6.3] 중복 코드 제거 및 최종 검토 완료 버전
import streamlit as st
import pandas as pd
import pandas_ta as ta
from pykrx import stock
import datetime
import json
import os
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time

# --- 1. 초기 설정 ---
st.set_page_config(page_title="실전 차트 시뮬레이터 V6.3", layout="wide")
st.title("📈 실전형 차트 기반 주식 시뮬레이터 V6.3")

INITIAL_CASH = 10_000_000
STATE_FILE = "sim_state.json"
MIN_DATA_PERIOD = 240
MIN_FUTURE_PERIOD = 100

# --- 2. 데이터 관련 함수 ---
@st.cache_resource
def get_all_tickers():
    return stock.get_market_ticker_list()

@st.cache_data
def get_stock_name(ticker):
    try: return stock.get_market_ticker_name(ticker)
    except: return "알 수 없는 종목"

@st.cache_data
def load_data(ticker, start, end):
    start_str, end_str = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    try:
        df = stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
        if df.empty: return pd.DataFrame()
        df.reset_index(inplace=True); df['날짜'] = pd.to_datetime(df['날짜'])
        df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"}, inplace=True)
        df.ta.macd(append=True)
        df.ta.rsi(append=True)
        df.ta.bbands(length=20, append=True)
        return df
    except Exception as e:
        st.error(f"'{ticker}' 데이터 로딩 중 오류: {e}"); return pd.DataFrame()

@st.cache_data
def get_fundamental_data(date_str, ticker):
    try:
        start_of_week = (datetime.datetime.strptime(date_str, "%Y%m%d") - datetime.timedelta(days=6)).strftime("%Y%m%d")
        funda = stock.get_market_fundamental_by_date(start_of_week, date_str, ticker)
        return funda.iloc[-1:] if not funda.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

# --- 3. 상태 관리 함수 ---
def load_state():
    default_state = {
        "cash": INITIAL_CASH, "holdings": {"quantity": 0, "avg_price": 0}, "trade_log": [],
        "day_index": MIN_DATA_PERIOD, "ticker": "005930",
        "start_date": datetime.date(2020, 1, 1).isoformat(), "end_date": datetime.date(2023, 12, 31).isoformat(),
        "daily_portfolio_value": [], "pending_orders": []
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                loaded_state = json.load(f); default_state.update(loaded_state)
        except (json.JSONDecodeError, FileNotFoundError): pass
    return default_state

def save_state(state):
    if isinstance(state.get("start_date"), datetime.date): state["start_date"] = state["start_date"].isoformat()
    if isinstance(state.get("end_date"), datetime.date): state["end_date"] = state["end_date"].isoformat()
    with open(STATE_FILE, "w") as f: json.dump(state, f, indent=4)

# --- 4. 핵심 로직 함수 ---
def calculate_performance(state, current_price):
    initial_asset = INITIAL_CASH; current_asset = state['cash'] + (state['holdings']['quantity'] * current_price)
    cumulative_return = ((current_asset / initial_asset) - 1) * 100 if initial_asset > 0 else 0
    sell_trades = [t for t in state['trade_log'] if '매도' in t['유형']]; wins = 0; total_profit_loss = 0
    if sell_trades:
        for trade in sell_trades:
            profit_loss = (trade['단가'] - trade.get('avg_price_at_sale', 0)) * trade['수량']
            if profit_loss > 0: wins += 1
            total_profit_loss += profit_loss
        win_rate = (wins / len(sell_trades)) * 100 if sell_trades else 0
    else: win_rate = 0
    value_history = pd.Series(state.get('daily_portfolio_value', [])); max_dd = 0
    if not value_history.empty:
        peak = value_history.expanding(min_periods=1).max(); drawdown = (value_history - peak) / peak
        max_dd = drawdown.min() * 100 if not drawdown.empty else 0
    return {"현재 총 자산": int(current_asset), "누적 수익률 (%)": round(cumulative_return, 2), "총 실현 손익": int(total_profit_loss), "승률 (%)": round(win_rate, 2), "최대 손실률 (MDD, %)": round(max_dd, 2), "총 매도 거래 횟수": len(sell_trades)}

def create_plotly_chart(df, trades):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=(None, 'Volume', 'MACD', 'RSI'), row_heights=[0.6, 0.1, 0.15, 0.15])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC', increasing_line_color='#d62728', decreasing_line_color='#1f77b4'), row=1, col=1)
    if 'BBL_20_2.0' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], mode='lines', line=dict(color='rgba(0,100,255,0.2)'), name='BB Lower'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], mode='lines', line=dict(color='rgba(0,100,255,0.2)'), fill='tonexty', name='BB Upper'), row=1, col=1)
    ma_periods = [5, 20, 60, 120]; ma_colors = ['#ff9900', '#00ced1', '#8a2be2', '#32cd32']
    for period, color in zip(ma_periods, ma_colors):
        if len(df) >= period: fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(window=period).mean(), mode='lines', name=f'{period}MA', line=dict(color=color, width=1.5)), row=1, col=1)
    volume_colors = np.where(df['Open'] <= df['Close'], '#d62728', '#1f77b4')
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=volume_colors), row=2, col=1)
    if 'MACD_12_26_9' in df.columns:
        macd_colors = np.where(df['MACDh_12_26_9'] < 0, '#1f77b4', '#d62728')
        fig.add_trace(go.Bar(x=df.index, y=df['MACDh_12_26_9'], name='MACD Hist', marker_color=macd_colors), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], mode='lines', name='MACD', line=dict(color='#ff9900')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], mode='lines', name='Signal', line=dict(color='#00ced1')), row=3, col=1)
    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], mode='lines', name='RSI', line=dict(color='purple')), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="blue", line_width=1, row=4, col=1)
    if trades:
        trade_df = pd.DataFrame(trades); trade_df['일자'] = pd.to_datetime(trade_df['일자'])
        buy_trades = trade_df[trade_df['유형'].str.contains('매수')]; sell_trades = trade_df[trade_df['유형'].str.contains('매도')]
        if not buy_trades.empty: fig.add_trace(go.Scatter(x=buy_trades['일자'], y=buy_trades['단가'], mode='markers', name='매수', marker=dict(symbol='triangle-up', color='#ff0000', size=12, line=dict(width=1, color='DarkSlateGrey'))), row=1, col=1)
        if not sell_trades.empty: fig.add_trace(go.Scatter(x=sell_trades['일자'], y=sell_trades['단가'], mode='markers', name='매도', marker=dict(symbol='triangle-down', color='#0000ff', size=12, line=dict(width=1, color='DarkSlateGrey'))), row=1, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=700, margin=dict(l=10, r=10, b=10, t=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(type='category'); fig.update_yaxes(showspikes=True, side='right')
    fig.update_xaxes(visible=False, row=1, col=1); fig.update_xaxes(visible=False, row=2, col=1); fig.update_xaxes(visible=False, row=3, col=1); fig.update_xaxes(showticklabels=False, row=4, col=1)
    return fig

# --- 5. Streamlit 앱 메인 로직 ---
if "state" not in st.session_state: st.session_state.state = load_state()
state = st.session_state.state

st.sidebar.header("🔧 시뮬레이션 설정")
ticker = st.sidebar.text_input("종목 코드", state.get("ticker", "005930"))
st.sidebar.markdown(f"**선택 종목:** {get_stock_name(ticker)} ({ticker})")
start_date = st.sidebar.date_input("시작 날짜", datetime.date.fromisoformat(state.get("start_date")))
end_date = st.sidebar.date_input("종료 날짜", datetime.date.fromisoformat(state.get("end_date")))
if (ticker != state.get("ticker") or start_date.isoformat() != state.get("start_date") or end_date.isoformat() != state.get("end_date")):
    state.update({"ticker": ticker, "start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "day_index": MIN_DATA_PERIOD, "pending_orders": [], "trade_log": []}); st.rerun()

st.sidebar.subheader("🎲 랜덤 리셋")
if st.sidebar.button("다른 종목/구간으로 새로 시작 (자산 유지)", key="random_reset_btn"):
    if state['holdings']['quantity'] > 0:
        full_df = load_data(state['ticker'], datetime.date.fromisoformat(state['start_date']), datetime.date.fromisoformat(state['end_date']))
        if not full_df.empty and state['day_index'] < len(full_df):
            last_price = full_df.iloc[state['day_index']]['Close']; state['cash'] += state['holdings']['quantity'] * last_price
            st.sidebar.info(f"기존 포지션 자동 정산 완료.")
    state.update({'holdings': {"quantity": 0, "avg_price": 0}, 'trade_log': [], 'daily_portfolio_value': [], 'day_index': MIN_DATA_PERIOD, 'pending_orders': []})
    all_tickers = get_all_tickers()
    while True:
        new_ticker = random.choice(all_tickers)
        full_history = load_data(new_ticker, datetime.date(2010, 1, 1), datetime.date.today())
        if len(full_history) > MIN_DATA_PERIOD + MIN_FUTURE_PERIOD:
            max_start_index = len(full_history) - (MIN_DATA_PERIOD + MIN_FUTURE_PERIOD)
            random_start_index = random.randint(0, max_start_index)
            state['ticker'] = new_ticker; state['start_date'] = full_history['날짜'].iloc[random_start_index].date().isoformat(); state['end_date'] = full_history['날짜'].iloc[-1].date().isoformat()
            break
    save_state(state); st.rerun()

st.sidebar.subheader("⚠ 위험 구역")
reset_confirmation = st.sidebar.checkbox("모든 매매 기록과 자산을 초기화하려면 체크하세요.", key="reset_confirm_checkbox")
if reset_confirmation:
    if st.sidebar.button("모든 기록 초기화 실행", type="primary", key="full_reset_btn"):
        initial_state = {
            "cash": INITIAL_CASH, "holdings": {"quantity": 0, "avg_price": 0}, "trade_log": [],
            "day_index": MIN_DATA_PERIOD, "ticker": "005930",
            "start_date": datetime.date(2020, 1, 1).isoformat(), "end_date": datetime.date(2023, 12, 31).isoformat(),
            "daily_portfolio_value": [], "pending_orders": []
        }
        st.session_state.state = initial_state; save_state(initial_state)
        st.sidebar.success("모든 기록이 초기화되었습니다."); time.sleep(2); st.rerun()

try:
    df = load_data(ticker, start_date, end_date)
    if df.empty or len(df) < MIN_DATA_PERIOD: st.warning("데이터가 부족합니다.")
    else:
        if state["day_index"] >= len(df): state["day_index"] = len(df) - 1
        visible_df = df.iloc[:state["day_index"] + 1].copy(); visible_df.set_index("날짜", inplace=True)
        current_date = visible_df.index[-1]; current_price = visible_df.iloc[-1]['Close']
        col1, col2 = st.columns([3.5, 1.5])

        with col1:
            st.subheader(f"📊 {get_stock_name(ticker)} ({ticker}) 차트"); plotly_fig = create_plotly_chart(visible_df, state.get("trade_log", [])); st.plotly_chart(plotly_fig, use_container_width=True)
            st.markdown("---"); st.subheader("🏢 기업 정보")
            date_str = current_date.strftime("%Y%m%d"); funda = get_fundamental_data(date_str, ticker)
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            if funda is not None and not funda.empty:
                f_col1.metric("PER", f"{funda['PER'].iloc[0]:.2f}" if funda['PER'].iloc[0] != 0 else "N/A"); f_col2.metric("PBR", f"{funda['PBR'].iloc[0]:.2f}" if funda['PBR'].iloc[0] != 0 else "N/A")
                f_col3.metric("EPS", f"{int(funda['EPS'].iloc[0])}" if funda['EPS'].iloc[0] != 0 else "N/A"); f_col4.metric("BPS", f"{int(funda['BPS'].iloc[0])}" if funda['BPS'].iloc[0] != 0 else "N/A")
            else:
                f_col1.metric("PER", "N/A"); f_col2.metric("PBR", "N/A"); f_col3.metric("EPS", "N/A"); f_col4.metric("BPS", "N/A")
        
        with col2:
            st.markdown(f"#### 📅 {current_date.date()} | 종가: {int(current_price):,}원"); st.markdown("---")
            if st.button("▶️ 다음 날로 이동", use_container_width=True):
                if state["day_index"] < len(df) - 1:
                    new_day_index = state["day_index"] + 1
                    next_day = df.iloc[new_day_index]
                    next_day_date_iso = next_day['날짜'].date().isoformat()
                    
                    for order in state["pending_orders"]:
                        if order['type'] == 'trailing_stop':
                            order['peak_price'] = max(order.get('peak_price', 0), next_day['High'])
                    
                    executed_order_ids = []
                    executed_qty = 0
                    for order in state["pending_orders"]:
                        if order['id'] in executed_order_ids:
                            continue
                        
                        if order['type'] == 'buy' and next_day['Low'] <= order['price'] and state['cash'] >= order['qty'] * order['price']:
                            cost = order['qty'] * order['price']
                            total_qty = state['holdings']['quantity'] + order['qty']
                            avg_price = (state['holdings']['avg_price'] * state['holdings']['quantity'] + cost) / total_qty
                            state.update({'holdings': {'quantity': int(total_qty), 'avg_price': float(avg_price)}, 'cash': float(state['cash'] - cost)})
                            state['trade_log'].append({"일자": next_day_date_iso, "유형": "지정가매수", "수량": order['qty'], "단가": int(order['price']), "금액": int(cost)})
                            st.toast(f"✅ [{next_day['날짜'].date()}] {int(order['price']):,}원 매수 체결!")
                            executed_order_ids.append(order['id'])
                        
                        elif order['type'] == 'sell' and next_day['High'] >= order['price'] and state['holdings']['quantity'] >= order['qty']:
                            revenue = order['qty'] * order['price']
                            current_avg_price = state['holdings']['avg_price']
                            state['cash'] = float(state['cash'] + revenue)
                            state['holdings']['quantity'] = int(state['holdings']['quantity'] - order['qty'])
                            if state['holdings']['quantity'] == 0: state['holdings']['avg_price'] = 0
                            state['trade_log'].append({"일자": next_day_date_iso, "유형": "지정가매도", "수량": order['qty'], "단가": int(order['price']), "금액": int(revenue), "avg_price_at_sale": float(current_avg_price)})
                            st.toast(f"✅ [{next_day['날짜'].date()}] {int(order['price']):,}원 매도 체결!")
                            executed_order_ids.append(order['id'])

                        elif order['type'] == 'stop_loss' and next_day['Low'] <= order['price'] and state['holdings']['quantity'] > 0:
                            executed_qty = state['holdings']['quantity']
                            sell_price = order['price']
                            revenue = executed_qty * sell_price
                            current_avg_price = state['holdings']['avg_price']
                            state['cash'] = float(state['cash'] + revenue)
                            state['holdings']['quantity'] = 0
                            state['holdings']['avg_price'] = 0
                            state['trade_log'].append({"일자": next_day_date_iso, "유형": "스탑로스매도", "수량": executed_qty, "단가": int(sell_price), "금액": int(revenue), "avg_price_at_sale": float(current_avg_price)})
                            st.toast(f"🚨 [{next_day['날짜'].date()}] {int(sell_price):,}원 스탑로스 체결!")
                            executed_order_ids.append(order['id'])
                        
                        elif order['type'] == 'trailing_stop' and state['holdings']['quantity'] > 0:
                            stop_price = order['peak_price'] * (1 - order['percentage'] / 100)
                            if next_day['Low'] <= stop_price:
                                executed_qty = state['holdings']['quantity']
                                sell_price = stop_price
                                revenue = executed_qty * sell_price
                                current_avg_price = state['holdings']['avg_price']
                                state['cash'] = float(state['cash'] + revenue)
                                state['holdings']['quantity'] = 0
                                state['holdings']['avg_price'] = 0
                                state['trade_log'].append({"일자": next_day_date_iso, "유형": "트레일링매도", "수량": executed_qty, "단가": int(sell_price), "금액": int(revenue), "avg_price_at_sale": float(current_avg_price)})
                                st.toast(f"✅ [{next_day['날짜'].date()}] {int(sell_price):,}원 트레일링 스탑 체결!")
                                executed_order_ids.append(order['id'])

                    state["pending_orders"] = [o for o in state["pending_orders"] if o['id'] not in executed_order_ids]
                    if executed_qty > 0:
                        state["pending_orders"] = [o for o in state["pending_orders"] if o['type'] == 'buy']
                    
                    state["day_index"] = new_day_index
                    
                    # --- [핵심 수정] ---
                    # MDD 계산용 자산 가치를 저장하기 전에, float()으로 변환합니다.
                    portfolio_value = state['cash'] + (state['holdings']['quantity'] * next_day['Close'])
                    state['daily_portfolio_value'].append(float(portfolio_value))
                    # --------------------
                    
                    st.rerun()
                else:
                    st.warning("시뮬레이션 기간의 마지막 날입니다.")
            
            st.markdown("---"); st.subheader("💰 자산 현황")
            c1, c2 = st.columns(2)
            c1.metric("현금 잔액", f"{int(state['cash']):,}원"); c2.metric("평가 금액", f"{int(state['holdings']['quantity'] * current_price):,}원")
            c1.metric("보유 수량", f"{state['holdings']['quantity']}주"); c2.metric("평균 단가", f"{int(state['holdings']['avg_price']):,}원")
            
            with st.expander("🛒 즉시 매매 (시장가)", expanded=False):
                tab1, tab2 = st.tabs(["수량으로 주문", "금액으로 주문"])
                with tab1:
                    max_buy_qty_tab1 = int(state['cash'] // current_price) if current_price > 0 else 0
                    buy_qty = st.number_input(f"매수 수량 (최대: {max_buy_qty_tab1}주)", min_value=0, max_value=max_buy_qty_tab1, step=10, key="buy_qty_market")
                    if st.button("🟩 수량으로 매수", use_container_width=True, key="buy_by_qty_btn"):
                        if buy_qty > 0:
                            cost = buy_qty * current_price; total_qty = state['holdings']['quantity'] + buy_qty; avg_price = (state['holdings']['avg_price'] * state['holdings']['quantity'] + cost) / total_qty
                            state.update({'holdings': {'quantity': int(total_qty), 'avg_price': float(avg_price)}, 'cash': float(state['cash'] - cost)}); state['trade_log'].append({"일자": current_date.date().isoformat(), "유형": "시장가매수", "수량": buy_qty, "단가": int(current_price), "금액": int(cost)}); st.rerun()
                    sell_qty = st.number_input(f"매도 수량 (보유: {state['holdings']['quantity']}주)", min_value=0, max_value=state['holdings']['quantity'], step=10, key="sell_qty_market")
                    if st.button("🟥 수량으로 매도", use_container_width=True, key="sell_by_qty_btn"):
                        if sell_qty > 0:
                            revenue = sell_qty * current_price; current_avg_price = state['holdings']['avg_price']; state['cash'] = float(state['cash'] + revenue); state['holdings']['quantity'] = int(state['holdings']['quantity'] - sell_qty)
                            if state['holdings']['quantity'] == 0: state['holdings']['avg_price'] = 0
                            state['trade_log'].append({"일자": current_date.date().isoformat(), "유형": "시장가매도", "수량": sell_qty, "단가": int(current_price), "금액": int(revenue), "avg_price_at_sale": float(current_avg_price)}); st.rerun()
                with tab2:
                    buy_amount = st.number_input("매수 금액 (원)", min_value=0, max_value=int(state['cash']), step=100000, key="buy_amount_market")
                    if st.button("🟩 금액으로 매수", use_container_width=True, key="buy_by_amount_btn"):
                        if buy_amount > 0 and current_price > 0:
                            buy_qty_from_amount = int(buy_amount // current_price)
                            if buy_qty_from_amount > 0:
                                cost = buy_qty_from_amount * current_price; total_qty = state['holdings']['quantity'] + buy_qty_from_amount; avg_price = (state['holdings']['avg_price'] * state['holdings']['quantity'] + cost) / total_qty
                                state.update({'holdings': {'quantity': int(total_qty), 'avg_price': float(avg_price)}, 'cash': float(state['cash'] - cost)})
                                state['trade_log'].append({"일자": current_date.date().isoformat(), "유형": "시장가매수", "수량": buy_qty_from_amount, "단가": int(current_price), "금액": int(cost)}); st.success(f"{buy_qty_from_amount}주 매수 실행 완료."); st.rerun()
                            else: st.warning("입력한 금액으로 최소 1주도 매수할 수 없습니다.")
                    max_sell_amount = int(state['holdings']['quantity'] * current_price)
                    sell_amount = st.number_input("매도 금액 (원)", min_value=0, max_value=max_sell_amount, step=100000, key="sell_amount_market")
                    if st.button("🟥 금액으로 매도", use_container_width=True, key="sell_by_amount_btn"):
                        if sell_amount > 0 and current_price > 0:
                            sell_qty_from_amount = int(sell_amount // current_price)
                            if sell_qty_from_amount > 0 and sell_qty_from_amount <= state['holdings']['quantity']:
                                revenue = sell_qty_from_amount * current_price; current_avg_price = state['holdings']['avg_price']; state['cash'] = float(state['cash'] + revenue); state['holdings']['quantity'] = int(state['holdings']['quantity'] - sell_qty_from_amount)
                                if state['holdings']['quantity'] == 0: state['holdings']['avg_price'] = 0
                                state['trade_log'].append({"일자": current_date.date().isoformat(), "유형": "시장가매도", "수량": sell_qty_from_amount, "단가": int(current_price), "금액": int(revenue), "avg_price_at_sale": float(current_avg_price)}); st.success(f"{sell_qty_from_amount}주 매도 실행 완료."); st.rerun()
                            elif sell_qty_from_amount > state['holdings']['quantity']: st.warning(f"입력한 금액은 보유 수량을 초과합니다. (최대: {max_sell_amount:,}원)")
                            else: st.warning("입력한 금액으로 최소 1주도 매도할 수 없습니다.")
            
            with st.expander("🕰️ 예약 매매 (지정가)", expanded=False):
                limit_buy_price = st.number_input("매수 희망 가격", value=int(current_price * 0.95), step=100); limit_buy_qty = st.number_input("매수 수량", min_value=0, step=10, key="limit_buy_qty")
                if st.button(" 예약 매수", use_container_width=True, key="limit_buy_btn"):
                    if limit_buy_price > 0 and limit_buy_qty > 0:
                        state["pending_orders"].append({"id": time.time(), "type": "buy", "price": limit_buy_price, "qty": limit_buy_qty}); st.success(f"{int(limit_buy_price):,}원에 {limit_buy_qty}주 매수 예약 완료!"); st.rerun()
                limit_sell_price = st.number_input("매도 희망 가격", value=int(current_price * 1.05), step=100); limit_sell_qty = st.number_input("매도 수량", min_value=0, step=10, key="limit_sell_qty")
                if st.button(" 예약 매도", use_container_width=True, key="limit_sell_btn"):
                    if limit_sell_price > 0 and limit_sell_qty > 0 and limit_sell_qty <= state['holdings']['quantity']:
                        state["pending_orders"].append({"id": time.time(), "type": "sell", "price": limit_sell_price, "qty": limit_sell_qty}); st.success(f"{int(limit_sell_price):,}원에 {limit_sell_qty}주 매도 예약 완료!"); st.rerun()
                    elif limit_sell_qty > state['holdings']['quantity']: st.warning("보유 수량보다 많이 예약할 수 없습니다.")
            
            with st.expander("🤖 자동 매매 (손절/익절)", expanded=True):
                st.info("자동 매매 주문은 현재 보유한 모든 수량에 대해 적용됩니다."); sl_price = st.number_input("손절 가격 (Stop-Loss)", value=0, step=100)
                if st.button("스탑로스 설정", use_container_width=True, key="sl_btn"):
                    if sl_price > 0 and state['holdings']['quantity'] > 0:
                        state["pending_orders"] = [o for o in state["pending_orders"] if o['type'] not in ['stop_loss', 'trailing_stop']]; state["pending_orders"].append({"id": time.time(), "type": "stop_loss", "price": sl_price, "qty": state['holdings']['quantity']})
                        st.success(f"{int(sl_price):,}원에 스탑로스 설정 완료!"); st.rerun()
                ts_percent = st.number_input("추적 손절 하락률 (%)", min_value=0.1, max_value=50.0, value=10.0, step=0.5, format="%.1f")
                if st.button("트레일링 스탑 설정", use_container_width=True, key="ts_btn"):
                    if ts_percent > 0 and state['holdings']['quantity'] > 0:
                        state["pending_orders"] = [o for o in state["pending_orders"] if o['type'] not in ['stop_loss', 'trailing_stop']]; state["pending_orders"].append({"id": time.time(), "type": "trailing_stop", "percentage": ts_percent, "qty": state['holdings']['quantity'], "peak_price": current_price})
                        st.success(f"고점 대비 {ts_percent}% 하락 시 매도 설정 완료!"); st.rerun()

            if state["pending_orders"]:
                st.markdown("---"); st.subheader("📋 예약 주문 목록")
                for order in state["pending_orders"]:
                    col_order, col_cancel = st.columns([4, 1]); order_type_map = {"buy": "지정가 매수", "sell": "지정가 매도", "stop_loss": "스탑로스", "trailing_stop": "트레일링 스탑"}; order_text = f" - {order_type_map.get(order['type'])}: {order['qty']}주"
                    if order['type'] in ['buy', 'sell', 'stop_loss']: order_text += f" @ {int(order['price']):,}원"
                    elif order['type'] == 'trailing_stop':
                        stop_price = int(order.get('peak_price', current_price) * (1 - order['percentage']/100)); order_text += f" ({order['percentage']}%) / 현재가: {stop_price:,}원"
                    col_order.write(order_text)
                    if col_cancel.button("취소", key=f"cancel_{order['id']}", use_container_width=True):
                        state["pending_orders"] = [o for o in state["pending_orders"] if o['id'] != order['id']]; st.rerun()
        
        st.markdown("<hr>", unsafe_allow_html=True); st.subheader("🚀 성과 리포트")
        performance = calculate_performance(state, current_price); perf_cols = st.columns(6)
        perf_cols[0].metric("현재 총 자산", f"{performance['현재 총 자산']:,}원"); perf_cols[1].metric("누적 수익률 (%)", f"{performance['누적 수익률 (%)']}%"); perf_cols[2].metric("총 실현 손익", f"{performance['총 실현 손익']:,}원")
        perf_cols[3].metric("승률 (%)", f"{performance['승률 (%)']}%"); perf_cols[4].metric("최대 손실률 (MDD, %)", f"{performance['최대 손실률 (MDD, %)']}%"); perf_cols[5].metric("총 매도 거래", f"{performance['총 매도 거래 횟수']}회")
        if state["trade_log"]:
            with st.expander("📒 매매 기록 상세보기"):
                log_df = pd.DataFrame(state["trade_log"]); display_cols = ['일자', '유형', '수량', '단가', '금액']; st.dataframe(log_df[display_cols], use_container_width=True, hide_index=True)
        save_state(state)
except Exception as e:
    st.error(f"❌ 시뮬레이션 실행 중 오류가 발생했습니다: {e}")
