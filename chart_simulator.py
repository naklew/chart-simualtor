# --- 사이드바 (모든 key 추가로 오류 최종 수정) ---
st.sidebar.header("🔧 시뮬레이션 설정")
ticker = st.sidebar.text_input("종목 코드", state.get("ticker", "005930"))
st.sidebar.markdown(f"**선택 종목:** {get_stock_name(ticker)} ({ticker})")

start_date_iso = state.get("start_date", datetime.date(2020, 1, 1).isoformat())
end_date_iso = state.get("end_date", datetime.date(2023, 12, 31).isoformat())
start_date = st.sidebar.date_input("시작 날짜", datetime.date.fromisoformat(start_date_iso))
end_date = st.sidebar.date_input("종료 날짜", datetime.date.fromisoformat(end_date_iso))

if (ticker != state.get("ticker") or start_date.isoformat() != start_date_iso or end_date.isoformat() != end_date_iso):
    state.update({"ticker": ticker, "start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "day_index": MIN_DATA_PERIOD, "pending_orders": [], "trade_log": []})
    st.rerun()

st.sidebar.subheader("🎲 랜덤 리셋")
if st.sidebar.button("다른 종목/구간으로 새로 시작 (자산 유지)", key="random_reset_btn"):
    if state['holdings']['quantity'] > 0:
        full_df = load_data(state['ticker'], datetime.date.fromisoformat(state['start_date']), datetime.date.fromisoformat(state['end_date']))
        if not full_df.empty and state['day_index'] < len(full_df):
            last_price = full_df.iloc[state['day_index']]['Close']
            state['cash'] += state['holdings']['quantity'] * last_price
            st.sidebar.info(f"기존 포지션 자동 정산 완료.")
    state.update({'holdings': {"quantity": 0, "avg_price": 0}, 'trade_log': [], 'daily_portfolio_value': [], 'day_index': MIN_DATA_PERIOD, 'pending_orders': []})
    all_tickers = get_all_tickers()
    while True:
        new_ticker = random.choice(all_tickers)
        full_history = load_data(new_ticker, datetime.date(2010, 1, 1), datetime.date.today())
        if len(full_history) > MIN_DATA_PERIOD + MIN_FUTURE_PERIOD:
            max_start_index = len(full_history) - (MIN_DATA_PERIOD + MIN_FUTURE_PERIOD)
            random_start_index = random.randint(0, max_start_index)
            state['ticker'] = new_ticker
            state['start_date'] = full_history['날짜'].iloc[random_start_index].date().isoformat()
            state['end_date'] = full_history['날짜'].iloc[-1].date().isoformat()
            break
    save_state(state)
    st.rerun()

st.sidebar.subheader("⚠️ 위험 구역")
# [수정] checkbox에도 고유한 key를 추가합니다.
reset_confirmation = st.sidebar.checkbox("모든 매매 기록과 자산을 초기화하려면 체크하세요.", key="reset_confirm_checkbox")
if reset_confirmation:
    if st.sidebar.button("모든 기록 초기화 실행", type="primary", key="full_reset_btn"):
        initial_state = {
            "cash": INITIAL_CASH,
            "holdings": {"quantity": 0, "avg_price": 0},
            "trade_log": [],
            "day_index": MIN_DATA_PERIOD,
            "ticker": "005930",
            "start_date": datetime.date(2020, 1, 1).isoformat(),
            "end_date": datetime.date(2023, 12, 31).isoformat(),
            "daily_portfolio_value": [],
            "pending_orders": []
        }
        st.session_state.state = initial_state
        save_state(initial_state)
        st.sidebar.success("모든 기록이 초기화되었습니다. 페이지가 새로고침됩니다.")
        time.sleep(2)
        st.rerun()
