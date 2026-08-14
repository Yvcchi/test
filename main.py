import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from vnstock import Quote, Finance, Market

# ==========================================
# HÀM TÍNH CHỈ BÁO KỸ THUẬT & THANH KHOẢN
# ==========================================
def calculate_technical_indicators(df_price):
    """Tính ADTV20, Volatility, MA200, RSI14, Momentum 6M từ DF giá lịch sử"""
    if df_price is None or len(df_price) < 200: # Cần tối thiểu 200 phiên cho MA200
        return None

    price_col = 'close' if 'close' in df_price.columns else df_price.columns[1]
    volume_col = 'volume' if 'volume' in df_price.columns else df_price.columns[2]

    df = df_price.copy()
    df['close_num'] = df[price_col].astype(float)
    df['vol_num'] = df[volume_col].astype(float)

    # 1. Thanh khoản trung bình 20 ngày (ADTV 20)
    adtv_20 = df['vol_num'].tail(20).mean()

    # 2. Độ biến động (Volatility 30 ngày) - Độ lệch chuẩn chuỗi lợi nhuận theo ngày
    daily_returns = df['close_num'].pct_change()
    volatility_30 = daily_returns.tail(30).std() * np.sqrt(252) # Chuẩn hóa năm

    # 3. MA200
    ma200 = df['close_num'].rolling(window=200).mean().iloc[-1]
    current_price = df['close_num'].iloc[-1]

    # 4. Momentum 6M (126 phiên giao dịch)
    price_6m_ago = df['close_num'].iloc[-126] if len(df) >= 126 else df['close_num'].iloc[0]
    momentum_6m = (current_price - price_6m_ago) / price_6m_ago

    # 5. RSI 14 ngày
    delta = df['close_num'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    rsi_14 = 100 - (100 / (1 + rs)).iloc[-1]

    return {
        'current_price': current_price,
        'adtv_20': adtv_20,
        'volatility_30': volatility_30,
        'ma200': ma200,
        'momentum_6m': momentum_6m,
        'rsi_14': rsi_14
    }

# ==========================================
# HÀM LẤY VÀ XỬ LÝ BCTC
# ==========================================
def extract_financial_ratios(df_ratio):
    """Trích xuất 13 chỉ số tài chính từ dataframe BCTC của vnstock"""
    if df_ratio is None or df_ratio.empty:
        return None

    row = df_ratio.iloc[0]
    
    # Hàm đọc chỉ số an toàn theo danh sách tên cột có thể xuất hiện
    def get_val(keys, default=np.nan):
        for k in keys:
            for col in df_ratio.columns:
                if k.lower() in str(col).lower():
                    try:
                        v = float(row[col])
                        if not np.isnan(v): return v
                    except: pass
        return default

    pe = get_val(['priceToEarning', 'p/e', 'pe'])
    pb = get_val(['priceToBook', 'p/b', 'pb'])
    div_yield = get_val(['dividendYield', 'ty_le_co_tuc', 'dividend'])
    ev_ebitda = get_val(['ev/ebitda', 'evebitda'])

    roe = get_val(['roe'])
    roa = get_val(['roa'])
    roic = get_val(['roic'])
    de = get_val(['debtToEquity', 'd/e', 'no_vcshe'])
    gross_margin = get_val(['grossMargin', 'bien_loi_nhuan_gop'])
    net_margin = get_val(['netMargin', 'bien_loi_nhuan_rong'])
    ocf_ni = get_val(['ocf/ni', 'cash_flow_quality'], default=1.0) # Dòng tiền OCF / NI

    rev_growth = get_val(['revenueGrowth', 'tang_truong_doanh_thu'])
    net_inc_growth = get_val(['netIncomeGrowth', 'tang_truong_loi_nhuan'])
    eps_growth = get_val(['epsGrowth', 'tang_truong_eps'])

    return {
        'PE': pe, 'PB': pb, 'Div_Yield': div_yield, 'EV_EBITDA': ev_ebitda,
        'ROE': roe, 'ROA': roa, 'ROIC': roic, 'DE': de,
        'Gross_Margin': gross_margin, 'Net_Margin': net_margin, 'OCF_NI': ocf_ni,
        'Rev_Growth': rev_growth, 'Net_Inc_Growth': net_inc_growth, 'EPS_Growth': eps_growth
    }

# ==========================================
# SÀNG LỌC VÀ CHẤM ĐIỂM QUANT MULTI-FACTOR
# ==========================================
def quant_multi_factor_screener(top_n=10):
    print("--- BẮT ĐẦU SÀNG LỌC QUANT MULTI-FACTOR TOÀN DIỆN (13 TIÊU CHÍ) ---")
    
    mkt = Market()
    df_symbols = mkt.listing_symbols()
    hose_tickers = df_symbols[df_symbols['organ_code'] == 'HOSE']['ticker'].tolist() if 'organ_code' in df_symbols.columns else df_symbols['ticker'].tolist()
    
    # Giới hạn lấy danh sách mẫu 30 mã lớn trên HOSE để đảm bảo thời gian chạy demo công khai
    sample_tickers = hose_tickers[:30] 
    
    raw_data = []
    today = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d') # Lấy 1 năm cho đủ 200 phiên

    for i, ticker in enumerate(sample_tickers):
        if i > 0 and i % 3 == 0:
            print("⏳ Đang tạm dừng 35 giây để reset Rate Limit...")
            time.sleep(35)

        print(f"[{i+1}/{len(sample_tickers)}] Đang xử lý mã: {ticker}...")
        
        try:
            # 1. Lấy giá lịch sử
            q = Quote(symbol=ticker, source='VCI')
            df_price = q.history(start=start_date, end=today, interval='1D')
            tech = calculate_technical_indicators(df_price)
            time.sleep(2)

            # 2. Lấy BCTC
            f = Finance(symbol=ticker, source='VCI')
            df_ratio = f.ratio(period='year', lang='vi')
            ratios = extract_financial_ratios(df_ratio)
            time.sleep(2)

            if tech is None or ratios is None:
                continue

            # ==========================================
            # BƯỚC LỌC CỨNG (HARD FILTERS / LIQUIDITY)
            # ==========================================
            # 1. Lọc thanh khoản ADTV20 < 100k cổ/ngày
            if tech['adtv_20'] < 100000:
                print(f"  ❌ Loại {ticker}: ADTV20 quá thấp ({tech['adtv_20']:.0f} cổ/ngày)")
                continue

            # 2. Lọc Bắt dao rơi (Giá < MA200 hoặc RSI < 35)
            if tech['current_price'] < tech['ma200'] or tech['rsi_14'] < 35:
                print(f"  ❌ Loại {ticker}: Xu hướng xấu (Giá < MA200 hoặc RSI < 35)")
                continue

            # 3. Lọc đòn bẩy quá cao (D/E > 2.5)
            if not np.isnan(ratios['DE']) and ratios['DE'] > 2.5:
                print(f"  ❌ Loại {ticker}: Nợ quá cao (D/E = {ratios['DE']:.2f})")
                continue

            # Gộp dữ liệu hợp lệ
            entry = {'Ticker': ticker, **tech, **ratios}
            raw_data.append(entry)
            print(f"  ✓ Qua vòng sơ loại: {ticker}")

        except Exception as e:
            print(f"  └─ Lỗi {ticker}: {e}")
            continue

    df = pd.DataFrame(raw_data)
    if df.empty:
        print("Không có cổ phiếu nào vượt qua vòng lọc cứng.")
        return

    # ==========================================
    # CHUẨN HÓA TÍNH Z-SCORE VÀ TỔNG HỢP ĐIỂM QUANT
    # ==========================================
    def z_score(series, invert=False):
        std = series.std()
        if std == 0 or np.isnan(std): return series * 0
        z = (series - series.mean()) / std
        return -z if invert else z

    # 1. Valuation Z-Score (P/E, P/B đảo chiều; Div_Yield thuận chiều)
    z_pe = z_score(df['PE'].fillna(df['PE'].median()), invert=True)
    z_pb = z_score(df['PB'].fillna(df['PB'].median()), invert=True)
    z_div = z_score(df['Div_Yield'].fillna(0))
    df['Z_Valuation'] = (z_pe + z_pb + z_div) / 3

    # 2. Quality Z-Score (ROE, ROIC, ROA, Margins, OCF)
    z_roe = z_score(df['ROE'].fillna(df['ROE'].median()))
    z_roic = z_score(df['ROIC'].fillna(df['ROIC'].median()))
    z_roa = z_score(df['ROA'].fillna(df['ROA'].median()))
    z_margin = z_score(df['Net_Margin'].fillna(df['Net_Margin'].median()))
    df['Z_Quality'] = (z_roe + z_roic + z_roa + z_margin) / 4

    # 3. Growth Z-Score
    z_rev_g = z_score(df['Rev_Growth'].fillna(0))
    z_net_g = z_score(df['Net_Inc_Growth'].fillna(0))
    df['Z_Growth'] = (z_rev_g + z_net_g) / 2

    # 4. Risk-Adjusted Momentum Z-Score (Momentum thuận, Volatility đảo)
    z_mom = z_score(df['momentum_6m'])
    z_vol = z_score(df['volatility_30'], invert=True)
    df['Z_Momentum'] = (z_mom + z_vol) / 2

    # TỔNG HỢP QUANT MULTI-FACTOR SCORE (Tỷ trọng: 25% Val, 35% Qual, 25% Growth, 15% Mom)
    df['Quant_Score'] = (0.25 * df['Z_Valuation'] + 
                         0.35 * df['Z_Quality'] + 
                         0.25 * df['Z_Growth'] + 
                         0.15 * df['Z_Momentum'])

    # Bảng kết quả xếp hạng
    df_ranked = df.sort_values(by='Quant_Score', ascending=False).reset_index(drop=True)
    top_stocks = df_ranked.head(top_n)

    print("\n================ TOP CỔ PHIẾU QUANT MULTI-FACTOR ================")
    print(top_stocks[['Ticker', 'current_price', 'PE', 'ROE', 'Rev_Growth', 'momentum_6m', 'Quant_Score']])

    top_stocks.to_csv('top_stocks.csv', index=False)
    print("\nĐã lưu kết quả Quant nâng cao vào top_stocks.csv")

if __name__ == '__main__':
    quant_multi_factor_screener(top_n=10)
