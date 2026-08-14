import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
from vnstock import Quote, Finance, Market, config

VNSTOCK_KEY = os.getenv('VNSTOCK_API_KEY', 'vnstock_9a7c79177954459d6c1109f6ee8590e6')
try:
    config.set_api_key(VNSTOCK_KEY)
    print(f"✓ Đã cấu hình Vnstock API Key thành công!")
except Exception as e:
    print(f"⚠️ Lỗi cấu hình API Key: {e}")
# ==========================================
# HÀM ĐỌC TRỌNG SỐ TỐI ƯU TỪ PSO (CONFIG)
# ==========================================
def load_optimal_weights():
    config_path = 'config.json'
    default_weights = {
        'Z_Valuation': 0.25,
        'Z_Quality': 0.35,
        'Z_Growth': 0.25,
        'Z_Momentum': 0.15
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                weights = config.get('weights', {})
                last_updated = config.get('last_updated', 'N/A')
                
                required_keys = ['Z_Valuation', 'Z_Quality', 'Z_Growth', 'Z_Momentum']
                if all(k in weights for k in required_keys):
                    print(f"✓ Đã tải thành công trọng số tối ưu PSO (Cập nhật ngày: {last_updated})")
                    print(f"  Trọng số: {weights}")
                    return weights
        except Exception as e:
            print(f"⚠️ Lỗi đọc {config_path}: {e}. Chuyển sang trọng số mặc định.")

    print("ℹ️ File config.json không khả dụng. Sử dụng bộ trọng số mặc định.")
    return default_weights

# ==========================================
# HÀM TÍNH CHỈ BÁO KỸ THUẬT & THANH KHOẢN
# ==========================================
def calculate_technical_indicators(df_price):
    if df_price is None or len(df_price) < 200:
        return None

    price_col = 'close' if 'close' in df_price.columns else df_price.columns[1]
    volume_col = 'volume' if 'volume' in df_price.columns else df_price.columns[2]

    df = df_price.copy()
    df['close_num'] = df[price_col].astype(float)
    df['vol_num'] = df[volume_col].astype(float)

    adtv_20 = df['vol_num'].tail(20).mean()
    daily_returns = df['close_num'].pct_change()
    volatility_30 = daily_returns.tail(30).std() * np.sqrt(252)

    ma200 = df['close_num'].rolling(window=200).mean().iloc[-1]
    current_price = df['close_num'].iloc[-1]

    price_6m_ago = df['close_num'].iloc[-126] if len(df) >= 126 else df['close_num'].iloc[0]
    momentum_6m = (current_price - price_6m_ago) / price_6m_ago

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
    if df_ratio is None or df_ratio.empty:
        return None

    row = df_ratio.iloc[0]
    
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
    ocf_ni = get_val(['ocf/ni', 'cash_flow_quality'], default=1.0)

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
def quant_multi_factor_screener(top_n=20):
    print("--- BẮT ĐẦU SÀNG LỌC QUANT MULTI-FACTOR TOÀN DIỆN ---")
    weights = load_optimal_weights()

    # Lấy danh sách cổ phiếu
    hose_tickers = ['ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
                    'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SSI', 'SSB', 'STB', 'TCB', 
                    'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VNM', 'VRE']
    
    try:
        from vnstock import listing_companies
        df_symbols = listing_companies()
        if 'ticker' in df_symbols.columns:
            if 'comGroupCode' in df_symbols.columns:
                hose_tickers = df_symbols[df_symbols['comGroupCode'] == 'HOSE']['ticker'].tolist()
            else:
                hose_tickers = df_symbols['ticker'].tolist()
    except Exception as e:
        print(f"Dùng danh sách mặc định do lỗi: {e}")

    sample_tickers = hose_tickers
    print(f"Đã chuẩn bị {len(sample_tickers)} mã cổ phiếu để kiểm tra.")

    raw_data = []
    today = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    for i, ticker in enumerate(sample_tickers):
        # Giảm time.sleep xuống 1.5 giây mỗi mã để tránh bị đứng quá lâu
        time.sleep(1.5)
        print(f"[{i+1}/{len(sample_tickers)}] Đang xử lý mã: {ticker}...")
        
        try:
            q = Quote(symbol=ticker, source='VCI')
            df_price = q.history(start=start_date, end=today, interval='1D')
            tech = calculate_technical_indicators(df_price)

            f = Finance(symbol=ticker, source='VCI')
            df_ratio = f.ratio(period='year', lang='vi')
            ratios = extract_financial_ratios(df_ratio)

            if tech is None or ratios is None:
                continue

            # Hard Filters (Lọc cứng)
            if tech['adtv_20'] < 100000: continue
            if tech['current_price'] < tech['ma200'] or tech['rsi_14'] < 35: continue
            if not np.isnan(ratios['DE']) and ratios['DE'] > 2.5: continue

            raw_data.append({'Ticker': ticker, **tech, **ratios})
            print(f"  ✓ Qua vòng sơ loại: {ticker}")

        except Exception as e:
            print(f"  └─ Bỏ qua {ticker} do lỗi: {e}")
            continue

    df = pd.DataFrame(raw_data)
    if df.empty:
        print("❌ Không có cổ phiếu nào vượt qua vòng lọc cứng.")
        pd.DataFrame([{'Ticker': 'N/A', 'Note': 'No qualified stocks found'}]).to_csv('top_stocks.csv', index=False)
        return

    # Hàm chuẩn hóa Z-Score
    def z_score(series, invert=False):
        std = series.std()
        if std == 0 or np.isnan(std): return series * 0
        z = (series - series.mean()) / std
        return -z if invert else z

    # 1. Valuation Z-Score
    z_pe = z_score(df['PE'].fillna(df['PE'].median()), invert=True)
    z_pb = z_score(df['PB'].fillna(df['PB'].median()), invert=True)
    z_div = z_score(df['Div_Yield'].fillna(0))
    df['Z_Valuation'] = (z_pe + z_pb + z_div) / 3

    # 2. Quality Z-Score
    z_roe = z_score(df['ROE'].fillna(df['ROE'].median()))
    z_roic = z_score(df['ROIC'].fillna(df['ROIC'].median()))
    z_roa = z_score(df['ROA'].fillna(df['ROA'].median()))
    z_margin = z_score(df['Net_Margin'].fillna(df['Net_Margin'].median()))
    df['Z_Quality'] = (z_roe + z_roic + z_roa + z_margin) / 4

    # 3. Growth Z-Score
    z_rev_g = z_score(df['Rev_Growth'].fillna(0))
    z_net_g = z_score(df['Net_Inc_Growth'].fillna(0))
    df['Z_Growth'] = (z_rev_g + z_net_g) / 2

    # 4. Momentum Z-Score
    z_mom = z_score(df['momentum_6m'])
    z_vol = z_score(df['volatility_30'], invert=True)
    df['Z_Momentum'] = (z_mom + z_vol) / 2

    # 5. Tính Quant Score
    df['Quant_Score'] = (
        weights['Z_Valuation'] * df['Z_Valuation'] + 
        weights['Z_Quality']   * df['Z_Quality']   + 
        weights['Z_Growth']    * df['Z_Growth']    + 
        weights['Z_Momentum']  * df['Z_Momentum']
    )

    df_ranked = df.sort_values(by='Quant_Score', ascending=False).reset_index(drop=True)
    top_stocks = df_ranked.head(top_n) if top_n else df_ranked

    print("\n================ TOP CỔ PHIẾU QUANT MULTI-FACTOR ================")
    print(top_stocks[['Ticker', 'current_price', 'PE', 'ROE', 'Rev_Growth', 'momentum_6m', 'Quant_Score']])

    # Ghi đè trực tiếp và ép cập nhật timestamp
    top_stocks.to_csv('top_stocks.csv', index=False)
    print(f"\n✅ Đã lưu thành công {len(top_stocks)} mã cổ phiếu vào top_stocks.csv lúc {datetime.now().strftime('%H:%M:%S')}")

if __name__ == '__main__':
    quant_multi_factor_screener(top_n=20)
