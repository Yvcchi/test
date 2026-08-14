import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from vnstock import Quote, Finance, Market

def fetch_data_safely(symbol):
    """Lấy BCTC và Giá lịch sử an toàn, kiểm soát nhịp độ gọi API"""
    today = datetime.now().strftime('%Y-%m-%d')
    six_months_ago = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    df_ratio = None
    df_price = None

    # 1. Lấy BCTC (dùng VCI, nếu lỗi thử TCBS)
    for src in ['VCI', 'TCBS']:
        try:
            f = Finance(symbol=symbol, source=src)
            df_ratio = f.ratio(period='year', lang='vi')
            if df_ratio is not None and not df_ratio.empty:
                break
        except Exception:
            pass
        time.sleep(2)

    time.sleep(2)

    # 2. Lấy giá lịch sử
    for src in ['VCI', 'TCBS']:
        try:
            q = Quote(symbol=symbol, source=src)
            df_price = q.history(start=six_months_ago, end=today, interval='1D')
            if df_price is not None and not df_price.empty:
                break
        except Exception:
            pass
        time.sleep(2)

    return df_ratio, df_price

def quant_stock_screener(top_n=10):
    print("--- BẮT ĐẦU QUÁ TRÌNH SÀNG LỌC TOÀN BỘ CỔ PHIẾU HOSE ---")
    
    # 1. Lấy danh sách toàn bộ cổ phiếu trên sàn HOSE
    try:
        mkt = Market()
        df_symbols = mkt.listing_symbols()
        if 'organ_code' in df_symbols.columns:
            hose_tickers = df_symbols[df_symbols['organ_code'] == 'HOSE']['ticker'].tolist()
        else:
            hose_tickers = df_symbols['ticker'].tolist()
        print(f" Tìm thấy {len(hose_tickers)} cổ phiếu trên sàn HOSE.")
    except Exception as e:
        print(f"Lỗi lấy danh sách cổ phiếu: {e}")
        return

    screening_results = []

    # 2. Vòng lặp quét qua từng mã cổ phiếu
    for i, ticker in enumerate(hose_tickers):
        # Sau mỗi 3 mã cổ phiếu, tạm dừng 35 giây để bộ đếm Rate Limit của Vnstock reset
        if i > 0 and i % 3 == 0:
            print(f"⏳ Đã xử lý {i}/{len(hose_tickers)} mã. Tạm dừng 35 giây để reset Rate Limit...")
            time.sleep(35)

        print(f"[{i+1}/{len(hose_tickers)}] Đang quét mã: {ticker}...")
        df_ratio, df_price = fetch_data_safely(ticker)

        if df_ratio is None or df_ratio.empty or df_price is None or df_price.empty or len(df_price) < 20:
            continue
            
        latest_ratio = df_ratio.iloc[0]
        
        # Đọc P/E và ROE
        pe = np.nan
        roe = np.nan
        for col in df_ratio.columns:
            col_str = str(col).lower()
            if 'price' in col_str or 'pe' in col_str or 'p/e' in col_str:
                try: pe = float(latest_ratio[col])
                except: pass
            if 'roe' in col_str:
                try: roe = float(latest_ratio[col])
                except: pass

        # Đọc Momentum 6M
        price_col = 'close' if 'close' in df_price.columns else df_price.columns[1]
        price_start = float(df_price.iloc[0][price_col])
        price_end = float(df_price.iloc[-1][price_col])
        momentum_6m = (price_end - price_start) / price_start

        # Bỏ qua các mã lỗ (PE <= 0) hoặc không có dữ liệu PE/ROE
        if np.isnan(pe) or pe <= 0 or np.isnan(roe):
            continue

        screening_results.append({
            'Ticker': ticker,
            'PE': pe,
            'ROE': roe,
            'Momentum_6M': momentum_6m
        })

    df_res = pd.DataFrame(screening_results)
    if df_res.empty:
        print("Không tìm thấy dữ liệu phù hợp.")
        pd.DataFrame([{'Ticker': 'N/A', 'Note': 'No data'}]).to_csv('top_stocks.csv', index=False)
        return

    # 3. Tính Quant Score (Z-Score Normalization)
    df_res['Z_PE'] = -1 * (df_res['PE'] - df_res['PE'].mean()) / (df_res['PE'].std() + 1e-6)
    df_res['Z_ROE'] = (df_res['ROE'] - df_res['ROE'].mean()) / (df_res['ROE'].std() + 1e-6)
    df_res['Z_Momentum'] = (df_res['Momentum_6M'] - df_res['Momentum_6M'].mean()) / (df_res['Momentum_6M'].std() + 1e-6)

    # Trọng số: 30% Value, 40% Quality, 30% Momentum
    df_res['Quant_Score'] = (0.30 * df_res['Z_PE']) + (0.40 * df_res['Z_ROE']) + (0.30 * df_res['Z_Momentum'])

    # 4. Xuất Top 10 Cổ Phếu Tốt Nhất ra CSV
    df_ranked = df_res.sort_values(by='Quant_Score', ascending=False).reset_index(drop=True)
    top_stocks = df_ranked.head(top_n)
    
    print("\n================ TOP CỔ PHIẾU HOSE ĐƯỢC LỌC ================")
    print(top_stocks[['Ticker', 'PE', 'ROE', 'Momentum_6M', 'Quant_Score']])
    
    top_stocks.to_csv('top_stocks.csv', index=False)
    print("\nĐã lưu kết quả thành công vào file top_stocks.csv")

if __name__ == '__main__':
    quant_stock_screener(top_n=10)
