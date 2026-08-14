import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Import trực tiếp các class chính từ gói vnstock
from vnstock import Quote, Finance, Market

def get_data_with_retry(fetch_func, max_retries=3, delay=35):
    """Hàm hỗ trợ: Tự động đợi và thử lại nếu bị dính Rate Limit"""
    for attempt in range(max_retries):
        try:
            res = fetch_func()
            if res is not None and not res.empty:
                return res
        except Exception as e:
            err_str = str(e)
            if "Rate limit" in err_str or "20/20" in err_str:
                print(f"⚠️ Dính Rate Limit. Đang chờ {delay} giây...")
                time.sleep(delay)
            else:
                raise e
    return None

def quant_stock_screener(top_n=5):
    print("--- BẮT ĐẦU QUÁ TRÌNH SÀNG LỌC CỔ PHIẾU QUANT ---")
    
    # 1. Khởi tạo module Market
    try:
        mkt = Market(source='VCI')
        df_companies = mkt.listing_symbols()
        if 'organ_code' in df_companies.columns:
            hose_tickers = df_companies[df_companies['organ_code'] == 'HOSE']['ticker'].tolist()
        else:
            hose_tickers = df_companies['ticker'].tolist()
    except Exception as e:
        print(f"Lỗi lấy danh sách cổ phiếu từ API: {e}. Sử dụng danh sách mặc định.")
        hose_tickers = ['VNM', 'HPG', 'FPT', 'TCB', 'MBB', 'MWG', 'MSN', 'REE', 'VHM', 'ACB']

    # Chọn 8 mã chạy demo để không bị vượt trần 20 requests/phút của gói Guest
    sample_tickers = hose_tickers[:8] if hose_tickers else ['VNM', 'HPG', 'FPT', 'TCB', 'MBB']
    
    screening_results = []
    today = datetime.now().strftime('%Y-%m-%d')
    six_months_ago = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

    for ticker in sample_tickers:
        try:
            print(f"Đang xử lý mã: {ticker}...")
            
            q = Quote(symbol=ticker, source='VCI')
            f = Finance(symbol=ticker, source='VCI')
            
            # 2. Lấy BCTC
            df_ratio = get_data_with_retry(lambda: f.ratio(period='year', lang='vi'))
            time.sleep(2) 
            
            # 3. Lấy giá lịch sử
            df_price = get_data_with_retry(lambda: q.history(start=six_months_ago, end=today, interval='1D'))

            if df_ratio is None or df_price is None or len(df_price) < 20:
                print(f"Bỏ qua mã {ticker} do thiếu dữ liệu.")
                time.sleep(2)
                continue
                
            latest_ratio = df_ratio.iloc[0]
            pe = float(latest_ratio.get('priceToEarning', latest_ratio.get('P/E', np.nan)))
            roe = float(latest_ratio.get('roe', latest_ratio.get('ROE', np.nan)))

            price_start = df_price.iloc[0]['close']
            price_end = df_price.iloc[-1]['close']
            momentum_6m = (price_end - price_start) / price_start

            if np.isnan(pe) or np.isnan(roe) or pe <= 0:
                time.sleep(2)
                continue

            screening_results.append({
                'Ticker': ticker,
                'PE': pe,
                'ROE': roe,
                'Momentum_6M': momentum_6m
            })
            
            time.sleep(3) # Nghỉ giữa các mã
            
        except Exception as e:
            print(f"Bỏ qua mã {ticker} do lỗi: {e}")
            time.sleep(3)
            continue

    df_res = pd.DataFrame(screening_results)
    if df_res.empty:
        print("Không tìm thấy dữ liệu phù hợp.")
        return

    # 4. Tính Quant Score (Z-Score)
    df_res['Z_PE'] = -1 * (df_res['PE'] - df_res['PE'].mean()) / (df_res['PE'].std() + 1e-6)
    df_res['Z_ROE'] = (df_res['ROE'] - df_res['ROE'].mean()) / (df_res['ROE'].std() + 1e-6)
    df_res['Z_Momentum'] = (df_res['Momentum_6M'] - df_res['Momentum_6M'].mean()) / (df_res['Momentum_6M'].std() + 1e-6)

    # Trọng số: 30% PE, 40% ROE, 30% Momentum
    df_res['Quant_Score'] = (0.30 * df_res['Z_PE']) + (0.40 * df_res['Z_ROE']) + (0.30 * df_res['Z_Momentum'])

    # 5. Xuất Top Cổ Phiếu ra CSV
    df_ranked = df_res.sort_values(by='Quant_Score', ascending=False).reset_index(drop=True)
    top_stocks = df_ranked.head(top_n)
    
    print("\n================ TOP CỔ PHIẾU ĐƯỢC LỌC ================")
    print(top_stocks[['Ticker', 'PE', 'ROE', 'Momentum_6M', 'Quant_Score']])
    
    top_stocks.to_csv('top_stocks.csv', index=False)
    print("\nĐã lưu kết quả thành công vào file top_stocks.csv")

if __name__ == '__main__':
    quant_stock_screener(top_n=5)
