import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from vnstock import Quote, Finance

def fetch_data_safely(symbol):
    """Lấy dữ liệu an toàn, chủ động chờ để tránh đụng trần 20 req/phút"""
    print(f"Đang xử lý mã: {symbol}...")
    
    # Lấy BCTC từ VCI
    try:
        f = Finance(symbol=symbol, source='VCI')
        df_ratio = f.ratio(period='year', lang='vi')
    except Exception as e:
        print(f"  └─ Lỗi lấy BCTC {symbol}: {e}")
        df_ratio = None

    # Nghỉ 3 giây giữa các lượt gọi API
    time.sleep(3)

    # Lấy giá lịch sử từ VCI
    today = datetime.now().strftime('%Y-%m-%d')
    six_months_ago = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    try:
        q = Quote(symbol=symbol, source='VCI')
        df_price = q.history(start=six_months_ago, end=today, interval='1D')
    except Exception as e:
        print(f"  └─ Lỗi lấy giá {symbol}: {e}")
        df_price = None

    # Nghỉ tiếp 4 giây để đảm bảo tổng thời gian xử lý 1 mã > 7 giây
    time.sleep(4)
    
    return df_ratio, df_price

def quant_stock_screener(top_n=5):
    print("--- BẮT ĐẦU QUÁ TRÌNH SÀNG LỌC CỔ PHIẾU QUANT ---")
    
    # Chọn danh sách 6 cổ phiếu hàng đầu (đủ tạo Top 5 và nằm an toàn trong giới hạn API)
    sample_tickers = ['VNM', 'HPG', 'FPT', 'TCB', 'MBB', 'MWG']
    
    screening_results = []

    for i, ticker in enumerate(sample_tickers):
        # Sau mỗi 3 mã, nghỉ 35 giây để Rate Limit của Vnstock reset lại hoàn toàn
        if i > 0 and i % 3 == 0:
            print("⏳ Đang tạm dừng 35 giây để reset giới hạn API (Rate Limit)...")
            time.sleep(35)

        df_ratio, df_price = fetch_data_safely(ticker)

        if df_ratio is None or df_ratio.empty or df_price is None or df_price.empty or len(df_price) < 20:
            print(f"Bỏ qua mã {ticker} do thiếu dữ liệu.")
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

        # Gán giá trị mặc định nếu thiếu chỉ số
        if np.isnan(pe) or pe <= 0: pe = 12.0
        if np.isnan(roe): roe = 15.0

        screening_results.append({
            'Ticker': ticker,
            'PE': pe,
            'ROE': roe,
            'Momentum_6M': momentum_6m
        })
        print(f"  ✓ {ticker}: PE={pe:.1f}, ROE={roe:.1f}%, Momentum={momentum_6m*100:.1f}%")

    df_res = pd.DataFrame(screening_results)
    if df_res.empty:
        print("Không tìm thấy dữ liệu phù hợp.")
        pd.DataFrame([{'Ticker': 'N/A', 'Note': 'No data'}]).to_csv('top_stocks.csv', index=False)
        return

    # Tính Quant Score (Z-Score)
    df_res['Z_PE'] = -1 * (df_res['PE'] - df_res['PE'].mean()) / (df_res['PE'].std() + 1e-6)
    df_res['Z_ROE'] = (df_res['ROE'] - df_res['ROE'].mean()) / (df_res['ROE'].std() + 1e-6)
    df_res['Z_Momentum'] = (df_res['Momentum_6M'] - df_res['Momentum_6M'].mean()) / (df_res['Momentum_6M'].std() + 1e-6)

    df_res['Quant_Score'] = (0.30 * df_res['Z_PE']) + (0.40 * df_res['Z_ROE']) + (0.30 * df_res['Z_Momentum'])

    # Xuất file CSV
    df_ranked = df_res.sort_values(by='Quant_Score', ascending=False).reset_index(drop=True)
    top_stocks = df_ranked.head(top_n)
    
    print("\n================ TOP CỔ PHIẾU ĐƯỢC LỌC ================")
    print(top_stocks[['Ticker', 'PE', 'ROE', 'Momentum_6M', 'Quant_Score']])
    
    top_stocks.to_csv('top_stocks.csv', index=False)
    print("\nĐã lưu kết quả thành công vào file top_stocks.csv")

if __name__ == '__main__':
    quant_stock_screener(top_n=5)
