import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from vnstock import Quote, Finance, Market

def fetch_ratio_with_fallback(symbol):
    """Lấy BCTC linh hoạt với xử lý lỗi timeout"""
    for source in ['VCI', 'TCBS']:
        try:
            f = Finance(symbol=symbol, source=source)
            df = f.ratio(period='year', lang='vi')
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f"  └─ Thử nguồn {source} cho {symbol} thất bại: {e}")
            time.sleep(2)
    return None

def fetch_price_with_fallback(symbol, start_date, end_date):
    """Lấy giá lịch sử linh hoạt với xử lý lỗi timeout"""
    for source in ['VCI', 'TCBS']:
        try:
            q = Quote(symbol=symbol, source=source)
            df = q.history(start=start_date, end=end_date, interval='1D')
            if df is not None and not df.empty:
                return df
        except Exception as e:
            time.sleep(2)
    return None

def quant_stock_screener(top_n=5):
    print("--- BẮT ĐẦU QUÁ TRÌNH SÀNG LỌC CỔ PHIẾU QUANT ---")
    
    # Danh sách các mã cổ phiếu hàng đầu trên HOSE để kiểm tra
    sample_tickers = ['VNM', 'HPG', 'FPT', 'TCB', 'MBB', 'MWG', 'MSN', 'REE', 'ACB', 'VIC']
    
    screening_results = []
    today = datetime.now().strftime('%Y-%m-%d')
    six_months_ago = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

    for ticker in sample_tickers:
        try:
            print(f"Đang xử lý mã: {ticker}...")
            
            # 1. Lấy dữ liệu BCTC & Giá
            df_ratio = fetch_ratio_with_fallback(ticker)
            time.sleep(1)
            df_price = fetch_price_with_fallback(ticker, six_months_ago, today)

            if df_ratio is None or df_ratio.empty or df_price is None or df_price.empty or len(df_price) < 20:
                print(f"Bỏ qua mã {ticker} do không lấy được dữ liệu đầy đủ.")
                time.sleep(2)
                continue
                
            latest_ratio = df_ratio.iloc[0]
            
            # 2. Tìm kiếm chỉ số PE và ROE linh hoạt theo tên cột
            pe = np.nan
            roe = np.nan
            
            for col in df_ratio.columns:
                col_str = str(col).lower()
                if 'price' in col_str or 'pe' in col_str or 'p/e' in col_str:
                    try:
                        val = float(latest_ratio[col])
                        if not np.isnan(val) and val > 0:
                            pe = val
                    except:
                        pass
                if 'roe' in col_str:
                    try:
                        val = float(latest_ratio[col])
                        if not np.isnan(val):
                            roe = val
                    except:
                        pass

            # Giá lịch sử để tính Momentum 6 tháng
            price_col = 'close' if 'close' in df_price.columns else df_price.columns[1]
            price_start = float(df_price.iloc[0][price_col])
            price_end = float(df_price.iloc[-1][price_col])
            momentum_6m = (price_end - price_start) / price_start

            # Mặc định giá trị giả định nếu thiếu chỉ số nhỏ để không loại bỏ toàn bộ danh mục
            if np.isnan(pe): pe = 12.0
            if np.isnan(roe): roe = 15.0

            screening_results.append({
                'Ticker': ticker,
                'PE': pe,
                'ROE': roe,
                'Momentum_6M': momentum_6m
            })
            print(f"  ✓ {ticker}: PE={pe:.1f}, ROE={roe:.1f}%, Momentum={momentum_6m*100:.1f}%")
            
            time.sleep(2)
            
        except Exception as e:
            print(f"Bỏ qua mã {ticker} do lỗi: {e}")
            time.sleep(2)
            continue

    df_res = pd.DataFrame(screening_results)
    if df_res.empty:
        print("Không tìm thấy dữ liệu phù hợp.")
        pd.DataFrame([{'Ticker': 'N/A', 'Note': 'No data fetched'}]).to_csv('top_stocks.csv', index=False)
        return

    # 3. Tính Quant Score (Z-Score)
    df_res['Z_PE'] = -1 * (df_res['PE'] - df_res['PE'].mean()) / (df_res['PE'].std() + 1e-6)
    df_res['Z_ROE'] = (df_res['ROE'] - df_res['ROE'].mean()) / (df_res['ROE'].std() + 1e-6)
    df_res['Z_Momentum'] = (df_res['Momentum_6M'] - df_res['Momentum_6M'].mean()) / (df_res['Momentum_6M'].std() + 1e-6)

    df_res['Quant_Score'] = (0.30 * df_res['Z_PE']) + (0.40 * df_res['Z_ROE']) + (0.30 * df_res['Z_Momentum'])

    # 4. Xuất Top Cổ Phiếu ra CSV
    df_ranked = df_res.sort_values(by='Quant_Score', ascending=False).reset_index(drop=True)
    top_stocks = df_ranked.head(top_n)
    
    print("\n================ TOP CỔ PHIẾU ĐƯỢC LỌC ================")
    print(top_stocks[['Ticker', 'PE', 'ROE', 'Momentum_6M', 'Quant_Score']])
    
    top_stocks.to_csv('top_stocks.csv', index=False)
    print("\nĐã lưu kết quả thành công vào file top_stocks.csv")

if __name__ == '__main__':
    quant_stock_screener(top_n=5)
