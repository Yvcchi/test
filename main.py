import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from vnstock import financial_ratio, stock_historical_data, listing_companies

def quant_stock_screener(top_n=5):
    print("--- BẮT ĐẦU QUÁ TRÌNH SÀNG LỌC CỔ PHIẾU QUANT ---")
    
    # 1. Lấy danh sách cổ phiếu trên sàn HOSE
    try:
        df_companies = listing_companies()
        hose_tickers = df_companies[df_companies['comGroupCode'] == 'HOSE']['ticker'].tolist()
    except Exception as e:
        print(f"Lỗi lấy danh sách cổ phiếu: {e}")
        hose_tickers = ['VNM', 'HPG', 'FPT', 'TCB', 'MBB', 'MWG', 'MSN', 'REE', 'VHM', 'ACB']

    # Chọn 20 mã tiêu biểu để chạy mẫu (hoặc bỏ [:20] để quét toàn bộ sàn)
    sample_tickers = hose_tickers[:20] 
    
    screening_results = []
    today = datetime.now().strftime('%Y-%m-%d')
    six_months_ago = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

    for ticker in sample_tickers:
        try:
            print(f"Đang xử lý mã: {ticker}...")
            
            # 2. Lấy BCTC & Tính các chỉ số
            df_ratio = financial_ratio(symbol=ticker, report_range='yearly', is_pro=False)
            if df_ratio.empty:
                continue
                
            latest_ratio = df_ratio.iloc[0]
            pe = float(latest_ratio.get('priceToEarning', np.nan))
            roe = float(latest_ratio.get('roe', np.nan))
            
            # 3. Lấy giá lịch sử tính Momentum 6M
            df_price = stock_historical_data(
                symbol=ticker, 
                start_date=six_months_ago, 
                end_date=today, 
                resolution='1D', 
                type='stock'
            )
            
            if len(df_price) < 20:
                continue
                
            price_start = df_price.iloc[0]['close']
            price_end = df_price.iloc[-1]['close']
            momentum_6m = (price_end - price_start) / price_start

            if np.isnan(pe) or np.isnan(roe) or pe <= 0:
                continue

            screening_results.append({
                'Ticker': ticker,
                'PE': pe,
                'ROE': roe,
                'Momentum_6M': momentum_6m
            })
            
        except Exception as e:
            print(f"Bỏ qua mã {ticker}: {e}")
            continue

    df_res = pd.DataFrame(screening_results)
    if df_res.empty:
        print("Không có dữ liệu hợp lệ.")
        return

    # 4. Tính Quant Score (Z-Score Normalization)
    df_res['Z_PE'] = -1 * (df_res['PE'] - df_res['PE'].mean()) / df_res['PE'].std()
    df_res['Z_ROE'] = (df_res['ROE'] - df_res['ROE'].mean()) / df_res['ROE'].std()
    df_res['Z_Momentum'] = (df_res['Momentum_6M'] - df_res['Momentum_6M'].mean()) / df_res['Momentum_6M'].std()

    # Trọng số: 30% Value, 40% Quality, 30% Momentum
    df_res['Quant_Score'] = (0.30 * df_res['Z_PE']) + (0.40 * df_res['Z_ROE']) + (0.30 * df_res['Z_Momentum'])

    # 5. Xếp hạng & Lưu kết quả
    df_ranked = df_res.sort_values(by='Quant_Score', ascending=False).reset_index(drop=True)
    top_stocks = df_ranked.head(top_n)
    
    print("\n================ TOP CỔ PHIẾU ĐƯỢC LỌC ================")
    print(top_stocks[['Ticker', 'PE', 'ROE', 'Momentum_6M', 'Quant_Score']])
    
    # Lưu file CSV để GitHub Bot commit ngược lại vào Repo
    top_stocks.to_csv('top_stocks.csv', index=False)
    print("\nĐã xuất kết quả ra file top_stocks.csv")

if __name__ == '__main__':
    quant_stock_screener(top_n=5)