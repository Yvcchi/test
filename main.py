import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# Import theo chuẩn API mới của vnstock
from vnstock import Vnstock

def quant_stock_screener(top_n=5):
    print("--- BẮT ĐẦU QUÁ TRÌNH SÀNG LỌC CỔ PHIẾU QUANT ---")
    
    # Khởi tạo đối tượng Vnstock
    stock = Vnstock()
    
    # 1. Lấy danh sách cổ phiếu HOSE
    try:
        df_companies = stock.listing.all_symbols()
        # Lọc danh sách thuộc sàn HOSE
        hose_tickers = df_companies[df_companies['organ_code'] == 'HOSE']['ticker'].tolist()
    except Exception as e:
        print(f"Lỗi lấy danh sách cổ phiếu: {e}")
        hose_tickers = ['VNM', 'HPG', 'FPT', 'TCB', 'MBB', 'MWG', 'MSN', 'REE', 'VHM', 'ACB']

    # Chọn 15 mã chạy demo để tránh bị chặn IP do spam request
    sample_tickers = hose_tickers[:15] if hose_tickers else ['VNM', 'HPG', 'FPT', 'TCB', 'MBB']
    
    screening_results = []
    today = datetime.now().strftime('%Y-%m-%d')
    six_months_ago = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

    for ticker in sample_tickers:
        try:
            print(f"Đang xử lý mã: {ticker}...")
            
            # Khởi tạo runner cho từng mã cổ phiếu
            stock_runner = Vnstock().stock(symbol=ticker, source='VCI')
            
            # 2. Lấy chỉ số BCTC (Financial Ratios)
            df_ratio = stock_runner.finance.ratio(period='year', lang='vi')
            if df_ratio.empty:
                continue
                
            # Lấy thông tin P/E và ROE
            # Lưu ý: Xử lý linh hoạt theo tên cột dữ liệu trả về từ vnstock
            latest_ratio = df_ratio.iloc[0]
            pe = float(latest_ratio.get('priceToEarning', latest_ratio.get('P/E', np.nan)))
            roe = float(latest_ratio.get('roe', latest_ratio.get('ROE', np.nan)))
            
            # 3. Lấy giá lịch sử tính Momentum 6M
            df_price = stock_runner.quote.history(start=six_months_ago, end=today, interval='1D')
            
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
            print(f"Bỏ qua mã {ticker} do lỗi: {e}")
            continue

    df_res = pd.DataFrame(screening_results)
    if df_res.empty:
        print("Không có dữ liệu hợp lệ được tìm thấy.")
        return

    # 4. Tính Quant Score (Z-Score Normalization)
    df_res['Z_PE'] = -1 * (df_res['PE'] - df_res['PE'].mean()) / (df_res['PE'].std() + 1e-6)
    df_res['Z_ROE'] = (df_res['ROE'] - df_res['ROE'].mean()) / (df_res['ROE'].std() + 1e-6)
    df_res['Z_Momentum'] = (df_res['Momentum_6M'] - df_res['Momentum_6M'].mean()) / (df_res['Momentum_6M'].std() + 1e-6)

    # Trọng số: 30% Value, 40% Quality, 30% Momentum
    df_res['Quant_Score'] = (0.30 * df_res['Z_PE']) + (0.40 * df_res['Z_ROE']) + (0.30 * df_res['Z_Momentum'])

    # 5. Xếp hạng & Lưu kết quả
    df_ranked = df_res.sort_values(by='Quant_Score', ascending=False).reset_index(drop=True)
    top_stocks = df_ranked.head(top_n)
    
    print("\n================ TOP CỔ PHIẾU ĐƯỢC LỌC ================")
    print(top_stocks[['Ticker', 'PE', 'ROE', 'Momentum_6M', 'Quant_Score']])
    
    top_stocks.to_csv('top_stocks.csv', index=False)
    print("\nĐã xuất kết quả ra file top_stocks.csv")

if __name__ == '__main__':
    quant_stock_screener(top_n=5)
