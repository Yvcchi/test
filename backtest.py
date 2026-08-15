import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from vnstock import Quote, config

# Cấu hình API Key nếu có
VNSTOCK_KEY = os.getenv('VNSTOCK_API_KEY', '')
if VNSTOCK_KEY:
    try:
        if hasattr(config, 'set_token'):
            config.set_token(VNSTOCK_KEY)
        elif hasattr(config, 'set_api_key'):
            config.set_api_key(VNSTOCK_KEY)
    except Exception:
        pass

def run_backtest_from_csv(csv_path='top_stocks.csv', initial_capital=100_000_000, dca_monthly=5_000_000, years=5):
    """
    Đọc danh sách cổ phiếu từ file CSV và giả lập hiệu suất đầu tư 5 năm quá khứ.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file {csv_path}. Hãy chạy main.py trước để tạo danh sách cổ phiếu!")
        return

    # 1. Đọc danh sách cổ phiếu từ top_stocks.csv
    df_top = pd.read_csv(csv_path)
    if 'Ticker' not in df_top.columns or df_top['Ticker'].iloc[0] == 'N/A':
        print("❌ File top_stocks.csv không chứa danh sách cổ phiếu hợp lệ.")
        return

    top_tickers = df_top['Ticker'].dropna().unique().tolist()
    print(f"🚀 BẮT ĐẦU BACKTEST CHO {len(top_tickers)} MÃ: {', '.join(top_tickers)}")
    print(f"⏱️ Khoảng thời gian: {years} năm gần nhất | Vốn ban đầu: {initial_capital:,.0f} VNĐ")

    # 2. Xác định mốc thời gian
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    price_series = {}

    # 3. Tải dữ liệu giá lịch sử cho từng mã
    for ticker in top_tickers:
        try:
            time.sleep(1) # Tránh bị dính Rate Limit
            q = Quote(symbol=ticker, source='VCI')
            df_hist = q.history(start=start_str, end=end_str, interval='1D')
            
            if df_hist is not None and not df_hist.empty:
                col_price = 'close' if 'close' in df_hist.columns else df_hist.columns[1]
                col_time = 'time' if 'time' in df_hist.columns else df_hist.columns[0]
                
                df_clean = df_hist[[col_time, col_price]].copy()
                df_clean[col_time] = pd.to_datetime(df_clean[col_time])
                df_clean[col_price] = df_clean[col_price].astype(float)
                
                price_series[ticker] = df_clean.set_index(col_time)[col_price]
                print(f"  ✓ Lấy dữ liệu thành công: {ticker}")
        except Exception as e:
            print(f"  └─ Lỗi lấy dữ liệu {ticker}: {e}")

    if not price_series:
        print("❌ Không lấy được dữ liệu lịch sử của mã nào.")
        return

    # Gộp thành 1 DataFrame chứa giá đóng cửa của tất cả các mã
    df_prices = pd.DataFrame(price_series).ffill().bfill()

    # ==========================================
    # KỊCH BẢN 1: MUA 1 LẦN BAN ĐẦU (LUMP-SUM)
    # ==========================================
    capital_per_stock = initial_capital / len(price_series)
    start_prices = df_prices.iloc[0]
    end_prices = df_prices.iloc[-1]
    
    shares_lumpsum = capital_per_stock / start_prices
    final_lumpsum_val = (shares_lumpsum * end_prices).sum()
    lumpsum_profit = final_lumpsum_val - initial_capital
    lumpsum_roi = (lumpsum_profit / initial_capital) * 100

    # ==========================================
    # KỊCH BẢN 2: TÍCH SẢN HÀNG THÁNG (DCA)
    # ==========================================
    # Lấy ngày giao dịch đầu tiên mỗi tháng
    df_monthly = df_prices.resample('MS').first().dropna()
    
    total_dca_invested = 0
    shares_dca = pd.Series(0.0, index=df_prices.columns)
    
    for _, row in df_monthly.iterrows():
        total_dca_invested += dca_monthly
        dca_per_stock = dca_monthly / len(df_prices.columns)
        shares_dca += (dca_per_stock / row)

    final_dca_val = (shares_dca * end_prices).sum()
    dca_profit = final_dca_val - total_dca_invested
    dca_roi = (dca_profit / total_dca_invested) * 100

    # ==========================================
    # KỊCH BẢN 3: HIỆU SUẤT TỪNG MÃ RIÊNG LẺ
    # ==========================================
    individual_performance = ((end_prices - start_prices) / start_prices * 100).round(2)

    # ==========================================
    # XUẤT KẾT QUẢ BÁO CÁO
    # ==========================================
    print("\n" + "="*70)
    print(f"📊 BÁO CÁO GIẢ LẬP ĐẦU TƯ BACKTEST ({start_str} -> {end_str})")
    print("="*70)
    print(f"1️⃣ KỊCH BẢN MUA 1 LẦN (LUMP-SUM):")
    print(f"   • Vốn đầu tư ban đầu : {initial_capital:,.0f} VNĐ")
    print(f"   • Giá trị danh mục hiện tại : {final_lumpsum_val:,.0f} VNĐ")
    print(f"   • Lợi nhuận ròng          : {lumpsum_profit:+,.0f} VNĐ ({lumpsum_roi:+.2f}%)")
    
    print(f"\n2️⃣ KỊCH BẢN TÍCH SẢN HÀNG THÁNG (DCA):")
    print(f"   • Mức nộp mỗi tháng       : {dca_monthly:,.0f} VNĐ/tháng")
    print(f"   • Tổng vốn đã nộp         : {total_dca_invested:,.0f} VNĐ")
    print(f"   • Giá trị danh mục hiện tại : {final_dca_val:,.0f} VNĐ")
    print(f"   • Lợi nhuận ròng          : {dca_profit:+,.0f} VNĐ ({dca_roi:+.2f}%)")

    print("\n3️⃣ HIỆU SUẤT TỪNG MÃ TRONG DANH MỤC (LUMP-SUM):")
    for ticker, roi in individual_performance.items():
        print(f"   • {ticker:<6}: {roi:+.2f}%")
    print("="*70)

if __name__ == '__main__':
    # Chạy giả lập độc lập
    run_backtest_from_csv(initial_capital=100_000_000, dca_monthly=5_000_000, years=5)
