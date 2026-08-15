import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from vnstock import Quote, Finance, config

# Cấu hình API Key
VNSTOCK_KEY = os.getenv('VNSTOCK_API_KEY', '')
if VNSTOCK_KEY:
    try:
        if hasattr(config, 'set_token'): config.set_token(VNSTOCK_KEY)
        elif hasattr(config, 'set_api_key'): config.set_api_key(VNSTOCK_KEY)
    except Exception: pass

def run_dynamic_backtest(
    initial_capital=100_000_000, 
    top_n=3,               # Số lượng mã chọn mỗi quý
    take_profit=0.15,      # Chốt lời khi tăng 15%
    stop_loss=0.07,        # Cắt lỗ khi giảm 7%
    years=5
):
    print(f"🚀 BẮT ĐẦU BACKTEST ĐỘNG THỜI GIAN REAL-TIME ({years} NĂM)")
    print(f"⚙️ Cấu hình: Top {top_n} mã/quý | Chốt lời: +{take_profit*100}% | Cắt lỗ: -{stop_loss*100}%")
    print("-" * 70)

    # Lấy danh sách mã giao dịch (Ví dụ các mã VN30/HOSE)
    sample_tickers = ['ACB', 'BID', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
                      'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SSI', 'STB', 
                      'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VNM', 'VRE']

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    # Tạo các mốc thời gian đầu quý trong 5 năm
    quarter_dates = pd.date_range(start=start_date, end=end_date, freq='QS')

    current_capital = initial_capital
    trade_history = []

    for i in range(len(quarter_dates) - 1):
        q_start = quarter_dates[i]
        q_end = quarter_dates[i+1]
        
        print(f"\n📅 --- QUÝ {q_start.strftime('%Y-Q%q')} ({q_start.strftime('%d/%m/%Y')} -> {q_end.strftime('%d/%m/%Y')}) ---")
        print(f"   Vốn đầu quý: {current_capital:,.0f} VNĐ")

        # ----------------------------------------------------
        # 1. Giả lập chấm điểm Quant để chọn Top N mã cho Quý
        # ----------------------------------------------------
        # (Trong thực tế sẽ gọi hàm chấm điểm chỉ số tài chính quý đó. 
        # Ở đây chọn ngẫu nhiên/hoặc lấy danh sách mã đại diện có dữ liệu)
        selected_stocks = sample_tickers[(i * top_n) % len(sample_tickers) : ((i * top_n) % len(sample_tickers)) + top_n]
        if len(selected_stocks) < top_n:
            selected_stocks = sample_tickers[:top_n]
            
        print(f"   🎯 Top {top_n} mã chọn cho quý: {', '.join(selected_stocks)}")

        capital_per_stock = current_capital / top_n
        quarter_end_capital = 0

        # ----------------------------------------------------
        # 2. Giả lập giao dịch từng mã trong Quý (Có TP / SL)
        # ----------------------------------------------------
        for ticker in selected_stocks:
            try:
                time.sleep(0.5)
                q = Quote(symbol=ticker, source='VCI')
                df = q.history(start=q_start.strftime('%Y-%m-%d'), end=q_end.strftime('%Y-%m-%d'), interval='1D')
                
                if df is None or df.empty or len(df) < 5:
                    quarter_end_capital += capital_per_stock
                    continue

                prices = df['close'].astype(float).values
                entry_price = prices[0]
                exit_price = prices[-1] # Mặc định giữ đến cuối quý
                reason = "Hết Quý (Tái cân bằng)"

                # Theo dõi biến động giá trong quý để Chốt lời / Cắt lỗ
                for price in prices[1:]:
                    return_pct = (price - entry_price) / entry_price
                    
                    if return_pct >= take_profit:
                        exit_price = entry_price * (1 + take_profit)
                        reason = f"🎯 Chốt lời (+{take_profit*100}%)"
                        break
                    elif return_pct <= -stop_loss:
                        exit_price = entry_price * (1 - stop_loss)
                        reason = f"🛑 Cắt lỗ (-{stop_loss*100}%)"
                        break

                # Tính vốn thu về từ mã này
                stock_return = (exit_price - entry_price) / entry_price
                final_stock_capital = capital_per_stock * (1 + stock_return)
                quarter_end_capital += final_stock_capital

                trade_history.append({
                    'Quarter': q_start.strftime('%Y-Q%q'),
                    'Ticker': ticker,
                    'Entry': entry_price,
                    'Exit': exit_price,
                    'Return': f"{stock_return*100:+.2f}%",
                    'Reason': reason
                })

            except Exception as e:
                quarter_end_capital += capital_per_stock
                print(f"   ⚠️ Lỗi mã {ticker}: {e}")

        # Cập nhật vốn tổng sau khi kết thúc Quý
        current_capital = quarter_end_capital

    # ----------------------------------------------------
    # 3. BÁO CÁO KẾT QUẢ TỔNG KẾT 5 NĂM
    # ----------------------------------------------------
    total_profit = current_capital - initial_capital
    total_roi = (total_profit / initial_capital) * 100

    print("\n" + "="*70)
    print("📊 BÁO CÁO TỔNG KẾT BACKTEST ĐỘNG THEO QUÝ")
    print("="*70)
    print(f"• Vốn ban đầu            : {initial_capital:,.0f} VNĐ")
    print(f"• Vốn cuối kỳ (sau 5 năm) : {current_capital:,.0f} VNĐ")
    print(f"• Tổng lợi nhuận Ròng    : {total_profit:+,.0f} VNĐ")
    print(f"• Tỷ lệ lợi nhuận (ROI)  : {total_roi:+.2f}%")
    print("="*70)

    # Hiển thị 10 giao dịch gần nhất
    df_trades = pd.DataFrame(trade_history)
    print("\n📜 LỊCH SỬ GIAO DỊCH GẦN ĐÂY:")
    print(df_trades.tail(10).to_string(index=False))

if __name__ == '__main__':
    run_dynamic_backtest(
        initial_capital=100_000_000, 
        top_n=3,           # Lấy Top 3 mã mỗi quý
        take_profit=0.15,  # Lãi 15% -> Bán chốt lời ngay
        stop_loss=0.07,    # Lỗ 7% -> Bán cắt lỗ ngay
        years=5
    )
