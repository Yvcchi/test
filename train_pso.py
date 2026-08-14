import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from vnstock import Quote, Finance

# ==============================================================================
# 1. HÀM CÀO & LƯU DỮ LIỆU LỊCH SỬ (FETCH & SAVE HISTORICAL DATA)
# ==============================================================================
def fetch_and_save_data(tickers, start_date='2021-01-01', data_dir='data'):
    """
    Cào dữ liệu Giá lịch sử và Chỉ số BCTC theo Quý cho danh sách mã.
    Lưu kết quả ra 2 file CSV trong thư mục data/.
    """
    os.makedirs(data_dir, exist_ok=True)
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    price_file = os.path.join(data_dir, 'historical_prices.csv')
    finance_file = os.path.join(data_dir, 'historical_financials.csv')

    print("=== [BƯỚC 1] TẢI DỮ LIỆU LỊCH SỬ TỪ VNSTOCK ===")
    
    price_data_list = []
    financial_data_list = []

    for i, ticker in enumerate(tickers):
        if i > 0 and i % 3 == 0:
            print("⏳ Đang tạm dừng 10s để reset Rate Limit...")
            time.sleep(10)
            
        print(f"[{i+1}/{len(tickers)}] Đang cào dữ liệu mã: {ticker}...")

        # A. Cào Dữ liệu Giá (OHLCV)
        try:
            q = Quote(symbol=ticker, source='VCI')
            df_price = q.history(start=start_date, end=end_date, interval='1D')
            if df_price is not None and not df_price.empty:
                df_price['Ticker'] = ticker
                price_data_list.append(df_price)
        except Exception as e:
            print(f"  └─ Lỗi cào giá {ticker}: {e}")

        time.sleep(1.5)

        # B. Cào Dữ liệu BCTC theo Quý
        try:
            f = Finance(symbol=ticker, source='VCI')
            df_ratio = f.ratio(period='quarter', lang='vi')
            if df_ratio is not None and not df_ratio.empty:
                df_ratio['Ticker'] = ticker
                financial_data_list.append(df_ratio)
        except Exception as e:
            print(f"  └─ Lỗi cào BCTC {ticker}: {e}")

        time.sleep(1.5)

    if price_data_list:
        full_price_df = pd.concat(price_data_list, ignore_index=True)
        full_price_df.to_csv(price_file, index=False)
        print(f"✓ Đã lưu file giá: {price_file}")

    if financial_data_list:
        full_finance_df = pd.concat(financial_data_list, ignore_index=True)
        full_finance_df.to_csv(finance_file, index=False)
        print(f"✓ Đã lưu file BCTC: {finance_file}")

    return price_file, finance_file


# ==============================================================================
# 2. XỬ LÝ DỮ LIỆU & TÍNH MA TRẬN Z-SCORE THEO QUÝ (FACTOR ENGINE)
# ==============================================================================
def process_factor_matrix(price_file, finance_file):
    """
    Đọc dữ liệu CSV, ghép nối BCTC & Giá theo mốc Quý,
    tính toán các chỉ số Z-Score cho Valuation, Quality, Growth, Momentum.
    """
    print("\n=== [BƯỚC 2] XỬ LÝ & TÍNH Z-SCORE CHUẨN QUANT ===")
    
    df_price = pd.read_csv(price_file)
    df_fin = pd.read_csv(finance_file)

    # 2.1 Chuẩn hóa cột thời gian & Quý
    time_col = 'time' if 'time' in df_price.columns else 'TradingDate'
    close_col = 'close' if 'close' in df_price.columns else df_price.columns[1]
    
    df_price[time_col] = pd.to_datetime(df_price[time_col])
    df_price['Quarter'] = df_price[time_col].dt.to_period('Q').astype(str)

    # 2.2 Lấy giá đóng cửa cuối mỗi Quý để tính Forward Returns
    q_prices = df_price.groupby(['Ticker', 'Quarter'])[close_col].last().reset_index()
    q_prices['Fwd_Return_3M'] = q_prices.groupby('Ticker')[close_col].pct_change().shift(-1)
    
    # Tính Momentum 3 tháng (Lợi nhuận quý vừa qua)
    q_prices['Momentum_3M'] = q_prices.groupby('Ticker')[close_col].pct_change()

    # 2.3 Trích xuất chỉ số BCTC từ file cào
    def extract_val(df, keys):
        for k in keys:
            for col in df.columns:
                if k.lower() in str(col).lower():
                    return pd.to_numeric(df[col], errors='coerce')
        return pd.Series(np.nan, index=df.index)

    df_fin['PE'] = extract_val(df_fin, ['priceToEarning', 'p/e', 'pe'])
    df_fin['ROE'] = extract_val(df_fin, ['roe'])
    df_fin['Rev_Growth'] = extract_val(df_fin, ['revenueGrowth', 'tang_truong_doanh_thu'])

    period_col = 'period' if 'period' in df_fin.columns else df_fin.columns[0]
    df_fin['Quarter'] = df_fin[period_col].astype(str)

    # 2.4 Merge BCTC và Giá
    merged = pd.merge(df_fin, q_prices, on=['Ticker', 'Quarter'], how='inner')
    merged = merged.dropna(subset=['PE', 'ROE', 'Rev_Growth', 'Momentum_3M', 'Fwd_Return_3M'])

    history_factors = {}
    history_returns = {}
    
    quarters = sorted(merged['Quarter'].unique())

    for idx, q in enumerate(quarters):
        df_q = merged[merged['Quarter'] == q].copy()
        if len(df_q) < 5:  # Bỏ qua quý quá ít cổ phiếu
            continue

        # Tính Z-Score Cross-sectional từng Quý
        # Valuation: P/E thấp tốt hơn -> Nhân -1
        df_q['Z_Valuation'] = -1 * (df_q['PE'] - df_q['PE'].mean()) / (df_q['PE'].std() + 1e-6)
        df_q['Z_Quality'] = (df_q['ROE'] - df_q['ROE'].mean()) / (df_q['ROE'].std() + 1e-6)
        df_q['Z_Growth'] = (df_q['Rev_Growth'] - df_q['Rev_Growth'].mean()) / (df_q['Rev_Growth'].std() + 1e-6)
        df_q['Z_Momentum'] = (df_q['Momentum_3M'] - df_q['Momentum_3M'].mean()) / (df_q['Momentum_3M'].std() + 1e-6)

        history_factors[idx] = df_q[['Ticker', 'Z_Valuation', 'Z_Quality', 'Z_Growth', 'Z_Momentum']].reset_index(drop=True)
        history_returns[idx] = pd.Series(df_q['Fwd_Return_3M'].values, index=df_q['Ticker'])

    print(f"✓ Đã hoàn tất xử lý dữ liệu cho {len(history_factors)} quý lịch sử.")
    return history_factors, history_returns


# ==============================================================================
# 3. BACKTEST ENGINE (ĐÁNH GIÁ TRỌNG SỐ TRONG QUÁ KHỨ)
# ==============================================================================
def run_backtest(weights, history_factors, history_returns, top_n=5, fee=0.0015):
    """
    Giả lập Backtest đảo danh mục theo từng Quý dựa trên trọng số weights.
    Trả về Sharpe Ratio, CAGR, Max Drawdown.
    """
    weights = np.array(weights)
    if weights.sum() == 0:
        return -999.0, 0, 0, []
    weights = weights / weights.sum()

    portfolio_returns = []

    for q_idx in history_factors.keys():
        df_f = history_factors[q_idx]
        rets = history_returns[q_idx]

        # Tính Quant Score cho từng cổ phiếu tại Quý q_idx
        scores = (weights[0] * df_f['Z_Valuation'] +
                  weights[1] * df_f['Z_Quality'] +
                  weights[2] * df_f['Z_Growth'] +
                  weights[3] * df_f['Z_Momentum'])

        # Lấy Top N mã điểm cao nhất
        top_indices = np.argsort(scores)[-top_n:]
        selected_tickers = df_f.iloc[top_indices]['Ticker'].values

        # Lợi nhuận trung bình danh mục quý (trừ phí giao dịch mua/bán)
        valid_rets = [rets[t] for t in selected_tickers if t in rets]
        if not valid_rets:
            continue

        q_ret = np.mean(valid_rets) - (2 * fee)
        portfolio_returns.append(q_ret)

    if len(portfolio_returns) < 2:
        return -999.0, 0, 0, []

    portfolio_returns = np.array(portfolio_returns)
    mean_ret = portfolio_returns.mean()
    std_ret = portfolio_returns.std() + 1e-6

    # Sharpe Ratio (Chuẩn hóa theo năm với 4 quý)
    sharpe = ((mean_ret - 0.01) / std_ret) * np.sqrt(4)

    # Cumulative Return, CAGR & Max Drawdown
    cum_returns = np.cumprod(1 + portfolio_returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - peak) / peak
    max_drawdown = drawdowns.min()
    cagr = (cum_returns[-1]) ** (4 / len(portfolio_returns)) - 1

    return sharpe, cagr, max_drawdown, portfolio_returns


# ==============================================================================
# 4. THUẬT TOÁN TỐI ƯU HÓA PSO (PARTICLE SWARM OPTIMIZATION)
# ==============================================================================
class PSOFactorOptimizer:
    def __init__(self, n_particles=25, max_iter=30):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.dim = 4  # [w_val, w_qual, w_growth, w_mom]

    def optimize(self, history_factors, history_returns):
        print("\n=== [BƯỚC 3] CHẠY THUẬT TOÁN PSO TỐI ƯU HÓA TRỌNG SỐ ===")
        
        # Khởi tạo vị trí và vận tốc ngẫu nhiên cho bầy đàn
        positions = np.random.uniform(0.1, 1.0, (self.n_particles, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, (self.n_particles, self.dim))
        positions = positions / positions.sum(axis=1, keepdims=True)

        pbest_pos = positions.copy()
        pbest_score = np.zeros(self.n_particles) - 999.0

        gbest_pos = np.ones(self.dim) / self.dim
        gbest_score = -999.0

        w_inertia = 0.5
        c1, c2 = 1.5, 1.5

        for it in range(self.max_iter):
            for i in range(self.n_particles):
                sharpe, _, _, _ = run_backtest(positions[i], history_factors, history_returns)

                # Cập nhật PBest
                if sharpe > pbest_score[i]:
                    pbest_score[i] = sharpe
                    pbest_pos[i] = positions[i].copy()

                # Cập nhật GBest
                if sharpe > gbest_score:
                    gbest_score = sharpe
                    gbest_pos = positions[i].copy()

            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            # Cập nhật Vận tốc & Vị trí mới
            velocities = (w_inertia * velocities +
                          c1 * r1 * (pbest_pos - positions) +
                          c2 * r2 * (gbest_pos - positions))

            positions = positions + velocities
            positions = np.clip(positions, 0.01, 1.0)
            positions = positions / positions.sum(axis=1, keepdims=True)

            if (it + 1) % 10 == 0 or it == 0:
                print(f"Iteration {it+1:02d}/{self.max_iter} | Best Sharpe Ratio: {gbest_score:.4f}")

        gbest_pos = gbest_pos / gbest_pos.sum()
        return gbest_pos, gbest_score


# ==============================================================================
# 5. LƯU BỘ TRỌNG SỐ TỐI ƯU VÀO CONFIG.JSON
# ==============================================================================
def save_config(opt_weights, config_path='config.json'):
    """Xuất bộ trọng số tìm được ra file config.json để main.py đọc"""
    config_data = {
        "weights": {
            "Z_Valuation": round(float(opt_weights[0]), 4),
            "Z_Quality": round(float(opt_weights[1]), 4),
            "Z_Growth": round(float(opt_weights[2]), 4),
            "Z_Momentum": round(float(opt_weights[3]), 4)
        },
        "last_updated": datetime.now().strftime('%Y-%m-%d')
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n✓ Đã xuất thành công bộ trọng số tối ưu vào file: {config_path}")


# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # 1. Danh sách mã dùng để huấn luyện mô hình
    train_tickers = ['HPG', 'FPT', 'VNM', 'TCB', 'MBB', 'MWG', 'SSI', 'REE', 'VHM', 'VIC',
                     'ACB', 'CTG', 'BID', 'MSN', 'GVR', 'VIB', 'TPB', 'STB', 'HDB', 'GAS']
    
    # 2. Đường dẫn thư mục dữ liệu
    price_file = os.path.join('data', 'historical_prices.csv')
    finance_file = os.path.join('data', 'historical_financials.csv')
    
    # Kiểm tra nếu chưa có data thì tự cào về
    if not (os.path.exists(price_file) and os.path.exists(finance_file)):
        price_file, finance_file = fetch_and_save_data(train_tickers)
    else:
        print("✓ Đã tìm thấy dữ liệu CSV sẵn có trong thư mục data/, bỏ qua bước cào lại.")

    # 3. Biến đổi dữ liệu thành Factor Matrix
    history_factors, history_returns = process_factor_matrix(price_file, finance_file)

    if not history_factors:
        print("❌ Dữ liệu không đủ để tiến hành Backtest & PSO!")
    else:
        # 4. Tối ưu hóa bằng PSO
        pso = PSOFactorOptimizer(n_particles=25, max_iter=30)
        opt_weights, best_sharpe = pso.optimize(history_factors, history_returns)

        # 5. Đánh giá lại kết quả
        _, opt_cagr, opt_mdd, _ = run_backtest(opt_weights, history_factors, history_returns)

        print("\n========================================================")
        print("         BÁO CÁO KẾT QUẢ HUẤN LUYỆN BẰNG PSO           ")
        print("========================================================")
        print(f"1. TRỌNG SỐ TỐI ƯU TÌM ĐƯỢC:")
        print(f"   ├─ Valuation (w1): {opt_weights[0]:.2%}")
        print(f"   ├─ Quality   (w2): {opt_weights[1]:.2%}")
        print(f"   ├─ Growth    (w3): {opt_weights[2]:.2%}")
        print(f"   └─ Momentum  (w4): {opt_weights[3]:.2%}")
        print("--------------------------------------------------------")
        print(f"2. CHỈ SỐ LỊCH SỬ HIỆU NĂNG (BACKTEST):")
        print(f"   ├─ Sharpe Ratio  : {best_sharpe:.4f}")
        print(f"   ├─ CAGR (Năm)    : {opt_cagr*100:.2f}%")
        print(f"   └─ Max Drawdown  : {opt_mdd*100:.2f}%")
        print("========================================================")

        # 6. Lưu kết quả ra config.json để main.py sử dụng
        save_config(opt_weights)
