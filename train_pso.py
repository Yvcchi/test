import json
import os
import numpy as np
import pandas as pd

# ==========================================
# 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
def load_historical_data():
    """
    Tải dữ liệu lịch sử yếu tố (Factors) và Lợi nhuận (Returns).
    Lưu ý: Thay thế hàm này bằng logic đọc file CSV/Database thực tế của bạn.
    """
    # GIẢ LẬP DỮ LIỆU ĐỂ DEMO (Thay thế phần này bằng code đọc dữ liệu thực tế)
    np.random.seed(42)
    quarters = [f"{y}Q{q}" for y in range(2021, 2026) for q in range(1, 5)]
    tickers = [f"STOCK_{i}" for i in range(50)]
    
    history_factors = {}
    history_returns = {}
    
    for q in quarters:
        # 4 nhóm yếu tố: Valuation, Quality, Growth, Momentum
        factors = np.random.randn(len(tickers), 4)
        # Chuẩn hóa Z-Score cho từng factor trong quý
        factors = (factors - factors.mean(axis=0)) / (factors.std(axis=0) + 1e-8)
        
        # Giả lập lợi nhuận thực tế quý tiếp theo (Fwd Return)
        # Tạo mối quan hệ thực tế: Quality và Growth có ảnh hưởng tích cực đến lợi nhuận
        true_weights = np.array([0.15, 0.40, 0.30, 0.15])
        returns = factors @ true_weights * 0.08 + np.random.randn(len(tickers)) * 0.05
        
        history_factors[q] = pd.DataFrame(factors, index=tickers, columns=['Z_Valuation', 'Z_Quality', 'Z_Growth', 'Z_Momentum'])
        history_returns[q] = pd.Series(returns, index=tickers)
        
    return history_factors, history_returns

# ==========================================
# 2. HÀM SIMULATION & BACKTEST (TÍNH FIT
# ==========================================
def run_backtest(weights, history_factors, history_returns, top_n=10):
    """
    Chạy Backtest giả lập dựa trên bộ trọng số và trả về chỉ số Sharpe & CAGR.
    """
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # Chuẩn hóa về tổng = 1.0
    
    portfolio_returns = []
    
    for q in sorted(history_factors.keys()):
        df_f = history_factors[q]
        ret_s = history_returns[q]
        
        # Tính điểm Quant tổng hợp cho từng cổ phiếu
        scores = df_f.values @ weights
        df_scores = pd.Series(scores, index=df_f.index)
        
        # Chọn Top N cổ phiếu có điểm cao nhất
        top_stocks = df_scores.nlargest(top_n).index
        
        # Lợi nhuận trung bình của danh mục trong quý đó
        q_return = ret_s.loc[top_stocks].mean()
        portfolio_returns.append(q_return)
        
    portfolio_returns = np.array(portfolio_returns)
    
    # Tính toán các chỉ số hiệu năng
    mean_ret = np.mean(portfolio_returns)
    std_ret = np.std(portfolio_returns) + 1e-8
    
    # Sharpe Ratio tính theo Quý (Annualized = * sqrt(4))
    sharpe_ratio = (mean_ret / std_ret) * np.sqrt(4)
    cagr = np.prod(1 + portfolio_returns) ** (4 / len(portfolio_returns)) - 1
    
    return sharpe_ratio, cagr

# ==========================================
# 3. THUẬT TOÁN PSO (PARTICLE SWARM OPTIMIZATION)
# ==========================================
def optimize_pso(history_factors, history_returns, num_particles=30, max_iter=40):
    """
    Huấn luyện bầy đàn PSO CHỈ TRÊN TẬP TRAIN để tìm bộ trọng số tối ưu.
    """
    dim = 4  # 4 nhóm yếu tố Quant
    
    # Khởi tạo vị trí ngẫu nhiên cho các hạt [0, 1]
    X = np.random.rand(num_particles, dim)
    X = X / X.sum(axis=1, keepdims=True)  # Chuẩn hóa tổng = 1
    
    V = np.random.uniform(-0.05, 0.05, (num_particles, dim))
    
    pbest_x = X.copy()
    pbest_fitness = np.array([run_backtest(x, history_factors, history_returns)[0] for x in X])
    
    gbest_idx = np.argmax(pbest_fitness)
    gbest_x = pbest_x[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]
    
    # Tham số PSO
    w = 0.6    # Hệ số quán tính (Inertia)
    c1 = 1.5   # Hệ số học cá nhân (Cognitive)
    c2 = 1.5   # Hệ số học cộng đồng (Social)
    
    print(f"🌀 Bắt đầu huấn luyện PSO ({max_iter} vòng lặp)...")
    for it in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            
            # Cập nhật vận tốc
            V[i] = w * V[i] + c1 * r1 * (pbest_x[i] - X[i]) + c2 * r2 * (gbest_x - X[i])
            
            # Cập nhật vị trí mới
            X[i] = X[i] + V[i]
            
            # Ràng buộc không âm
            X[i] = np.maximum(X[i], 0.001)
            X[i] = X[i] / np.sum(X[i])  # Chuẩn hóa tổng = 1.0
            
            # Tính điểm Fitness mới
            fit, _ = run_backtest(X[i], history_factors, history_returns)
            
            # Cập nhật pbest
            if fit > pbest_fitness[i]:
                pbest_fitness[i] = fit
                pbest_x[i] = X[i].copy()
                
                # Cập nhật gbest
                if fit > gbest_fitness:
                    gbest_fitness = fit
                    gbest_x = X[i].copy()
                    
        if (it + 1) % 10 == 0 or it == max_iter - 1:
            print(f"   ► Iteration {it+1:02d}/{max_iter:02d} | Train Best Sharpe: {gbest_fitness:.4f}")
            
    return gbest_x

# ==========================================
# 4. LUỒNG CHÍNH VÀ KIỂM THỬ OUT-OF-SAMPLE
# ==========================================
def save_config(weights, filepath="config.json"):
    """Lưu bộ trọng số tối ưu ra file json."""
    weights = weights / np.sum(weights)
    config_data = {
        "weights": {
            "Z_Valuation": round(float(weights[0]), 4),
            "Z_Quality": round(float(weights[1]), 4),
            "Z_Growth": round(float(weights[2]), 4),
            "Z_Momentum": round(float(weights[3]), 4)
        }
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)
    print(f"\n💾 Đã lưu file cấu hình thành công: {filepath}")

if __name__ == "__main__":
    print("========================================================")
    print("      QUY TRÌNH HUẤN LUYỆN PSO & KIỂM THỬ TỰ ĐỘNG      ")
    print("========================================================\n")
    
    # 1. Load dữ liệu
    history_factors, history_returns = load_historical_data()
    all_quarters = sorted(list(history_factors.keys()))
    
    # 2. Chia dữ liệu theo tỷ lệ 80% Train - 20% Test (Out-of-Sample)
    split_idx = int(len(all_quarters) * 0.8)
    train_quarters = all_quarters[:split_idx]
    test_quarters = all_quarters[split_idx:]
    
    train_factors = {q: history_factors[q] for q in train_quarters}
    train_returns = {q: history_returns[q] for q in train_quarters}
    
    test_factors = {q: history_factors[q] for q in test_quarters}
    test_returns = {q: history_returns[q] for q in test_quarters}
    
    print(f"📊 Dữ liệu huấn luyện (Train): {len(train_quarters)} Quý ({train_quarters[0]} -> {train_quarters[-1]})")
    print(f"🧪 Dữ liệu kiểm thử  (Test) : {len(test_quarters)} Quý ({test_quarters[0]} -> {test_quarters[-1]})\n")
    
    # 3. Chạy PSO chỉ trên tập TRAIN
    best_weights = optimize_pso(train_factors, train_returns, num_particles=30, max_iter=40)
    
    # 4. Đánh giá kiểm thử (Out-of-Sample Test)
    sharpe_train, cagr_train = run_backtest(best_weights, train_factors, train_returns)
    sharpe_test, cagr_test = run_backtest(best_weights, test_factors, test_returns)
    
    # 5. In báo cáo kiểm thử ra GitHub Actions Log
    print("\n========================================================")
    print("        BÁO CÁO KIỂM THỬ CHỐNG OVERFITTING (TEST)       ")
    print("========================================================")
    print(f"1. TẬP TRAIN (Dữ liệu học)  : Sharpe Ratio = {sharpe_train:6.2f} | CAGR = {cagr_train*100:6.2f}%")
    print(f"2. TẬP TEST  (Dữ liệu thực) : Sharpe Ratio = {sharpe_test:6.2f} | CAGR = {cagr_test*100:6.2f}%")
    print("--------------------------------------------------------")
    
    # Đánh giá tiêu chuẩn kiểm thử
    if sharpe_test > 0.5 and (sharpe_train - sharpe_test) < 1.0:
        print("✅ XÁC NHẬN: Mô hình ĐẠT YÊU CẦU! Không bị Overfitting.")
        save_config(best_weights)
    else:
        print("❌ CẢNH BÁO: Mô hình thất bại trong kiểm thử (Bị Overfitting hoặc Sharpe quá thấp)!")
        print("🚨 Huấn luyện bị từ chối. Không xuất file config.json.")
        # Thoát với mã lỗi 1 để thông báo thất bại cho GitHub Action
        exit(1)
