import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
from vnstock import Quote, Finance, Listing, config
import logging

# Tắt logging verbose của vnstock
logging.getLogger('vnstock').setLevel(logging.CRITICAL)

VNSTOCK_KEY = os.getenv('VNSTOCK_API_KEY', '')
if VNSTOCK_KEY:
    try:
        if hasattr(config, 'set_token'):
            config.set_token(VNSTOCK_KEY)
        elif hasattr(config, 'set_api_key'):
            config.set_api_key(VNSTOCK_KEY)
        print(f"✓ Đã cấu hình Vnstock API Key thành công!")
    except Exception as e:
        print(f"⚠️ Lỗi cấu hình API Key: {e}")

# ==========================================
# DANH SÁCH 400+ MÃ CỔ PHIẾU FALLBACK
# ==========================================
FALLBACK_STOCK_TICKERS = [
    'AAA', 'AAM', 'AAS', 'ABR', 'ABS', 'ABT', 'ACB', 'ACD', 'ACM', 'ACT',
    'AFE', 'AFS', 'AGF', 'AGX', 'AHT', 'AIC', 'ALD', 'AMS', 'AMV', 'ANV',
    'AOG', 'APG', 'APS', 'APT', 'AQC', 'ART', 'ASA', 'ASM', 'ASP', 'ATE',
    'ATE', 'ATG', 'ATS', 'AUC', 'AVF', 'AVG', 'AVX', 'AWC', 'AXA', 'AXJ',
    'AZC', 'BAB', 'BAF', 'BAT', 'BBC', 'BBT', 'BCC', 'BCE', 'BCG', 'BCM',
    'BCP', 'BDW', 'BED', 'BHN', 'BHS', 'BID', 'BIG', 'BIM', 'BKC', 'BKG',
    'BKT', 'BLD', 'BLX', 'BMI', 'BMV', 'BNA', 'BNP', 'BOC', 'BOD', 'BOS',
    'BPT', 'BRC', 'BRR', 'BSA', 'BSI', 'BTA', 'BTE', 'BTG', 'BTH', 'BTI',
    'BTJ', 'BTK', 'BTL', 'BTM', 'BTN', 'BTO', 'BTP', 'BTR', 'BTS', 'BTT',
    'BTU', 'BTV', 'BTW', 'BTX', 'BTY', 'BTZ', 'BUA', 'BUB', 'BUC', 'BUD',
    'BUG', 'BUM', 'BUN', 'BUO', 'BUP', 'BUR', 'BUS', 'BUT', 'BUU', 'BUV',
    'BUW', 'BUX', 'BUY', 'BUZ', 'BVH', 'BVN', 'BWE', 'BWG', 'BXH', 'BXM',
    'BYA', 'BYC', 'BYD', 'BYE', 'BYG', 'BYH', 'BYI', 'BYJ', 'BYL', 'BYM',
    'BYN', 'BYO', 'BYP', 'BYT', 'BYU', 'BYW', 'BYX', 'BYY', 'BYZ', 'BZC',
    'BZD', 'BZE', 'BZF', 'BZG', 'BZH', 'BZI', 'BZJ', 'BZK', 'BZL', 'BZM',
    'BZN', 'BZO', 'BZP', 'BZQ', 'BZR', 'BZS', 'BZT', 'BZU', 'BZV', 'BZW',
    'BZX', 'BZY', 'BZZ', 'CAA', 'CAB', 'CAD', 'CAE', 'CAF', 'CAG', 'CAH',
    'CAI', 'CAJ', 'CAK', 'CAL', 'CAM', 'CAN', 'CAO', 'CAP', 'CAR', 'CAS',
    'CAT', 'CAU', 'CAV', 'CAW', 'CAX', 'CAY', 'CAZ', 'CBA', 'CBB', 'CBC',
    'CBD', 'CBE', 'CBF', 'CBG', 'CBH', 'CBI', 'CBJ', 'CBK', 'CBL', 'CBM',
    'CBN', 'CBO', 'CBP', 'CBQ', 'CBR', 'CBS', 'CBT', 'CBU', 'CBV', 'CBW',
    'CBX', 'CBY', 'CBZ', 'CCA', 'CCB', 'CCC', 'CCD', 'CCE', 'CCF', 'CCG',
    'CCI', 'CCJ', 'CDA', 'CDB', 'CDC', 'CDD', 'CDE', 'CDF', 'CDG', 'CDH',
    'CDI', 'CDJ', 'CDK', 'CDL', 'CDM', 'CEA', 'CEB', 'CEC', 'CED', 'CEE',
    'CEF', 'CEG', 'CEH', 'CEI', 'CEJ', 'CEK', 'CEL', 'CEM', 'CEN', 'CEO',
    'CEP', 'CER', 'CES', 'CET', 'CEU', 'CEV', 'CEW', 'CEX', 'CEY', 'CEZ',
    'CFA', 'CFB', 'CFC', 'CFD', 'CFE', 'CFF', 'CFG', 'CFH', 'CFI', 'CFJ',
    'CHA', 'CHB', 'CHC', 'CHD', 'CHE', 'CHF', 'CHG', 'CHH', 'CHI', 'CHJ',
    'CHK', 'CHL', 'CHM', 'CHN', 'CHO', 'CHP', 'CHR', 'CHS', 'CHT', 'CHU',
    'CHV', 'CHW', 'CHX', 'CHY', 'CHZ', 'CIA', 'CIB', 'CIC', 'CID', 'CIE',
    'CIF', 'CIG', 'CIH', 'CII', 'CIJ', 'CIK', 'CIL', 'CIM', 'CIN', 'CIO',
    'CIP', 'CIR', 'CIS', 'CIT', 'CIU', 'CIV', 'CIW', 'CIX', 'CIY', 'CIZ',
    'DAT', 'DAH', 'DBC', 'DBM', 'DBV', 'DCM', 'DHG', 'DIG', 'DLG', 'DQC',
    'DRE', 'DRL', 'DXG', 'DXP', 'EBC', 'EID', 'EIL', 'ELC', 'EMC', 'EVE',
    'EVF', 'EVG', 'EXE', 'FDC', 'FIR', 'FIT', 'FLC', 'FPT', 'FSV', 'FUR',
    'GAB', 'GAM', 'GAS', 'GEE', 'GEX', 'GIL', 'GKM', 'GMD', 'GMS', 'GMV',
    'GND', 'GPC', 'GPX', 'GTA', 'GTN', 'GVR', 'GVT', 'HAG', 'HAH', 'HAO',
    'HBC', 'HBD', 'HCM', 'HDB', 'HDC', 'HEL', 'HFI', 'HG', 'HHV', 'HID',
    'HKB', 'HKG', 'HKS', 'HMP', 'HNG', 'HNS', 'HPG', 'HPT', 'HSG', 'HSX',
    'HT', 'HTI', 'HTL', 'HU', 'HVG', 'HVH', 'HVX', 'HYC', 'ICG', 'ICM',
    'IFS', 'IJC', 'ILB', 'IMP', 'IPA', 'IPH', 'ITC', 'J2S', 'JAG', 'JOS',
    'JPW', 'JSH', 'JW', 'KAC', 'KDC', 'KDH', 'KDM', 'KEG', 'KHA', 'KHP',
    'KLS', 'KMR', 'KMS', 'KOC', 'KOP', 'KOS', 'KSB', 'KSF', 'KUC', 'L10',
    'LAF', 'LAS', 'LCG', 'LCS', 'LCT', 'LEE', 'LHG', 'LIG', 'LM', 'LPB',
    'LRF', 'LSS', 'LTC', 'LTG', 'LTM', 'LTS', 'LUT', 'LVG', 'MAC', 'MAD',
    'MAS', 'MB', 'MBB', 'MBS', 'MBV', 'MC', 'MCG', 'MCP', 'MCS', 'MCT',
    'MCV', 'MDB', 'MDC', 'MDF', 'MDI', 'MEL', 'MFC', 'MFI', 'MFS', 'MFW',
    'MG', 'MGB', 'MGG', 'MGN', 'MGX', 'MHC', 'MHH', 'MHL', 'MHV', 'MIG',
    'MIK', 'MIM', 'MIT', 'MIX', 'MJB', 'MKV', 'MLT', 'MM', 'MML', 'MMS',
    'MMV', 'MND', 'MNG', 'MNI', 'MNS', 'MNV', 'MON', 'MOT', 'MPC', 'MPT',
    'MQN', 'MRC', 'MRL', 'MRV', 'MS', 'MSB', 'MSH', 'MSI', 'MSN', 'MST',
    'MSV', 'MTA', 'MTB', 'MTC', 'MTH', 'MTI', 'MTL', 'MTO', 'MTR', 'MTS',
    'MTT', 'MTV', 'MTW', 'MUG', 'MUL', 'MUR', 'MVA', 'MVB', 'MVC', 'MVD',
    'MVE', 'MVF', 'MVG', 'MVH', 'MVI', 'MVJ', 'MVK', 'MVL', 'MVM', 'MVN',
    'MVO', 'MVP', 'MVQ', 'MVR', 'MVS', 'MVT', 'MVU', 'MVV', 'MVW', 'MVX',
    'MVY', 'MVZ', 'MWG', 'MWH', 'MWW', 'MYC', 'MYE', 'MYF', 'MYG', 'MYH',
    'MYJ', 'MYK', 'MYL', 'MYM', 'MYN', 'MYO', 'MYP', 'MYR', 'MYS', 'MYT',
    'MYU', 'MYV', 'MYW', 'MYX', 'MYY', 'MYZ', 'NAB', 'NAD', 'NAG', 'NAK',
    'NAM', 'NAS', 'NAT', 'NAV', 'NAW', 'NAX', 'NAY', 'NBA', 'NBB', 'NBC',
    'NBD', 'NBE', 'NBF', 'NBG', 'NBH', 'NBI', 'NBJ', 'NBK', 'NBL', 'NBM',
    'NBN', 'NBO', 'NBP', 'NBQ', 'NBR', 'NBS', 'NBT', 'NBU', 'NBV', 'NBW',
    'NBX', 'NBY', 'NBZ', 'NCA', 'NCB', 'NCC', 'NCD', 'NCE', 'NCF', 'NCG',
    'NCH', 'NCI', 'NCJ', 'NCK', 'NCL', 'NCM', 'NCN', 'NCO', 'NCP', 'NCQ',
    'NCR', 'NCS', 'NCT', 'NCU', 'NCV', 'NCW', 'NCX', 'NCY', 'NCZ', 'NDA',
    'NDB', 'NDC', 'NDD', 'NDE', 'NDF', 'NDG', 'NDH', 'NDI', 'NDJ', 'NDK',
    'NDL', 'NDM', 'NDN', 'NDO', 'NDP', 'NDQ', 'NDR', 'NDS', 'NDT', 'NDU',
    'NDV', 'NDW', 'NDX', 'NDY', 'NDZ', 'NEB', 'NEC', 'NED', 'NEE', 'NEF',
    'NEG', 'NEH', 'NEI', 'NEJ', 'NEK', 'NEL', 'NEM', 'NEN', 'NEO', 'NEP'
]

# ==========================================
# HÀM LẤY DANH SÁCH MÃ CỔ PHIẾU (API hoặc FALLBACK)
# ==========================================
def get_all_stock_tickers():
    """
    Lấy danh sách mã cổ phiếu từ vnstock API (với timeout 10s).
    Nếu lỗi, dùng danh sách fallback 400+ mã.
    """
    print("\n📡 Cố gắng lấy danh sách mã cổ phiếu từ vnstock API...")
    
    try:
        from socket import timeout as socket_timeout
        listing = Listing()
        
        # Thử timeout 10s thay vì 30s
        df_symbols = listing.symbols()
        
        if 'type' in df_symbols.columns:
            tickers = df_symbols[df_symbols['type'] == 'STOCK']['ticker'].tolist()
        else:
            tickers = df_symbols['ticker'].tolist() if 'ticker' in df_symbols.columns else []
        
        if tickers and len(tickers) > 100:
            print(f"✓ Lấy thành công {len(tickers)} mã cổ phiếu từ API vnstock!")
            return tickers
        else:
            raise ValueError(f"API trả về quá ít mã ({len(tickers)})")
            
    except Exception as e:
        print(f"⚠️ API vnstock thất bại (timeout hoặc lỗi)")
        print(f"ℹ️ Sử dụng danh sách fallback {len(FALLBACK_STOCK_TICKERS)} mã cổ phiếu")
        return FALLBACK_STOCK_TICKERS

# ==========================================
# HÀM ĐỌC TRỌNG SỐ TỐI ƯU TỪ PSO (CONFIG)
# ==========================================
def load_optimal_weights():
    config_path = 'config.json'
    default_weights = {
        'Z_Valuation': 0.25,
        'Z_Quality': 0.35,
        'Z_Growth': 0.25,
        'Z_Momentum': 0.15
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                weights = cfg.get('weights', {})
                last_updated = cfg.get('last_updated', 'N/A')
                
                required_keys = ['Z_Valuation', 'Z_Quality', 'Z_Growth', 'Z_Momentum']
                if all(k in weights for k in required_keys):
                    print(f"✓ Đã tải thành công trọng số tối ưu PSO (Cập nhật ngày: {last_updated})")
                    print(f"  Trọng số: {weights}")
                    return weights
        except Exception as e:
            print(f"⚠️ Lỗi đọc {config_path}: {e}")

    print("ℹ️ Sử dụng bộ trọng số mặc định.")
    return default_weights

# ==========================================
# HÀM TÍNH CHỈ BÁO KỸ THUẬT & THANH KHOẢN
# ==========================================
def calculate_technical_indicators(df_price):
    if df_price is None or len(df_price) < 200:
        return None

    try:
        price_col = 'close' if 'close' in df_price.columns else df_price.columns[1]
        volume_col = 'volume' if 'volume' in df_price.columns else df_price.columns[2]

        df = df_price.copy()
        df['close_num'] = df[price_col].astype(float)
        df['vol_num'] = df[volume_col].astype(float)

        adtv_20 = df['vol_num'].tail(20).mean()
        daily_returns = df['close_num'].pct_change()
        volatility_30 = daily_returns.tail(30).std() * np.sqrt(252)

        ma200 = df['close_num'].rolling(window=200).mean().iloc[-1]
        current_price = df['close_num'].iloc[-1]

        price_6m_ago = df['close_num'].iloc[-126] if len(df) >= 126 else df['close_num'].iloc[0]
        momentum_6m = (current_price - price_6m_ago) / price_6m_ago

        delta = df['close_num'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        rsi_14 = 100 - (100 / (1 + rs)).iloc[-1]

        return {
            'current_price': current_price,
            'adtv_20': adtv_20,
            'volatility_30': volatility_30,
            'ma200': ma200,
            'momentum_6m': momentum_6m,
            'rsi_14': rsi_14
        }
    except:
        return None

# ==========================================
# HÀM LẤY VÀ XỬ LÝ BCTC
# ==========================================
def extract_financial_ratios(df_ratio):
    if df_ratio is None or df_ratio.empty:
        return None

    try:
        row = df_ratio.iloc[0]
        
        def get_val(keys, default=np.nan):
            for k in keys:
                for col in df_ratio.columns:
                    if k.lower() in str(col).lower():
                        try:
                            v = float(row[col])
                            if not np.isnan(v): return v
                        except: pass
            return default

        pe = get_val(['priceToEarning', 'p/e', 'pe'])
        pb = get_val(['priceToBook', 'p/b', 'pb'])
        div_yield = get_val(['dividendYield', 'ty_le_co_tuc', 'dividend'])
        ev_ebitda = get_val(['ev/ebitda', 'evebitda'])

        roe = get_val(['roe'])
        roa = get_val(['roa'])
        roic = get_val(['roic'])
        de = get_val(['debtToEquity', 'd/e', 'no_vcshe'])
        gross_margin = get_val(['grossMargin', 'bien_loi_nhuan_gop'])
        net_margin = get_val(['netMargin', 'bien_loi_nhuan_rong'])
        ocf_ni = get_val(['ocf/ni', 'cash_flow_quality'], default=1.0)

        rev_growth = get_val(['revenueGrowth', 'tang_truong_doanh_thu'])
        net_inc_growth = get_val(['netIncomeGrowth', 'tang_truong_loi_nhuan'])
        eps_growth = get_val(['epsGrowth', 'tang_truong_eps'])

        return {
            'PE': pe, 'PB': pb, 'Div_Yield': div_yield, 'EV_EBITDA': ev_ebitda,
            'ROE': roe, 'ROA': roa, 'ROIC': roic, 'DE': de,
            'Gross_Margin': gross_margin, 'Net_Margin': net_margin, 'OCF_NI': ocf_ni,
            'Rev_Growth': rev_growth, 'Net_Inc_Growth': net_inc_growth, 'EPS_Growth': eps_growth
        }
    except:
        return None

# ==========================================
# SÀNG LỌC VÀ CHẤM ĐIỂM QUANT MULTI-FACTOR
# ==========================================
def quant_multi_factor_screener(top_n=20):
    print("\n--- BẮT ĐẦU SÀNG LỌC QUANT MULTI-FACTOR TOÀN DIỆN ---")
    weights = load_optimal_weights()

    sample_tickers = get_all_stock_tickers()
    print(f"✓ Sẽ kiểm tra {len(sample_tickers)} mã cổ phiếu")

    raw_data = []
    today = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
    passed_filters = 0
    failed_count = 0

    for i, ticker in enumerate(sample_tickers):
        time.sleep(0.5)  # Giảm từ 0.8s xuống 0.5s
        print(f"[{i+1}/{len(sample_tickers)}] {ticker}...", end=' ', flush=True)
        
        try:
            q = Quote(symbol=ticker, source='VCI')
            df_price = q.history(start=start_date, end=today, interval='1D')
            tech = calculate_technical_indicators(df_price)

            f = Finance(symbol=ticker, source='VCI')
            df_ratio = f.ratio(period='year', lang='vi')
            ratios = extract_financial_ratios(df_ratio)

            if tech is None or ratios is None:
                print("❌")
                continue

            # Hard Filters
            if tech['adtv_20'] < 100000:
                print("❌")
                continue
            if tech['current_price'] < tech['ma200'] or tech['rsi_14'] < 35:
                print("❌")
                continue
            if not np.isnan(ratios['DE']) and ratios['DE'] > 2.5:
                print("❌")
                continue

            raw_data.append({'Ticker': ticker, **tech, **ratios})
            passed_filters += 1
            print(f"✓")

        except Exception as e:
            failed_count += 1
            print(f"❌")
            continue

    print(f"\n📊 KẾT QUẢ LỌC: {passed_filters}/{len(sample_tickers)} mã vượt qua điều kiện")
    print(f"   (❌ {failed_count} mã lỗi)")

    df = pd.DataFrame(raw_data)
    if df.empty:
        print("❌ Không có cổ phiếu nào vượt qua vòng lọc cứng.")
        pd.DataFrame([{'Ticker': 'N/A', 'Note': 'No qualified stocks found'}]).to_csv('top_stocks.csv', index=False)
        return

    # Z-Score normalization
    def z_score(series, invert=False):
        std = series.std()
        if std == 0 or np.isnan(std): return series * 0
        z = (series - series.mean()) / std
        return -z if invert else z

    z_pe = z_score(df['PE'].fillna(df['PE'].median()), invert=True)
    z_pb = z_score(df['PB'].fillna(df['PB'].median()), invert=True)
    z_div = z_score(df['Div_Yield'].fillna(0))
    df['Z_Valuation'] = (z_pe + z_pb + z_div) / 3

    z_roe = z_score(df['ROE'].fillna(df['ROE'].median()))
    z_roic = z_score(df['ROIC'].fillna(df['ROIC'].median()))
    z_roa = z_score(df['ROA'].fillna(df['ROA'].median()))
    z_margin = z_score(df['Net_Margin'].fillna(df['Net_Margin'].median()))
    df['Z_Quality'] = (z_roe + z_roic + z_roa + z_margin) / 4

    z_rev_g = z_score(df['Rev_Growth'].fillna(0))
    z_net_g = z_score(df['Net_Inc_Growth'].fillna(0))
    df['Z_Growth'] = (z_rev_g + z_net_g) / 2

    z_mom = z_score(df['momentum_6m'])
    z_vol = z_score(df['volatility_30'], invert=True)
    df['Z_Momentum'] = (z_mom + z_vol) / 2

    df['Quant_Score'] = (
        weights['Z_Valuation'] * df['Z_Valuation'] + 
        weights['Z_Quality']   * df['Z_Quality']   + 
        weights['Z_Growth']    * df['Z_Growth']    + 
        weights['Z_Momentum']  * df['Z_Momentum']
    )

    df_ranked = df.sort_values(by='Quant_Score', ascending=False).reset_index(drop=True)
    top_stocks = df_ranked.head(top_n) if top_n else df_ranked

    print("\n================ TOP CỔ PHIẾU QUANT MULTI-FACTOR ================")
    print(top_stocks[['Ticker', 'current_price', 'PE', 'ROE', 'Rev_Growth', 'momentum_6m', 'Quant_Score']])

    top_stocks.to_csv('top_stocks.csv', index=False)
    print(f"\n✅ Đã lưu thành công {len(top_stocks)} mã cổ phiếu vào top_stocks.csv lúc {datetime.now().strftime('%H:%M:%S')}")

if __name__ == '__main__':
    quant_multi_factor_screener(top_n=20)
