import pandas as pd
import numpy as np
import streamlit as st

# ★★★ 📌【重要】このURLを公開されたCSVファイルのURLに書き換えてください ★★★
# 例: CSV_URL = "https://raw.githubusercontent.com/user/repo/main/data.csv"
CSV_URL = "https://github.com/nuremoh868-boop/baseball-analyzer/blob/main/%E9%87%8E%E7%90%83%E5%8F%AF%E8%A6%96%E5%8C%96%E3%83%84%E3%83%BC%E3%83%AB%E4%BB%AE%E3%83%87%E3%83%BC%E3%82%BF.csv"
CSV_ENCODING = 'shift_jis'
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# ======================================================================
# データロード＆計算ロジック（URL読み込み用に分離）
# ======================================================================

# @st.cache_data デコレーターはそのまま維持し、高速化します。
@st.cache_data 
def clean_data(df):
    """データクリーニング関数 (変更なし)"""
    # ... (前回の clean_data 関数の中身をそのまま貼り付け)
    df = df.copy() 
    key_cols = ['game_id', 'game_play_num']
    for col in key_cols:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=[col])
        df.loc[:, col] = df[col].astype(int)
        
    df = df.sort_values(key_cols).reset_index(drop=True)
    df = df.drop_duplicates(subset=key_cols, keep='first').copy()
    
    df.loc[:, 'result_of_pa_clean'] = df['result_of_pa'].astype(str).str.strip().fillna('')
    
    error_cols = ['catcher_error', 'first_error', 'second_error', 'third_error', 
                  'short_error', 'left_error', 'center_error', 'right_error']
    for col in error_cols:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
    df.loc[:, 'batter_name'] = df['batter_name'].astype(str).str.strip().fillna('')
    if 'runs_against' in df.columns:
        df.loc[:, 'runs_against'] = pd.to_numeric(df['runs_against'], errors='coerce').fillna(0)
    if 'earned_run' in df.columns:
        df.loc[:, 'earned_run'] = pd.to_numeric(df['earned_run'], errors='coerce').fillna(0)

    return df.drop(columns=['result_of_pa'], errors='ignore')

@st.cache_data
def track_pitcher_responsibility_final_unified(df):
    """R/ER責任追跡ロジック (変更なし)"""
    # ... (前回の track_pitcher_responsibility_final_unified 関数の中身をそのまま貼り付け)
    df_dict = df.set_index(['game_id', 'game_play_num']).to_dict('index')
    pitcher_R_total = {}
    pitcher_ER_total = {}
    scoring_plays = df[df['runs_against'] > 0].copy()

    for i in scoring_plays.index:
        current_row = scoring_plays.loc[i]
        game_id = current_row['game_id']
        play_num = current_row['game_play_num']
        r_scored_in_play = current_row['runs_against']
        er_scored_in_play = current_row['earned_run'] 
        responsible_runners = []
        if current_row.get('batter_runner_to') == '4':
            responsible_runners.append('BATTER_' + str(game_id) + str(play_num))
        
        prev_play_num = play_num - 1
        prev_row = df_dict.get((game_id, prev_play_num))
        if prev_row is not None:
            if current_row.get('third_runner_to') == '4':
                scorer_name = prev_row.get('third_runner_name')
                if pd.notna(scorer_name): responsible_runners.append(scorer_name)
            if current_row.get('second_runner_to') == '4':
                scorer_name = prev_row.get('second_runner_name')
                if pd.notna(scorer_name): responsible_runners.append(scorer_name)
            if current_row.get('first_runner_to') == '4':
                scorer_name = prev_row.get('first_runner_name')
                if pd.notna(scorer_name): responsible_runners.append(scorer_name)
        
        final_R_pitchers = set()
        for runner_key in set(responsible_runners):
            if runner_key.startswith('BATTER_'):
                pitcher = current_row.get('pitcher_name')
                final_R_pitchers.add(pitcher)
                continue
            runner_name = runner_key
            outfield_runner = df[(df['game_id'] == game_id) & (df['batter_name'] == runner_name)]
            if outfield_runner.empty: continue
            
            outfield_rows = outfield_runner[
                outfield_runner['result_of_pa_clean'].str.contains('安|２|３|本')
            ]
            
            if not outfield_rows.empty:
                outfield_row = outfield_rows.iloc[0] 
                pitcher = outfield_row.get('pitcher_name')
                final_R_pitchers.add(pitcher)

        r_per_pitcher = r_scored_in_play / max(1, len(final_R_pitchers))
        er_per_pitcher = er_scored_in_play / max(1, len(final_R_pitchers))

        for pitcher in final_R_pitchers:
            pitcher_R_total[pitcher] = pitcher_R_total.get(pitcher, 0) + r_per_pitcher
            pitcher_ER_total[pitcher] = pitcher_ER_total.get(pitcher, 0) + er_per_pitcher
    
    pitcher_R_total = {k: round(v) for k, v in pitcher_R_total.items()}
    pitcher_ER_total = {k: round(v) for k, v in pitcher_ER_total.items()}
    
    return pitcher_R_total, pitcher_ER_total

def assign_error_to_fielder(row):
    """エラー付与ロジック (変更なし)"""
    # ... (前回の assign_error_to_fielder 関数の中身をそのまま貼り付け)
    if not row['result_of_pa_clean'].endswith('失'):
        return None

    fielder_map = {
        'catcher_error': 'catcher_name',
        'first_error': 'first_name',
        'second_error': 'second_name',
        'third_error': 'third_name',
        'short_error': 'short_name',
        'left_error': 'left_name',
        'center_error': 'center_name',
        'right_error': 'right_name',
    }
    
    error_cols = list(fielder_map.keys())
    error_pos_found = [col for col in error_cols if row[col] == 1]
    
    if len(error_pos_found) == 1:
        error_fielder_col = fielder_map[error_pos_found[0]]
        return row[error_fielder_col]
        
    elif len(error_pos_found) == 0:
        return row['pitcher_name']
        
    else:
        return None 

@st.cache_data
def calculate_stats(df_raw):
    """成績計算ロジック (変更なし)"""
    
    # 1. データのクリーンアップ
    df = clean_data(df_raw)

    # --------------------------------------------------
    # 野手成績の集計・計算
    # --------------------------------------------------
    HIT_PATTERN = r'.*(安|２|３|本)'
    SINGLE_PATTERN = r'.*安'
    DOUBLE_PATTERN = r'.*２'
    TRIPLE_PATTERN = r'.*３'
    HOME_RUN_PATTERN = r'.*本'

    df_pa_ended = df[df['result_of_pa_clean'].str.len() == 2].copy()

    # エラー (E) の集計ロジック 
    df_errors = df_pa_ended[df_pa_ended['result_of_pa_clean'].str.endswith('失')].copy()
    if not df_errors.empty:
        df_errors.loc[:, 'error_fielder'] = df_errors.apply(assign_error_to_fielder, axis=1)
        error_counts = df_errors.dropna(subset=['error_fielder']).groupby('error_fielder').size()
        error_df = error_counts.reset_index(name='E_tracked').rename(columns={'error_fielder': 'batter_name'})
    else:
        error_df = pd.DataFrame(columns=['batter_name', 'E_tracked'])
        
    # 打席関連の集計
    batter_stats_raw = df_pa_ended.groupby('batter_name').agg(
        PA=('result_of_pa_clean', 'count'), 
        H=('result_of_pa_clean', lambda x: x.str.contains(HIT_PATTERN, regex=True).sum()), 
        Singles=('result_of_pa_clean', lambda x: x.str.contains(SINGLE_PATTERN, regex=True).sum()), 
        Doubles=('result_of_pa_clean', lambda x: x.str.contains(DOUBLE_PATTERN, regex=True).sum()), 
        Triples=('result_of_pa_clean', lambda x: x.str.contains(TRIPLE_PATTERN, regex=True).sum()), 
        HR=('result_of_pa_clean', lambda x: x.str.contains(HOME_RUN_PATTERN, regex=True).sum()), 
        BB=('result_of_pa_clean', lambda x: (x == '四球').sum()), 
        HBP=('result_of_pa_clean', lambda x: (x == '死球').sum()), 
        SO=('result_of_pa_clean', lambda x: (x == '三振').sum()), 
        RBI=('rbi', 'sum'), 
        G=('game_id', pd.Series.nunique), 
        SB=('first_stolen_base', 'sum'), 
        CS=('first_caught_stealing', 'sum'), 
        SH=('bunt', lambda x: (x == 1).sum()), 
        SF=('result_of_pa_clean', lambda x: (x == '犠飛').sum()), 
    ).reset_index()

    # Eの集計結果をマージ
    batter_stats_raw = pd.merge(batter_stats_raw, error_df, on='batter_name', how='left').fillna(0)
    batter_stats_raw = batter_stats_raw.rename(columns={'E_tracked': 'E'})
    batter_stats_raw['E'] = batter_stats_raw['E'].astype(int) 

    # 指標の計算
    batter_stats_raw['AB'] = batter_stats_raw['PA'] - batter_stats_raw['BB'] - batter_stats_raw['HBP'] - batter_stats_raw['SH'] - batter_stats_raw['SF']
    batter_stats_raw['AB'] = batter_stats_raw['AB'].clip(lower=0) 
    batter_stats_raw['AVG'] = (batter_stats_raw['H'] / batter_stats_raw['AB']).replace([np.inf, -np.inf], 0).fillna(0).round(3)
    batter_stats_raw['TB'] = batter_stats_raw['Singles'] + 2 * batter_stats_raw['Doubles'] + 3 * batter_stats_raw['Triples'] + 4 * batter_stats_raw['HR']
    batter_stats_raw['SLG'] = (batter_stats_raw['TB'] / batter_stats_raw['AB']).replace([np.inf, -np.inf], 0).fillna(0).round(3)
    batter_stats_raw['OBP'] = ((batter_stats_raw['H'] + batter_stats_raw['BB'] + batter_stats_raw['HBP']) / 
                               (batter_stats_raw['AB'] + batter_stats_raw['BB'] + batter_stats_raw['HBP'] + batter_stats_raw['SF'])
                              ).replace([np.inf, -np.inf], 0).fillna(0).round(3)
    batter_stats_raw['OPS'] = (batter_stats_raw['OBP'] + batter_stats_raw['SLG']).round(3)


    # --------------------------------------------------
    # 投手成績の集計・計算 
    # --------------------------------------------------
    pitcher_groups = ['game_id', 'inning', 'top_bottom']
    df['out_gained'] = 0
    df['out_diff'] = df.groupby(pitcher_groups)['out'].diff().fillna(0)
    df.loc[df['out_diff'] > 0, 'out_gained'] = df['out_diff']
    inning_max_out = df.groupby(pitcher_groups)['out'].max()
    def calculate_needed_outs(max_out):
        if max_out == 2: return 1
        elif max_out == 1: return 2
        elif max_out == 0: return 3
        return 0
    needed_outs = inning_max_out.apply(calculate_needed_outs)
    innings_to_correct = needed_outs[needed_outs > 0].reset_index()
    extra_outs_by_pitcher = {} 
    for index, row in innings_to_correct.iterrows():
        game_id, inning, top_bottom = row['game_id'], row['inning'], row['top_bottom']
        outs_to_add = row['out'] 
        last_row_in_inning = df[(df['game_id'] == game_id) & (df['inning'] == inning) & (df['top_bottom'] == top_bottom)].iloc[-1]
        pitcher = last_row_in_inning['pitcher_name']
        extra_outs_by_pitcher[pitcher] = extra_outs_by_pitcher.get(pitcher, 0) + outs_to_add
    total_outs = df.groupby('pitcher_name')['out_gained'].sum()
    total_outs_series = pd.Series(extra_outs_by_pitcher)
    total_outs = total_outs.add(total_outs_series, fill_value=0).fillna(0)

    # 投手成績集計
    pitcher_stats_raw = df_pa_ended.groupby('pitcher_name').agg(
        TBF=('batter_name', 'count'),
        Pitches=('pitches_of_pa', 'sum'),
        H_allowed=('result_of_pa_clean', lambda x: x.str.contains(HIT_PATTERN, regex=True).sum()), 
        Singles_allowed=('result_of_pa_clean', lambda x: x.str.contains(SINGLE_PATTERN, regex=True).sum()), 
        Doubles_allowed=('result_of_pa_clean', lambda x: x.str.contains(DOUBLE_PATTERN, regex=True).sum()), 
        Triples_allowed=('result_of_pa_clean', lambda x: x.str.contains(TRIPLE_PATTERN, regex=True).sum()), 
        HR_allowed=('result_of_pa_clean', lambda x: x.str.contains(HOME_RUN_PATTERN, regex=True).sum()), 
        K=('result_of_pa_clean', lambda x: (x == '三振').sum()), 
        BB_allowed=('result_of_pa_clean', lambda x: (x == '四球').sum()), 
        HBP_allowed=('result_of_pa_clean', lambda x: (x == '死球').sum()),
        R=('runs_against', 'sum'), 
        ER=('earned_run', 'sum'), 
        WP=('wild_pitch', 'sum'),
        BALK=('balk', 'sum'),
        G=('game_id', pd.Series.nunique),
    ).reset_index()

    # 投手成績にエラーをマージ
    pitcher_error_df = error_df.rename(columns={'batter_name': 'pitcher_name', 'E_tracked': 'E_p'}).set_index('pitcher_name')
    pitcher_stats_raw = pitcher_stats_raw.set_index('pitcher_name')
    pitcher_stats_raw = pitcher_stats_raw.merge(pitcher_error_df, left_index=True, right_index=True, how='left').fillna(0)
    pitcher_stats_raw = pitcher_stats_raw.reset_index().rename(columns={'E_p': 'E'})
    pitcher_stats_raw['E'] = pitcher_stats_raw['E'].astype(int)

    # R/ER責任追跡ロジックの適用
    pitcher_R_total, pitcher_ER_total = track_pitcher_responsibility_final_unified(df)
    r_df = pd.DataFrame(pitcher_R_total.items(), columns=['pitcher_name', 'R_tracked'])
    er_df = pd.DataFrame(pitcher_ER_total.items(), columns=['pitcher_name', 'ER_tracked'])
    pitcher_stats_raw = pitcher_stats_raw.drop(columns=['R', 'ER'], errors='ignore')
    pitcher_stats_raw = pitcher_stats_raw.merge(r_df, on='pitcher_name', how='outer').fillna(0)
    pitcher_stats_raw = pitcher_stats_raw.merge(er_df, on='pitcher_name', how='outer').fillna(0)
    pitcher_stats_raw = pitcher_stats_raw.rename(columns={'R_tracked': 'R', 'ER_tracked': 'ER'})

    # 指標の再計算
    pitcher_stats_raw['Total_Outs'] = pitcher_stats_raw['pitcher_name'].map(total_outs).fillna(0)
    numerical_ip = pitcher_stats_raw['Total_Outs'] / 3
    
    # 表示用のIPを計算
    def format_ip(total_outs):
        full_innings = int(total_outs // 3)
        partial_outs = int(total_outs % 3)
        return f"{full_innings}.{partial_outs}"
    
    pitcher_stats_raw['IP'] = pitcher_stats_raw['Total_Outs'].apply(format_ip)
    
    # 率の計算 (分母が0の場合は0を適用)
    pitcher_stats_raw['ERA'] = ((pitcher_stats_raw['ER'] * 9) / numerical_ip).replace([np.inf, -np.inf], 0).fillna(0).round(2)
    pitcher_stats_raw['K/9'] = ((pitcher_stats_raw['K'] * 9) / numerical_ip).replace([np.inf, -np.inf], 0).fillna(0).round(2)
    pitcher_stats_raw['BB/9'] = ((pitcher_stats_raw['BB_allowed'] * 9) / numerical_ip).replace([np.inf, -np.inf], 0).fillna(0).round(2)
    pitcher_stats_raw['WHIP'] = ((pitcher_stats_raw['H_allowed'] + pitcher_stats_raw['BB_allowed']) / numerical_ip).replace([np.inf, -np.inf], 0).fillna(0).round(2)
    pitcher_stats_raw['K-BB%'] = (((pitcher_stats_raw['K'] - pitcher_stats_raw['BB_allowed']) / pitcher_stats_raw['TBF']) * 100).replace([np.inf, -np.inf], 0).fillna(0).round(1)

    return batter_stats_raw, pitcher_stats_raw

@st.cache_data
def load_and_calculate_data(csv_url, csv_encoding):
    """URLからデータを読み込み、成績を計算する一連の処理"""
    try:
        # ★★★ ここでURLから直接ファイルを読み込みます ★★★
        df_raw = pd.read_csv(csv_url, encoding=csv_encoding, low_memory=False)
        st.info(f"データソース: `{csv_url}` から読み込みました。")
    except Exception as e:
        st.error(f"❌ データの自動読み込みに失敗しました。URLまたはファイル形式を確認してください。")
        st.code(f"エラー詳細: {e}")
        return None, None
        
    # 既存の計算ロジックを呼び出す
    return calculate_stats(df_raw)

def format_stats_for_display(df, is_batter=True):
    """表示形式調整ロジック (変更なし)"""
    # ... (前回の format_stats_for_display 関数の中身をそのまま貼り付け)
    if df.empty:
        return df

    if is_batter:
        output_cols = ['batter_name', 'G', 'PA', 'AB', 'H', 'HR', 'RBI', 'SO', 'BB', 'E', 
                       'AVG', 'OBP', 'SLG', 'OPS'] # 表示する列を絞り込み
        df_display = df[output_cols].copy()
        df_display = df_display.rename(columns={'batter_name': '選手名'})
        
        # 率を小数点3桁表示に
        for col in ['AVG', 'OBP', 'SLG', 'OPS']:
            df_display.loc[:, col] = df_display[col].round(3).map('{:.3f}'.format)
        # カウントを整数表示に
        for col in [c for c in df_display.columns if c not in ['選手名'] and c not in ['AVG', 'OBP', 'SLG', 'OPS']]:
            df_display.loc[:, col] = df_display[col].astype(int)
        
    else:
        output_cols = ['pitcher_name', 'G', 'IP', 'TBF', 'H_allowed', 'HR_allowed', 'K', 'BB_allowed', 'R', 'ER', 'E', 
                       'ERA', 'WHIP', 'K/9', 'BB/9'] # 表示する列を絞り込み
        df_display = df[output_cols].copy()
        df_display = df_display.rename(columns={'pitcher_name': '選手名', 'H_allowed': '被安打', 'HR_allowed': '被本塁打', 'BB_allowed': '与四球'})
        
        # IPは文字列としてそのまま、率は小数点2桁表示に
        for col in ['ERA', 'WHIP', 'K/9', 'BB/9']:
            df_display.loc[:, col] = df_display[col].round(2)

        # カウントを整数表示に
        int_cols = [c for c in df_display.columns if c not in ['選手名', 'IP'] and c not in ['ERA', 'WHIP', 'K/9', 'BB/9']]
        for col in int_cols:
             df_display.loc[:, col] = df_display[col].astype(int)
    
    return df_display

# ======================================================================
# Streamlit アプリケーションのメイン関数
# ======================================================================

def main():
    st.set_page_config(layout="centered", page_title="野球データ分析ツール")
    
    st.title("⚾ 野球データ分析ツール (Webデータ自動読み込み版)")
    st.markdown("---")

    # URL設定確認
    if "【ここに公開されたCSVファイルのURLを貼り付けてください】" in CSV_URL:
        st.error("⚠️ コード内の `CSV_URL` 定数に、公開されたCSVファイルのURLを貼り付けてください。")
        st.markdown("**ヒント**: GitHub Rawリンクが最も確実です。")
        return

    # 1. データ読み込みと計算の実行
    with st.spinner("🌐 URLからデータを取得し、成績を計算中..."):
        # load_and_calculate_data 関数がURLからデータを読み込み、計算を行う
        batter_stats, pitcher_stats = load_and_calculate_data(CSV_URL, CSV_ENCODING)
    
    if batter_stats is None or pitcher_stats is None:
        return # エラー処理は関数内で実行済み

    st.success("✅ Webからのデータ読み込みと成績計算が完了しました。")
    st.markdown("---")

    # 2. 選手名検索欄の作成
    st.subheader("選手成績検索")
    
    all_players = sorted(list(set(batter_stats['batter_name'].tolist() + pitcher_stats['pitcher_name'].tolist())))
    
    # セレクトボックス（ドロップダウンリスト）で選手名を選択
    player_name = st.selectbox(
        "選手名を選択してください", 
        options=['--- 全て表示 ---'] + all_players
    )

    # 3. 結果の表示
    if player_name == '--- 全て表示 ---':
        st.subheader("📋 全野手成績一覧")
        st.dataframe(format_stats_for_display(batter_stats, is_batter=True), use_container_width=True)
        
        st.subheader("📊 全投手成績一覧")
        st.dataframe(format_stats_for_display(pitcher_stats, is_batter=False), use_container_width=True)
    
    elif player_name:
        
        # --- 野手成績の表示 ---
        batter_result = batter_stats[batter_stats['batter_name'] == player_name]
        if not batter_result.empty:
            st.subheader(f"⚾️ 野手成績: {player_name}")
            st.dataframe(format_stats_for_display(batter_result, is_batter=True), hide_index=True, use_container_width=True)
        
        # --- 投手成績の表示 ---
        pitcher_result = pitcher_stats[pitcher_stats['pitcher_name'] == player_name]
        if not pitcher_result.empty:
            st.subheader(f"📊 投手成績: {player_name}")
            st.dataframe(format_stats_for_display(pitcher_result, is_batter=False), hide_index=True, use_container_width=True)
        
        if batter_result.empty and pitcher_result.empty:
            st.warning(f"選手名 '{player_name}' の成績は見つかりませんでした。")

# Streamlitアプリの実行
if __name__ == '__main__':
    main()