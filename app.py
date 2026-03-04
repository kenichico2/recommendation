"""
協調フィルタリングによる履修科目レコメンデーション
― グロービス経営大学院の科目を例に ―

相関係数を使ったユーザーベース協調フィルタリングの仕組みを
ステップごとに学べるインタラクティブ教材です。
"""

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title="協調フィルタリング教材",
    page_icon="📚",
    layout="wide",
)

# ============================================================
# データ定義
# ============================================================

COURSES = [
    "クリティカル・シンキング",
    "ビジネス定量分析",
    "マーケティング・経営戦略基礎",
    "マーケティング",
    "経営戦略",
    "事業創造",
    "アカウンティング基礎",
    "ファイナンス基礎",
    "アカウンティング",
    "ファイナンス",
    "企業分析とバリュエーション",
    "組織行動とリーダーシップ",
    "人材マネジメント",
    "パワーと影響力",
    "テクノベート・ストラテジー",
]

COURSE_AREAS = {
    "クリティカル・シンキング": "思考",
    "ビジネス定量分析": "思考",
    "マーケティング・経営戦略基礎": "マーケ・戦略",
    "マーケティング": "マーケ・戦略",
    "経営戦略": "マーケ・戦略",
    "事業創造": "マーケ・戦略",
    "アカウンティング基礎": "会計・財務",
    "ファイナンス基礎": "会計・財務",
    "アカウンティング": "会計・財務",
    "ファイナンス": "会計・財務",
    "企業分析とバリュエーション": "会計・財務",
    "組織行動とリーダーシップ": "人・組織",
    "人材マネジメント": "人・組織",
    "パワーと影響力": "人・組織",
    "テクノベート・ストラテジー": "テクノベート",
}

COURSE_LEVELS = {
    "クリティカル・シンキング": "基本",
    "ビジネス定量分析": "応用",
    "マーケティング・経営戦略基礎": "基本",
    "マーケティング": "応用",
    "経営戦略": "応用",
    "事業創造": "展開",
    "アカウンティング基礎": "基本",
    "ファイナンス基礎": "基本",
    "アカウンティング": "応用",
    "ファイナンス": "応用",
    "企業分析とバリュエーション": "展開",
    "組織行動とリーダーシップ": "基本",
    "人材マネジメント": "応用",
    "パワーと影響力": "応用",
    "テクノベート・ストラテジー": "応用",
}

AREA_COLORS = {
    "思考": "#FF6B6B",
    "マーケ・戦略": "#4ECDC4",
    "会計・財務": "#45B7D1",
    "人・組織": "#96CEB4",
    "テクノベート": "#DDA0DD",
}


def build_enrollment_data() -> pd.DataFrame:
    """20名のダミー履修データを生成する。"""
    data = {
        "学生A（戦略志向）":            [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        "学生B（財務志向）":            [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "学生C（人・組織志向）":        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        "学生D（バランス型）":          [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
        "学生E（戦略+財務型）":         [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        "学生F（組織+戦略型）":         [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        "学生G（財務特化型）":          [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        "学生H（全方位型）":            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        "学生I（初期段階）":            [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "学生J（戦略志向）":            [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        "学生K（財務+テクノベート型）": [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        "学生L（人・組織+戦略型）":     [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        "学生M（後期段階）":            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        "学生N（マーケ特化型）":        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "学生O（テクノベート志向）":    [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "学生P（財務+人組織型）":       [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
        "学生Q（戦略志向）":            [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        "学生R（初期段階）":            [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "学生S（会計+マーケ型）":       [1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        "学生T（変革リーダー型）":      [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
    }
    return pd.DataFrame(data, index=COURSES).T


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """学生間のピアソン相関係数を計算する。"""
    return df.T.corr()


def recommend_courses(
    target: str,
    df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    top_n: int = 5,
    min_corr: float = 0.0,
) -> pd.DataFrame:
    """
    協調フィルタリングによるレコメンド。

    Score(u, j) = Σ r_uv * e_vj  /  Σ |r_uv|
    (v は target と正の相関 > min_corr を持つ学生)
    """
    target_enroll = df.loc[target]
    not_enrolled = target_enroll[target_enroll == 0].index.tolist()

    if not not_enrolled:
        return pd.DataFrame()

    corrs = corr_matrix.loc[target].drop(target)
    similar = corrs[corrs > min_corr]

    scores = {}
    contributors = {}

    for course in not_enrolled:
        w_sum = 0.0
        w_total = 0.0
        names = []
        for student, r in similar.items():
            e = df.loc[student, course]
            w_sum += r * e
            w_total += abs(r)
            if e == 1:
                names.append((student, r))
        scores[course] = w_sum / w_total if w_total > 0 else 0.0
        contributors[course] = names

    result = pd.DataFrame({
        "スコア": scores,
        "領域": {c: COURSE_AREAS[c] for c in not_enrolled},
        "レベル": {c: COURSE_LEVELS[c] for c in not_enrolled},
    })
    result["推薦に寄与した類似学生"] = {
        c: ", ".join(f"{n}(r={r:.2f})" for n, r in contributors[c])
        if contributors[c] else "—"
        for c in not_enrolled
    }
    return result.sort_values("スコア", ascending=False).head(top_n)


# ============================================================
# セッション初期化
# ============================================================
if "df" not in st.session_state:
    st.session_state.df = build_enrollment_data()
    st.session_state.corr = compute_correlation(st.session_state.df)


# ============================================================
# UI
# ============================================================

st.title("📚 協調フィルタリングによる履修科目レコメンデーション")
st.caption("― グロービス経営大学院の科目を例に ―")

tabs = st.tabs([
    "① アルゴリズム解説",
    "② 履修データ",
    "③ 相関係数",
    "④ レコメンド結果",
    "⑤ 自分で試す",
    "⑥ 計算過程の詳細",
])

# ============================================================
# Tab 1: アルゴリズム解説
# ============================================================
with tabs[0]:
    st.header("協調フィルタリングとは？")

    st.markdown("""
**「あなたと似た人が選んだものは、あなたにも合うだろう」** という直感に基づくレコメンデーション手法です。

Amazonの「この商品を買った人はこんな商品も買っています」や、Netflixのおすすめ映画の裏側で使われている技術の基礎になっています。
""")

    st.subheader("今回のアルゴリズムの流れ")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
#### Step 1
**履修データの準備**

各学生の履修履歴をダミー変数（0/1）の行列で表現します。
""")
    with col2:
        st.markdown("""
#### Step 2
**類似度の計算**

学生間のピアソン相関係数を計算し、履修パターンの類似度とします。
""")
    with col3:
        st.markdown("""
#### Step 3
**スコアの算出**

未履修科目ごとに、類似学生の履修状況を相関係数で加重平均します。
""")
    with col4:
        st.markdown("""
#### Step 4
**おすすめ提示**

スコアの高い科目を「おすすめ」として提示します。
""")

    st.divider()

    st.subheader("数式")
    st.latex(
        r"\text{Score}(u, j) = "
        r"\frac{\displaystyle\sum_{v \in N(u)} r_{uv} \cdot e_{vj}}"
        r"{\displaystyle\sum_{v \in N(u)} |r_{uv}|}"
    )

    st.markdown("""
| 記号 | 意味 |
|------|------|
| $u$ | レコメンド対象の学生 |
| $j$ | 未履修科目 |
| $N(u)$ | 学生 $u$ と **正の相関** を持つ他の学生の集合 |
| $r_{uv}$ | 学生 $u$ と学生 $v$ の **ピアソン相関係数** |
| $e_{vj}$ | 学生 $v$ の科目 $j$ の履修有無（0 or 1）|
""")

    st.info("""
**なぜ相関係数を使うのか？**

相関係数は各学生の平均的な履修数（多い/少ない）を差し引いた上で
パターンの類似性を測定するため、
「たくさん履修している学生」と「少なく履修している学生」の間でも公平に比較できます。
""")

# ============================================================
# Tab 2: 履修データ
# ============================================================
with tabs[1]:
    st.header("Step 1: 履修データの確認")

    df = st.session_state.df

    st.markdown("""
グロービス経営大学院のカリキュラムを参考に、**15科目 × 20名**の学生の履修履歴データです。
各セルの値は **1（履修済み）** または **0（未履修）** です。
""")

    # 科目体系テーブル
    with st.expander("📖 科目体系を見る"):
        area_table = pd.DataFrame({
            "科目名": COURSES,
            "領域": [COURSE_AREAS[c] for c in COURSES],
            "レベル": [COURSE_LEVELS[c] for c in COURSES],
        })
        st.dataframe(area_table, hide_index=True, use_container_width=True)

    # 履修マトリクスをスタイリング付きで表示
    st.subheader("履修マトリクス")

    styled = df.style.map(
        lambda v: "background-color: #FFDDC1; font-weight: bold" if v == 1
        else "background-color: #F0F0F0; color: #AAA"
    )
    st.dataframe(styled, use_container_width=True, height=600)

    # 統計情報
    st.subheader("統計情報")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**科目別の履修者数**")
        course_counts = df.sum().reset_index()
        course_counts.columns = ["科目", "履修者数"]
        course_counts["領域"] = course_counts["科目"].map(COURSE_AREAS)
        chart = (
            alt.Chart(course_counts)
            .mark_bar()
            .encode(
                y=alt.Y("科目:N", sort="-x", title=None),
                x=alt.X("履修者数:Q", title="履修者数"),
                color=alt.Color(
                    "領域:N",
                    scale=alt.Scale(
                        domain=list(AREA_COLORS.keys()),
                        range=list(AREA_COLORS.values()),
                    ),
                ),
                tooltip=["科目", "履修者数", "領域"],
            )
            .properties(height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    with col_b:
        st.markdown("**学生別の履修科目数**")
        student_counts = df.sum(axis=1).reset_index()
        student_counts.columns = ["学生", "履修科目数"]
        chart2 = (
            alt.Chart(student_counts)
            .mark_bar(color="#45B7D1")
            .encode(
                y=alt.Y("学生:N", sort="-x", title=None),
                x=alt.X("履修科目数:Q", title="履修科目数"),
                tooltip=["学生", "履修科目数"],
            )
            .properties(height=400)
        )
        st.altair_chart(chart2, use_container_width=True)

# ============================================================
# Tab 3: 相関係数
# ============================================================
with tabs[2]:
    st.header("Step 2: 学生間の相関係数")

    corr_matrix = st.session_state.corr

    st.markdown("""
**ピアソン相関係数**で学生間の履修パターンの類似度を測定します。

$$r_{xy} = \\frac{\\sum(x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum(x_i - \\bar{x})^2} \\cdot \\sqrt{\\sum(y_i - \\bar{y})^2}}$$

- **+1 に近い** → 履修パターンが非常に似ている
- **0 に近い** → 履修パターンに関連がない
- **-1 に近い** → 履修パターンが正反対
""")

    st.subheader("相関係数マトリクス")

    # ヒートマップ風の色付き表示
    styled_corr = corr_matrix.round(2).style.background_gradient(
        cmap="RdBu_r", vmin=-1, vmax=1
    ).format("{:.2f}")
    st.dataframe(styled_corr, use_container_width=True, height=600)

    # 手計算の確認
    st.subheader("🔍 手計算で確認してみよう")

    col1, col2 = st.columns(2)
    students = list(df.index)
    with col1:
        s1 = st.selectbox("学生1", students, index=0, key="corr_s1")
    with col2:
        s2 = st.selectbox("学生2", students, index=5, key="corr_s2")

    if s1 == s2:
        st.warning("同じ学生が選択されています。異なる学生を選んでください。")
    else:
        vec1 = df.loc[s1]
        vec2 = df.loc[s2]
        mean1 = vec1.mean()
        mean2 = vec2.mean()
        dev1 = vec1 - mean1
        dev2 = vec2 - mean2

        detail = pd.DataFrame({
            "科目": COURSES,
            f"{s1.split('（')[0]}": vec1.values.astype(int),
            f"{s2.split('（')[0]}": vec2.values.astype(int),
            "偏差1": dev1.values.round(4),
            "偏差2": dev2.values.round(4),
            "偏差の積": (dev1.values * dev2.values).round(4),
        })

        st.dataframe(detail, hide_index=True, use_container_width=True)

        num = (dev1 * dev2).sum()
        den = np.sqrt((dev1**2).sum()) * np.sqrt((dev2**2).sum())
        r_manual = num / den if den != 0 else 0

        st.markdown(f"""
| 項目 | 値 |
|------|----|
| {s1.split('（')[0]} の平均（履修率）| {mean1:.4f} |
| {s2.split('（')[0]} の平均（履修率）| {mean2:.4f} |
| 分子（偏差の積の合計）| {num:.4f} |
| 分母（標準偏差の積）| {den:.4f} |
| **手計算の相関係数** | **{r_manual:.4f}** |
| pandas計算の相関係数 | {corr_matrix.loc[s1, s2]:.4f} |
""")
        st.success("✅ 一致していることが確認できます")

# ============================================================
# Tab 4: レコメンド結果
# ============================================================
with tabs[3]:
    st.header("Step 3–4: レコメンド結果")

    df = st.session_state.df
    corr_matrix = st.session_state.corr

    target = st.selectbox("レコメンド対象の学生を選択", list(df.index), key="rec_target")
    min_c = st.slider("最低相関係数の閾値", 0.0, 0.8, 0.0, 0.05, key="rec_min_corr")
    top_n = st.slider("おすすめ表示件数", 1, 10, 5, key="rec_top_n")

    enrolled = df.loc[target]

    # 現在の履修状況
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("📋 現在の履修科目")
        for course in enrolled[enrolled == 1].index:
            area = COURSE_AREAS[course]
            color = AREA_COLORS.get(area, "#999")
            st.markdown(
                f'<span style="color:{color}; font-weight:bold">●</span> {course}'
                f' <small style="color:gray">[{area}]</small>',
                unsafe_allow_html=True,
            )

        st.markdown(f"**履修済み: {int(enrolled.sum())} / {len(COURSES)} 科目**")

    with col_right:
        st.subheader("🎯 おすすめ科目")

        recs = recommend_courses(target, df, corr_matrix, top_n=top_n, min_corr=min_c)

        if recs.empty:
            st.info("全科目を履修済みです 🎓")
        else:
            for i, (course, row) in enumerate(recs.iterrows(), 1):
                area = row["領域"]
                color = AREA_COLORS.get(area, "#999")
                score = row["スコア"]

                st.markdown(
                    f"**{i}. {course}**"
                    f' <small style="color:gray">[{area} / {row["レベル"]}]</small>'
                    f" — スコア: **{score:.3f}**",
                    unsafe_allow_html=True,
                )
                st.progress(min(score, 1.0))
                with st.expander("推薦に寄与した類似学生"):
                    st.write(row["推薦に寄与した類似学生"])

    # 類似する学生
    st.subheader("👥 類似する学生 TOP 5")
    corrs = corr_matrix.loc[target].drop(target).sort_values(ascending=False)
    top5_corrs = corrs.head(5).reset_index()
    top5_corrs.columns = ["学生", "相関係数"]
    top5_corrs["相関係数"] = top5_corrs["相関係数"].round(3)
    st.dataframe(top5_corrs, hide_index=True, use_container_width=True)

    # スコアの可視化
    st.subheader("📊 全未履修科目のスコア")
    all_recs = recommend_courses(target, df, corr_matrix, top_n=100, min_corr=min_c)
    if not all_recs.empty:
        chart_data = all_recs.reset_index()
        chart_data.columns = ["科目", "スコア", "領域", "レベル", "寄与学生"]
        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                y=alt.Y("科目:N", sort="-x", title=None),
                x=alt.X("スコア:Q", title="レコメンデーションスコア",
                         scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "領域:N",
                    scale=alt.Scale(
                        domain=list(AREA_COLORS.keys()),
                        range=list(AREA_COLORS.values()),
                    ),
                ),
                tooltip=["科目", "スコア", "領域", "レベル"],
            )
            .properties(height=max(len(chart_data) * 30, 200))
        )
        st.altair_chart(chart, use_container_width=True)

# ============================================================
# Tab 5: 自分で試す
# ============================================================
with tabs[4]:
    st.header("🧪 自分の履修パターンで試してみよう")

    st.markdown("""
あなた自身の（仮想の）履修パターンを入力して、おすすめ科目を確認してみましょう。
""")

    # 領域ごとにグループ化して表示
    my_enrollment = {}
    areas_grouped = {}
    for c in COURSES:
        area = COURSE_AREAS[c]
        if area not in areas_grouped:
            areas_grouped[area] = []
        areas_grouped[area].append(c)

    for area, area_courses in areas_grouped.items():
        color = AREA_COLORS.get(area, "#999")
        st.markdown(
            f'<span style="color:{color}; font-weight:bold; font-size:1.1em">'
            f"■ {area}領域</span>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(area_courses))
        for i, course in enumerate(area_courses):
            level = COURSE_LEVELS[course]
            with cols[i]:
                my_enrollment[course] = st.checkbox(
                    f"{course}\n({level})", key=f"my_{course}"
                )

    my_vec = [int(my_enrollment[c]) for c in COURSES]
    n_enrolled = sum(my_vec)

    st.markdown(f"**選択中: {n_enrolled} 科目**")

    if n_enrolled == 0:
        st.warning("1つ以上の科目を選択してください。")
    elif n_enrolled == len(COURSES):
        st.success("全科目を履修済みです 🎓")
    else:
        # 一時DataFrameにあなたを追加
        df_me = st.session_state.df.copy()
        df_me.loc["あなた"] = my_vec
        corr_me = compute_correlation(df_me)

        min_c_me = st.slider(
            "最低相関係数の閾値", 0.0, 0.8, 0.0, 0.05, key="my_min_corr"
        )

        # 類似学生
        my_corrs = corr_me.loc["あなた"].drop("あなた").sort_values(ascending=False)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("👥 あなたに似た学生")
            for name, r in my_corrs.head(5).items():
                bar_color = "#4CAF50" if r > 0.3 else "#FFC107" if r > 0 else "#F44336"
                st.markdown(
                    f'<span style="color:{bar_color}">●</span> {name}: '
                    f"r = **{r:.3f}**",
                    unsafe_allow_html=True,
                )

        with col2:
            st.subheader("🎯 あなたへのおすすめ科目")
            my_recs = recommend_courses(
                "あなた", df_me, corr_me, top_n=5, min_corr=min_c_me
            )
            if not my_recs.empty:
                for i, (course, row) in enumerate(my_recs.iterrows(), 1):
                    st.markdown(
                        f"**{i}. {course}**"
                        f' <small style="color:gray">[{row["領域"]} / '
                        f'{row["レベル"]}]</small>'
                        f" — スコア: **{row['スコア']:.3f}**",
                        unsafe_allow_html=True,
                    )
                    st.progress(min(row["スコア"], 1.0))
            else:
                st.info("レコメンドが生成できませんでした。")

# ============================================================
# Tab 6: 計算過程の詳細
# ============================================================
with tabs[5]:
    st.header("🔬 計算過程の詳細")

    st.markdown("""
特定の学生・科目について、スコアがどのように計算されるか
一つ一つの掛け算と合計を確認できます。
""")

    df = st.session_state.df
    corr_matrix = st.session_state.corr

    target_detail = st.selectbox(
        "学生を選択", list(df.index), key="detail_target"
    )
    min_c_detail = st.slider(
        "最低相関係数", 0.0, 0.8, 0.0, 0.05, key="detail_min_corr"
    )

    target_enroll = df.loc[target_detail]
    not_enrolled = target_enroll[target_enroll == 0].index.tolist()

    if not not_enrolled:
        st.info("この学生は全科目を履修済みです。")
    else:
        course_detail = st.selectbox(
            "スコアを確認する科目（未履修科目）", not_enrolled, key="detail_course"
        )

        corrs = corr_matrix.loc[target_detail].drop(target_detail)
        positive = corrs[corrs > min_c_detail].sort_values(ascending=False)

        st.subheader(
            f"「{course_detail}」のスコア計算"
        )

        rows = []
        w_sum = 0.0
        w_total = 0.0
        for student, r in positive.items():
            e = int(df.loc[student, course_detail])
            contribution = r * e
            w_sum += contribution
            w_total += abs(r)
            rows.append({
                "学生": student,
                "相関係数 (r)": round(r, 4),
                "この科目の履修": "● 履修済" if e == 1 else "○ 未履修",
                "r × 履修": round(contribution, 4),
            })

        calc_df = pd.DataFrame(rows)
        st.dataframe(calc_df, hide_index=True, use_container_width=True)

        score = w_sum / w_total if w_total > 0 else 0
        st.markdown(f"""
---
**計算結果:**

$$\\text{{Score}} = \\frac{{\\sum r_{{uv}} \\times e_{{vj}}}}{{\\sum |r_{{uv}}|}} = \\frac{{{w_sum:.4f}}}{{{w_total:.4f}}} = {score:.4f}$$

| 項目 | 値 |
|------|----|
| 加重和（分子） | {w_sum:.4f} |
| 重み合計（分母）| {w_total:.4f} |
| **スコア** | **{score:.4f}** |
| 使用した類似学生数 | {len(positive)}名 |
| うち履修済みの学生数 | {sum(1 for _, r_row in positive.items() if df.loc[r_row if isinstance(r_row, str) else _, course_detail] == 1) if False else calc_df["この科目の履修"].str.contains("履修済").sum()}名 |
""")

# ============================================================
# フッター
# ============================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.85em">
    協調フィルタリング教材 | グロービス経営大学院の科目を素材にした学習用デモ<br>
    参考: <a href="https://mba.globis.ac.jp/curriculum/">グロービス経営大学院 カリキュラム</a>
</div>
""", unsafe_allow_html=True)
