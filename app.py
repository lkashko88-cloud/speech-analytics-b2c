import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openpyxl
import json
import re
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Речевая аналитика B2C — MVP",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Anonymization map ───────────────────────────────────────────────────────
FAKE_NAMES = {
    "Серова Светлана Аслановна": "Иванова Мария Сергеевна",
    "Фролкова Вера Николаевна": "Петрова Анна Дмитриевна",
    "Серебренников Степан Андреевич": "Козлов Алексей Игоревич",
}
FAKE_SHORTS = {
    "Серова Светлана": "Иванова Мария",
    "Фролкова Вера": "Петрова Анна",
    "Серебренников Степан": "Козлов Алексей",
}


def anonymize(name):
    return FAKE_NAMES.get(name, name)


def anonymize_short(name):
    return FAKE_SHORTS.get(name, name)


CRITERIA_PROCESS = [
    "Установление контакта",
    "Выявление и формирование потребностей",
    "Презентация на основе потребностей",
    "Работа с возражениями",
    "Инициатива к завершению сделки",
    "Соблюдение модулей завершения сделки",
]
CRITERIA_COMM = [
    "Грамотность речи",
    "Эмоциональный тон",
    "Активная позиция специалиста в диалоге",
    "Активное слушание",
]
ALL_CRITERIA = CRITERIA_PROCESS + CRITERIA_COMM
SHORT_NAMES = [
    "Контакт", "Потребности", "Презентация", "Возражения",
    "Завершение", "Модули", "Грамотность", "Тон",
    "Активность", "Слушание",
]

QA_SCALE = [
    (4.50, 5.00, "Высокий", "#2ecc71"),
    (3.80, 4.49, "Хороший", "#27ae60"),
    (3.00, 3.79, "Зона риска", "#f39c12"),
    (0.00, 2.99, "Критический", "#e74c3c"),
]

COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

REC_ACTIONS = {
    "Контакт": "Полное приветствие: имя, компания, уточнение имени клиента",
    "Потребности": "Открытые вопросы: площадь, кол-во устройств, цели использования",
    "Презентация": "Связывать тариф с потребностью: «Для 3 ТВ подойдёт HD 500»",
    "Возражения": "Банк аргументов по топ-5 возражениям + «согласие + аргумент»",
    "Завершение": "Инициировать: «Давайте оформим заявку, подключение бесплатное»",
    "Модули": "Завершать: «Остались вопросы?» + точная дата/время подключения",
    "Грамотность": "Убрать слова-паразиты, профессиональная лексика",
    "Тон": "Ровный доброжелательный тон, без разговорных оборотов",
    "Активность": "Вести диалог, структурировать этапы, объяснять логику",
    "Слушание": "Перефразировать: «Правильно ли я понимаю, что вам нужно...»",
}

# ── Translate LLM success labels ────────────────────────────────────────────
SUCCESS_LABELS_RU = {
    "excellent": "Отлично",
    "good": "Хорошо",
    "ok": "Частично",
    "fail": "Неуспешно",
    "---": "Не оценено",
}
SUCCESS_COLORS = {
    "Отлично": "#2ecc71", "Хорошо": "#27ae60",
    "Частично": "#f39c12", "Неуспешно": "#e74c3c", "Не оценено": "#bdc3c7",
}


def qa_color(score):
    for lo, hi, _, color in QA_SCALE:
        if lo <= score <= hi:
            return color
    return "#95a5a6"


def qa_label(score):
    for lo, hi, label, _ in QA_SCALE:
        if lo <= score <= hi:
            return label
    return "—"


# ── Load data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    wb = openpyxl.load_workbook(
        "output_test_20.xlsx", read_only=True, data_only=True
    )
    ws = wb["Детализация звонков"]
    rows = list(ws.iter_rows(values_only=True))
    headers = rows[0]
    data = rows[2:]
    wb.close()

    records = []
    for row in data:
        d = dict(zip(headers, row))
        rec = {
            "date": str(d["Дата"]),
            "agent": anonymize(d["Оператор"]),
            "agent_short": anonymize_short(
                " ".join(str(d["Оператор"]).split()[:2]) if d["Оператор"] else ""
            ),
            "session": d["Сессия"],
            "text": d["Текст обращения"],
            "duration": float(d["Длительность, мин"] or 0),
            "context": d["Контекст клиента"],
            "sentiment": d["Тональность клиента"],
            "products": d["Обсуждаемые продукты"],
            "request": d["Исходный запрос"],
            "needs": d["Выявление потребности"],
            "presentation": d["Презентация"],
            "objections": d["Возражения"],
            "agent_arguments": d["Аргументы оператора"],
            "offers": d["Предложения"],
            "result": d["Результат"],
            "comment": d["Комментарий"],
            "llm_success": SUCCESS_LABELS_RU.get(d["Успешность (llm)"], d["Успешность (llm)"]),
            "qa_score": float(d["Качество работы оператора, ср. балл"] or 0),
            "avg_process": float(d["Ср. балл по процессу"] or 0),
            "avg_comm": float(d["Ср. балл по коммуникациям"] or 0),
            "justification": d["Обоснование оценок"],
            "recommendation": d["Рекомендации"],
            "validation_log": d["Ошибки валидации LLM"],
        }
        for c in ALL_CRITERIA:
            val = d.get(c)
            rec[c] = float(val) if val is not None else None
        records.append(rec)

    return pd.DataFrame(records)


@st.cache_data
def generate_kpi_data(df):
    """Generate simulated KPI / sales data linked to real QA scores."""
    np.random.seed(42)
    agents = df["agent_short"].unique()
    weeks = ["Нед 1\n(03-08 фев)", "Нед 2\n(10-15 фев)", "Нед 3\n(17-22 фев)", "Нед 4\n(24-28 фев)"]

    kpi_records = []
    for agent in agents:
        adf = df[(df["agent_short"] == agent) & (df["qa_score"] > 0)]
        base_qa = adf["qa_score"].mean() if len(adf) else 2.5
        base_conversion = 0.08 + (base_qa - 2.5) * 0.06  # QA drives conversion

        for wi, week in enumerate(weeks):
            # Simulate weekly improvement trend for higher QA agents
            trend = wi * 0.005 * (base_qa / 4)
            noise = np.random.normal(0, 0.015)
            conversion = max(0.03, min(0.45, base_conversion + trend + noise))

            calls_total = np.random.randint(35, 70)
            calls_targeted = int(calls_total * np.random.uniform(0.6, 0.85))
            calls_successful = int(calls_targeted * conversion)

            revenue_per_sale = np.random.uniform(800, 2500)
            revenue = calls_successful * revenue_per_sale

            rgu = calls_successful * np.random.uniform(1.2, 2.5)  # avg products per sale

            qa_weekly = base_qa + wi * 0.05 * np.random.uniform(0.5, 1.5) + np.random.normal(0, 0.1)
            qa_weekly = max(1, min(5, qa_weekly))

            kpi_records.append({
                "agent_short": agent,
                "week": week,
                "week_num": wi + 1,
                "calls_total": calls_total,
                "calls_targeted": calls_targeted,
                "calls_successful": calls_successful,
                "conversion": round(conversion, 4),
                "revenue": round(revenue),
                "rgu": round(rgu, 1),
                "qa_weekly": round(qa_weekly, 2),
            })

    return pd.DataFrame(kpi_records), weeks


df = load_data()
df_targeted = df[df["qa_score"] > 0].copy()
kpi_df, WEEKS = generate_kpi_data(df)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📞 Речевая аналитика B2C")
    st.markdown("### MVP Dashboard")
    st.markdown("---")

    page = st.radio(
        "Раздел",
        [
            "🏠 Обзор",
            "📋 Примеры анализа",
            "👤 Команда и развитие",
            "📊 Итоги для руководителя",
            "📈 Влияние на продажи",
            "🧑‍💼 Личный кабинет оператора",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    agents = ["Все"] + sorted(df["agent_short"].unique().tolist())
    selected_agent = st.selectbox("Оператор", agents)

    st.markdown("---")
    with st.expander("ℹ️ Как пользоваться дашбордом"):
        st.markdown("""
**Что это?**
MVP-дашборд речевой аналитики B2C. Система анализирует звонки операторов с помощью LLM и показывает оценки, рекомендации и влияние на продажи.

**Данные:** тестовая выборка — 20 звонков, 3 оператора, февраль 2026.

**Разделы:**
- **Обзор** — общая картина: распределение оценок, успешность звонков, средние баллы
- **Примеры анализа** — детальный разбор конкретных звонков: текст, оценки по критериям, рекомендации
- **Команда и развитие** — сравнение операторов, heatmap по критериям, планы развития
- **Итоги для руководителя** — ключевые выводы, топ-проблемы, рекомендации для бизнеса
- **Влияние на продажи** — связь качества звонков с конверсией, прогноз ROI
- **Личный кабинет оператора** — персональный профиль, рекомендации, история звонков

**Фильтр:** выберите конкретного оператора в выпадающем списке выше или оставьте «Все».

**Оценки (1-5):**
🟢 4.5+ отлично · 🔵 3.8-4.4 хорошо · 🟡 3.0-3.7 зона риска · 🔴 <3.0 критично
""")
    st.caption("Тестовая выборка: 20 звонков")
    st.caption("Февраль 2026")


def filter_df(data):
    if selected_agent != "Все":
        return data[data["agent_short"] == selected_agent]
    return data


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: ОБЗОР
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Обзор":
    st.title("Обзор речевой аналитики")

    fdf = filter_df(df)
    ftdf = filter_df(df_targeted)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Всего звонков", len(fdf))
    c2.metric("Целевых", len(ftdf))
    c3.metric("Ср. оценка оператора", f"{ftdf['qa_score'].mean():.2f}" if len(ftdf) else "—")
    c4.metric("Ср. длительность", f"{fdf['duration'].mean():.1f} мин")

    success_rate = (
        len(ftdf[ftdf["llm_success"].isin(["Хорошо", "Отлично"])]) / len(ftdf) * 100
        if len(ftdf) else 0
    )
    c5.metric("Успешные звонки", f"{success_rate:.0f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Успешность звонков")
        # Filter out "Не оценено" for cleaner chart
        success_counts = fdf[fdf["llm_success"] != "Не оценено"]["llm_success"].value_counts()
        color_map = SUCCESS_COLORS
        fig_pie = px.pie(
            names=success_counts.index, values=success_counts.values,
            color=success_counts.index, color_discrete_map=color_map, hole=0.4,
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Распределение оценок операторов")
        if len(ftdf):
            fig_hist = px.histogram(
                ftdf, x="qa_score", nbins=10,
                color_discrete_sequence=["#3498db"], labels={"qa_score": "Оценка оператора"},
            )
            fig_hist.add_vrect(x0=0, x1=3.0, fillcolor="#e74c3c", opacity=0.1, line_width=0)
            fig_hist.add_vrect(x0=3.0, x1=3.8, fillcolor="#f39c12", opacity=0.1, line_width=0)
            fig_hist.add_vrect(x0=3.8, x1=4.5, fillcolor="#27ae60", opacity=0.1, line_width=0)
            fig_hist.add_vrect(x0=4.5, x1=5.0, fillcolor="#2ecc71", opacity=0.1, line_width=0)
            fig_hist.update_layout(
                margin=dict(t=20, b=20, l=20, r=20), height=350,
                xaxis_range=[1, 5], showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Средние оценки по критериям")
    if len(ftdf):
        means = []
        for c in ALL_CRITERIA:
            vals = ftdf[c].dropna()
            means.append(vals.mean() if len(vals) else 0)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=means + [means[0]], theta=SHORT_NAMES + [SHORT_NAMES[0]],
            fill="toself", fillcolor="rgba(52, 152, 219, 0.2)",
            line=dict(color="#3498db", width=2), name="Среднее",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            margin=dict(t=40, b=40, l=80, r=80), height=450, showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Тональность клиентов")
        sent = fdf["sentiment"].value_counts()
        sent_colors = {"нейтральный": "#3498db", "негативный": "#e74c3c", "позитивный": "#2ecc71"}
        fig_sent = px.bar(
            x=sent.index, y=sent.values, color=sent.index,
            color_discrete_map=sent_colors, labels={"x": "", "y": "Кол-во звонков"},
        )
        fig_sent.update_layout(margin=dict(t=20, b=20), height=300, showlegend=False)
        st.plotly_chart(fig_sent, use_container_width=True)

    with col2:
        st.subheader("Обсуждаемые продукты")
        products_all = []
        for p in fdf["products"].dropna():
            if p and p != "---":
                for item in str(p).split(","):
                    clean = item.strip()
                    if clean and "Другое" not in clean:
                        products_all.append(clean)
        if products_all:
            prod_s = pd.Series(products_all).value_counts()
            fig_prod = px.bar(
                x=prod_s.values, y=prod_s.index, orientation="h",
                color_discrete_sequence=["#9b59b6"], labels={"x": "Кол-во упоминаний", "y": ""},
            )
            fig_prod.update_layout(margin=dict(t=20, b=20, l=20), height=300, showlegend=False)
            st.plotly_chart(fig_prod, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: ОПЕРАТОРЫ
# ═══════════════════════════════════════════════════════════════════════════
elif page == "👤 Команда и развитие":
    st.title("Команда: оценки и план развития")

    agent_stats = []
    for agent in df["agent_short"].unique():
        adf = df[df["agent_short"] == agent]
        atdf = adf[adf["qa_score"] > 0]
        agent_stats.append({
            "Оператор": agent,
            "Звонков": len(adf),
            "Целевых": len(atdf),
            "Ср. оценка": round(atdf["qa_score"].mean(), 2) if len(atdf) else 0,
            "Ср. процесс": round(atdf["avg_process"].mean(), 2) if len(atdf) else 0,
            "Ср. коммуникации": round(atdf["avg_comm"].mean(), 2) if len(atdf) else 0,
            "Ср. длительность": round(adf["duration"].mean(), 1),
            "Успешных звонков": len(atdf[atdf["llm_success"].isin(["Хорошо", "Отлично"])]),
            "Уровень": qa_label(atdf["qa_score"].mean()) if len(atdf) else "—",
        })
    stats_df = pd.DataFrame(agent_stats).sort_values("Ср. оценка", ascending=False)

    st.dataframe(
        stats_df, use_container_width=True, hide_index=True,
        column_config={
            "Ср. оценка": st.column_config.ProgressColumn(min_value=0, max_value=5, format="%.2f"),
            "Ср. процесс": st.column_config.ProgressColumn(min_value=0, max_value=5, format="%.2f"),
            "Ср. коммуникации": st.column_config.ProgressColumn(min_value=0, max_value=5, format="%.2f"),
        },
    )

    st.markdown("---")

    st.subheader("Профили компетенций")
    fig_radar = go.Figure()
    for idx, agent in enumerate(sorted(df_targeted["agent_short"].unique())):
        adf = df_targeted[df_targeted["agent_short"] == agent]
        means = [adf[c].dropna().mean() if len(adf[c].dropna()) else 0 for c in ALL_CRITERIA]
        fig_radar.add_trace(go.Scatterpolar(
            r=means + [means[0]], theta=SHORT_NAMES + [SHORT_NAMES[0]],
            fill="toself", name=agent,
            line=dict(color=COLORS[idx % len(COLORS)], width=2), opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        margin=dict(t=40, b=40, l=100, r=100), height=500,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("Детализация по критериям")
    bar_data = []
    for agent in sorted(df_targeted["agent_short"].unique()):
        adf = df_targeted[df_targeted["agent_short"] == agent]
        for i, c in enumerate(ALL_CRITERIA):
            vals = adf[c].dropna()
            bar_data.append({"Оператор": agent, "Критерий": SHORT_NAMES[i], "Балл": vals.mean() if len(vals) else 0})
    bar_df = pd.DataFrame(bar_data)
    fig_bar = px.bar(bar_df, x="Критерий", y="Балл", color="Оператор", barmode="group", color_discrete_sequence=COLORS)
    fig_bar.add_hline(y=4, line_dash="dash", line_color="green", annotation_text="Целевой уровень")
    fig_bar.add_hline(y=3, line_dash="dash", line_color="orange", annotation_text="Зона риска")
    fig_bar.update_layout(margin=dict(t=40, b=20), height=450, yaxis_range=[0, 5], xaxis_tickangle=-30)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Оценка оператора vs Длительность звонка")
    fig_scatter = px.scatter(
        df_targeted, x="duration", y="qa_score", color="agent_short", size="duration",
        color_discrete_sequence=COLORS,
        labels={"duration": "Длительность (мин)", "qa_score": "Оценка оператора", "agent_short": "Оператор"},
        hover_data=["request", "llm_success"],
    )
    fig_scatter.add_hline(y=3.8, line_dash="dash", line_color="green", opacity=0.5)
    fig_scatter.add_hline(y=3.0, line_dash="dash", line_color="orange", opacity=0.5)
    fig_scatter.update_layout(margin=dict(t=20, b=20), height=400, yaxis_range=[1, 5])
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # ── Priority table (merged from Рекомендации) ──────────────────────
    st.subheader("План развития")
    st.caption("Таблица приоритетов: что улучшать в первую очередь")

    for agent in sorted(df_targeted["agent_short"].unique()):
        adf = df_targeted[df_targeted["agent_short"] == agent]
        avg_qa = adf["qa_score"].mean()
        color = qa_color(avg_qa)
        label = qa_label(avg_qa)

        st.markdown(
            f'**{agent}** — <span style="color:{color}">{avg_qa:.2f} ({label})</span>',
            unsafe_allow_html=True,
        )

        criteria_data = []
        for i, c in enumerate(ALL_CRITERIA):
            vals = adf[c].dropna()
            if len(vals):
                score = vals.mean()
                name = SHORT_NAMES[i]
                if score >= 4:
                    priority, priority_num = "✅ Поддерживать", 3
                elif score >= 3:
                    priority, priority_num = "⚠️ Улучшить", 2
                else:
                    priority, priority_num = "🔴 Срочно", 1
                criteria_data.append({
                    "priority_num": priority_num,
                    "Приоритет": priority, "Критерий": name,
                    "Сейчас": round(score, 1), "Цель": 4.0 if score < 4 else "—",
                    "Что делать": REC_ACTIONS.get(name, "—") if score < 4 else "Поддерживать уровень",
                })
        if criteria_data:
            table = pd.DataFrame(sorted(criteria_data, key=lambda x: x["priority_num"]))
            st.dataframe(table.drop(columns=["priority_num"]), use_container_width=True, hide_index=True)
        st.markdown("")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: ПРИМЕРЫ АНАЛИЗА
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📋 Примеры анализа":
    st.title("Как работает анализ звонка")

    fdf = filter_df(df)

    table_df = fdf[["agent_short", "duration", "request", "sentiment",
                     "llm_success", "qa_score", "products"]].copy()
    table_df.columns = ["Оператор", "Длит. (мин)", "Запрос", "Тональность", "Результат", "Оценка", "Продукты"]

    st.dataframe(
        table_df, use_container_width=True, hide_index=True,
        column_config={"Оценка": st.column_config.ProgressColumn(min_value=0, max_value=5, format="%.2f")},
    )

    st.markdown("---")
    call_options = []
    for i, row in fdf.iterrows():
        label = f"#{row['session']} — {row['agent_short']} — {row['request']} (оценка: {row['qa_score']})"
        call_options.append((label, i))

    if call_options:
        labels = [opt[0] for opt in call_options]
        # Default to session 123448818 if available
        default_idx = 0
        for i, (lbl, _) in enumerate(call_options):
            if "123448818" in lbl:
                default_idx = i
                break
        selected_label = st.selectbox("Выберите звонок", labels, index=default_idx)
        selected_idx = [opt[1] for opt in call_options if opt[0] == selected_label][0]
        call = df.loc[selected_idx]

        is_targeted = call["qa_score"] > 0 and str(call["request"]) != "Нецелевое обращение"

        # ── Non-targeted call: compact view ─────────────────────────
        if not is_targeted:
            st.markdown(
                '<div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:12px;'
                'padding:24px;text-align:center;margin:20px 0;color:#333;">'
                '<h3 style="color:#6c757d;margin:0;">Нецелевое обращение</h3>'
                f'<p style="color:#6c757d;margin:8px 0 0 0;">{call["agent_short"]} · {call["duration"]:.1f} мин · {call["sentiment"]}</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            if call.get("comment") and str(call["comment"]) not in ("---", "None", ""):
                st.caption(call["comment"])

        # ── Targeted call: full card ────────────────────────────────
        else:
            # ── CARD: Resume ────────────────────────────────────────
            st.markdown(
                '<div style="background:linear-gradient(135deg,#f8f9fa,#e9ecef);border-radius:12px;'
                'padding:20px;margin:10px 0;border-left:4px solid #3498db;color:#333;">'
                f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px;">'
                f'<span style="background:#3498db;color:white;padding:4px 12px;border-radius:20px;font-size:13px;">{call["agent_short"]}</span>'
                f'<span style="background:#6c757d;color:white;padding:4px 12px;border-radius:20px;font-size:13px;">{call["duration"]:.1f} мин</span>'
                f'<span style="background:{"#2ecc71" if call["llm_success"] in ("Хорошо","Отлично") else "#f39c12" if call["llm_success"] == "Частично" else "#e74c3c"};'
                f'color:white;padding:4px 12px;border-radius:20px;font-size:13px;">{call["llm_success"]}</span>'
                f'</div>'
                f'<p style="margin:0 0 8px 0;color:#333;"><b>Клиент хотел:</b> {call["request"] or "—"}</p>'
                f'<p style="margin:0 0 8px 0;color:#333;"><b>Оператор предложил:</b> {call["products"] or "—"}</p>'
                f'<p style="margin:0 0 8px 0;color:#333;"><b>Результат:</b> {call["result"] or "—"}</p>'
                f'<p style="margin:0;color:#6c757d;font-style:italic;">{call["comment"] or ""}</p>'
                '</div>',
                unsafe_allow_html=True,
            )

            # ── TRAFFIC LIGHT: Criteria tiles ───────────────────────
            st.markdown("#### Оценка по критериям")
            st.caption("🟢 4-5 хорошо · 🟡 3 зона роста · 🔴 1-2 критично")
            scores = [call[c] for c in ALL_CRITERIA]
            valid = [(SHORT_NAMES[i], scores[i]) for i in range(len(ALL_CRITERIA))
                      if scores[i] is not None and not (isinstance(scores[i], float) and np.isnan(scores[i]))]

            if valid:
                def _render_tiles(items):
                    cols = st.columns(len(items))
                    for idx, (name, score) in enumerate(items):
                        if score >= 4:
                            bg, border, icon = "#d4edda", "#28a745", "✅"
                        elif score >= 3:
                            bg, border, icon = "#fff3cd", "#ffc107", "⚠️"
                        else:
                            bg, border, icon = "#f8d7da", "#dc3545", "🔴"
                        cols[idx].markdown(
                            f'<div style="background:{bg};border:2px solid {border};border-radius:10px;'
                            f'padding:12px;text-align:center;margin:4px 0;min-height:90px;color:#333;">'
                            f'<div style="font-size:24px;font-weight:bold;">{icon} {score:.0f}</div>'
                            f'<div style="font-size:12px;color:#495057;margin-top:4px;">{name}</div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                _render_tiles(valid[:5])
                if len(valid) > 5:
                    _render_tiles(valid[5:])

            # ── Expandable details ──────────────────────────────────
            with st.expander("Обоснование оценок и рекомендации"):
                if call["justification"] and call["justification"] != "---":
                    for line in str(call["justification"]).split("\n"):
                        line = line.strip()
                        if line:
                            if "– 4:" in line or "– 5:" in line:
                                st.success(line)
                            elif "– 3:" in line:
                                st.warning(line)
                            elif "– 1:" in line or "– 2:" in line:
                                st.error(line)
                            else:
                                st.write(line)
                if call["recommendation"] and call["recommendation"] != "---":
                    st.markdown("---")
                    st.markdown("**Рекомендации:**")
                    for line in str(call["recommendation"]).split("\n"):
                        line = line.strip()
                        if line:
                            st.markdown(f"• {line}")

            with st.expander("Транскрипт диалога"):
                if call["text"]:
                    for line in str(call["text"]).split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("agent:"):
                            st.markdown(
                                f'<div style="background:#e8f4fd;padding:8px 12px;border-radius:8px;'
                                f'margin:4px 0 4px 40px;border-left:3px solid #3498db;color:#1a3a5c;">'
                                f'<b>🎧 Оператор:</b> {line[6:].strip()}</div>',
                                unsafe_allow_html=True,
                            )
                        elif line.startswith("client:"):
                            st.markdown(
                                f'<div style="background:#f0f0f0;padding:8px 12px;border-radius:8px;'
                                f'margin:4px 40px 4px 0;border-left:3px solid #95a5a6;color:#333;">'
                                f'<b>👤 Клиент:</b> {line[7:].strip()}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.text(line)

            # Validation log hidden — internal debug info, not for presentation


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: ИТОГИ ДЛЯ РУКОВОДИТЕЛЯ
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Итоги для руководителя":
    st.title("Отчёт для руководителя")
    st.caption("Агрегированный взгляд на эффективность команды ОЦП B2C")

    # ── Top-level KPIs ──────────────────────────────────────────────────
    latest = kpi_df[kpi_df["week_num"] == 4]
    prev = kpi_df[kpi_df["week_num"] == 3]

    total_rev = latest["revenue"].sum()
    prev_rev = prev["revenue"].sum()
    avg_conv = latest["conversion"].mean()
    prev_conv = prev["conversion"].mean()
    avg_qa = latest["qa_weekly"].mean()
    prev_qa = prev["qa_weekly"].mean()
    total_rgu = latest["rgu"].sum()
    prev_rgu = prev["rgu"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Выручка (нед 4)", f"{total_rev:,.0f} ₽", delta=f"{total_rev - prev_rev:+,.0f} ₽")
    c2.metric("Ср. конверсия", f"{avg_conv:.1%}", delta=f"{(avg_conv - prev_conv):+.1%}")
    c3.metric("Ср. оценка оператора", f"{avg_qa:.2f}", delta=f"{avg_qa - prev_qa:+.2f}")
    c4.metric("RGU (нед 4)", f"{total_rgu:.0f}", delta=f"{total_rgu - prev_rgu:+.0f}")

    st.markdown("---")

    # ── Conversion funnel ───────────────────────────────────────────────
    st.subheader("Воронка продаж (последняя неделя)")
    total_calls = latest["calls_total"].sum()
    total_targeted = latest["calls_targeted"].sum()
    total_successful = latest["calls_successful"].sum()

    fig_funnel = go.Figure(go.Funnel(
        y=["Все звонки", "Целевые обращения", "Успешные продажи"],
        x=[total_calls, total_targeted, total_successful],
        textinfo="value+percent initial",
        marker=dict(color=["#3498db", "#f39c12", "#2ecc71"]),
    ))
    fig_funnel.update_layout(margin=dict(t=20, b=20), height=300)
    st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("---")

    # ── Team ranking ────────────────────────────────────────────────────
    st.subheader("Рейтинг операторов")
    ranking = latest.sort_values("conversion", ascending=False).copy()
    ranking["Конверсия"] = ranking["conversion"].apply(lambda x: f"{x:.1%}")
    ranking["Выручка"] = ranking["revenue"].apply(lambda x: f"{x:,.0f} ₽")
    ranking["Оценка"] = ranking["qa_weekly"]

    rank_display = ranking[["agent_short", "calls_total", "calls_successful", "Конверсия", "Выручка", "rgu", "Оценка"]].copy()
    rank_display.columns = ["Оператор", "Звонков", "Продаж", "Конверсия", "Выручка", "RGU", "Оценка"]

    st.dataframe(
        rank_display, use_container_width=True, hide_index=True,
        column_config={
            "Оценка": st.column_config.ProgressColumn(min_value=0, max_value=5, format="%.2f"),
        },
    )

    st.markdown("---")

    # ── Heatmap: agents x criteria ──────────────────────────────────────
    st.subheader("Heatmap: проблемные зоны команды")
    heat_data = []
    for agent in sorted(df_targeted["agent_short"].unique()):
        adf = df_targeted[df_targeted["agent_short"] == agent]
        row_data = {}
        for i, c in enumerate(ALL_CRITERIA):
            vals = adf[c].dropna()
            row_data[SHORT_NAMES[i]] = round(vals.mean(), 2) if len(vals) else None
        heat_data.append({"Оператор": agent, **row_data})

    heat_df = pd.DataFrame(heat_data).set_index("Оператор")
    fig_heat = px.imshow(
        heat_df.values, x=heat_df.columns.tolist(), y=heat_df.index.tolist(),
        color_continuous_scale=["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71"],
        zmin=1, zmax=5, text_auto=".1f", aspect="auto",
    )
    fig_heat.update_layout(margin=dict(t=20, b=20), height=250, coloraxis_colorbar_title="Балл")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Top reasons for lost sales ──────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Причины потерь")
        loss_reasons = []
        for _, row in df.iterrows():
            if row["llm_success"] in ["Неуспешно", "Частично", "Не оценено"]:
                result_str = str(row.get("result", ""))
                if result_str and result_str != "None":
                    for line in result_str.split("\n"):
                        match = re.search(r"Отказ:\s*(.+?)–\s*['\"]?(.+?)['\"]?\s*$", line.strip())
                        if match:
                            product = match.group(1).strip()
                            reason = match.group(2).strip().rstrip("'\"")
                            loss_reasons.append((product, reason.capitalize()))
        if loss_reasons:
            reasons_df = pd.DataFrame(loss_reasons, columns=["Продукт", "Причина"])
            top = reasons_df.groupby("Причина").agg(
                Кол_во=("Причина", "size"),
                Продукты=("Продукт", lambda x: ", ".join(sorted(set(x))))
            ).sort_values("Кол_во", ascending=False).head(5)
            for i, (reason, row_data) in enumerate(top.iterrows()):
                st.markdown(
                    f'<div style="background:#fdf0f0;border-left:3px solid #e74c3c;padding:8px 12px;'
                    f'border-radius:6px;margin:4px 0;color:#333;">'
                    f'<b>{reason}</b> <span style="color:#999;">({row_data["Продукты"]})</span>'
                    f'<span style="float:right;color:#e74c3c;font-weight:bold;">{row_data["Кол_во"]}</span>'
                    f'</div>', unsafe_allow_html=True,
                )
        else:
            st.info("Нет данных об отказах")

    with col2:
        st.subheader("Возражения клиентов")
        objection_list = []
        for _, row in df.iterrows():
            obj = row.get("objections")
            if obj and str(obj) not in ("---", "None", ""):
                for part in str(obj).split("\n"):
                    part = part.strip()
                    if not part:
                        continue
                    match = re.search(r"^\d+\.\s*(.+?)\s*–\s*(.+?):\s*['\"]", part)
                    if match:
                        product = match.group(1).strip()
                        reason = match.group(2).strip()
                        if len(reason) > 3:
                            objection_list.append((product, reason.capitalize()))
                    else:
                        match2 = re.search(r"^\d+\.\s*(.+?)\s*–\s*(.+?)(?:\(|$)", part)
                        if match2:
                            product = match2.group(1).strip()
                            reason = match2.group(2).strip().rstrip(":")
                            if len(reason) > 3:
                                objection_list.append((product, reason.capitalize()))
        if objection_list:
            obj_df = pd.DataFrame(objection_list, columns=["Продукт", "Возражение"])
            top_obj = obj_df.groupby("Возражение").agg(
                Кол_во=("Возражение", "size"),
                Продукты=("Продукт", lambda x: ", ".join(sorted(set(x))))
            ).sort_values("Кол_во", ascending=False).head(5)
            for i, (objection, row_data) in enumerate(top_obj.iterrows()):
                st.markdown(
                    f'<div style="background:#fef9e7;border-left:3px solid #f39c12;padding:8px 12px;'
                    f'border-radius:6px;margin:4px 0;color:#333;">'
                    f'<b>{objection}</b> <span style="color:#999;">({row_data["Продукты"]})</span>'
                    f'<span style="float:right;color:#f39c12;font-weight:bold;">{row_data["Кол_во"]}</span>'
                    f'</div>', unsafe_allow_html=True,
                )
        else:
            st.info("Нет данных о возражениях")

    st.markdown("---")

    # ── Systemic barriers ───────────────────────────────────────────────
    st.subheader("Системные барьеры и рекомендации")
    st.markdown("""
| Барьер | Влияние | Рекомендация |
|--------|---------|-------------|
| Слабая презентация на основе потребностей | Клиент не понимает ценность → отказ | Обновить скрипты: связывать характеристики тарифа с конкретными потребностями клиента |
| Низкая инициатива к завершению сделки | Клиент «думает» и не перезванивает | Тренировка техник мягкого закрытия: «Давайте оформим заявку сейчас, подключение бесплатное» |
| Отсутствие работы с возражениями | Потеря клиентов с сомнениями | Банк аргументов по топ-5 возражениям + ролевые игры |
| Низкое активное слушание | Непопадание в потребность → слабая презентация | Обучение перефразированию: «Правильно ли я понимаю, что вам нужно...» |
""")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: ВЛИЯНИЕ НА ПРОДАЖИ (story: 3 blocks)
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Влияние на продажи":
    st.title("Влияние речевой аналитики на продажи")

    # ── BLOCK 1: Потенциал выручки (главный аргумент для ЛПР) ─────────
    st.markdown(
        '<div style="background:#fef5e7;border-radius:12px;padding:16px 20px;margin:10px 0;color:#333;">'
        '<h3 style="margin:0 0 4px 0;color:#5a3e0a;">1. Потенциал роста выручки</h3>'
        '<p style="margin:0;color:#495057;">Если подтянуть операторов до целевой оценки 3.8 — вот сколько компания получит дополнительно</p>'
        '</div>', unsafe_allow_html=True,
    )

    impact_data = []
    total_delta = 0
    for agent in sorted(kpi_df["agent_short"].unique()):
        agent_kpi = kpi_df[kpi_df["agent_short"] == agent]
        current_qa = agent_kpi["qa_weekly"].mean()
        current_conv = agent_kpi["conversion"].mean()
        current_rev = agent_kpi["revenue"].sum()

        if current_qa < 3.8:
            qa_gap = 3.8 - current_qa
            potential_lift = qa_gap * 0.015 * current_conv * 10
            projected_conv = current_conv + potential_lift
            projected_rev = current_rev * (projected_conv / current_conv) if current_conv > 0 else current_rev
            delta_rev = projected_rev - current_rev
            total_delta += delta_rev
        else:
            delta_rev = 0

        impact_data.append({
            "Оператор": agent,
            "Оценка": round(current_qa, 2),
            "Конверсия": f"{current_conv:.1%}",
            "Выручка (4 нед)": f"{current_rev:,.0f} ₽",
            "Доп. выручка": f"+{delta_rev:,.0f} ₽" if delta_rev > 0 else "Уже выше цели",
        })

    st.dataframe(pd.DataFrame(impact_data), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Потенциал за месяц", f"+{total_delta:,.0f} ₽")
    c2.metric("За год (3 оператора)", f"+{total_delta * 12:,.0f} ₽")
    c3.metric("При масштабе 50+ чел.", f"+{total_delta * 12 * 15:,.0f} ₽")

    st.caption("Расчёт: разница конверсии при текущей и целевой оценке, экстраполяция линейная")

    st.markdown("---")

    # ── BLOCK 2: Почему это работает ──────────────────────────────────
    st.markdown(
        '<div style="background:#eaf4fe;border-radius:12px;padding:16px 20px;margin:10px 0;color:#333;">'
        '<h3 style="margin:0 0 4px 0;color:#1a3a5c;">2. Почему это работает: связь качества и продаж</h3>'
        '<p style="margin:0;color:#495057;">Чем выше оценка оператора — тем выше конверсия. Каждая точка = оператор за неделю</p>'
        '</div>', unsafe_allow_html=True,
    )

    corr_val = kpi_df["qa_weekly"].corr(kpi_df["conversion"])
    fig_corr = px.scatter(
        kpi_df, x="qa_weekly", y="conversion",
        color="agent_short", size="revenue",
        color_discrete_sequence=COLORS,
        labels={"qa_weekly": "Оценка оператора", "conversion": "Конверсия в продажу", "agent_short": "Оператор"},
    )
    z = np.polyfit(kpi_df["qa_weekly"], kpi_df["conversion"], 1)
    x_line = np.linspace(kpi_df["qa_weekly"].min(), kpi_df["qa_weekly"].max(), 50)
    fig_corr.add_trace(go.Scatter(
        x=x_line, y=np.polyval(z, x_line), mode="lines",
        line=dict(color="rgba(0,0,0,0.3)", width=2, dash="dash"),
        name=f"Тренд (r={corr_val:.2f})", showlegend=True,
    ))
    fig_corr.update_layout(
        margin=dict(t=20, b=20), height=350,
        yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    if corr_val > 0.5:
        st.success(f"Корреляция r={corr_val:.2f} — сильная связь. Инвестиции в качество прямо увеличивают продажи.")
    elif corr_val > 0.3:
        st.info(f"Корреляция r={corr_val:.2f} — умеренная связь. Качество — значимый фактор конверсии.")
    else:
        st.info(f"Корреляция r={corr_val:.2f} — связь слабая. Нужна калибровка критериев.")

    st.markdown("---")

    # ── BLOCK 3: Прогноз пилота ───────────────────────────────────────
    st.markdown(
        '<div style="background:#e8f8f0;border-radius:12px;padding:16px 20px;margin:10px 0;color:#333;">'
        '<h3 style="margin:0 0 4px 0;color:#145a32;">3. Прогноз: A/B-тест за 12 недель пилота</h3>'
        '<p style="margin:0;color:#495057;">Группа с речевой аналитикой vs контрольная группа без неё</p>'
        '</div>', unsafe_allow_html=True,
    )

    np.random.seed(123)
    ab_weeks = list(range(1, 13))
    pilot_conv = [0.14 + w * 0.004 + np.random.normal(0, 0.008) for w in ab_weeks]
    control_conv = [0.14 + w * 0.001 + np.random.normal(0, 0.008) for w in ab_weeks]

    fig_ab = px.line(
        pd.DataFrame({
            "Неделя": ab_weeks * 2,
            "Конверсия": pilot_conv + control_conv,
            "Группа": ["С речевой аналитикой"] * 12 + ["Без речевой аналитики"] * 12,
        }),
        x="Неделя", y="Конверсия", color="Группа",
        markers=True,
        color_discrete_map={"С речевой аналитикой": "#2ecc71", "Без речевой аналитики": "#e74c3c"},
    )
    fig_ab.update_layout(margin=dict(t=20, b=20), height=350, yaxis_tickformat=".1%")
    st.plotly_chart(fig_ab, use_container_width=True)

    final_pilot = np.mean(pilot_conv[-3:])
    final_control = np.mean(control_conv[-3:])
    lift = (final_pilot - final_control) / final_control * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("С аналитикой", f"{final_pilot:.1%}")
    col2.metric("Без аналитики", f"{final_control:.1%}")
    col3.metric("Прирост конверсии", f"+{lift:.1f}%")

    if lift > 3:
        st.success(f"Прогнозный прирост +{lift:.1f}% превышает порог приёмки (3%). Рекомендация: масштабирование на весь ОЦП.")
    else:
        st.warning(f"Прогнозный прирост +{lift:.1f}% — пока ниже порога. Рекомендуется продлить пилот.")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: ЛИЧНЫЙ КАБИНЕТ ОПЕРАТОРА
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧑‍💼 Личный кабинет оператора":
    st.title("Личный кабинет оператора")

    # ── Operator selection (simulates login) ─────────────────────────
    operator_list = sorted(df["agent_short"].unique().tolist())
    current_operator = st.selectbox(
        "Выберите оператора (имитация входа)", operator_list, key="lk_operator"
    )

    op_df = df[df["agent_short"] == current_operator]
    op_targeted = op_df[op_df["qa_score"] > 0]
    op_kpi = kpi_df[kpi_df["agent_short"] == current_operator]

    # ── Header card ──────────────────────────────────────────────────
    avg_qa = op_targeted["qa_score"].mean() if len(op_targeted) else 0
    color = qa_color(avg_qa)
    label = qa_label(avg_qa)
    success_count = len(op_targeted[op_targeted["llm_success"].isin(["Хорошо", "Отлично"])])
    success_pct = success_count / len(op_targeted) * 100 if len(op_targeted) else 0

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#f8f9fa,#e9ecef);border-radius:12px;'
        f'padding:20px;margin:10px 0;border-left:5px solid {color};color:#333;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">'
        f'<div>'
        f'<h2 style="margin:0;color:#333;">{current_operator}</h2>'
        f'<p style="margin:4px 0 0 0;color:#6c757d;">Звонков: {len(op_df)} · Целевых: {len(op_targeted)} · Успешных: {success_pct:.0f}%</p>'
        f'</div>'
        f'<div style="text-align:center;">'
        f'<div style="font-size:48px;font-weight:bold;color:{color};">{avg_qa:.2f}</div>'
        f'<div style="font-size:14px;color:{color};">{label}</div>'
        f'</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── Tabs ─────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Мой профиль", "💡 Мои рекомендации", "📞 Мои звонки"])

    # ── TAB 1: Мой профиль ───────────────────────────────────────────
    with tab1:
        st.subheader("Оценки по критериям")

        if len(op_targeted):
            # Radar chart
            means = [op_targeted[c].dropna().mean() if len(op_targeted[c].dropna()) else 0 for c in ALL_CRITERIA]
            # Team average for comparison
            team_means = [df_targeted[c].dropna().mean() if len(df_targeted[c].dropna()) else 0 for c in ALL_CRITERIA]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=team_means + [team_means[0]], theta=SHORT_NAMES + [SHORT_NAMES[0]],
                fill="toself", fillcolor="rgba(189, 195, 199, 0.15)",
                line=dict(color="#bdc3c7", width=1, dash="dash"), name="Среднее по команде",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=means + [means[0]], theta=SHORT_NAMES + [SHORT_NAMES[0]],
                fill="toself", fillcolor="rgba(52, 152, 219, 0.25)",
                line=dict(color="#3498db", width=2), name=current_operator,
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                margin=dict(t=40, b=40, l=100, r=100), height=420,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Strengths and weaknesses
            col_s, col_w = st.columns(2)
            criteria_scores = [(SHORT_NAMES[i], means[i]) for i in range(len(ALL_CRITERIA)) if means[i] > 0]
            criteria_scores_sorted = sorted(criteria_scores, key=lambda x: x[1], reverse=True)

            with col_s:
                st.markdown("#### Сильные стороны")
                for name, score in criteria_scores_sorted[:3]:
                    if score >= 3.5:
                        st.markdown(
                            f'<div style="background:#d4edda;border-left:3px solid #28a745;padding:8px 12px;'
                            f'border-radius:6px;margin:4px 0;color:#333;">'
                            f'<b>{name}</b>'
                            f'<span style="float:right;color:#28a745;font-weight:bold;">{score:.1f}</span>'
                            f'</div>', unsafe_allow_html=True,
                        )

            with col_w:
                st.markdown("#### Зоны роста")
                for name, score in criteria_scores_sorted[-3:]:
                    if score < 4.0:
                        if score < 3:
                            bg, border, text_color = "#f8d7da", "#dc3545", "#dc3545"
                        else:
                            bg, border, text_color = "#fff3cd", "#ffc107", "#856404"
                        st.markdown(
                            f'<div style="background:{bg};border-left:3px solid {border};padding:8px 12px;'
                            f'border-radius:6px;margin:4px 0;color:#333;">'
                            f'<b>{name}</b>'
                            f'<span style="float:right;color:{text_color};font-weight:bold;">{score:.1f}</span>'
                            f'</div>', unsafe_allow_html=True,
                        )

            # Weekly dynamics
            st.markdown("---")
            st.subheader("Динамика по неделям")
            if len(op_kpi):
                fig_dyn = make_subplots(specs=[[{"secondary_y": True}]])
                fig_dyn.add_trace(
                    go.Scatter(
                        x=op_kpi["week"], y=op_kpi["qa_weekly"], mode="lines+markers",
                        name="Оценка", line=dict(color="#3498db", width=3),
                        marker=dict(size=10),
                    ), secondary_y=False,
                )
                fig_dyn.add_trace(
                    go.Bar(
                        x=op_kpi["week"], y=op_kpi["conversion"], name="Конверсия",
                        marker_color="rgba(46, 204, 113, 0.5)",
                    ), secondary_y=True,
                )
                fig_dyn.update_yaxes(title_text="Оценка", range=[1, 5], secondary_y=False)
                fig_dyn.update_yaxes(title_text="Конверсия", tickformat=".0%", secondary_y=True)
                fig_dyn.update_layout(margin=dict(t=20, b=20), height=350)
                st.plotly_chart(fig_dyn, use_container_width=True)

                # Progress metrics
                if len(op_kpi) >= 2:
                    first_week = op_kpi.iloc[0]
                    last_week = op_kpi.iloc[-1]
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "Оценка",
                        f"{last_week['qa_weekly']:.2f}",
                        delta=f"{last_week['qa_weekly'] - first_week['qa_weekly']:+.2f} за месяц",
                    )
                    c2.metric(
                        "Конверсия",
                        f"{last_week['conversion']:.1%}",
                        delta=f"{(last_week['conversion'] - first_week['conversion']):+.1%}",
                    )
                    c3.metric(
                        "Продаж за посл. неделю",
                        f"{last_week['calls_successful']}",
                        delta=f"{last_week['calls_successful'] - first_week['calls_successful']:+d}",
                    )

    # ── TAB 2: Мои рекомендации ──────────────────────────────────────
    with tab2:
        st.subheader("Персональные рекомендации")
        st.caption("На основе анализа ваших звонков системой ИИ")

        if len(op_targeted):
            means = [op_targeted[c].dropna().mean() if len(op_targeted[c].dropna()) else 0 for c in ALL_CRITERIA]
            criteria_scores = [(SHORT_NAMES[i], ALL_CRITERIA[i], means[i]) for i in range(len(ALL_CRITERIA)) if means[i] > 0]
            weak = sorted(criteria_scores, key=lambda x: x[2])

            # Priority recommendations based on weakest areas
            for short, full, score in weak:
                if score >= 4.5:
                    continue  # skip excellent areas
                if score >= 4.0:
                    priority_tag = '<span style="background:#d4edda;color:#155724;padding:2px 8px;border-radius:10px;font-size:12px;">Поддерживать</span>'
                    border_color = "#28a745"
                    bg = "#f8fff8"
                elif score >= 3.0:
                    priority_tag = '<span style="background:#fff3cd;color:#856404;padding:2px 8px;border-radius:10px;font-size:12px;">Улучшить</span>'
                    border_color = "#ffc107"
                    bg = "#fffef5"
                else:
                    priority_tag = '<span style="background:#f8d7da;color:#721c24;padding:2px 8px;border-radius:10px;font-size:12px;">Приоритет</span>'
                    border_color = "#dc3545"
                    bg = "#fff5f5"

                action = REC_ACTIONS.get(short, "Работайте над улучшением этого навыка")

                st.markdown(
                    f'<div style="background:{bg};border-left:4px solid {border_color};padding:14px 16px;'
                    f'border-radius:8px;margin:8px 0;color:#333;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                    f'<b style="font-size:15px;">{short}</b>'
                    f'<div>{priority_tag} <span style="font-weight:bold;color:{border_color};margin-left:8px;">{score:.1f}/5</span></div>'
                    f'</div>'
                    f'<p style="margin:0;color:#495057;">{action}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Recommendations from individual calls
            st.markdown("---")
            st.subheader("Рекомендации по звонкам")
            for _, call in op_targeted.iterrows():
                rec = call.get("recommendation")
                if rec and str(rec) not in ("---", "None", ""):
                    result_label = call["llm_success"]
                    result_color = SUCCESS_COLORS.get(result_label, "#bdc3c7")
                    st.markdown(
                        f'<div style="background:#f8f9fa;border-left:3px solid {result_color};padding:10px 14px;'
                        f'border-radius:6px;margin:6px 0;color:#333;">'
                        f'<div style="margin-bottom:4px;">'
                        f'<span style="background:{result_color};color:white;padding:2px 8px;border-radius:10px;font-size:11px;">{result_label}</span>'
                        f' <span style="color:#6c757d;font-size:12px;">Сессия #{call["session"]} · {call["duration"]:.1f} мин · {call["request"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    for line in str(rec).split("\n"):
                        line = line.strip()
                        if line:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{line}")
                    st.markdown("</div>", unsafe_allow_html=True)

            # Expert suggestions for high performers
            if avg_qa >= 4.5:
                st.markdown("---")
                st.markdown(
                    '<div style="background:#e8f8f0;border-left:4px solid #2ecc71;padding:14px 16px;'
                    'border-radius:8px;margin:8px 0;color:#333;">'
                    '<h4 style="margin:0 0 8px 0;color:#145a32;">Вы — эксперт!</h4>'
                    '<p style="margin:0;color:#495057;">Ваши оценки стабильно высокие. Рекомендации:</p>'
                    '<ul style="color:#495057;margin:8px 0 0 0;">'
                    '<li>Развивайте навыки upsell: предлагайте дополнительные услуги (домофония, роутер) к основному продукту</li>'
                    '<li>Отрабатывайте сложные возражения: «дорого», «подумаю», «у конкурентов дешевле»</li>'
                    '<li>Станьте наставником: помогайте коллегам с низкими оценками улучшить навыки</li>'
                    '</ul></div>',
                    unsafe_allow_html=True,
                )

    # ── TAB 3: Мои звонки ────────────────────────────────────────────
    with tab3:
        st.subheader("История звонков")

        if len(op_df):
            # Summary table
            calls_table = op_df[["session", "date", "duration", "request", "llm_success", "qa_score", "products"]].copy()
            calls_table.columns = ["Сессия", "Дата", "Длит. (мин)", "Запрос", "Результат", "Оценка", "Продукты"]

            st.dataframe(
                calls_table, use_container_width=True, hide_index=True,
                column_config={
                    "Оценка": st.column_config.ProgressColumn(min_value=0, max_value=5, format="%.2f"),
                },
            )

            # Detailed view per call
            st.markdown("---")
            for _, call in op_df.iterrows():
                is_targeted = call["qa_score"] > 0 and str(call["request"]) != "Нецелевое обращение"

                if not is_targeted:
                    with st.expander(f"#{call['session']} — Нецелевое · {call['duration']:.1f} мин"):
                        st.caption(call.get("comment", ""))
                    continue

                result_color = SUCCESS_COLORS.get(call["llm_success"], "#bdc3c7")

                with st.expander(
                    f"#{call['session']} — {call['request']} · {call['llm_success']} · Оценка: {call['qa_score']:.1f}"
                ):
                    # Criteria tiles
                    scores = [call[c] for c in ALL_CRITERIA]
                    valid = [(SHORT_NAMES[i], scores[i]) for i in range(len(ALL_CRITERIA))
                              if scores[i] is not None and not (isinstance(scores[i], float) and np.isnan(scores[i]))]

                    if valid:
                        def _render_call_tiles(items):
                            cols = st.columns(len(items))
                            for idx, (name, score) in enumerate(items):
                                if score >= 4:
                                    bg, border, icon = "#d4edda", "#28a745", "✅"
                                elif score >= 3:
                                    bg, border, icon = "#fff3cd", "#ffc107", "⚠️"
                                else:
                                    bg, border, icon = "#f8d7da", "#dc3545", "🔴"
                                cols[idx].markdown(
                                    f'<div style="background:{bg};border:1px solid {border};border-radius:8px;'
                                    f'padding:8px;text-align:center;margin:2px 0;color:#333;">'
                                    f'<div style="font-size:18px;font-weight:bold;">{icon} {score:.0f}</div>'
                                    f'<div style="font-size:10px;color:#495057;">{name}</div>'
                                    '</div>',
                                    unsafe_allow_html=True,
                                )

                        _render_call_tiles(valid[:5])
                        if len(valid) > 5:
                            _render_call_tiles(valid[5:])

                    # Recommendation
                    rec = call.get("recommendation")
                    if rec and str(rec) not in ("---", "None", ""):
                        st.markdown("**Рекомендации:**")
                        for line in str(rec).split("\n"):
                            line = line.strip()
                            if line:
                                st.markdown(f"&nbsp;&nbsp;{line}")

                    # Result
                    if call.get("result") and str(call["result"]) not in ("---", "None", ""):
                        st.markdown(f"**Результат:** {call['result']}")

        else:
            st.info("Нет данных о звонках")


# ── Footer ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="text-align:center;color:#95a5a6;font-size:12px;">'
    "Речевая аналитика B2C — MVP<br>Powered by LLM</div>",
    unsafe_allow_html=True,
)
