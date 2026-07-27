import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

sns.set_style("whitegrid")
ACCENT = "#B22222"   # Eicher / Royal Enfield red

st.set_page_config(page_title="Eicher Motors Dashboard", layout="wide", page_icon="🏍️")

# ---------- Load data ----------
@st.cache_data
def load_data():
    return pd.read_csv("eicher_financials_clean.csv")

data = load_data()

# ---------- Sidebar navigation ----------
st.sidebar.title("🏍️ Eicher Motors")
st.sidebar.caption("Company Financial Performance Dashboard")
section = st.sidebar.radio(
    "Go to section",
    ["Overview", "Quarter Comparison", "Charts", "Prediction Models", "Data Table"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Prepared by Debjyoti Mukherjee | CA2 Project")

# ---------- Header ----------
st.markdown(f"<h1 style='color:{ACCENT};'>Financial Dashboard: Eicher Motors</h1>", unsafe_allow_html=True)
st.caption("Data source: Screener.in (public quarterly financial statements) | NSE: EICHERMOT")

latest = data.iloc[-1]
prev = data.iloc[-2]

# ===================== OVERVIEW =====================
if section == "Overview":
    k1, k2, k3 = st.columns(3)
    k1.metric("Latest Sales (₹ Cr)", f"{latest['Sales']:,.0f}",
              f"{latest['Sales_Growth']:.1f}% vs last qtr")
    k2.metric("Latest Net Profit (₹ Cr)", f"{latest['Net_Profit']:,.0f}",
              f"{latest['Net_Profit_Growth']:.1f}% vs last qtr")
    k3.metric("Latest OPM %", f"{latest['OPM_Percent']:.2f}%")

    st.markdown("### What this quarter looked like")
    direction = "up" if latest['Net_Profit'] > prev['Net_Profit'] else "down"
    st.write(
        f"In **{latest['Period']}**, Eicher Motors reported Sales of ₹{latest['Sales']:,.0f} Cr "
        f"and Net Profit of ₹{latest['Net_Profit']:,.0f} Cr, moving **{direction}** from the "
        f"previous quarter. Operating margin stood at **{latest['OPM_Percent']:.1f}%**."
    )

    grew_share = (data['Profit_Trend'] == 'Profit Grew').mean() * 100
    st.info(f"Across the {len(data)} quarters in this dataset, profit grew quarter-on-quarter "
            f"in **{grew_share:.0f}%** of them.")

    st.markdown("### Sales & Net Profit — full history")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(data['Period'], data['Sales'], marker='o', color=ACCENT, label='Sales')
    ax.plot(data['Period'], data['Net_Profit'], marker='s', color='black', label='Net Profit')
    ax.set_ylabel("₹ Crore")
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

# ===================== QUARTER COMPARISON =====================
elif section == "Quarter Comparison":
    st.markdown("### Compare any two quarters side by side")
    col_a, col_b = st.columns(2)
    q1 = col_a.selectbox("Quarter A", data['Period'], index=len(data) - 2)
    q2 = col_b.selectbox("Quarter B", data['Period'], index=len(data) - 1)

    row1 = data[data['Period'] == q1].iloc[0]
    row2 = data[data['Period'] == q2].iloc[0]

    compare_cols = ['Sales', 'Operating_Profit', 'OPM_Percent', 'Other_Income',
                     'Interest', 'Depreciation', 'Profit_Before_Tax', 'Net_Profit', 'EPS']

    comp_df = pd.DataFrame({
        "Metric": compare_cols,
        q1: [row1[c] for c in compare_cols],
        q2: [row2[c] for c in compare_cols],
    })
    comp_df["Change"] = comp_df[q2] - comp_df[q1]
    comp_df["Change %"] = (comp_df["Change"] / comp_df[q1].replace(0, pd.NA)) * 100

    st.dataframe(comp_df.style.format({q1: "{:.2f}", q2: "{:.2f}", "Change": "{:.2f}", "Change %": "{:.1f}%"}),
                 use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = ["Sales", "Operating_Profit", "Net_Profit"]
    width = 0.35
    positions = range(len(x))
    ax.bar([p - width/2 for p in positions], [row1[c] for c in x], width, label=q1, color="gray")
    ax.bar([p + width/2 for p in positions], [row2[c] for c in x], width, label=q2, color=ACCENT)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(x)
    ax.set_ylabel("₹ Crore")
    ax.legend()
    st.pyplot(fig)

# ===================== CHARTS =====================
elif section == "Charts":
    st.markdown("### Operating Profit Margin (%) by quarter")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Period', y='OPM_Percent', data=data, color=ACCENT, ax=ax1)
    plt.xticks(rotation=45)
    ax1.set_ylabel("OPM %")
    st.pyplot(fig1)

    st.markdown("### Net Profit spread: growth quarters vs decline quarters")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.boxplot(x='Profit_Trend', y='Net_Profit', data=data,
                palette={'Profit Grew': ACCENT, 'Profit Declined': 'gray'}, ax=ax2)
    st.pyplot(fig2)

    st.markdown("### Quarter-on-quarter growth rates")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    x = range(len(data))
    ax3.bar([i - 0.2 for i in x], data['Sales_Growth'], width=0.4, label='Sales Growth %', color='black')
    ax3.bar([i + 0.2 for i in x], data['Net_Profit_Growth'], width=0.4, label='Net Profit Growth %', color=ACCENT)
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(data['Period'], rotation=45)
    ax3.legend()
    st.pyplot(fig3)

# ===================== PREDICTION MODELS =====================
elif section == "Prediction Models":
    st.markdown("### Predicting Profit_Trend: Logistic Regression vs Decision Tree")
    st.caption("Drivers used: Sales_Growth, OPM_Percent, Interest, Other_Income")

    drivers = ['Sales_Growth', 'OPM_Percent', 'Interest', 'Other_Income']
    model_data = data.dropna(subset=drivers + ['Profit_Trend'])
    X = model_data[drivers]
    y = model_data['Profit_Trend']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    logit = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)

    logit_acc = accuracy_score(y_test, logit.predict(X_test))
    tree_acc = accuracy_score(y_test, tree.predict(X_test))

    comp = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree"],
        "Accuracy": [f"{logit_acc:.0%}", f"{tree_acc:.0%}"]
    })
    st.table(comp)

    winner = "Decision Tree" if tree_acc > logit_acc else "Logistic Regression"
    st.success(f"On this test split, **{winner}** performs better.")

    with st.expander("See feature importance (Decision Tree)"):
        importance = pd.DataFrame({
            "Feature": drivers,
            "Importance": tree.feature_importances_
        }).sort_values("Importance", ascending=False)
        st.bar_chart(importance.set_index("Feature"))

    st.warning("With only 12 quarterly rows, these accuracy numbers are a small-sample illustration, "
               "not a dependable forecast. A multi-year dataset would be needed for real prediction use.")

# ===================== DATA TABLE =====================
elif section == "Data Table":
    st.markdown("### Full cleaned dataset")
    st.dataframe(data, use_container_width=True)
    st.download_button("⬇️ Download this data as CSV", data.to_csv(index=False),
                        file_name="eicher_financials_clean.csv", mime="text/csv")

st.markdown("---")
st.caption("Prepared by Debjyoti Mukherjee | CA2 Project | Company Financial Performance Dashboard")
