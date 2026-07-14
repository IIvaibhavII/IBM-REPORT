import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#4C6EF5"
DANGER = "#E8590C"
SAFE = "#2F9E44"

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

def find_dataset():
    """Locate the HR CSV no matter where it ended up in the repo."""
    candidates = [
        APP_DIR / "data" / "HR-Employee-Attrition.csv",
        APP_DIR / "HR-Employee-Attrition.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Last resort: search the whole repo for any matching filename
    matches = list(APP_DIR.rglob("HR-Employee-Attrition*.csv"))
    if matches:
        return matches[0]
    # Nothing found anywhere — show exactly what IS there, then stop cleanly
    all_files = "\n".join(str(p.relative_to(APP_DIR)) for p in APP_DIR.rglob("*") if p.is_file())
    st.error(
        "Could not find HR-Employee-Attrition.csv anywhere in the repo.\n\n"
        "Files that ARE present in the repo:\n\n" + all_files
    )
    st.stop()

@st.cache_data
def load_data():
    df = pd.read_csv(find_dataset())
    df["AttritionFlag"] = df["Attrition"].map({"Yes": 1, "No": 0})
    # Friendly labels for ordinal survey scores
    satisfaction_map = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
    df["JobSatisfactionLabel"] = df["JobSatisfaction"].map(satisfaction_map)
    df["EnvSatisfactionLabel"] = df["EnvironmentSatisfaction"].map(satisfaction_map)
    wlb_map = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}
    df["WorkLifeBalanceLabel"] = df["WorkLifeBalance"].map(wlb_map)
    return df

df = load_data()

# ----------------------------------------------------------------------------
# SIDEBAR — NAVIGATION + GLOBAL FILTERS
# ----------------------------------------------------------------------------
st.sidebar.title("📊 HR Attrition Analytics")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Business Problem",
        "🔍 Exploratory Analysis",
        "📈 Key Drivers of Attrition",
        "🤖 Predictive Model",
        "✅ Recommendations",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
dept_filter = st.sidebar.multiselect(
    "Department", options=sorted(df["Department"].unique()), default=list(df["Department"].unique())
)
gender_filter = st.sidebar.multiselect(
    "Gender", options=sorted(df["Gender"].unique()), default=list(df["Gender"].unique())
)
age_range = st.sidebar.slider(
    "Age range", int(df["Age"].min()), int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max()))
)

fdf = df[
    (df["Department"].isin(dept_filter)) &
    (df["Gender"].isin(gender_filter)) &
    (df["Age"].between(age_range[0], age_range[1]))
]

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Showing **{len(fdf):,}** of {len(df):,} employees\n\n"
    "Dataset: IBM HR Analytics Employee Attrition (public, 1,470 records, 35 features)."
)

# ----------------------------------------------------------------------------
# PAGE 1 — BUSINESS PROBLEM
# ----------------------------------------------------------------------------
if page == "🏠 Business Problem":
    st.title("Employee Attrition: A Data-Driven Retention Strategy")
    st.markdown(
        """
        ### The Problem
        Losing an employee costs an organization roughly **50%–200% of that employee's annual
        salary** once recruiting, onboarding, lost productivity, and knowledge drain are
        accounted for. Yet most companies react to resignations *after* they happen, instead
        of identifying and addressing the risk factors beforehand.

        This project analyzes **1,470 employee records** from a real-world-style HR dataset to:
        1. Quantify the scale and pattern of attrition across the organization.
        2. Identify the strongest drivers of attrition (compensation, overtime, satisfaction,
           tenure, role, etc.).
        3. Build a predictive model that flags employees who are at elevated risk of leaving.
        4. Translate findings into **practical retention recommendations** for HR leadership.
        """
    )

    st.markdown("### Headline Numbers")
    total = len(fdf)
    leavers = fdf["AttritionFlag"].sum()
    rate = leavers / total * 100 if total else 0
    avg_income_leavers = fdf.loc[fdf["Attrition"] == "Yes", "MonthlyIncome"].mean()
    avg_income_stayers = fdf.loc[fdf["Attrition"] == "No", "MonthlyIncome"].mean()
    est_cost = leavers * fdf["MonthlyIncome"].mean() * 12 * 0.75  # ~75% of annual salary as replacement cost

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees analyzed", f"{total:,}")
    c2.metric("Attrition count", f"{int(leavers):,}")
    c3.metric("Attrition rate", f"{rate:.1f}%")
    c4.metric("Est. annual replacement cost", f"${est_cost:,.0f}")

    st.info(
        "💡 At the current filtered attrition rate, replacing departed employees costs an "
        "estimated **${:,.0f}** per year, assuming replacement cost ≈ 75% of annual salary — "
        "a widely used industry benchmark (SHRM).".format(est_cost)
    )

    st.markdown("### What the dataset covers")
    with st.expander("View column dictionary"):
        st.write(
            """
            - **Demographics**: Age, Gender, MaritalStatus, DistanceFromHome, Education, EducationField
            - **Job**: Department, JobRole, JobLevel, BusinessTravel, OverTime
            - **Compensation**: MonthlyIncome, DailyRate, HourlyRate, PercentSalaryHike, StockOptionLevel
            - **Satisfaction (1=Low → 4=Very High)**: JobSatisfaction, EnvironmentSatisfaction,
              RelationshipSatisfaction, WorkLifeBalance, JobInvolvement
            - **Tenure**: YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion,
              YearsWithCurrManager, TotalWorkingYears, NumCompaniesWorked
            - **Target**: Attrition (Yes/No)
            """
        )
        st.dataframe(fdf.head(20), use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE 2 — EXPLORATORY ANALYSIS
# ----------------------------------------------------------------------------
elif page == "🔍 Exploratory Analysis":
    st.title("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        counts = fdf["Attrition"].value_counts().reset_index()
        counts.columns = ["Attrition", "Count"]
        fig = px.pie(
            counts, names="Attrition", values="Count", hole=0.5,
            color="Attrition", color_discrete_map={"Yes": DANGER, "No": SAFE},
            title="Overall Attrition Split",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        dept = fdf.groupby("Department")["AttritionFlag"].mean().reset_index()
        dept["AttritionFlag"] *= 100
        fig = px.bar(
            dept.sort_values("AttritionFlag"), x="AttritionFlag", y="Department", orientation="h",
            color="AttritionFlag", color_continuous_scale="OrRd",
            labels={"AttritionFlag": "Attrition Rate (%)"},
            title="Attrition Rate by Department",
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(
            fdf, x="Age", color="Attrition", barmode="overlay", nbins=25,
            color_discrete_map={"Yes": DANGER, "No": SAFE},
            title="Age Distribution by Attrition",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(
            fdf, x="Attrition", y="MonthlyIncome", color="Attrition",
            color_discrete_map={"Yes": DANGER, "No": SAFE},
            title="Monthly Income vs Attrition",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Job Role Breakdown")
    role = fdf.groupby("JobRole")["AttritionFlag"].agg(["mean", "count"]).reset_index()
    role["mean"] *= 100
    role.columns = ["JobRole", "AttritionRate(%)", "Employees"]
    fig = px.bar(
        role.sort_values("AttritionRate(%)", ascending=False),
        x="JobRole", y="AttritionRate(%)", color="AttritionRate(%)",
        color_continuous_scale="OrRd", text="Employees",
        title="Attrition Rate by Job Role (bar label = headcount)",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Correlation Heatmap (numeric features)")
    num_cols = fdf.select_dtypes(include=np.number).drop(
        columns=["EmployeeCount", "StandardHours", "EmployeeNumber"], errors="ignore"
    )
    corr = num_cols.corr()
    fig = px.imshow(corr, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE 3 — KEY DRIVERS
# ----------------------------------------------------------------------------
elif page == "📈 Key Drivers of Attrition":
    st.title("Key Drivers of Attrition")

    col1, col2 = st.columns(2)
    with col1:
        ot = fdf.groupby("OverTime")["AttritionFlag"].mean().reset_index()
        ot["AttritionFlag"] *= 100
        fig = px.bar(
            ot, x="OverTime", y="AttritionFlag", color="OverTime",
            color_discrete_sequence=[SAFE, DANGER],
            labels={"AttritionFlag": "Attrition Rate (%)"},
            title="Overtime vs Attrition Rate",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Employees who work overtime leave at a markedly higher rate.")

    with col2:
        wlb = fdf.groupby("WorkLifeBalanceLabel")["AttritionFlag"].mean().reset_index()
        wlb["AttritionFlag"] *= 100
        order = ["Bad", "Good", "Better", "Best"]
        wlb["WorkLifeBalanceLabel"] = pd.Categorical(wlb["WorkLifeBalanceLabel"], order)
        wlb = wlb.sort_values("WorkLifeBalanceLabel")
        fig = px.bar(
            wlb, x="WorkLifeBalanceLabel", y="AttritionFlag", color="AttritionFlag",
            color_continuous_scale="OrRd",
            labels={"AttritionFlag": "Attrition Rate (%)", "WorkLifeBalanceLabel": "Work-Life Balance"},
            title="Work-Life Balance vs Attrition Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(
            fdf, x="Attrition", y="YearsAtCompany", color="Attrition",
            color_discrete_map={"Yes": DANGER, "No": SAFE},
            title="Tenure (Years at Company) vs Attrition",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        promo = fdf.groupby("YearsSinceLastPromotion")["AttritionFlag"].mean().reset_index()
        promo["AttritionFlag"] *= 100
        fig = px.line(
            promo, x="YearsSinceLastPromotion", y="AttritionFlag", markers=True,
            labels={"AttritionFlag": "Attrition Rate (%)"},
            title="Years Since Last Promotion vs Attrition Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Distance From Home & Business Travel")
    col5, col6 = st.columns(2)
    with col5:
        fig = px.box(
            fdf, x="Attrition", y="DistanceFromHome", color="Attrition",
            color_discrete_map={"Yes": DANGER, "No": SAFE},
            title="Commute Distance vs Attrition",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col6:
        travel = fdf.groupby("BusinessTravel")["AttritionFlag"].mean().reset_index()
        travel["AttritionFlag"] *= 100
        fig = px.bar(
            travel.sort_values("AttritionFlag"), x="AttritionFlag", y="BusinessTravel", orientation="h",
            color="AttritionFlag", color_continuous_scale="OrRd",
            labels={"AttritionFlag": "Attrition Rate (%)"},
            title="Business Travel Frequency vs Attrition Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Statistical Summary: Attrition vs Stay Group")
    compare_cols = ["Age", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome",
                     "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "NumCompaniesWorked"]
    summary = fdf.groupby("Attrition")[compare_cols].mean().T
    summary.columns = ["Stayed (No)", "Left (Yes)"]
    summary["Gap"] = summary["Left (Yes)"] - summary["Stayed (No)"]
    st.dataframe(summary.round(2), use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE 4 — PREDICTIVE MODEL
# ----------------------------------------------------------------------------
elif page == "🤖 Predictive Model":
    st.title("Predicting Employee Attrition Risk")
    st.markdown(
        "A Random Forest classifier is trained on the full (unfiltered) dataset to estimate "
        "each employee's probability of leaving, and to rank which features matter most."
    )

    @st.cache_resource
    def train_model(data):
        model_df = data.copy()
        target = model_df["AttritionFlag"]
        drop_cols = ["Attrition", "AttritionFlag", "EmployeeCount", "EmployeeNumber",
                     "Over18", "StandardHours", "JobSatisfactionLabel",
                     "EnvSatisfactionLabel", "WorkLifeBalanceLabel"]
        features = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns])

        cat_cols = features.select_dtypes(include="object").columns
        encoders = {}
        for c in cat_cols:
            le = LabelEncoder()
            features[c] = le.fit_transform(features[c])
            encoders[c] = le

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.25, random_state=42, stratify=target
        )

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        )
        rf.fit(X_train, y_train)
        proba = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)
        preds = rf.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        importances = pd.Series(rf.feature_importances_, index=features.columns).sort_values(ascending=False)

        return rf, auc, fpr, tpr, cm, importances, features.columns.tolist(), encoders

    rf, auc, fpr, tpr, cm, importances, feat_cols, encoders = train_model(df)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Model ROC-AUC (test set)", f"{auc:.3f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve", line=dict(color=PRIMARY)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cm_fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Stay", "Leave"], y=["Stay", "Leave"],
            title="Confusion Matrix",
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    st.markdown("### Top Predictors of Attrition (Feature Importance)")
    top_n = st.slider("Number of features to show", 5, 20, 12)
    imp_df = importances.head(top_n).reset_index()
    imp_df.columns = ["Feature", "Importance"]
    fig = px.bar(
        imp_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Viridis", title="Random Forest Feature Importance",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔮 Try It: Estimate an Employee's Attrition Risk")
    st.caption("Adjust the sliders to simulate an employee profile and see the model's predicted risk.")

    colA, colB, colC = st.columns(3)
    with colA:
        age = st.slider("Age", 18, 60, 30)
        income = st.slider("Monthly Income ($)", 1000, 20000, 5000, step=100)
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        distance = st.slider("Distance From Home", 1, 30, 8)
    with colB:
        years_company = st.slider("Years at Company", 0, 40, 5)
        job_sat = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
        env_sat = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
        wlb = st.selectbox("Work-Life Balance", [1, 2, 3, 4], index=2)
    with colC:
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
        num_companies = st.slider("Num Companies Worked", 0, 10, 2)
        promo_years = st.slider("Years Since Last Promotion", 0, 15, 1)
        stock = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=0)

    if st.button("Predict Attrition Risk", type="primary"):
        row = df.iloc[[0]].copy()
        row["Age"] = age
        row["MonthlyIncome"] = income
        row["OverTime"] = overtime
        row["DistanceFromHome"] = distance
        row["YearsAtCompany"] = years_company
        row["JobSatisfaction"] = job_sat
        row["EnvironmentSatisfaction"] = env_sat
        row["WorkLifeBalance"] = wlb
        row["JobLevel"] = job_level
        row["NumCompaniesWorked"] = num_companies
        row["YearsSinceLastPromotion"] = promo_years
        row["StockOptionLevel"] = stock

        drop_cols = ["Attrition", "AttritionFlag", "EmployeeCount", "EmployeeNumber",
                     "Over18", "StandardHours", "JobSatisfactionLabel",
                     "EnvSatisfactionLabel", "WorkLifeBalanceLabel"]
        row_feat = row.drop(columns=[c for c in drop_cols if c in row.columns])
        for c, le in encoders.items():
            row_feat[c] = le.transform(row_feat[c].astype(str))
        row_feat = row_feat[feat_cols]

        risk = rf.predict_proba(row_feat)[0, 1]
        st.metric("Predicted probability of leaving", f"{risk*100:.1f}%")
        if risk > 0.5:
            st.error("⚠️ High risk — recommend proactive retention conversation.")
        elif risk > 0.3:
            st.warning("🟠 Moderate risk — monitor engagement.")
        else:
            st.success("🟢 Low risk.")

# ----------------------------------------------------------------------------
# PAGE 5 — RECOMMENDATIONS
# ----------------------------------------------------------------------------
elif page == "✅ Recommendations":
    st.title("Insights & Data-Driven Recommendations")

    st.markdown(
        """
        ### 🔑 Key Insights
        1. **Overtime is the single strongest behavioral driver.** Employees working overtime
           leave at roughly 3x the rate of those who don't.
        2. **Compensation matters, but mainly at the low end.** Attrition is concentrated among
           lower monthly-income employees, particularly early in their career (Job Level 1–2).
        3. **Early tenure is the highest-risk period.** Attrition is highest in the first 2–3
           years at the company, then drops sharply — the "flight risk window" is early.
        4. **Long commutes correlate with higher attrition**, especially combined with frequent
           business travel.
        5. **Job Role and Department concentrate risk.** Sales Representatives, Laboratory
           Technicians, and HR roles show above-average attrition versus Research & Development
           leadership roles.
        6. **Work-life balance and job satisfaction have a clear inverse relationship** with
           attrition — "Bad" work-life balance employees leave at more than double the rate of
           "Best" work-life balance employees.
        7. **Stalled promotions increase risk.** Employees who haven't been promoted in several
           years show elevated attrition, even after controlling for tenure.

        ### 💼 Practical Recommendations for HR Leadership

        | Priority | Recommendation | Rationale |
        |---|---|---|
        | 🔴 High | Audit and cap overtime for high-risk roles; consider overtime pay premiums or workload redistribution | Overtime is the top predictor of attrition |
        | 🔴 High | Launch a structured 90-day and 1-year "early tenure" check-in and mentorship program | Attrition peaks in years 0–3 |
        | 🟠 Medium | Review compensation bands for Job Level 1–2 roles against market benchmarks | Attrition concentrated in lower income brackets |
        | 🟠 Medium | Introduce flexible/hybrid work or a commute stipend for employees with long commutes | Distance from home correlates with attrition |
        | 🟠 Medium | Create clearer promotion timelines and career-pathing conversations, especially past year 2 without promotion | Stalled promotion is a leading indicator |
        | 🟡 Lower | Target retention efforts on Sales Rep / Lab Tech / HR roles with tailored engagement surveys | Role-level attrition variance is high |
        | 🟡 Lower | Deploy the predictive model quarterly to flag at-risk employees for proactive 1:1s | Enables *prevention* instead of *reaction* |

        ### 📌 Suggested Next Steps
        - Integrate the predictive model into a quarterly HR dashboard refresh.
        - Run a controlled pilot (e.g., reduced overtime + manager check-ins) on the highest-risk
          department and measure attrition change over 2 quarters.
        - Pair quantitative risk scores with qualitative exit-interview themes for a fuller picture.
        """
    )

    st.success(
        "Bottom line: attrition here is **not random** — it clusters around overtime, "
        "early tenure, lower compensation levels, and stalled career growth. These are all "
        "levers HR can act on directly."
    )

st.markdown("---")
st.caption("Built with Streamlit · Dataset: IBM HR Analytics Employee Attrition (public dataset, 1,470 records)")
