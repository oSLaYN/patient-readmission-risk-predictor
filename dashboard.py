import warnings
warnings.filterwarnings("ignore")

import traceback
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Set page configuration
st.set_page_config(page_title="Patient Readmission Risk Predictor", page_icon="🏥", layout="wide")

# Constants
RANDOM_STATE = 42

# -------------------------------------------------------------------------------------------------
# 1. Data Loading & Clean-Up
# -------------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and Mapping Dataset...")
def load_data():
    try:
        # Load main dataset
        df = pd.read_csv("data/diabetic_data.csv", na_values='?')

        # Load mapping file (no header — it has 3 blocks separated by empty rows)
        mapping = pd.read_csv("data/IDS_mapping.csv", header=None)

        # Find empty rows (act as block separators)
        empty_rows = mapping[mapping.isna().all(axis=1)].index.tolist()

        # Split into 3 blocks
        b1 = mapping.iloc[0:empty_rows[0], :].copy()
        b2 = mapping.iloc[empty_rows[0] + 1:empty_rows[1], :].copy()
        b3 = mapping.iloc[empty_rows[1] + 1:, :].copy()

        # Assign column names
        b1.columns = ["admission_type_id", "admission_type"]
        b2.columns = ["discharge_disposition_id", "discharge_disposition"]
        b3.columns = ["admission_source_id", "admission_source"]

        # Convert IDs to numeric (drop NaN rows)
        for b, col_id in [
            (b1, "admission_type_id"),
            (b2, "discharge_disposition_id"),
            (b3, "admission_source_id"),
        ]:
            b[col_id] = pd.to_numeric(b[col_id], errors="coerce")
            b.dropna(subset=[col_id], inplace=True)

        # Coerce main dataframe ID columns
        for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Merge descriptive labels into main dataframe
        df = df.merge(b1, on="admission_type_id", how="left")
        df = df.merge(b2, on="discharge_disposition_id", how="left")
        df = df.merge(b3, on="admission_source_id", how="left")

        # Target column: binary classification — '<30' vs anything else
        df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)

        # Drop identifiers and now-redundant numeric ID columns
        df.drop(
            columns=['encounter_id', 'patient_nbr',
                     'admission_type_id', 'discharge_disposition_id', 'admission_source_id'],
            inplace=True, errors='ignore'
        )

        # Drop high-missing columns (>40% missing)
        df.drop(columns=['weight', 'payer_code', 'medical_specialty'], inplace=True, errors='ignore')

        # Drop invalid gender rows
        df = df[df['gender'] != 'Unknown/Invalid']

        # Convert age bracket strings -> numeric midpoints
        age_map = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        df['age_num'] = df['age'].map(age_map)
        df.drop(columns=['age'], inplace=True)

        # Fill remaining NaN with column mode
        for col in ['race', 'diag_1', 'diag_2', 'diag_3']:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)

        # Cardinality reduction: keep top-20 diagnosis values, merge rest → "Other"
        # This prevents a feature-count explosion during One-Hot Encoding
        for col in ['diag_1', 'diag_2', 'diag_3']:
            if col in df.columns:
                top_20 = df[col].value_counts().nlargest(20).index
                df.loc[~df[col].isin(top_20), col] = 'Other'

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.code(traceback.format_exc())
        return None


# -------------------------------------------------------------------------------------------------
# 2. Preprocessing
# -------------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Preprocessing Data for Training...")
def preprocess_data(df):
    if df is None:
        return None, None, None, None, None

    target_col = 'readmitted_binary'
    drop_cols = ['readmitted', target_col]

    df_features = df.drop(columns=drop_cols, errors='ignore')
    target = df[target_col]

    # One-Hot Encode all remaining categorical columns
    categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)

    # Drop constant columns — they cause divide-by-zero in StandardScaler
    df_encoded = df_encoded.loc[:, df_encoded.nunique() > 1]

    X = df_encoded.values.astype(np.float32)
    y = target.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_names = df_encoded.columns.tolist()

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


# -------------------------------------------------------------------------------------------------
# 3. Model Training (cached so they only run once per session)
# -------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training ML Models (LR + RF)...")
def train_ml_models(_X_train, _y_train):
    """Underscore-prefix tells Streamlit not to hash these numpy arrays."""
    lr_model = LogisticRegression(
        class_weight='balanced', max_iter=200, n_jobs=-1, random_state=RANDOM_STATE
    )
    lr_model.fit(_X_train, _y_train)

    rf_model = RandomForestClassifier(
        n_estimators=50, class_weight='balanced', max_depth=8,
        n_jobs=-1, random_state=RANDOM_STATE
    )
    rf_model.fit(_X_train, _y_train)

    return lr_model, rf_model


@st.cache_resource(show_spinner="Training Deep Learning Model (Keras)...")
def train_dl_model(_X_train, _y_train):
    tf.random.set_seed(RANDOM_STATE)

    cw_values = compute_class_weight(
        class_weight="balanced", classes=np.unique(_y_train), y=_y_train
    )
    class_weights = dict(zip(np.unique(_y_train), cw_values))

    n_features = _X_train.shape[1]

    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_auc", patience=3, restore_best_weights=True, mode="max"
    )

    model.fit(
        _X_train, _y_train,
        validation_split=0.15,
        epochs=10,
        batch_size=1024,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=0
    )

    return model


# -------------------------------------------------------------------------------------------------
# UI Pages
# -------------------------------------------------------------------------------------------------
def page_overview(df):
    st.header("Dataset Overview")
    st.markdown(
        "Below is a cleaned view of the dataset (identifiers dropped, missing values handled, "
        "age converted to numeric midpoints, diagnosis codes bucketed to top-20 categories)."
    )
    st.dataframe(df.head(200), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        rate = df['readmitted_binary'].mean() * 100
        st.metric("30-Day Readmission Rate", f"{rate:.2f}%")

    dist = df['readmitted_binary'].value_counts(normalize=True) * 100
    st.subheader("Target Distribution")
    st.write(f"**Not readmitted / >30 days (0):** {dist.get(0, 0):.2f}%")
    st.write(f"**Readmitted <30 days (1):** {dist.get(1, 0):.2f}%")


def page_eda(df):
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Readmission Class Balance")
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ['Not Readmitted\n(>30d / No)', 'Readmitted\n(<30d)']
        counts = df['readmitted_binary'].value_counts().sort_index()
        ax.bar(labels, counts.values, color=['steelblue', 'tomato'], edgecolor='white')
        ax.set_ylabel("Number of Patients")
        ax.set_title("30-Day Readmission Counts")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Age Distribution by Readmission")
        if 'age_num' in df.columns:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.boxplot(data=df, x='readmitted_binary', y='age_num', palette='Set2', ax=ax2)
            ax2.set_xticklabels(['Not Readmitted (<30d)', 'Readmitted (<30d)'])
            ax2.set_xlabel("")
            ax2.set_ylabel("Age (midpoint)")
            ax2.set_title("Age vs Readmission")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    # --- Readmission rate by race ---
    if 'race' in df.columns:
        st.subheader("30-Day Readmission Rate by Race")
        race_rates = (
            df.groupby("race")["readmitted_binary"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "readmission_rate", "count": "n_patients"})
            .sort_values("readmission_rate", ascending=False)
            .reset_index()
        )
        fig_r, ax_r = plt.subplots(figsize=(8, 4))
        ax_r.bar(race_rates["race"], race_rates["readmission_rate"], color="steelblue", edgecolor="white")
        ax_r.axhline(df["readmitted_binary"].mean(), color="red", linestyle="--",
                     label=f'Overall mean ({df["readmitted_binary"].mean():.3f})')
        ax_r.set_xlabel("Race")
        ax_r.set_ylabel("Readmission Rate")
        ax_r.set_title("30-Day Readmission Rate by Race")
        ax_r.legend()
        plt.tight_layout()
        st.pyplot(fig_r)
        plt.close(fig_r)

    # --- Numeric correlation heatmap ---
    st.subheader("Feature Correlation Heatmap (Numeric Features)")
    numerical_df = df.select_dtypes(include=[np.number])
    corr = numerical_df.corr()
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax3,
                annot_kws={"size": 7}, linewidths=0.3)
    ax3.set_title("Pearson Correlation Matrix")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)


def page_models(df):
    st.header("Model Evaluation & Feature Importance")

    with st.spinner("Preprocessing…"):
        result = preprocess_data(df)
    if result[0] is None:
        st.error("Preprocessing failed.")
        return

    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = result

    with st.spinner("Training ML models (first run only)…"):
        lr_model, rf_model = train_ml_models(X_train_scaled, y_train)

    dl_model = None
    if TF_AVAILABLE:
        with st.spinner("Training Keras model (first run only)…"):
            dl_model = train_dl_model(X_train_scaled, y_train)

    # Feature Importance
    st.markdown("### 1. Feature Importance")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 10 — Random Forest (Impurity-based)**")
        rf_imp = (
            pd.Series(rf_model.feature_importances_, index=feature_names)
            .sort_values(ascending=False)
            .head(10)
        )
        fig_rf, ax_rf = plt.subplots(figsize=(6, 4))
        sns.barplot(x=rf_imp.values, y=rf_imp.index, palette='Blues_r', ax=ax_rf)
        ax_rf.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig_rf)
        plt.close(fig_rf)

    with col2:
        st.markdown("**Top 10 — Logistic Regression (|Coefficient|)**")
        lr_imp = (
            pd.Series(np.abs(lr_model.coef_[0]), index=feature_names)
            .sort_values(ascending=False)
            .head(10)
        )
        fig_lr, ax_lr = plt.subplots(figsize=(6, 4))
        sns.barplot(x=lr_imp.values, y=lr_imp.index, palette='Greens_r', ax=ax_lr)
        ax_lr.set_xlabel("|Coefficient|")
        plt.tight_layout()
        st.pyplot(fig_lr)
        plt.close(fig_lr)

    st.markdown("---")
    st.markdown("### 2. Model Performance Comparison")

    # Predictions
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

    rf_preds = rf_model.predict(X_test_scaled)
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

    n_cols = 3 if dl_model else 2
    cols = st.columns(n_cols)

    with cols[0]:
        st.info("**Logistic Regression**")
        st.metric("ROC-AUC", f"{roc_auc_score(y_test, lr_probs):.4f}")
        st.metric("Accuracy", f"{accuracy_score(y_test, lr_preds):.4f}")
        st.text(classification_report(y_test, lr_preds))

    with cols[1]:
        st.success("**Random Forest**")
        st.metric("ROC-AUC", f"{roc_auc_score(y_test, rf_probs):.4f}")
        st.metric("Accuracy", f"{accuracy_score(y_test, rf_preds):.4f}")
        st.text(classification_report(y_test, rf_preds))

    if dl_model:
        dl_probs = dl_model.predict(X_test_scaled, verbose=0).flatten()
        dl_preds = (dl_probs > 0.5).astype(int)
        with cols[2]:
            st.warning("**Deep Learning (Keras)**")
            st.metric("ROC-AUC", f"{roc_auc_score(y_test, dl_probs):.4f}")
            st.metric("Accuracy", f"{accuracy_score(y_test, dl_preds):.4f}")
            st.text(classification_report(y_test, dl_preds))

    # ROC Curves
    st.markdown("### 3. ROC Curves")
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    models_to_plot = [
        ("Logistic Regression", lr_probs),
        ("Random Forest", rf_probs),
    ]
    if dl_model:
        models_to_plot.append(("Keras Neural Network", dl_probs))

    for name, probs in models_to_plot:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], 'k--', label="Random Baseline")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves — Model Comparison")
    ax_roc.legend(loc='lower right')
    plt.tight_layout()
    st.pyplot(fig_roc)
    plt.close(fig_roc)


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------
def main():
    st.title("🏥 Patient Readmission Risk Predictor")
    st.markdown(
        "Interactive dashboard for the **Diabetes 130-US Hospitals** dataset. "
        "Compares Logistic Regression, Random Forest, and Keras Neural Network "
        "on predicting 30-day hospital readmission."
    )

    st.sidebar.title("Navigation")
    pages = ["Dataset Overview", "Exploratory Data Analysis", "Model Performance & Features"]
    selection = st.sidebar.radio("Go to", pages)

    df = load_data()
    if df is None:
        st.stop()

    if selection == "Dataset Overview":
        page_overview(df)
    elif selection == "Exploratory Data Analysis":
        page_eda(df)
    elif selection == "Model Performance & Features":
        page_models(df)


if __name__ == "__main__":
    main()
