import warnings
warnings.filterwarnings("ignore")

import traceback
import os
import joblib
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
from xgboost import XGBClassifier

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
EXPECTED_FEATURES = [
    'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
    'race_Asian', 'race_Caucasian', 'race_Hispanic', 'race_Other', 'race_Unknown',
    'gender_Male', 'gender_Unknown/Invalid', 'diag_1_Diabetes', 'diag_1_Digestive',
    'diag_1_Genitourinary', 'diag_1_Injury', 'diag_1_Missing', 'diag_1_Musculoskeletal',
    'diag_1_Neoplasms', 'diag_1_Other', 'diag_1_Respiratory', 'diag_2_Diabetes',
    'diag_2_Digestive', 'diag_2_Genitourinary', 'diag_2_Injury', 'diag_2_Missing',
    'diag_2_Musculoskeletal', 'diag_2_Neoplasms', 'diag_2_Other', 'diag_2_Respiratory',
    'diag_3_Diabetes', 'diag_3_Digestive', 'diag_3_Genitourinary', 'diag_3_Injury',
    'diag_3_Missing', 'diag_3_Musculoskeletal', 'diag_3_Neoplasms', 'diag_3_Other',
    'diag_3_Respiratory', 'max_glu_serum_>300', 'max_glu_serum_Norm', 'A1Cresult_>8',
    'A1Cresult_Norm', 'metformin_No', 'metformin_Steady', 'metformin_Up',
    'repaglinide_No', 'repaglinide_Steady', 'repaglinide_Up', 'nateglinide_No',
    'nateglinide_Steady', 'nateglinide_Up', 'chlorpropamide_No', 'chlorpropamide_Steady',
    'chlorpropamide_Up', 'glimepiride_No', 'glimepiride_Steady', 'glimepiride_Up',
    'acetohexamide_Steady', 'glipizide_No', 'glipizide_Steady', 'glipizide_Up',
    'glyburide_No', 'glyburide_Steady', 'glyburide_Up', 'tolbutamide_Steady',
    'pioglitazone_No', 'pioglitazone_Steady', 'pioglitazone_Up', 'rosiglitazone_No',
    'rosiglitazone_Steady', 'rosiglitazone_Up', 'acarbose_No', 'acarbose_Steady',
    'acarbose_Up', 'miglitol_No', 'miglitol_Steady', 'miglitol_Up', 'troglitazone_Steady',
    'tolazamide_Steady', 'tolazamide_Up', 'insulin_No', 'insulin_Steady', 'insulin_Up',
    'glyburide-metformin_No', 'glyburide-metformin_Steady', 'glyburide-metformin_Up',
    'glipizide-metformin_Steady', 'glimepiride-pioglitazone_Steady',
    'metformin-rosiglitazone_Steady', 'metformin-pioglitazone_Steady', 'change_No',
    'diabetesMed_Yes', 'admission_type_Emergency', 'admission_type_Newborn',
    'admission_type_Not Available', 'admission_type_Not Mapped',
    'admission_type_Trauma Center', 'admission_type_Urgent',
    'discharge_disposition_Discharged to home', 'discharge_disposition_Discharged/transferred to ICF',
    'discharge_disposition_Discharged/transferred to SNF',
    'discharge_disposition_Discharged/transferred to a federal health care facility.',
    'discharge_disposition_Discharged/transferred to a long term care hospital.',
    'discharge_disposition_Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.',
    'discharge_disposition_Discharged/transferred to another rehab fac including rehab units of a hospital .',
    'discharge_disposition_Discharged/transferred to another short term hospital',
    'discharge_disposition_Discharged/transferred to another type of inpatient care institution',
    'discharge_disposition_Discharged/transferred to home under care of Home IV provider',
    'discharge_disposition_Discharged/transferred to home with home health service',
    'discharge_disposition_Discharged/transferred within this institution to Medicare approved swing bed',
    'discharge_disposition_Discharged/transferred/referred another institution for outpatient services',
    'discharge_disposition_Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
    'discharge_disposition_Discharged/transferred/referred to this institution for outpatient services',
    'discharge_disposition_Expired', 'discharge_disposition_Expired at home. Medicaid only, hospice.',
    'discharge_disposition_Expired in a medical facility. Medicaid only, hospice.',
    'discharge_disposition_Hospice / home', 'discharge_disposition_Hospice / medical facility',
    'discharge_disposition_Left AMA',
    'discharge_disposition_Neonate discharged to another hospital for neonatal aftercare',
    'discharge_disposition_Not Mapped',
    'discharge_disposition_Still patient or expected to return for outpatient services',
    'admission_source_ Emergency Room', 'admission_source_ Extramural Birth',
    'admission_source_ Not Available', 'admission_source_ Not Mapped',
    'admission_source_ Physician Referral', 'admission_source_ Sick Baby',
    'admission_source_ Transfer from Ambulatory Surgery Center',
    'admission_source_ Transfer from a Skilled Nursing Facility (SNF)',
    'admission_source_ Transfer from another health care facility',
    'admission_source_ Transfer from critial access hospital',
    'admission_source_ Transfer from hospital inpt/same fac reslt in a sep claim',
    'admission_source_Clinic Referral', 'admission_source_HMO Referral',
    'admission_source_Normal Delivery', 'admission_source_Transfer from a hospital'
]

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

        # Keep redundant numeric ID columns as they were used in training
        df.drop(
            columns=['encounter_id', 'patient_nbr', 'readmitted'],
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

        # ICD-9 code grouping (Matching main.ipynb logic)
        def map_icd9(val):
            if pd.isna(val) or val == '?':
                return 'Missing'
            val_str = str(val).upper()
            if val_str.startswith('V') or val_str.startswith('E'):
                return 'Other'
            try:
                v = float(val_str)
                if 250 <= v < 251: return 'Diabetes'
                elif (390 <= v <= 459) or v == 785: return 'Circulatory'
                elif (460 <= v <= 519) or v == 786: return 'Respiratory'
                elif (520 <= v <= 579) or v == 787: return 'Digestive'
                elif 800 <= v <= 999: return 'Injury'
                elif 710 <= v <= 739: return 'Musculoskeletal'
                elif (580 <= v <= 629) or v == 788: return 'Genitourinary'
                elif 140 <= v <= 239: return 'Neoplasms'
                else: return 'Other'
            except ValueError:
                return 'Other'

        for col in ['diag_1', 'diag_2', 'diag_3']:
            if col in df.columns:
                df[col] = df[col].apply(map_icd9)

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

    # Feature Alignment: ensure all expected columns exist, fill missing with 0
    for col in EXPECTED_FEATURES:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reorder columns to match training set exactly
    df_encoded = df_encoded[EXPECTED_FEATURES]

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
@st.cache_resource(show_spinner="Preparing ML Models (Loading from disk or training)...")
def get_ml_models(_X_train, _y_train):
    """Loads optimized models from disk if available, otherwise falls back to training."""
    if os.path.exists('models/logistic_regression.joblib') and os.path.exists('models/xgboost.joblib'):
        lr_model = joblib.load('models/logistic_regression.joblib')
        rf_model = joblib.load('models/random_forest.joblib')
        xgb_model = joblib.load('models/xgboost.joblib')
        return lr_model, rf_model, xgb_model

    # Fallback: Train if models directory not found
    lr_model = LogisticRegression(
        class_weight='balanced', max_iter=200, n_jobs=-1, random_state=RANDOM_STATE
    )
    lr_model.fit(_X_train, _y_train)

    rf_model = RandomForestClassifier(
        n_estimators=50, class_weight='balanced', max_depth=8,
        n_jobs=-1, random_state=RANDOM_STATE
    )
    rf_model.fit(_X_train, _y_train)
    
    cw_values = compute_class_weight('balanced', classes=np.unique(_y_train), y=_y_train)
    class_weights = dict(zip(np.unique(_y_train), cw_values))
    xgb_model = XGBClassifier(
        n_estimators=50, max_depth=6, learning_rate=0.1,
        scale_pos_weight=class_weights[1] / class_weights[0],
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'
    )
    xgb_model.fit(_X_train, _y_train)

    return lr_model, rf_model, xgb_model


@st.cache_resource(show_spinner="Preparing Deep Learning Model (Keras)...")
def get_dl_model(_X_train, _y_train):
    if os.path.exists('models/keras_model.h5'):
        model = keras.models.load_model('models/keras_model.h5')
        return model

    # Fallback: Train if not found
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

    with st.spinner("Preparing ML models (First run may take a moment if tuning hasn't been saved)…"):
        lr_model, rf_model, xgb_model = get_ml_models(X_train_scaled, y_train)

    dl_model = None
    if TF_AVAILABLE:
        with st.spinner("Preparing Keras model…"):
            dl_model = get_dl_model(X_train_scaled, y_train)

    # Feature Importance
    st.markdown("### 1. Feature Importance")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Top 10 — Random Forest (Impurity-based)**")
        rf_imp = (
            pd.Series(rf_model.feature_importances_, index=feature_names)
            .sort_values(ascending=False)
            .head(10)
        )
        fig_rf, ax_rf = plt.subplots(figsize=(5, 3.5))
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
        fig_lr, ax_lr = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x=lr_imp.values, y=lr_imp.index, palette='Greens_r', ax=ax_lr)
        ax_lr.set_xlabel("|Coefficient|")
        plt.tight_layout()
        st.pyplot(fig_lr)
        plt.close(fig_lr)

    with col3:
        st.markdown("**Top 10 — XGBoost (Weight)**")
        xgb_imp = (
            pd.Series(xgb_model.feature_importances_, index=feature_names)
            .sort_values(ascending=False)
            .head(10)
        )
        fig_xgb, ax_xgb = plt.subplots(figsize=(5, 3.5))
        sns.barplot(x=xgb_imp.values, y=xgb_imp.index, palette='Oranges_r', ax=ax_xgb)
        ax_xgb.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig_xgb)
        plt.close(fig_xgb)

    st.markdown("---")
    st.markdown("### 2. Model Performance Comparison")

    # Predictions
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

    rf_preds = rf_model.predict(X_test_scaled)
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

    xgb_preds = xgb_model.predict(X_test_scaled)
    xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

    n_cols = 4 if dl_model else 3
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

    with cols[2]:
        st.error("**XGBoost**")
        st.metric("ROC-AUC", f"{roc_auc_score(y_test, xgb_probs):.4f}")
        st.metric("Accuracy", f"{accuracy_score(y_test, xgb_preds):.4f}")
        st.text(classification_report(y_test, xgb_preds))

    if dl_model:
        dl_probs = dl_model.predict(X_test_scaled, verbose=0).flatten()
        dl_preds = (dl_probs > 0.5).astype(int)
        with cols[3]:
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
        ("XGBoost", xgb_probs),
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
        "Compares Logistic Regression, Random Forest, XGBoost, and Keras Neural Network "
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
