import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from config import (
    PROCESSED_DATA_DIR,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    MULTI_FEATURES,
    DROP_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
)

from data_ingestion import run_ingestion
from utils.logger import setup_logger

logger = setup_logger()


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes/No → 1/0 and gender → 1/0 in place."""
    yes_no_cols = [c for c in BINARY_FEATURES if df[c].dtype == object]
    for col in yes_no_cols:
        if col == "gender":
            df[col] = (df[col] == "Male").astype(int)
        else:
            df[col] = (df[col] == "Yes").astype(int)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Yes").astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Numeric: median impute + standard scale. Categorical: OHE."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, MULTI_FEATURES),
        ],
        remainder="passthrough",
    )

    return preprocessor


def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame]:

    if not PROCESSED_DATA_DIR:
        logger.warning("Folder isn't available!")
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = run_ingestion()

    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    df = encode_target(df)
    df = encode_binary(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    ohe_cols = (
        preprocessor.named_transformers_["cat"]
        .named_steps["ohe"]
        .get_feature_names_out(MULTI_FEATURES)
        .tolist()
    )
    passthrough_cols = [
        c for c in X_train.columns if c not in NUMERIC_FEATURES + MULTI_FEATURES
    ]
    all_cols = NUMERIC_FEATURES + ohe_cols + passthrough_cols

    train_df = pd.DataFrame(X_train_t, columns=all_cols)
    train_df[TARGET_COLUMN] = y_train.reset_index(drop=True)

    test_df = pd.DataFrame(X_test_t, columns=all_cols)
    test_df[TARGET_COLUMN] = y_test.reset_index(drop=True)

    # 8. Persist
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    logger.info(f"Saved train -> {TRAIN_DATA_PATH}")
    logger.info(f"Saved test  -> {TEST_DATA_PATH}")
    logger.info(f"Saved preprocessor -> {PREPROCESSOR_PATH}")
    logger.info("---> Preprocessing Completed...")

    return train_df, test_df


if __name__ == "__main__":
    run_preprocessing()
