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
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    BINARY_FEATURES,
    MULTI_FEATURES,
    DROP_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    RAW_DATA_PATH,
    PREPROCESSOR_PATH,
)
import joblib
from data_ingestion import DataIngestion
from utils.logger import setup_logger


class DataPreprocessor:

    def __init__(self, df: DataIngestion):
        self.df = df.df
        self.logger = setup_logger()

    def _encode_binary(self) -> pd.DataFrame:
        
        self.df["TotalCharges"] = pd.to_numeric(
            self.df["TotalCharges"], errors="coerce"
        ).fillna(0)
        self.df = self.df.drop(columns=DROP_COLUMNS)

        self.yes_no_cols = [c for c in BINARY_FEATURES if self.df[c].dtype == object]

        for col in self.yes_no_cols:
            if col == "gender":
                self.df[col] = (
                    self.df[col].map({"Male": 1, "Female": 0}).fillna(0).astype(int)
                )
            else:
                self.df[col] = (
                    self.df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
                )
        return self.df

    def _encode_target(self) -> pd.DataFrame:
        self.df = self._encode_binary()

        if self.df[TARGET_COLUMN].dtype == object:
            self.df[TARGET_COLUMN] = (
                self.df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
            )

        return self.df

    def setup_preprocessing(self) -> ColumnTransformer:
        self.df = self._encode_target()

        self.numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        self.categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        self.preprocessor = ColumnTransformer(
            [
                ("num", self.numeric_pipeline, NUMERIC_FEATURES),
                ("cat", self.categorical_pipeline, MULTI_FEATURES),
            ],
            remainder="passthrough",
        )

        return self.preprocessor

    def build_preprocessor(self):
        self.preprocessor = self.setup_preprocessing()
        self.logger.info("🚀 ===> Preprocessor Stage: Started Processing ")

        if PROCESSED_DATA_DIR.exists():
            self.logger.warning("⚠️ ===> Processed file already exists.")
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.logger.info("ℹ️ ===> Build processed file.")

        self.X = self.df.drop(columns=[TARGET_COLUMN])
        self.y = self.df[TARGET_COLUMN]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=self.y,
        )

        self.logger.info(f"Train: {len(self.X_train):,}  |  Test: {len(self.X_test):,}")

        self._X_train = self.preprocessor.fit_transform(self.X_train)
        self._X_test = self.preprocessor.transform(self.X_test)

        self.ohe_cols = (
            self.preprocessor.named_transformers_["cat"]
            .named_steps["ohe"]
            .get_feature_names_out(MULTI_FEATURES)
            .tolist()
        )

        self.passthrough_cols = [
            c
            for c in self.X_train.columns
            if c not in NUMERIC_FEATURES + MULTI_FEATURES
        ]

        self.all_cols = NUMERIC_FEATURES + self.ohe_cols + self.passthrough_cols

        self.train_df = pd.DataFrame(self._X_train, columns=self.all_cols)  # type: ignore
        self.train_df[TARGET_COLUMN] = self.y_train.reset_index(drop=True)

        self.test_df = pd.DataFrame(self._X_test, columns=self.all_cols)  # type: ignore
        self.test_df[TARGET_COLUMN] = self.y_test.reset_index(drop=True)

        self.train_df.to_csv(TRAIN_DATA_PATH, index=False)
        self.test_df.to_csv(TEST_DATA_PATH, index=False)

        if not PREPROCESSOR_PATH.exists():
            PREPROCESSOR_PATH.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.preprocessor, PREPROCESSOR_PATH / "preprocessor.joblib")
        self.logger.info(" Saved Preprocessor ===> ℹ️")
        self.logger.info(f" Saved Train Data ===> ℹ️")
        self.logger.info(f" Saved Test Data ===> ℹ️")
        self.logger.info("✅ ===> Preprocessing Stage: Completed Processing")

        return self.train_df, self.test_df


if __name__ == "__main__":
    ingestion = DataIngestion(RAW_DATA_PATH)
    df = ingestion.load_dataset()
    validated_df = ingestion.validate_data()

    preprocessor = DataPreprocessor(ingestion)
    train_df, test_df = preprocessor.build_preprocessor()
