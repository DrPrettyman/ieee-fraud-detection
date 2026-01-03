"""
Custom sklearn transformers for IEEE-CIS Fraud Detection 
feature engineering.
"""

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class ColumnNormalizer(BaseEstimator, TransformerMixin):
    """Standardize column names for consistency.

    Renames columns matching 'id-XX' pattern to 'id_XX' to ensure
    consistent naming between transaction and identity datasets.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.columns = [
            re.sub(r'^id-(\d+)', r'id_\1', col)
            for col in X.columns
        ]
        return X


class TimeFeatures(BaseEstimator, TransformerMixin):
    """Add cyclical time features from TransactionDT."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        hour = (X['TransactionDT'] / 3600) % 24
        day = (X['TransactionDT'] / 86400) % 7

        X['hod_sin'] = np.sin(2 * np.pi * hour / 24)
        X['hod_cos'] = np.cos(2 * np.pi * hour / 24)
        X['dow_sin'] = np.sin(2 * np.pi * day / 7)
        X['dow_cos'] = np.cos(2 * np.pi * day / 7)

        return X


class EmailFeatures(BaseEstimator, TransformerMixin):
    """Add email-based features."""

    FREE_EMAILS = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
                   'aol.com', 'icloud.com', 'mail.com', 'protonmail.com']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['email_match'] = (X['P_emaildomain'] == X['R_emaildomain']).astype(int)
        X['P_email_is_free'] = X['P_emaildomain'].isin(self.FREE_EMAILS).astype(int)
        X['R_email_is_free'] = X['R_emaildomain'].isin(self.FREE_EMAILS).astype(int)
        X['P_email_missing'] = X['P_emaildomain'].isna().astype(int)
        X['R_email_missing'] = X['R_emaildomain'].isna().astype(int)

        return X


class CardFeatures(BaseEstimator, TransformerMixin):
    """Add card and device features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['is_new_card'] = (X['D1'] <= 7).astype(int)
        X['has_identity'] = X['DeviceType'].notna().astype(int)
        X['is_mobile'] = (X['DeviceType'] == 'mobile').astype(int)

        return X


class AmountFeatures(BaseEstimator, TransformerMixin):
    """Add transaction amount features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])
        X['TransactionAmt_decimal'] = (X['TransactionAmt'] % 1).round(2)
        X['TransactionAmt_is_round'] = (X['TransactionAmt'] % 1 == 0).astype(int)

        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency encode high-cardinality categorical columns.

    Replaces categorical values with their frequency (count or ratio) from
    the training data. Useful for columns like card1, addr1 where:
    - Label encoding implies false ordinal relationships
    - One-hot encoding creates too many columns
    - Frequency captures "how common is this value" signal

    Parameters
    ----------
    cols : list of str
        Columns to frequency encode.
    drop_original : bool, default=True
        Whether to drop the original columns after encoding.
    normalize : bool, default=True
        If True, use frequency ratio (0-1). If False, use raw counts.
    """

    def __init__(self, cols=None, drop_original=True, normalize=True):
        self.cols = cols or ['card1', 'addr1']
        self.drop_original = drop_original
        self.normalize = normalize

    def fit(self, X, y=None):
        self.freq_maps_ = {}
        n_rows = len(X)

        for col in self.cols:
            if col not in X.columns:
                continue
            counts = X[col].value_counts()
            if self.normalize:
                self.freq_maps_[col] = (counts / n_rows).to_dict()
            else:
                self.freq_maps_[col] = counts.to_dict()

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.cols:
            if col not in X.columns or col not in self.freq_maps_:
                continue

            freq_map = self.freq_maps_[col]
            # Unseen values get frequency 0 (rarest possible)
            X[f'{col}_freq'] = X[col].map(freq_map).fillna(0)

            if self.drop_original:
                X = X.drop(columns=[col])

        return X


class AggregationFeatures(BaseEstimator, TransformerMixin):
    """Add aggregation features - statistics per customer UID.

    Creates a composite UID from multiple columns (default: card1 + addr1)
    to better identify unique customers, then calculates:
    - Mean transaction amount for that customer
    - Std of transaction amount for that customer
    - Count of transactions for that customer

    Then we compute deviation features:
    - Z-score: how unusual is this transaction compared to customer's
                typical spending

    Parameters
    ----------
    uid_cols : list of str, default=['card1', 'addr1']
        Columns to combine into a unique customer ID.
        Top Kaggle solutions used:
        - ['card1', 'addr1'] - simple and effective
        - ['card1', 'addr1', 'P_emaildomain'] - 1st place solution
    """

    def __init__(self, uid_cols=None):
        self.uid_cols = uid_cols or ['card1', 'addr1']

    def _create_uid(self, X):
        """Create composite UID from specified columns."""
        uid = X[self.uid_cols[0]].astype(str)
        for col in self.uid_cols[1:]:
            uid = uid + '_' + X[col].fillna('nan').astype(str)
        return uid

    @property
    def _uid_name(self):
        """Generate a name for the UID based on component columns."""
        return '_'.join(self.uid_cols)

    def fit(self, X, y=None):
        X = X.copy()
        X['_uid'] = self._create_uid(X)

        # Calculate global stats for fallback
        self.global_mean_ = X['TransactionAmt'].mean()
        self.global_std_ = X['TransactionAmt'].std()

        # Calculate per-customer statistics from training data
        self.uid_stats_ = X.groupby('_uid')['TransactionAmt'].agg(['mean', 'std', 'count'])
        self.uid_stats_.columns = [f'{self._uid_name}_amt_mean',
                                   f'{self._uid_name}_amt_std',
                                   f'{self._uid_name}_txn_count']
        # Fill NaN std (customers with single transaction) with global std
        self.uid_stats_[f'{self._uid_name}_amt_std'].fillna(self.global_std_, inplace=True)

        return self

    def transform(self, X):
        X = X.copy()
        X['_uid'] = self._create_uid(X)

        # Merge customer statistics
        X = X.merge(self.uid_stats_, left_on='_uid', right_index=True, how='left')

        # Fill missing (unseen customers) with global stats
        X[f'{self._uid_name}_amt_mean'].fillna(self.global_mean_, inplace=True)
        X[f'{self._uid_name}_amt_std'].fillna(self.global_std_, inplace=True)
        X[f'{self._uid_name}_txn_count'].fillna(1, inplace=True)

        # Compute z-score: how unusual is this transaction for this customer
        X[f'{self._uid_name}_amt_zscore'] = (
            (X['TransactionAmt'] - X[f'{self._uid_name}_amt_mean']) /
            X[f'{self._uid_name}_amt_std'].clip(lower=1e-6)  # Avoid division by zero
        )

        # Drop temporary UID column
        X = X.drop(columns=['_uid'])

        return X


class MissingIndicators(BaseEstimator, TransformerMixin):
    """Add missing value pattern features."""

    def __init__(self, d_cols=None):
        self.d_cols = d_cols or ['D8', 'D7', 'D2']

    def fit(self, X, y=None):
        self.v_cols_ = [col for col in X.columns if re.match(r'V\d+', col)]
        return self

    def transform(self, X):
        X = X.copy()
        # Only use V cols that exist in this dataset
        v_cols_present = [c for c in self.v_cols_ if c in X.columns]
        X['v_missing_count'] = X[v_cols_present].isna().sum(axis=1)

        for col in self.d_cols:
            if col in X.columns:
                X[f'{col}_missing'] = X[col].isna().astype(int)

        return X


class AsCategory(BaseEstimator, TransformerMixin):
    """Convert numeric columns to string for categorical encoding.

    Use this for low-cardinality numeric columns that are actually categorical
    (e.g., id_32 with values 0, 16, 24, 32 representing screen color depth).

    Parameters
    ----------
    cols : list of str
        Columns to convert to string dtype.
    """

    def __init__(self, cols=None):
        self.cols = cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in X.columns:
                X[col] = X[col].astype(str)
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Label encode categorical columns.

    Best for: Tree-based models (LightGBM, XGBoost, Random Forest)
    that can learn splits on arbitrary thresholds.
    """

    def __init__(self):
        self.encoders_ = {}
        self.cat_cols_ = []

    def fit(self, X, y=None):
        self.cat_cols_ = X.select_dtypes(include=['object']).columns.tolist()
        for col in self.cat_cols_:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cat_cols_:
            if col not in X.columns:
                continue  # Skip columns not present in this dataset
            le = self.encoders_[col]
            X[col] = X[col].astype(str)
            # Handle unseen categories
            unseen = set(X[col].unique()) - set(le.classes_)
            if unseen:
                le.classes_ = np.append(le.classes_, list(unseen))
            X[col] = le.transform(X[col])
        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode categorical columns.

    Best for: Linear models (Logistic Regression, Linear SVM)
    that would incorrectly interpret label-encoded values as ordinal.

    Parameters
    ----------
    max_categories : int, default=50
        Maximum number of categories to one-hot encode per column.
        Columns with more categories are frequency-encoded instead
        to avoid explosion of features.
    drop_original : bool, default=True
        Whether to drop the original columns after encoding.
    """

    def __init__(self, max_categories=50, drop_original=True):
        self.max_categories = max_categories
        self.drop_original = drop_original

    def fit(self, X, y=None):
        self.cat_cols_ = X.select_dtypes(include=['object']).columns.tolist()
        self.categories_ = {}
        self.high_cardinality_ = []

        for col in self.cat_cols_:
            unique_vals = X[col].astype(str).unique()
            if len(unique_vals) <= self.max_categories:
                # One-hot encode: store categories
                self.categories_[col] = sorted(unique_vals)
            else:
                # Too many categories: will use frequency encoding
                self.high_cardinality_.append(col)
                counts = X[col].value_counts(normalize=True)
                self.categories_[col] = counts.to_dict()

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.cat_cols_:
            if col not in X.columns:
                continue

            X[col] = X[col].astype(str)

            if col in self.high_cardinality_:
                # Frequency encode high-cardinality columns
                freq_map = self.categories_[col]
                X[f'{col}_freq'] = X[col].map(freq_map).fillna(0)
            else:
                # One-hot encode
                categories = self.categories_[col]
                for cat in categories:
                    X[f'{col}_{cat}'] = (X[col] == cat).astype(int)

            if self.drop_original:
                X = X.drop(columns=[col])

        return X
