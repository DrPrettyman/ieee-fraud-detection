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
    """Add email-based features.

    EDA Insights (01_eda.ipynb, Section 8 - Email Domain Analysis):
    - Email match (P_emaildomain == R_emaildomain) shows 9.7% fraud rate
      vs 2.2% when different (Insight 6)
    - outlook.com has highest fraud rate at 9.5% for P_emaildomain
    - For R_emaildomain: outlook.com (16.5%), icloud.com (12.9%), gmail.com (11.9%)
    - These rates are 2-5x the baseline ~3.5% fraud rate
    """

    FREE_EMAILS = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
                   'aol.com', 'icloud.com', 'mail.com', 'protonmail.com']

    # Domains with fraud rate significantly above baseline (from EDA cell 39)
    # P_emaildomain: outlook.com (9.5%), hotmail.com (5.3%)
    # R_emaildomain: outlook.com (16.5%), icloud.com (12.9%), gmail.com (11.9%)
    HIGH_RISK_P_DOMAINS = ['outlook.com', 'hotmail.com']
    HIGH_RISK_R_DOMAINS = ['outlook.com', 'icloud.com', 'gmail.com', 'hotmail.com']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Original features
        X['email_match'] = (X['P_emaildomain'] == X['R_emaildomain']).astype(int)
        X['P_email_is_free'] = X['P_emaildomain'].isin(self.FREE_EMAILS).astype(int)
        X['R_email_is_free'] = X['R_emaildomain'].isin(self.FREE_EMAILS).astype(int)
        X['P_email_missing'] = X['P_emaildomain'].isna().astype(int)
        X['R_email_missing'] = X['R_emaildomain'].isna().astype(int)

        # New: High-risk domain flags (from EDA insight)
        X['P_email_high_risk'] = X['P_emaildomain'].isin(self.HIGH_RISK_P_DOMAINS).astype(int)
        X['R_email_high_risk'] = X['R_emaildomain'].isin(self.HIGH_RISK_R_DOMAINS).astype(int)

        return X


class CardFeatures(BaseEstimator, TransformerMixin):
    """Add card and device features.

    EDA Insights (01_eda.ipynb, Section 12 & Insights 6, 8):
    - D1 represents "days since card was first used"
    - New cards (D1 < 7 days) have elevated fraud rates (cell 54)
    - Fraud rate by D1 bucket shows clear pattern:
      0-7d: highest, then decreases with card age
    - Mobile devices show 10.2% fraud vs 6.5% desktop (1.6x higher)
    - Transactions WITH identity data have 7.8% fraud (vs 2.1% without)
    """

    # Card age buckets from EDA cell 54
    D1_BINS = [-1, 0, 7, 30, 90, 365, float('inf')]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Original features
        X['is_new_card'] = (X['D1'] <= 7).astype(int)
        X['has_identity'] = X['DeviceType'].notna().astype(int)
        X['is_mobile'] = (X['DeviceType'] == 'mobile').astype(int)

        # New: More granular card age buckets (from EDA)
        # Buckets: 0 (same day), 1-7d, 8-30d, 31-90d, 91-365d, 365d+
        X['card_age_bucket'] = np.digitize(
            X['D1'].fillna(-1),  # Missing D1 gets bucket 0
            bins=self.D1_BINS
        ) - 1

        return X


class DeviceFeatures(BaseEstimator, TransformerMixin):
    """Extract OS/browser features from DeviceInfo.

    EDA Insights (01_eda.ipynb, Section 9 & cell 45):
    - DeviceInfo contains mixed device/browser/OS information
    - Windows: 47,722 txns, 6.5% fraud rate
    - iOS Device: 19,782 txns, 6.3% fraud rate
    - MacOS: 12,573 txns, 2.2% fraud rate (below average)
    - SM-J700M (Android): 549 txns, 10.9% fraud rate (high risk)

    Parsing DeviceInfo to extract OS provides cleaner signal than
    raw high-cardinality DeviceInfo string.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        device = X['DeviceInfo'].fillna('unknown').astype(str).str.lower()

        # OS detection (from EDA device analysis)
        X['is_windows'] = device.str.contains('windows').astype(int)
        X['is_macos'] = device.str.contains('macos|mac os').astype(int)
        X['is_ios'] = device.str.contains('ios').astype(int)
        X['is_android'] = device.str.contains('android|sm-|samsung|lg-|moto').astype(int)
        X['is_linux'] = device.str.contains('linux').astype(int)

        # Browser detection (common patterns in DeviceInfo)
        X['is_chrome'] = device.str.contains('chrome').astype(int)
        X['is_safari'] = device.str.contains('safari').astype(int)
        X['is_firefox'] = device.str.contains('firefox|rv:').astype(int)
        X['is_edge'] = device.str.contains('edge|trident').astype(int)

        return X


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Create interaction features between categorical variables.

    EDA Insights (01_eda.ipynb, Sections 6-7):
    - card4 (network): Discover has 7.7% fraud vs ~3% for others (Insight 4)
    - ProductCD: Product C has highest fraud rate (Insight 5)
    - DeviceType: Mobile 10.2% vs Desktop 6.5% (Insight 6)

    Interaction rationale:
    - Fraud patterns may differ by card_network × product combination
    - A Discover card buying Product C on mobile might have very different
      risk than Visa buying Product W on desktop
    - String concatenation creates new categorical for tree models to split on
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Card network × Product interactions
        X['card4_ProductCD'] = (
            X['card4'].fillna('unk').astype(str) + '_' +
            X['ProductCD'].fillna('unk').astype(str)
        )
        X['card6_ProductCD'] = (
            X['card6'].fillna('unk').astype(str) + '_' +
            X['ProductCD'].fillna('unk').astype(str)
        )

        # Device × Product interactions
        X['DeviceType_ProductCD'] = (
            X['DeviceType'].fillna('unk').astype(str) + '_' +
            X['ProductCD'].fillna('unk').astype(str)
        )

        # Card network × Device interactions
        X['card4_DeviceType'] = (
            X['card4'].fillna('unk').astype(str) + '_' +
            X['DeviceType'].fillna('unk').astype(str)
        )

        return X


class AmountFeatures(BaseEstimator, TransformerMixin):
    """Add transaction amount features.

    EDA Insights (01_eda.ipynb, Section 4 & Insight 2):
    - Fraud rate increases with amount: 2.4% ($0-25) → 5.5% ($500-1000)
    - BUT drops back to 3.4% for $1000+ (non-monotonic relationship)
    - Mean fraud amount ($149) slightly higher than legitimate ($135)
    - Explicit buckets help tree models find these non-linear patterns

    Cents patterns (fraud detection best practice):
    - Fraudsters often use round amounts or specific cent values
    - .00 cents (round amounts) and .99 cents (fake pricing) are signals
    """

    # Buckets from EDA cell 18 fraud rate analysis
    AMT_BINS = [0, 25, 50, 100, 200, 500, 1000, float('inf')]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Original features
        X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])

        # Amount bucket (from EDA - non-monotonic fraud rate by amount)
        X['amt_bucket'] = np.digitize(X['TransactionAmt'], bins=self.AMT_BINS) - 1

        # Cents analysis (fraud pattern detection)
        cents = (X['TransactionAmt'] * 100 % 100).astype(int)
        X['cents'] = cents
        X['cents_00'] = (cents == 0).astype(int)  # Round amounts
        X['cents_99'] = (cents == 99).astype(int)  # Fake pricing pattern
        X['cents_95'] = (cents == 95).astype(int)  # Common pricing pattern

        # Keep is_round as it's more interpretable than cents_00
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


class VFeatureBlocks(BaseEstimator, TransformerMixin):
    """Identify which V feature blocks are missing.

    EDA Insight (01_eda.ipynb, Insight 7):
    - 339 V features with highly correlated missing patterns
    - Missing value co-occurrence heatmap shows clear block structure
    - This suggests V features were collected in groups by Vesta
    - Knowing WHICH block is missing may be more predictive than just count

    The block boundaries were identified from the missing correlation heatmap
    in cell 49 of the EDA notebook.
    """

    # Blocks identified from missing correlation heatmap in EDA
    V_BLOCKS = {
        'V1_11': [f'V{i}' for i in range(1, 12)],
        'V12_34': [f'V{i}' for i in range(12, 35)],
        'V35_52': [f'V{i}' for i in range(35, 53)],
        'V53_74': [f'V{i}' for i in range(53, 75)],
        'V75_94': [f'V{i}' for i in range(75, 95)],
        'V95_137': [f'V{i}' for i in range(95, 138)],
        'V138_166': [f'V{i}' for i in range(138, 167)],
        'V167_216': [f'V{i}' for i in range(167, 217)],
        'V217_278': [f'V{i}' for i in range(217, 279)],
        'V279_339': [f'V{i}' for i in range(279, 340)],
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for block_name, cols in self.V_BLOCKS.items():
            present = [c for c in cols if c in X.columns]
            if present:
                # Block is "missing" if ALL columns in it are null
                X[f'{block_name}_missing'] = X[present].isna().all(axis=1).astype(int)
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
