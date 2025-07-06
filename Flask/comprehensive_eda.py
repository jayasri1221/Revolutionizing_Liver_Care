import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import requests
from io import StringIO

warnings.filterwarnings('ignore')


plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def load_data_from_url():
    """Load data from the provided URL"""
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Copy%20of%20HealthCareData-S5E2HCICgABoVLci9LAnnZpYdxG6w4.csv"

    try:
        response = requests.get(url)
        response.raise_for_status()


        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)

        print("✅ Data loaded successfully from URL")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def basic_data_info(df):
    """Display basic information about the dataset"""
    print("🔬 HEALTHCARE DATA ANALYSIS - COMPREHENSIVE EDA")
    print("=" * 60)

    print(f"📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"📋 Memory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    print("\n📋 Column Information:")
    print("-" * 40)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")

    print(f"\n📊 Data Types:")
    print(df.dtypes.value_counts())

    print(f"\n🔍 Missing Values:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Percentage': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

    if not missing_df.empty:
        print(missing_df)
    else:
        print("No missing values found!")

    return missing_df


def clean_and_preprocess_data(df):
    """Clean and preprocess the healthcare data"""
    print("\n🧹 DATA CLEANING AND PREPROCESSING")
    print("=" * 50)


    df_clean = df.copy()


    df_clean.columns = df_clean.columns.str.strip()


    print("🔧 Cleaning specific columns...")


    if 'A/G Ratio' in df_clean.columns:
        df_clean['A/G Ratio'] = (
            df_clean['A/G Ratio']
            .astype(str)
            .str.extract(r'([\d.]+)')[0]
            .astype(str)
            .str.rstrip('.')
        )
        df_clean['A/G Ratio'] = pd.to_numeric(df_clean['A/G Ratio'], errors='coerce')


    if 'Blood pressure (mmhg)' in df_clean.columns:
        # Extract systolic and diastolic pressure
        bp_split = df_clean['Blood pressure (mmhg)'].str.split('/', expand=True)
        if bp_split.shape[1] >= 2:
            df_clean['Systolic_BP'] = pd.to_numeric(bp_split[0], errors='coerce')
            df_clean['Diastolic_BP'] = pd.to_numeric(bp_split[1], errors='coerce')


    categorical_mappings = {
        'positive': 1, 'negative': 0,
        'yes': 1, 'no': 0,
        'diffuse liver': 1, 'normal': 0,
        'male': 1, 'female': 0,
        'rural': 0, 'urban': 1
    }

    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':

            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

            df_clean[col] = df_clean[col].map(categorical_mappings).fillna(df_clean[col])


    numeric_columns = []
    categorical_columns = []

    for col in df_clean.columns:
        if col not in ['S.NO']:

            original_col = df_clean[col].copy()
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')


            if df_clean[col].isnull().sum() > len(df_clean) * 0.8:
                df_clean[col] = original_col
                categorical_columns.append(col)
            else:
                numeric_columns.append(col)

    print(f"✅ Identified {len(numeric_columns)} numeric columns")
    print(f"✅ Identified {len(categorical_columns)} categorical columns")

    return df_clean, numeric_columns, categorical_columns


def descriptive_statistics(df, numeric_columns, categorical_columns):
    """Activity 1: Comprehensive Descriptive Statistical Analysis"""
    print("\n📈 ACTIVITY 1: DESCRIPTIVE STATISTICAL ANALYSIS")
    print("=" * 60)


    if numeric_columns:
        print("\n📊 Numerical Features - Descriptive Statistics:")
        print("-" * 50)
        desc_stats = df[numeric_columns].describe()
        print(desc_stats)


        print("\n📊 Additional Statistical Measures:")
        print("-" * 40)
        additional_stats = pd.DataFrame({
            'Skewness': df[numeric_columns].skew(),
            'Kurtosis': df[numeric_columns].kurtosis(),
            'Variance': df[numeric_columns].var()
        })
        print(additional_stats)


    if categorical_columns:
        print("\n📝 Categorical Features - Descriptive Statistics:")
        print("-" * 50)
        for col in categorical_columns[:10]:  # Show first 10 categorical columns
            if col in df.columns:
                print(f"\n🔍 {col}:")
                print(f"   Unique values: {df[col].nunique()}")
                print(f"   Most frequent: {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}")
                print(f"   Value counts:")
                print(df[col].value_counts().head())

    return desc_stats if numeric_columns else None


def univariate_analysis(df, numeric_columns, categorical_columns):
    """Activity 2.1: Univariate Analysis with Fixed Visualizations"""
    print("\n📊 ACTIVITY 2.1: UNIVARIATE ANALYSIS")
    print("=" * 50)


    if numeric_columns:
        print("📈 Creating distribution plots for numerical features...")


        valid_numeric_cols = []
        for col in numeric_columns[:12]:
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 10 and col_data.var() > 0:
                    valid_numeric_cols.append(col)

        if valid_numeric_cols:
            n_cols = 3
            n_rows = (len(valid_numeric_cols) + n_cols - 1) // n_cols


            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
            fig.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold', y=0.98)


            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for i, col in enumerate(valid_numeric_cols):
                if i < len(axes):
                    ax = axes[i]


                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        sns.histplot(data=df, x=col, kde=True, ax=ax, alpha=0.7)


                        mean_val = col_data.mean()
                        median_val = col_data.median()

                        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8,
                                   label=f'Mean: {mean_val:.2f}')
                        ax.axvline(median_val, color='green', linestyle='--', alpha=0.8,
                                   label=f'Median: {median_val:.2f}')

                        ax.set_title(f'Distribution of {col}', fontsize=11, pad=10)
                        ax.legend(fontsize=8)
                        ax.tick_params(axis='x', rotation=45, labelsize=8)
                        ax.tick_params(axis='y', labelsize=8)


            for i in range(len(valid_numeric_cols), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)
            plt.show()


            print("📊 Creating box plots for numerical features...")

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
            fig.suptitle('Box Plots of Numerical Features', fontsize=16, fontweight='bold', y=0.98)


            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for i, col in enumerate(valid_numeric_cols):
                if i < len(axes):
                    ax = axes[i]

                    try:

                        col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()

                        if len(col_data) > 0 and col_data.var() > 0:
                            sns.boxplot(y=col_data, ax=ax)
                            ax.set_title(f'Box Plot of {col}', fontsize=11, pad=10)
                            ax.tick_params(axis='y', labelsize=8)
                        else:
                            ax.text(0.5, 0.5, f'No valid data\nfor {col}',
                                    ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(f'Box Plot of {col}', fontsize=11, pad=10)

                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error plotting\n{col}',
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'Box Plot of {col}', fontsize=11, pad=10)


            for i in range(len(valid_numeric_cols), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)
            plt.show()


    if categorical_columns:
        print("📊 Creating count plots for categorical features...")


        valid_cat_cols = []
        for col in categorical_columns[:8]:
            if col in df.columns:
                unique_count = df[col].nunique()
                if 1 < unique_count <= 20:
                    valid_cat_cols.append(col)

        if valid_cat_cols:
            n_cols = 2
            n_rows = (len(valid_cat_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
            fig.suptitle('Count Plots for Categorical Features', fontsize=16, fontweight='bold', y=0.98)


            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for i, col in enumerate(valid_cat_cols):
                if i < len(axes):
                    ax = axes[i]

                    try:

                        sns.countplot(data=df, x=col, ax=ax)
                        ax.set_title(f'Count of {col}', fontsize=11, pad=10)
                        ax.tick_params(axis='x', rotation=45, labelsize=8)
                        ax.tick_params(axis='y', labelsize=8)


                        total = len(df[col].dropna())
                        if total > 0:
                            for p in ax.patches:
                                if p.get_height() > 0:
                                    percentage = f'{100 * p.get_height() / total:.1f}%'
                                    ax.annotate(percentage,
                                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                                ha='center', va='bottom', fontsize=8)

                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error plotting\n{col}',
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'Count of {col}', fontsize=11, pad=10)


            for i in range(len(valid_cat_cols), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)
            plt.show()


def bivariate_analysis(df, numeric_columns):
    """Activity 2.2: Bivariate Analysis"""
    print("\n📈 ACTIVITY 2.2: BIVARIATE ANALYSIS")
    print("=" * 50)

    if len(numeric_columns) < 2:
        print("❌ Not enough numerical columns for bivariate analysis")
        return {}, {}


    valid_numeric_cols = []
    for col in numeric_columns:
        if col in df.columns:
            col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(col_data) > 10 and col_data.var() > 0:
                valid_numeric_cols.append(col)

    if len(valid_numeric_cols) < 2:
        print("❌ Not enough valid numerical columns for bivariate analysis")
        return {}, {}


    print("🔗 Correlation Analysis:")
    correlation_matrix = df[valid_numeric_cols].corr()


    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if not np.isnan(corr_val) and abs(corr_val) > 0.5:
                high_corr_pairs.append({
                    'Feature 1': correlation_matrix.columns[i],
                    'Feature 2': correlation_matrix.columns[j],
                    'Correlation': corr_val
                })

    if high_corr_pairs:
        print("\n🔗 Highly Correlated Feature Pairs (|correlation| > 0.5):")
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x['Correlation']), reverse=True):
            print(f"   {pair['Feature 1']} ↔ {pair['Feature 2']}: {pair['Correlation']:.3f}")


    if high_corr_pairs:
        print("\n📊 Creating scatter plots for highly correlated pairs...")
        top_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['Correlation']), reverse=True)[:6]

        n_cols = 3
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
        fig.suptitle('Scatter Plots of Highly Correlated Features', fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for i, pair in enumerate(top_pairs):
            if i < len(axes):
                ax = axes[i]
                x_col, y_col = pair['Feature 1'], pair['Feature 2']

                try:

                    plot_data = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()

                    if len(plot_data) > 0:
                        sns.scatterplot(data=plot_data, x=x_col, y=y_col, ax=ax, alpha=0.6)
                        ax.set_title(f'{x_col} vs {y_col}\nCorr: {pair["Correlation"]:.3f}', fontsize=10)


                        sns.regplot(data=plot_data, x=x_col, y=y_col, ax=ax, scatter=False, color='red')
                        ax.tick_params(axis='both', labelsize=8)
                    else:
                        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{x_col} vs {y_col}', fontsize=10)

                except Exception as e:
                    ax.text(0.5, 0.5, f'Error: {str(e)[:20]}...', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{x_col} vs {y_col}', fontsize=10)


        for i in range(len(top_pairs), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()


    print("\n🔍 Outlier Detection using IQR Method:")
    outlier_summary = {}

    for col in valid_numeric_cols[:10]:
        if col in df.columns:
            col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()

            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1

                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_percentage = (outlier_count / len(col_data)) * 100

                    if outlier_count > 0:
                        outlier_summary[col] = {
                            'count': outlier_count,
                            'percentage': outlier_percentage,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        }
                        print(f"   {col}: {outlier_count} outliers ({outlier_percentage:.1f}%)")

    return correlation_matrix, outlier_summary


def multivariate_analysis(df, numeric_columns):
    """Activity 2.3: Multivariate Analysis"""
    print("\n🔗 ACTIVITY 2.3: MULTIVARIATE ANALYSIS")
    print("=" * 50)


    valid_numeric_cols = []
    for col in numeric_columns:
        if col in df.columns:
            col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(col_data) > 10 and col_data.var() > 0:
                valid_numeric_cols.append(col)

    if len(valid_numeric_cols) < 3:
        print("❌ Not enough valid numerical columns for multivariate analysis")
        return


    print("🌡️ Creating correlation heatmap...")
    correlation_matrix = df[valid_numeric_cols].corr()

    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8},
                mask=mask,
                fmt='.2f',
                annot_kws={'size': 8})

    plt.title('Correlation Heatmap - Multivariate Analysis', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.show()


    if len(valid_numeric_cols) >= 4:
        print("📊 Creating pairplot for selected features...")
        selected_features = valid_numeric_cols[:4]  # Select first 4 features


        pairplot_data = df[selected_features].replace([np.inf, -np.inf], np.nan).dropna()

        if len(pairplot_data) > 10:
            try:
                g = sns.pairplot(pairplot_data, diag_kind='hist', plot_kws={'alpha': 0.6, 's': 20})
                g.fig.suptitle('Pairplot of Selected Numerical Features', y=1.02, fontsize=16, fontweight='bold')
                plt.show()
            except Exception as e:
                print(f"⚠️ Could not create pairplot: {e}")


def target_variable_analysis(df):
    """Analyze the target variable and its relationships"""
    print("\n🎯 TARGET VARIABLE ANALYSIS")
    print("=" * 50)


    target_candidates = [col for col in df.columns if any(keyword in col.lower()
                                                          for keyword in
                                                          ['target', 'outcome', 'predict', 'result', 'cirrhosis'])]

    if not target_candidates:
        print("❌ No target variable found")
        return None

    target_col = target_candidates[0]
    print(f"🎯 Target variable identified: {target_col}")


    print(f"\n📊 Target Variable Distribution:")
    target_counts = df[target_col].value_counts()
    print(target_counts)


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))


    target_counts.plot(kind='bar', ax=axes[0])
    axes[0].set_title(f'Distribution of {target_col}')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)


    target_counts.plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
    axes[1].set_title(f'Proportion of {target_col}')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.show()

    return target_col


def statistical_tests(df, numeric_columns, target_col=None):
    """Perform statistical tests"""
    print("\n📊 STATISTICAL TESTS")
    print("=" * 50)


    valid_numeric_cols = []
    for col in numeric_columns[:5]:
        if col in df.columns:
            col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(col_data) > 10:
                valid_numeric_cols.append(col)


    print("🔍 Normality Tests (Shapiro-Wilk):")
    print("-" * 40)

    for col in valid_numeric_cols:
        data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) > 3 and len(data) <= 5000:
            try:
                stat, p_value = stats.shapiro(data[:5000])
                is_normal = p_value > 0.05
                print(f"   {col}: p-value = {p_value:.4f} {'(Normal)' if is_normal else '(Not Normal)'}")
            except Exception as e:
                print(f"   {col}: Could not perform test - {e}")


def data_splitting_analysis(df, numeric_columns, target_col=None):
    """Data splitting for machine learning"""
    print("\n🎯 DATA SPLITTING FOR MACHINE LEARNING")
    print("=" * 50)

    if not target_col:
        print("❌ No target variable available for splitting")
        return


    valid_numeric_cols = []
    for col in numeric_columns:
        if col in df.columns and col != target_col and col != 'S.NO':
            col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(col_data) > 10:
                valid_numeric_cols.append(col)

    if len(valid_numeric_cols) == 0:
        print("❌ No suitable features found for machine learning")
        return


    X = df[valid_numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(df[valid_numeric_cols].median())
    y = df[target_col].fillna(df[target_col].mode().iloc[0] if not df[target_col].mode().empty else 0)

    print(f"📊 Features selected: {len(valid_numeric_cols)}")
    print(f"🎯 Target variable: {target_col}")
    print(f"📈 Dataset size: {len(X)} samples")


    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        print(f"\n📊 Data Split Results:")
        print(f"   Training set: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)")
        print(f"   Testing set: {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)")
        print(f"   Features: {X_train.shape[1]}")


        print(f"\n🔧 Feature Correlation with Target:")
        feature_target_corr = []
        for feature in valid_numeric_cols:
            if feature in df.columns:
                corr = df[feature].corr(df[target_col])
                if not np.isnan(corr):
                    feature_target_corr.append((feature, abs(corr)))


        feature_target_corr.sort(key=lambda x: x[1], reverse=True)

        print("   Top 10 features by correlation with target:")
        for i, (feature, corr) in enumerate(feature_target_corr[:10], 1):
            print(f"   {i:2d}. {feature}: {corr:.3f}")

        return X_train, X_test, y_train, y_test, valid_numeric_cols

    except Exception as e:
        print(f"❌ Error in data splitting: {e}")
        return None


def generate_summary_report(df, numeric_columns, categorical_columns, outlier_summary, target_col):
    """Generate a comprehensive summary report"""
    print("\n📋 COMPREHENSIVE ANALYSIS SUMMARY REPORT")
    print("=" * 60)

    print(f"📊 Dataset Overview:")
    print(f"   • Total samples: {len(df):,}")
    print(f"   • Total features: {len(df.columns)}")
    print(f"   • Numerical features: {len(numeric_columns)}")
    print(f"   • Categorical features: {len(categorical_columns)}")
    print(f"   • Target variable: {target_col if target_col else 'Not identified'}")

    print(f"\n🔍 Data Quality Assessment:")
    missing_count = df.isnull().sum().sum()
    print(f"   • Missing values: {missing_count:,} ({missing_count / df.size * 100:.2f}%)")
    print(f"   • Complete cases: {len(df.dropna()):,} ({len(df.dropna()) / len(df) * 100:.1f}%)")

    if outlier_summary:
        print(f"\n⚠️ Outlier Summary:")
        total_outliers = sum([info['count'] for info in outlier_summary.values()])
        print(f"   • Features with outliers: {len(outlier_summary)}")
        print(f"   • Total outlier instances: {total_outliers:,}")

        print("   • Top features with outliers:")
        sorted_outliers = sorted(outlier_summary.items(), key=lambda x: x[1]['count'], reverse=True)
        for feature, info in sorted_outliers[:5]:
            print(f"     - {feature}: {info['count']} outliers ({info['percentage']:.1f}%)")

    print(f"\n✅ Analysis completed successfully!")
    print(f"   • All statistical analyses performed")
    print(f"   • Visualizations generated with proper formatting")
    print(f"   • Data ready for machine learning")


def main():
    """Main function to run the comprehensive EDA"""
    print("🚀 Starting Comprehensive Healthcare Data Analysis...")


    df = load_data_from_url()
    if df is None:
        return


    missing_df = basic_data_info(df)


    df_clean, numeric_columns, categorical_columns = clean_and_preprocess_data(df)


    desc_stats = descriptive_statistics(df_clean, numeric_columns, categorical_columns)


    univariate_analysis(df_clean, numeric_columns, categorical_columns)


    correlation_matrix, outlier_summary = bivariate_analysis(df_clean, numeric_columns)


    multivariate_analysis(df_clean, numeric_columns)


    target_col = target_variable_analysis(df_clean)

    statistical_tests(df_clean, numeric_columns, target_col)


    split_results = data_splitting_analysis(df_clean, numeric_columns, target_col)

    generate_summary_report(df_clean, numeric_columns, categorical_columns, outlier_summary, target_col)

    print("\n🎉 Comprehensive EDA Analysis Complete!")
    print("All visualizations have been displayed with proper formatting and spacing.")


if __name__ == "__main__":
    main()
