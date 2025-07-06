import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
import warnings

warnings.filterwarnings('ignore')


def quick_analysis():
    """Quick EDA analysis"""
    print("🔬 QUICK HEALTHCARE DATA ANALYSIS")
    print("=" * 40)

    # Load data
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Copy%20of%20HealthCareData-S5E2HCICgABoVLci9LAnnZpYdxG6w4.csv"

    try:
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text))
        print(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Basic info
    print(f"\n📊 Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Show first few rows
    print(f"\n📋 First 5 rows:")
    print(df.head())

    # Data types
    print(f"\n📊 Data types:")
    print(df.dtypes.value_counts())

    # Basic statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 Basic statistics for numerical columns:")
        print(df[numeric_cols].describe())

    print("\n✅ Quick analysis complete!")


if __name__ == "__main__":
    quick_analysis()
