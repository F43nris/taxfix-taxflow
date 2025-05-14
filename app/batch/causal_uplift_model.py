import os
import shutil
import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Create results directory dynamically
results_dir = os.path.join(os.getenv('HOME'), 'taxfix', 'taxfix-taxflow', 'app', 'data', 'processed', 'tax_uplift_results')
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)  # Remove if exists
os.makedirs(results_dir)
os.makedirs(os.path.join(results_dir, 'uplift_curves'), exist_ok=True)

# Load your data dynamically
df = pd.read_csv(os.path.join(os.getenv('HOME'), 'taxfix', 'taxfix-taxflow', 'notebooks', 'processed_data', 'full_joined_data.csv'))

# Calculate derived measures
df['refund_per_income'] = df['refund_amount'] / df['total_income']
df['deduction_ratio'] = df['total_deductions'] / df['total_income']

# Generate category statistics to use throughout analysis
# Aggregate at user-category level to avoid double-counting refunds
user_cat = df.groupby(['user_id', 'category']).agg({
    'amount': 'sum',
    'refund_amount': 'first'  # refund_amount is user-level, so take first (or mean)
}).reset_index()

# Now, for each category, sum the amounts and refunds (refunds only counted once per user)
category_stats = user_cat.groupby('category').agg({
    'amount': 'sum',
    'refund_amount': 'sum'
}).reset_index()

# Calculate derived metrics - show actual tax benefit ratio (no capping)
category_stats['tax_benefit_ratio'] = (category_stats['refund_amount'] / category_stats['amount']) * 100
category_stats['spending_share'] = category_stats['amount'] / category_stats['amount'].sum()

# Save category statistics
category_stats.to_csv(os.path.join(results_dir, 'category_statistics.csv'), index=False)
print(f"Saved category statistics to {os.path.join(results_dir, 'category_statistics.csv')}")

# Define confidence levels based on sample size
def get_confidence_level(sample_size):
    if sample_size < 5:
        return "Very Low"
    elif sample_size < 10:
        return "Low"
    elif sample_size < 30:
        return "Medium"
    else:
        return "High"

# Calculate user count per category for confidence level
user_counts = user_cat.groupby('category')['user_id'].nunique().reset_index()
user_counts.columns = ['category', 'user_count']
category_stats = category_stats.merge(user_counts, on='category', how='left')
category_stats['confidence_level'] = category_stats['user_count'].apply(get_confidence_level)

# Function to prepare user-year level data
def prepare_user_year_data(df):
    """Aggregate transaction data to user-year level and create treatment indicators"""
    
    # Get all unique categories
    categories = df['category'].unique()
    print(f"Found {len(categories)} unique categories")
    
    # Create user-year level dataset
    user_year_df = df[['user_id', 'tax_year', 'total_income', 'total_deductions', 
                       'refund_amount', 'occupation_category']].drop_duplicates()
    
    # Add derived outcome measures
    user_year_df['refund_per_income'] = user_year_df['refund_amount'] / user_year_df['total_income']
    user_year_df['deduction_ratio'] = user_year_df['total_deductions'] / user_year_df['total_income']
    
    # Create treatment indicators for each category
    for category in categories:
        # For each user-year, check if they have transactions in this category
        cat_txns = df[df['category'] == category].groupby(['user_id', 'tax_year']).size().reset_index()
        cat_txns.columns = ['user_id', 'tax_year', 'txn_count']
        
        # Create treatment indicator (1 if user has ≥1 transaction in category)
        cat_txns['has_txn'] = 1
        
        # Merge with user-year data
        user_year_df = user_year_df.merge(
            cat_txns[['user_id', 'tax_year', 'has_txn']], 
            on=['user_id', 'tax_year'], 
            how='left'
        )
        
        # Fill NAs with 0 (users without transactions in this category)
        cat_column = f'has_txn_{category.replace(" ", "_").lower()}'
        user_year_df[cat_column] = user_year_df['has_txn'].fillna(0).astype(int)
        user_year_df = user_year_df.drop('has_txn', axis=1)
        
        # Also add spending amount in each category
        cat_amount = df[df['category'] == category].groupby(['user_id', 'tax_year'])['amount'].sum().reset_index()
        cat_amount.columns = ['user_id', 'tax_year', 'amount']
        
        # Merge with user-year data
        user_year_df = user_year_df.merge(
            cat_amount[['user_id', 'tax_year', 'amount']], 
            on=['user_id', 'tax_year'], 
            how='left'
        )
        
        # Fill NAs with 0 (users without transactions in this category)
        amount_column = f'spent_{category.replace(" ", "_").lower()}'
        user_year_df[amount_column] = user_year_df['amount'].fillna(0)
        user_year_df = user_year_df.drop('amount', axis=1)
    
    return user_year_df, categories

# Function to calculate Qini coefficient (uplift AUC)
def calculate_qini(uplift, treatment, outcome):
    """Calculate Qini coefficient (measure of uplift model quality)"""
    df_qini = pd.DataFrame({'uplift': uplift, 'treatment': treatment, 'outcome': outcome})
    df_qini = df_qini.sort_values('uplift', ascending=False).reset_index(drop=True)
    
    # Calculate cumulative metrics
    df_qini['n_treat'] = df_qini['treatment'].cumsum()
    df_qini['n_ctrl'] = (1 - df_qini['treatment']).cumsum()
    df_qini['y_treat'] = (df_qini['treatment'] * df_qini['outcome']).cumsum()
    df_qini['y_ctrl'] = ((1 - df_qini['treatment']) * df_qini['outcome']).cumsum()
    
    # Calculate uplift curve points
    n_all = len(df_qini)
    df_qini['x'] = np.arange(1, n_all + 1) / n_all
    
    # Avoid division by zero
    df_qini['uplift_curve'] = np.where(
        df_qini['n_treat'] > 0, 
        (df_qini['y_treat'] / df_qini['n_treat'] - 
         df_qini['y_ctrl'] / np.maximum(df_qini['n_ctrl'], 1)) * df_qini['n_treat'],
        0
    )
    
    # Random policy line (straight line from origin to end point)
    y_final = df_qini['uplift_curve'].iloc[-1]
    random_line = df_qini['x'] * y_final
    
    # Calculate Qini coefficient (AUC above random)
    qini = np.trapz(df_qini['uplift_curve']) - np.trapz(random_line)
    return qini, df_qini

# Prepare features for modeling - REMOVED age_range, family_status, and region
def prepare_features(df):
    """Extract features for causal modeling"""
    # Select numeric features and one-hot encode categorical variables - REMOVED age_range, family_status, and region
    feature_df = pd.get_dummies(
        df[['occupation_category', 'total_income']],
        columns=['occupation_category'],
        drop_first=True
    )
    return feature_df

# Main function to run causal uplift modeling - simplified approach
def run_causal_uplift_analysis(df, outcome_var='refund_per_income'):
    """Run causal uplift analysis for each category, regardless of sample size"""
    
    # Prepare user-year level data
    user_year_df, categories = prepare_user_year_data(df)
    
    # Prepare features
    X = prepare_features(user_year_df)
    
    # Store results
    results = []
    
    # For each category
    for category in categories:
        cat_safe = category.replace(" ", "_").lower()
        cat_col = f'has_txn_{cat_safe}'
        
        # Get treatment count
        treatment_count = user_year_df[cat_col].sum()
        confidence_level = get_confidence_level(treatment_count)
        
        print(f"Processing category: {category} (n={treatment_count}, confidence={confidence_level})")
        
        # Prepare data for this category
        T = user_year_df[cat_col]  # Treatment: has transaction in category
        Y = user_year_df[outcome_var]  # Outcome: refund per income
        
        try:
            # Adapt model parameters based on sample size for better stability
            if treatment_count < 5:
                # For very small samples, use more regularization and simpler models
                model = CausalForestDML(
                    model_y=LassoCV(cv=3),  # More regularized model for outcome
                    model_t=LassoCV(cv=3),  # More regularized model for treatment
                    n_estimators=100,  # Fewer trees for stability
                    min_samples_leaf=1,  # Allow leaf nodes with just 1 sample
                    max_features=0.5,  # More feature subsampling for regularization
                    verbose=0,
                    random_state=42
                )
            else:
                # Standard model for larger samples
                model = CausalForestDML(
                    model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=2),
                    model_t=RandomForestRegressor(n_estimators=100, min_samples_leaf=2),
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features=0.8,
                    verbose=0,
                    random_state=42
                )
            
            # Fit the model
            model.fit(Y, T, X=X)
            
            # Calculate treatment effects
            te_pred = model.effect(X)
            
            # Calculate confidence intervals
            lower, upper = model.effect_interval(X, alpha=0.1)  # Use 90% CI for more stability
            
            # Calculate Qini coefficient (uplift AUC)
            qini_coef, qini_data = calculate_qini(te_pred, T, Y)
            
            # Get mean uplift effect and confidence interval
            mean_effect = np.mean(te_pred)
            mean_lower = np.mean(lower)
            mean_upper = np.mean(upper)
            
            print(f"  Average effect: {mean_effect:.4f} (90% CI: {mean_lower:.4f} to {mean_upper:.4f})")
            print(f"  Qini coefficient: {qini_coef:.4f}")
            
            # Create results for this category
            cat_results = pd.DataFrame({
                'user_id': user_year_df['user_id'],
                'tax_year': user_year_df['tax_year'],
                f'uplift_{cat_safe}_pct': te_pred,
                f'ci90_low_{cat_safe}': lower,
                f'ci90_high_{cat_safe}': upper
            })
            
            # Add category info and confidence level
            cat_results['category'] = category
            cat_results['qini_coefficient'] = qini_coef
            cat_results['treatment_count'] = treatment_count
            cat_results['confidence_level'] = confidence_level
            cat_results['mean_effect'] = mean_effect
            cat_results['mean_lower_ci'] = mean_lower
            cat_results['mean_upper_ci'] = mean_upper
            
            results.append(cat_results)
            
            # Plot uplift curve
            plt.figure(figsize=(10, 6))
            plt.plot(qini_data['x'], qini_data['uplift_curve'], label='Uplift curve')
            plt.plot(qini_data['x'], qini_data['x'] * qini_data['uplift_curve'].iloc[-1], 
                    '--', label='Random policy')
            plt.xlabel('Fraction of population targeted')
            plt.ylabel('Cumulative uplift')
            plt.title(f'Uplift curve for {category} (n={treatment_count}, Qini={qini_coef:.4f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            curve_path = os.path.join(results_dir, 'uplift_curves', f'uplift_curve_{cat_safe}.png')
            plt.savefig(curve_path)
            plt.close()
            print(f"  Saved uplift curve to {curve_path}")
            
        except Exception as e:
            print(f"  Error analyzing {category}: {str(e)}")
            # Create a placeholder result with NaNs
            cat_results = pd.DataFrame({
                'user_id': user_year_df['user_id'].unique()[0:1],
                'tax_year': user_year_df[user_year_df['user_id'] == user_year_df['user_id'].unique()[0]]['tax_year'].values[0:1],
                'category': category,
                'qini_coefficient': np.nan,
                'treatment_count': treatment_count,
                'confidence_level': confidence_level,
                'mean_effect': np.nan,
                'mean_lower_ci': np.nan,
                'mean_upper_ci': np.nan,
                f'uplift_{cat_safe}_pct': [np.nan],
                f'ci90_low_{cat_safe}': [np.nan],
                f'ci90_high_{cat_safe}': [np.nan]
            })
            results.append(cat_results)
    
    # Save category performance summary
    if results:
        # Create comprehensive category summary
        cat_summary = pd.DataFrame([
            {
                'category': r['category'].iloc[0], 
                'treatment_count': r['treatment_count'].iloc[0],
                'confidence_level': r['confidence_level'].iloc[0],
                'qini_coefficient': r['qini_coefficient'].iloc[0] if not pd.isna(r['qini_coefficient'].iloc[0]) else 0,
                'mean_effect': r['mean_effect'].iloc[0] if not pd.isna(r['mean_effect'].iloc[0]) else 0,
                'mean_lower_ci': r['mean_lower_ci'].iloc[0] if not pd.isna(r['mean_lower_ci'].iloc[0]) else 0,
                'mean_upper_ci': r['mean_upper_ci'].iloc[0] if not pd.isna(r['mean_upper_ci'].iloc[0]) else 0
            } 
            for r in results
        ])
        # Sort by confidence level first, then by mean effect
        confidence_order = {"High": 0, "Medium": 1, "Low": 2, "Very Low": 3}
        cat_summary['confidence_order'] = cat_summary['confidence_level'].map(confidence_order)
        cat_summary = cat_summary.sort_values(['confidence_order', 'mean_effect'], ascending=[True, False])
        cat_summary = cat_summary.drop('confidence_order', axis=1)
        cat_summary.to_csv(os.path.join(results_dir, 'category_performance.csv'), index=False)
        
    # Combine all results
    if results:
        all_results = pd.concat(results)
    else:
        print("WARNING: No valid results generated")
        all_results = pd.DataFrame(columns=['user_id', 'tax_year', 'category', 'qini_coefficient'])
    
    # Prepare combined user-year results
    user_year_results = user_year_df[['user_id', 'tax_year']].copy()
    
    # Add uplift results for each category
    for category in categories:
        cat_safe = category.replace(" ", "_").lower()
        cat_data = all_results[all_results['category'] == category]
        
        if not cat_data.empty and f'uplift_{cat_safe}_pct' in cat_data.columns:
            # Keep only the essential prediction columns
            cols_to_keep = ['user_id', 'tax_year', f'uplift_{cat_safe}_pct', 
                           f'ci90_low_{cat_safe}', f'ci90_high_{cat_safe}']
            cols_available = [c for c in cols_to_keep if c in cat_data.columns]
            
            if len(cols_available) > 2:  # If we have at least one prediction column
                user_year_results = user_year_results.merge(
                    cat_data[cols_available],
                    on=['user_id', 'tax_year'], 
                    how='left'
                )
    
    return user_year_results, all_results, cat_summary

def generate_enhanced_tax_insights(user_year_results, df, category_summary):
    """Generate more impactful tax optimization insights with scaled effects and comparative framing"""
    
    # Get average transaction amount per category
    avg_amounts = df.groupby('category')['amount'].mean().to_dict()
    
    # Get refund efficiency metrics - use user-level aggregation to avoid double-counting
    user_cat = df.groupby(['user_id', 'category']).agg({
        'amount': 'sum',
        'refund_amount': 'first'  # refund_amount is user-level, so take first (or mean)
    }).reset_index()
    refund_efficiency = user_cat.groupby('category').agg({
        'amount': 'sum',
        'refund_amount': 'sum'
    })
    # Calculate the actual tax benefit as a percentage of the expense (no capping)
    refund_efficiency['tax_benefit_pct'] = (refund_efficiency['refund_amount'] / refund_efficiency['amount']) * 100
    refund_efficiency = refund_efficiency['tax_benefit_pct'].to_dict()
    
    # Get average income for scaling effects
    avg_income = df['total_income'].mean()
    reference_income = 75000  # Use €75K as standard reference income
    
    # Merge category summary info for confidence levels
    category_info = category_summary[['category', 'confidence_level', 'mean_effect', 'treatment_count', 'qini_coefficient']].set_index('category').to_dict('index')
    
    all_insights = []
    
    # Find all categories with uplift estimates
    uplift_cols = [col for col in user_year_results.columns if col.startswith('uplift_')]
    
    for uplift_col in uplift_cols:
        cat_safe = uplift_col.replace('uplift_', '').replace('_pct', '')
        
        # Find original category name
        category = None
        for cat in df['category'].unique():
            if cat_safe == cat.replace(" ", "_").lower():
                category = cat
                break
        
        if category is None:
            continue
        
        # Get confidence level and other info for this category
        if category in category_info:
            confidence_level = category_info[category]['confidence_level']
            treatment_count = category_info[category]['treatment_count']
            qini_coefficient = category_info[category].get('qini_coefficient', 0)
            mean_effect = category_info[category]['mean_effect']
        else:
            confidence_level = "Unknown"
            treatment_count = 0
            qini_coefficient = 0
            mean_effect = 0
        
        # Get CI bounds if available
        ci_low_col = f'ci90_low_{cat_safe}'
        ci_high_col = f'ci90_high_{cat_safe}'
        
        # Get suggested amount (either category average or fixed €150)
        suggested_amount = avg_amounts.get(category, 150)
        
        # Calculate effects using mean effect from category summary
        pct_effect = mean_effect * 100  # Convert to percentage
        income_impact = reference_income * mean_effect  # Effect on reference income
        expected_refund = suggested_amount * mean_effect
        
        # Get tax benefit percentage for this category
        tax_benefit_pct = refund_efficiency.get(category, 0)
        
        # Format CI text if available
        if ci_low_col in user_year_results.columns and ci_high_col in user_year_results.columns:
            ci_low = user_year_results[ci_low_col].mean()
            ci_high = user_year_results[ci_high_col].mean()
            ci_range = (ci_high - ci_low) * 100  # Show as percentage points
            ci_text = f"±{ci_range/2:.2f}%"
            stat_sig = ci_low > 0
        else:
            ci_text = ""
            stat_sig = False
        
        # Enhanced messages with scaled effects and comparative framing
        if confidence_level == "Very Low":
            if pct_effect < 0:  # Negative effect
                message = (
                    f"⚠️ Adding €{suggested_amount:.0f} in {category} might decrease your tax refund. "
                    f"Users with {category} expenses typically receive {tax_benefit_pct:.1f}% of their spending back in tax benefits. "
                    f"(Based on very limited data - only {treatment_count} samples)"
                )
            elif pct_effect < 0.1:  # Very small effect
                message = (
                    f"💡 Adding €{suggested_amount:.0f} in {category} might increase your tax refund slightly. "
                    f"Users with {category} expenses typically receive {tax_benefit_pct:.1f}% of their spending back in tax benefits. "
                    f"(Based on very limited data - only {treatment_count} samples)"
                )
            else:
                message = (
                    f"💡 Adding €{suggested_amount:.0f} in {category} might increase your refund by approximately {pct_effect:.2f}% of your income. "
                    f"For someone earning €{reference_income:,}, this could mean about €{income_impact:.0f} in additional refunds. "
                    f"(Based on very limited data - only {treatment_count} samples)"
                )
                    
        elif confidence_level == "Low":
            if pct_effect < 0:  # Negative effect
                message = (
                    f"⚠️ Adding €{suggested_amount:.0f} in {category} could reduce your tax refund. "
                    f"Our analysis shows users claiming {category} expenses receive {tax_benefit_pct:.1f}% of their spending back in tax benefits. "
                    f"(Based on limited data from {treatment_count} tax situations)"
                )
            elif pct_effect < 0.1:
                message = (
                    f"💰 Adding €{suggested_amount:.0f} in {category} could boost your tax refund slightly. "
                    f"Our analysis shows users claiming {category} expenses receive {tax_benefit_pct:.1f}% of their spending back in tax benefits. "
                    f"(Based on limited data from {treatment_count} tax situations)"
                )
            else:
                message = (
                    f"💰 Adding €{suggested_amount:.0f} in {category} could increase your refund by up to {pct_effect:.2f}% of your income ({ci_text}). "
                    f"For a €{reference_income:,} income, this represents approximately €{income_impact:.0f} in additional refunds. "
                    f"(Based on {treatment_count} tax situations)"
                )
                    
        elif confidence_level == "Medium":
            if pct_effect < 0:  # Negative effect
                message = (
                    f"⚠️ Adding €{suggested_amount:.0f} in {category} would likely decrease your refund by {abs(pct_effect):.2f}% of your income ({ci_text}). "
                    f"This represents a statistically significant negative effect of approximately €{abs(income_impact):.0f} for someone earning €{reference_income:,}. "
                    f"Users with {category} expenses typically receive {tax_benefit_pct:.1f}% of their spending back in tax benefits."
                )
            else:
                message = (
                    f"✅ Adding €{suggested_amount:.0f} in {category} would likely increase your refund by {pct_effect:.2f}% of your income ({ci_text}). "
                    f"This represents a statistically significant effect worth approximately €{income_impact:.0f} for someone earning €{reference_income:,}. "
                    f"Users with {category} expenses typically receive {tax_benefit_pct:.1f}% of their spending back in tax benefits."
                )
                    
        else:  # High confidence
            if pct_effect < 0:  # Negative effect
                message = (
                    f"⚠️ Adding €{suggested_amount:.0f} in {category} will decrease your refund by {abs(pct_effect):.2f}% of your income ({ci_text}). "
                    f"This represents a proven negative effect of approximately €{abs(income_impact):.0f} for someone earning €{reference_income:,}. "
                    f"Our comprehensive analysis shows this negative effect is statistically significant and reliable."
                )
            else:
                message = (
                    f"⭐ Adding €{suggested_amount:.0f} in {category} will increase your refund by {pct_effect:.2f}% of your income ({ci_text}). "
                    f"This represents a proven tax benefit worth approximately €{income_impact:.0f} for someone earning €{reference_income:,}. "
                    f"Our comprehensive analysis shows this effect is statistically significant and reliable."
                )
        
        insight = {
            'category': category,
            'uplift_pct': mean_effect,
            'uplift_pct_formatted': f"{pct_effect:.2f}%",
            'suggested_amount': suggested_amount,
            'expected_refund': expected_refund,
            'income_impact': income_impact,
            'tax_benefit_pct': tax_benefit_pct,
            'confidence_level': confidence_level,
            'message': message,
            'treatment_count': treatment_count,
            'insight_type': 'causal'
        }
        
        all_insights.append(insight)
    
    # Convert to DataFrame for easier analysis
    if all_insights:
        insights_df = pd.DataFrame(all_insights)
        # Sort by confidence level and then by absolute income impact
        confidence_order = {"High": 0, "Medium": 1, "Low": 2, "Very Low": 3}
        insights_df['confidence_order'] = insights_df['confidence_level'].map(confidence_order)
        insights_df['abs_impact'] = insights_df['income_impact'].abs()
        insights_df = insights_df.sort_values(['confidence_order', 'abs_impact'], ascending=[True, False])
        insights_df = insights_df.drop(['confidence_order', 'abs_impact'], axis=1)
    else:
        insights_df = pd.DataFrame(columns=['category', 'uplift_pct', 'uplift_pct_formatted',
                                           'suggested_amount', 'expected_refund', 'income_impact', 'tax_benefit_pct',
                                           'confidence_level', 'message', 'insight_type', 'treatment_count'])
    
    return insights_df

# Run the full analysis
print("Running causal uplift analysis on all categories...")
user_year_results, all_results, cat_summary = run_causal_uplift_analysis(df)

# Generate enhanced insights with scaled effects and comparative framing
print("Generating enhanced tax optimization insights...")
tax_insights = generate_enhanced_tax_insights(user_year_results, df, cat_summary)

# Save all results to dedicated folder
user_year_results.to_csv(os.path.join(results_dir, 'user_year_results.csv'), index=False)
all_results.to_csv(os.path.join(results_dir, 'all_model_results.csv'), index=False)
tax_insights.to_csv(os.path.join(results_dir, 'tax_optimization_insights.csv'), index=False)

print(f"Analysis complete! Results saved to folder: {results_dir}")

# Create summary report with key findings
with open(os.path.join(results_dir, 'summary_report.txt'), 'w') as f:
    f.write("TAX OPTIMIZATION CAUSAL UPLIFT ANALYSIS\n")
    f.write("======================================\n\n")
    
    f.write("IMPORTANT NOTE ON CONFIDENCE LEVELS:\n")
    f.write("- High confidence: 30+ samples - Strong statistical support\n")
    f.write("- Medium confidence: 10-29 samples - Moderate statistical support\n")
    f.write("- Low confidence: 5-9 samples - Limited statistical support\n")
    f.write("- Very Low confidence: <5 samples - Minimal statistical support\n\n")
    
    # Category counts by confidence level
    confidence_counts = cat_summary['confidence_level'].value_counts().reindex(['High', 'Medium', 'Low', 'Very Low'])
    f.write("CATEGORIES BY CONFIDENCE LEVEL:\n")
    for conf, count in confidence_counts.items():
        f.write(f"- {conf} confidence: {count} categories\n")
    
    f.write("\n")
    
    # Top performing categories
    f.write("TOP PERFORMING CATEGORIES (by mean effect, grouped by confidence):\n\n")
    
    # High confidence categories
    high_conf = cat_summary[cat_summary['confidence_level'] == 'High'].sort_values('mean_effect', ascending=False)
    if not high_conf.empty:
        f.write("HIGH CONFIDENCE CATEGORIES:\n")
        for i, (_, row) in enumerate(high_conf.head(3).iterrows(), 1):
            f.write(f"{i}. {row['category']}: Effect = {row['mean_effect']:.4f}, Samples = {row['treatment_count']}\n")
        f.write("\n")
    
    # Medium confidence categories
    med_conf = cat_summary[cat_summary['confidence_level'] == 'Medium'].sort_values('mean_effect', ascending=False)
    if not med_conf.empty:
        f.write("MEDIUM CONFIDENCE CATEGORIES:\n")
        for i, (_, row) in enumerate(med_conf.head(3).iterrows(), 1):
            f.write(f"{i}. {row['category']}: Effect = {row['mean_effect']:.4f}, Samples = {row['treatment_count']}\n")
        f.write("\n")
    
    # Lower confidence categories (combined low & very low)
    low_conf = cat_summary[cat_summary['confidence_level'].isin(['Low', 'Very Low'])].sort_values('mean_effect', ascending=False)
    if not low_conf.empty:
        f.write("LOWER CONFIDENCE CATEGORIES (treat with caution):\n")
        for i, (_, row) in enumerate(low_conf.head(3).iterrows(), 1):
            f.write(f"{i}. {row['category']}: Effect = {row['mean_effect']:.4f}, Samples = {row['treatment_count']}, "
                   f"Confidence = {row['confidence_level']}\n")
        f.write("\n")
    
    # Enhanced insights metrics
    f.write(f"GENERATED INSIGHTS: {len(tax_insights)}\n")
    
    # Confidence level breakdown for insights
    for level in ['High', 'Medium', 'Low', 'Very Low']:
        count = sum(tax_insights['confidence_level'] == level)
        f.write(f"  {level} confidence insights: {count}\n")
    f.write("\n")
    
    # Enhanced summary statistics for expected effects
    if not tax_insights.empty:
        f.write(f"Average income effect (%): {tax_insights['uplift_pct'].mean()*100:.2f}%\n")
        f.write(f"Average income impact (€): €{tax_insights['income_impact'].mean():.2f}\n")
        f.write(f"Average tax benefit percentage: {tax_insights['tax_benefit_pct'].mean():.2f}%\n")
        f.write(f"Max income impact (€): €{tax_insights['income_impact'].max():.2f}\n")
    
    f.write("\n")
    f.write("Files generated:\n")
    f.write("- user_year_results.csv: User-year level analysis results\n")
    f.write("- all_model_results.csv: Detailed model results for each category\n")
    f.write("- tax_optimization_insights.csv: Enhanced insights with scaled effects and comparative framing\n")
    f.write("- category_statistics.csv: Summary statistics for each category\n")
    f.write("- category_performance.csv: Causal model performance by category\n")
    f.write("- uplift_curves/: Folder containing uplift curves for all categories\n")
    f.write("\n")
    f.write("IMPORTANT: For categories with Very Low or Low confidence ratings, the recommendations should be interpreted\n")
    f.write("as directional guidance only, not precise predictions. More data would be needed for stronger conclusions.\n")

print(f"Enhanced summary report created at {os.path.join(results_dir, 'summary_report.txt')}")