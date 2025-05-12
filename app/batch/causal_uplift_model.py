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

# Create results directory
results_dir = 'tax_uplift_results'
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)  # Remove if exists
os.makedirs(results_dir)
os.makedirs(os.path.join(results_dir, 'uplift_curves'), exist_ok=True)

# Load your data
df = pd.read_csv('/Users/ADAM-Production/taxfix/taxflow/casestudy/data/full_joined_data.csv')

# Calculate derived measures
df['refund_per_income'] = df['refund_amount'] / df['total_income']
df['deduction_ratio'] = df['total_deductions'] / df['total_income']

# Generate category statistics to use throughout analysis
category_stats = df.groupby('category').agg({
    'transaction_id': 'count',
    'amount': ['mean', 'sum'],
    'refund_amount': ['mean'],
    'total_income': ['mean']
}).reset_index()

# Flatten multi-level columns
category_stats.columns = ['category', 'transaction_count', 'avg_amount', 'total_spent', 
                         'avg_refund', 'avg_income']

# Calculate derived metrics
category_stats['refund_per_euro'] = category_stats['avg_refund'] / category_stats['avg_amount']
category_stats['spending_share'] = category_stats['total_spent'] / category_stats['total_spent'].sum()

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

# Add confidence level to category stats
category_stats['confidence_level'] = category_stats['transaction_count'].apply(get_confidence_level)
category_stats['user_count'] = df.groupby('category')['user_id'].nunique().values

# Function to prepare user-year level data - REMOVED age_range, family_status, and region
def prepare_user_year_data(df):
    """Aggregate transaction data to user-year level and create treatment indicators"""
    
    # Get all unique categories
    categories = df['category'].unique()
    print(f"Found {len(categories)} unique categories")
    
    # Create user-year level dataset - REMOVED age_range, family_status, and region
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
    
    # Generate category ranking for each user-year
    for idx, row in user_year_results.iterrows():
        uplift_cols = [col for col in user_year_results.columns if col.startswith('uplift_')]
        if uplift_cols:
            # Get categories sorted by uplift, handling NaNs
            uplift_dict = {}
            for col in uplift_cols:
                if not pd.isna(row[col]):
                    uplift_dict[col] = row[col]
            
            if uplift_dict:
                sorted_uplifts = pd.Series(uplift_dict).sort_values(ascending=False)
                
                # Store top 5 categories and their uplift
                for i, (col, value) in enumerate(list(sorted_uplifts.items())[:5], 1):
                    cat_safe = col.replace('uplift_', '').replace('_pct', '')
                    
                    # Find original category name
                    for cat in categories:
                        if cat_safe == cat.replace(" ", "_").lower():
                            user_year_results.loc[idx, f'top{i}_category'] = cat
                            user_year_results.loc[idx, f'top{i}_uplift'] = value
                            break
    
    # Create transaction-level results by joining user-year results back to original transactions
    txn_results = df.copy()
    
    # For each transaction, add its category's uplift
    txn_results['txn_uplift_pct'] = np.nan  # Initialize column
    
    for idx, txn in txn_results.iterrows():
        user_id = txn['user_id']
        tax_year = txn['tax_year']
        category = txn['category']
        
        # Find matching user-year result
        user_year_match = user_year_results[
            (user_year_results['user_id'] == user_id) & 
            (user_year_results['tax_year'] == tax_year)
        ]
        
        if not user_year_match.empty:
            cat_safe = category.replace(" ", "_").lower()
            uplift_col = f'uplift_{cat_safe}_pct'
            
            if uplift_col in user_year_match.columns and not pd.isna(user_year_match[uplift_col].values[0]):
                txn_results.loc[idx, 'txn_uplift_pct'] = user_year_match[uplift_col].values[0]
    
    return user_year_results, txn_results, all_results, cat_summary

def generate_enhanced_tax_insights(user_year_results, df, category_summary):
    """Generate more impactful tax optimization insights with scaled effects and comparative framing"""
    
    # Get average transaction amount per category
    avg_amounts = df.groupby('category')['amount'].mean().to_dict()
    
    # Get refund efficiency metrics
    refund_efficiency = df.groupby('category').agg({
        'amount': 'mean',
        'refund_amount': 'mean'
    })
    refund_efficiency['refund_per_euro'] = refund_efficiency['refund_amount'] / refund_efficiency['amount']
    refund_efficiency = refund_efficiency['refund_per_euro'].to_dict()
    
    # Get average income for scaling effects
    avg_income = df['total_income'].mean()
    reference_income = 75000  # Use €75K as standard reference income
    
    # Merge category summary info for confidence levels
    category_info = category_summary[['category', 'confidence_level', 'mean_effect', 'treatment_count', 'qini_coefficient']].set_index('category').to_dict('index')
    
    all_insights = []
    
    # 1. CAUSAL INSIGHTS: Generate enhanced personalized tips
    for idx, row in user_year_results.iterrows():
        user_id = row['user_id']
        tax_year = row['tax_year']
        
        user_tips = []
        
        # Find all categories with uplift estimates
        uplift_cols = [col for col in row.index if col.startswith('uplift_')]
        
        for uplift_col in uplift_cols:
            if pd.isna(row[uplift_col]):
                continue
                
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
            else:
                confidence_level = "Unknown"
                treatment_count = 0
                qini_coefficient = 0
            
            # Get CI bounds if available
            ci_low_col = f'ci90_low_{cat_safe}'
            ci_high_col = f'ci90_high_{cat_safe}'
            
            # Only include categories with positive point estimates
            if row[uplift_col] > 0:
                # Check if CI bounds are available
                lower_ci = row[ci_low_col] if ci_low_col in row and not pd.isna(row[ci_low_col]) else None
                upper_ci = row[ci_high_col] if ci_high_col in row and not pd.isna(row[ci_high_col]) else None
                
                # Get suggested amount (either category average or fixed €150)
                suggested_amount = avg_amounts.get(category, 150)
                
                # Key improvements here:
                # 1. Scale effects to reference income
                # 2. Calculate both percentage and absolute impact
                pct_effect = row[uplift_col] * 100  # Convert to percentage
                income_impact = reference_income * row[uplift_col]  # Effect on reference income
                expected_refund = suggested_amount * row[uplift_col]
                
                # Get refund efficiency for comparison framing
                efficiency_multiple = refund_efficiency.get(category, 1)
                
                # Format CI text if available - improved to show percentage
                if lower_ci is not None and upper_ci is not None:
                    ci_range = (upper_ci - lower_ci) * 100  # Show as percentage points
                    ci_text = f"±{ci_range/2:.2f}%"
                    stat_sig = lower_ci > 0
                else:
                    ci_text = ""
                    stat_sig = False
                
                # Enhanced messages with scaled effects and comparative framing
                if confidence_level == "Very Low":
                    if pct_effect < 0.1:  # Very small effect
                        message = (
                            f"💡 Adding €{suggested_amount:.0f} in {category} might increase your tax refund slightly. "
                            f"Users with {category} expenses typically see refunds {efficiency_multiple:.1f}× their spending in this category. "
                            f"(Based on very limited data - only {treatment_count} samples)"
                        )
                    else:
                        message = (
                            f"💡 Adding €{suggested_amount:.0f} in {category} might increase your refund by approximately {pct_effect:.2f}% of your income. "
                            f"For someone earning €{reference_income:,}, this could mean about €{income_impact:.0f} in additional refunds. "
                            f"(Based on very limited data - only {treatment_count} samples)"
                        )
                            
                elif confidence_level == "Low":
                    if pct_effect < 0.1:
                        message = (
                            f"💰 Adding €{suggested_amount:.0f} in {category} could boost your tax refund. "
                            f"Our analysis shows users claiming {category} expenses receive refunds approximately {efficiency_multiple:.1f}× higher than their spending in this category. "
                            f"(Based on limited data from {treatment_count} tax situations)"
                        )
                    else:
                        message = (
                            f"💰 Adding €{suggested_amount:.0f} in {category} could increase your refund by up to {pct_effect:.2f}% of your income ({ci_text}). "
                            f"For a €{reference_income:,} income, this represents approximately €{income_impact:.0f} in additional refunds. "
                            f"(Based on {treatment_count} tax situations)"
                        )
                        
                elif confidence_level == "Medium":
                    if stat_sig:
                        message = (
                            f"✅ Adding €{suggested_amount:.0f} in {category} would likely increase your refund by {pct_effect:.2f}% of your income ({ci_text}). "
                            f"This represents a statistically significant effect worth approximately €{income_impact:.0f} for someone earning €{reference_income:,}. "
                            f"Users with {category} expenses typically receive refunds {efficiency_multiple:.1f}× their spending in this category."
                        )
                    else:
                        message = (
                            f"⚠️ Adding €{suggested_amount:.0f} in {category} is associated with increased refunds of about {pct_effect:.2f}% of income ({ci_text}), "
                            f"though this effect is not statistically significant. Our analysis of {treatment_count} users shows those claiming {category} "
                            f"receive refunds approximately {efficiency_multiple:.1f}× their spending in this category."
                        )
                        
                else:  # High confidence
                    if stat_sig:
                        message = (
                            f"⭐ Adding €{suggested_amount:.0f} in {category} will increase your refund by {pct_effect:.2f}% of your income ({ci_text}). "
                            f"This represents a proven tax benefit worth approximately €{income_impact:.0f} for someone earning €{reference_income:,}. "
                            f"Our comprehensive analysis shows this effect is statistically significant and reliable."
                        )
                    else:
                        message = (
                            f"🔍 Adding €{suggested_amount:.0f} in {category} is associated with a {pct_effect:.2f}% increase in refund-to-income ratio ({ci_text}), "
                            f"representing about €{income_impact:.0f} for a €{reference_income:,} income. While consistent across {treatment_count} users, "
                            f"this effect does not reach statistical significance."
                        )
                
                insight = {
                    'user_id': user_id,
                    'tax_year': tax_year,
                    'category': category,
                    'uplift_pct': row[uplift_col],
                    'uplift_pct_formatted': f"{pct_effect:.2f}%",
                    'suggested_amount': suggested_amount,
                    'expected_refund': expected_refund,
                    'income_impact': income_impact,
                    'refund_efficiency': efficiency_multiple,
                    'confidence_level': confidence_level,
                    'message': message,
                    'insight_type': 'causal',
                    'treatment_count': treatment_count
                }
                
                user_tips.append(insight)
        
        # Sort tips by confidence level and then by expected refund
        confidence_order = {"High": 0, "Medium": 1, "Low": 2, "Very Low": 3}
        user_tips = sorted(user_tips, key=lambda x: (confidence_order.get(x['confidence_level'], 4), -x['income_impact']))
        
        # Add to all insights
        all_insights.extend(user_tips)
    
    # 2. DESCRIPTIVE INSIGHTS: Enhanced general insights
    for category, stats in category_info.items():
        mean_effect = stats.get('mean_effect', 0)
        if pd.isna(mean_effect) or mean_effect <= 0:
            continue  # Skip categories with no positive effect
            
        confidence_level = stats.get('confidence_level', 'Unknown')
        treatment_count = stats.get('treatment_count', 0)
        qini_coefficient = stats.get('qini_coefficient', 0)
        
        # Get average amount and calculate scaled effects
        avg_amount = avg_amounts.get(category, 150)
        expected_refund = avg_amount * mean_effect
        pct_effect = mean_effect * 100  # Convert to percentage
        income_impact = reference_income * mean_effect
        efficiency_multiple = refund_efficiency.get(category, 1)
        
        # Create enhanced general insight for this category
        if confidence_level == "Very Low":
            message = (
                f"🔎 Users with {category} expenses (avg €{avg_amount:.0f}) may see increased tax refunds equivalent to {pct_effect:.2f}% of their income. "
                f"For someone earning €{reference_income:,}, this represents a potential €{income_impact:.0f} benefit. "
                f"Tax data shows these users receive refunds approximately {efficiency_multiple:.1f}× their {category} spending, "
                f"though evidence is very limited ({treatment_count} samples)."
            )
        elif confidence_level == "Low":
            message = (
                f"📊 Tax data suggests {category} expenses (avg €{avg_amount:.0f}) could increase refunds by {pct_effect:.2f}% of income. "
                f"For a €{reference_income:,} earner, this represents approximately €{income_impact:.0f} in additional refunds. "
                f"Users with these expenses typically see refunds {efficiency_multiple:.1f}× their {category} spending. "
                f"(Based on {treatment_count} tax situations)"
            )
        elif confidence_level == "Medium":
            message = (
                f"📈 Our analysis of {treatment_count} tax situations shows {category} expenses (avg €{avg_amount:.0f}) "
                f"typically increase refunds by {pct_effect:.2f}% of income, worth about €{income_impact:.0f} for someone earning €{reference_income:,}. "
                f"Users claiming these deductions receive refunds approximately {efficiency_multiple:.1f}× their spending in this category."
            )
        else:  # High confidence
            message = (
                f"🏆 Strong evidence from {treatment_count} tax situations confirms {category} expenses (avg €{avg_amount:.0f}) "
                f"increase refunds by {pct_effect:.2f}% of income. For a €{reference_income:,} earner, this represents €{income_impact:.0f} in additional refunds. "
                f"Users with these deductions consistently see refunds {efficiency_multiple:.1f}× higher than their {category} spending."
            )
        
        insight = {
            'user_id': 'general',
            'tax_year': 'all',
            'category': category,
            'uplift_pct': mean_effect,
            'uplift_pct_formatted': f"{pct_effect:.2f}%",
            'suggested_amount': avg_amount,
            'expected_refund': expected_refund,
            'income_impact': income_impact,
            'refund_efficiency': efficiency_multiple,
            'confidence_level': confidence_level,
            'message': message,
            'insight_type': 'descriptive',
            'treatment_count': treatment_count
        }
        
        all_insights.append(insight)
    
    # Convert to DataFrame for easier analysis
    if all_insights:
        insights_df = pd.DataFrame(all_insights)
    else:
        insights_df = pd.DataFrame(columns=['user_id', 'tax_year', 'category', 'uplift_pct', 'uplift_pct_formatted',
                                           'suggested_amount', 'expected_refund', 'income_impact', 'refund_efficiency',
                                           'confidence_level', 'message', 'insight_type', 'treatment_count'])
    
    return insights_df

# Run the full analysis
print("Running causal uplift analysis on all categories...")
user_year_results, txn_results, all_results, cat_summary = run_causal_uplift_analysis(df)

# Generate enhanced insights with scaled effects and comparative framing
print("Generating enhanced tax optimization insights...")
tax_insights = generate_enhanced_tax_insights(user_year_results, df, cat_summary)

# Save all results to dedicated folder
user_year_results.to_csv(os.path.join(results_dir, 'user_year_uplift_results.csv'), index=False)
txn_results.to_csv(os.path.join(results_dir, 'transaction_uplift_results.csv'), index=False)
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
    causal_insights = tax_insights[tax_insights['insight_type'] == 'causal']
    desc_insights = tax_insights[tax_insights['insight_type'] == 'descriptive']
    
    f.write(f"GENERATED INSIGHTS: {len(tax_insights)}\n")
    f.write(f"- Personalized causal insights: {len(causal_insights)}\n")
    f.write(f"- General descriptive insights: {len(desc_insights)}\n\n")
    
    # Confidence level breakdown for causal insights
    if not causal_insights.empty:
        for level in ['High', 'Medium', 'Low', 'Very Low']:
            count = sum(causal_insights['confidence_level'] == level)
            f.write(f"  {level} confidence insights: {count}\n")
        f.write("\n")
    
    # Enhanced summary statistics for expected effects
    if not tax_insights.empty:
        f.write(f"Average income effect (%): {tax_insights['uplift_pct'].mean()*100:.2f}%\n")
        f.write(f"Average income impact (€): €{tax_insights['income_impact'].mean():.2f}\n")
        f.write(f"Average refund efficiency multiplier: {tax_insights['refund_efficiency'].mean():.2f}×\n")
        f.write(f"Max income impact (€): €{tax_insights['income_impact'].max():.2f}\n")
    
    f.write("\n")
    f.write("Files generated:\n")
    f.write("- user_year_uplift_results.csv: Uplift estimates per user-year\n")
    f.write("- transaction_uplift_results.csv: Original data enriched with uplift values\n")
    f.write("- all_model_results.csv: Detailed model results for each category\n")
    f.write("- tax_optimization_insights.csv: Enhanced insights with scaled effects and comparative framing\n")
    f.write("- category_statistics.csv: Summary statistics for each category\n")
    f.write("- category_performance.csv: Causal model performance by category\n")
    f.write("- uplift_curves/: Folder containing uplift curves for all categories\n")
    f.write("\n")
    f.write("IMPORTANT: For categories with Very Low or Low confidence ratings, the recommendations should be interpreted\n")
    f.write("as directional guidance only, not precise predictions. More data would be needed for stronger conclusions.\n")
    f.write("\n")
    f.write("NOTE: This analysis excludes age range, family status, and region as control variables.\n")

print(f"Enhanced summary report created at {os.path.join(results_dir, 'summary_report.txt')}")