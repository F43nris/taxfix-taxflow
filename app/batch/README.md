# Tax Optimization Causal Uplift Analysis

## Overview
This analysis uses causal inference techniques to identify which expense categories have the most significant impact on tax refunds. By employing a causal forest approach, we can estimate the true causal effect of different expense categories on tax refunds, accounting for confounding factors and providing more reliable insights than traditional correlation-based analysis.

## Methodology

### Data Preparation
1. **Data Aggregation**: Transaction data is aggregated at the category level to identify patterns in spending and refunds
2. **Feature Engineering**: Key metrics are calculated including:
   - Refund per income ratio
   - Deduction ratio
   - Category-specific spending patterns
   - Tax benefit percentage (percentage of expense amount received back in tax benefits)

### Causal Modeling Approach
The analysis uses a Causal Forest DML (Double Machine Learning) approach, which:
1. **Handles Confounding**: Accounts for factors that might influence both spending and refunds
2. **Estimates Treatment Effects**: Calculates the true causal impact of each expense category
3. **Provides Confidence Intervals**: Quantifies the uncertainty in our estimates

### Confidence Levels
Results are categorized by confidence levels based on sample size:
- **High**: 30+ samples - Strong statistical support
- **Medium**: 10-29 samples - Moderate statistical support
- **Low**: 5-9 samples - Limited statistical support
- **Very Low**: <5 samples - Minimal statistical support

## Key Findings

### High-Impact Categories
1. **Travel Expenses** (Very Low Confidence)
   - Highest potential impact: 0.38% of income
   - Average spending: €472
   - Tax benefit: 28.6% of spending

2. **Insurance** (Very Low Confidence)
   - Impact: 0.26% of income
   - Average spending: €193
   - Tax benefit: 57.9% of spending

3. **Business Meals** (Low Confidence)
   - Impact: 0.21% of income
   - Average spending: €67
   - Tax benefit: 18.2% of spending

### Categories with Negative Impact
1. **Rental Expenses** (Very Low Confidence)
   - Negative impact: -0.37% of income
   - Average spending: €1,098
   - Tax benefit: 91.2% of spending

2. **Property Expenses** (Very Low Confidence)
   - Negative impact: -0.37% of income
   - Average spending: €241
   - Tax benefit: 41.4% of spending

3. **Home Office** (Low Confidence)
   - Negative impact: -0.06% of income
   - Average spending: €270
   - Tax benefit: 35.8% of spending

### Most Beneficial Categories
1. **Transportation**: 14.1% tax benefit
2. **Medical**: 10.1% tax benefit
3. **Work Equipment**: 23.0% tax benefit

## Understanding Tax Benefits

### Tax Benefit Percentage
The tax benefit percentage shows what portion of your expense you receive back in tax benefits. For example:
- A 30% tax benefit means you get back 30% of what you spent
- A 50% tax benefit means you get back half of what you spent
- A 100% tax benefit means you get back the full amount you spent

### Effect Sizes
- Effects are expressed as percentage of income
- For a reference income of €75,000:
  - 0.1% effect = €75 additional refund
  - 0.5% effect = €375 additional refund
  - 1.0% effect = €750 additional refund

## Limitations

1. **Sample Size**: Many categories have limited data points
2. **Temporal Effects**: Results may vary by tax year
3. **Individual Variation**: Effects may differ based on personal circumstances
4. **Tax Law Changes**: Results may need updating with tax law changes

## Usage Recommendations

1. **High Confidence Categories**:
   - Use for strategic tax planning
   - Consider as reliable optimization opportunities

2. **Medium Confidence Categories**:
   - Use for general guidance
   - Monitor for changes in effectiveness

3. **Low/Very Low Confidence Categories**:
   - Use as directional indicators
   - Collect more data for better insights
   - Consider as experimental opportunities

## Files in this Directory

- `tax_optimization_insights.csv`: Detailed category-level insights
- `category_statistics.csv`: Basic statistics for each category
- `category_performance.csv`: Model performance metrics
- `uplift_curves/`: Visualizations of uplift effects
- `summary_report.txt`: Comprehensive analysis summary

## Technical Details

### Model Parameters
- Causal Forest DML with Random Forest base learners
- Confidence intervals at 90% level
- Treatment effect estimation using double machine learning
- Feature importance analysis for key drivers

### Data Requirements
- Transaction-level spending data
- Tax refund information
- Income data
- Category classifications

### Update Frequency
- Analysis should be rerun quarterly
- Confidence levels should be monitored
- New categories should be evaluated as data becomes available

# Hierarchical Clustering Tax Peer Group Analysis

## Overview
This analysis uses hierarchical clustering to group users into tax peer groups based on their expense patterns, income, and demographic features. The goal is to identify how each user compares to their peers and to generate personalized tax optimization recommendations.

## Methodology
- **Data Preparation:**
  - Transaction-level data is aggregated by user and tax year.
  - Expense categories are normalized as a percentage of each user's income.
  - Demographic features (occupation, age range, family status, region) are included if available.
- **Feature Engineering:**
  - Principal Component Analysis (PCA) reduces dimensionality for clustering.
- **Clustering:**
  - Hierarchical clustering (Ward's method) is used to form peer groups.
  - The optimal number of clusters is determined using silhouette scores.
- **Gap & Recommendation Generation:**
  - For each user and category, the gap between their spending and the cluster average is calculated.
  - Recommendations are generated for all users and categories, with confidence levels reflecting the strength of the evidence.

## Cluster Explanations
Based on the current analysis, the following clusters were identified (see `hierarchical_cluster_profiles.txt`):

### Cluster 0 (3 members)
- **Income Bands:** Band 2, 4, 5 (mid to high income)
- **Occupation:** 100% Self-Employed
- **Top Expenses:**
  - Self-Employment: 21.7% of income
  - Child Care: 3.3%
  - Travel: 2.1%
- **Interpretation:**
  - This cluster represents self-employed users with significant business-related expenses and moderate child care and travel deductions.

### Cluster 1 (9 members)
- **Income Bands:** Broadly distributed (Bands 1–5)
- **Occupations:** IT Professional, Education, Manufacturing, Healthcare, Finance
- **Top Expenses:**
  - Child Care: 5.3% of income
  - Professional Development: 1.9%
  - Transportation: 1.4%
- **Interpretation:**
  - This is a diverse cluster with a mix of professions and income levels, but a strong focus on child care and professional development deductions.

### Cluster 2 (1 member)
- **Income Band:** Band 4 (upper-middle income)
- **Occupation:** Healthcare
- **Top Expenses:**
  - Medical: 3.8% of income
  - Transportation: 1.0%
- **Interpretation:**
  - A single healthcare professional with high medical and transportation expenses.

### Cluster 3 (2 members)
- **Income Band:** Band 1 (lowest income)
- **Occupation:** Retail
- **Top Expenses:**
  - Child Care: 5.5% of income
  - Work Clothing: 1.4%
  - Transportation: 1.1%
- **Interpretation:**
  - Low-income retail workers with notable child care and work clothing deductions.

### Cluster 4 (1 member)
- **Income Band:** Band 5 (highest income)
- **Occupation:** Finance
- **Top Expenses:**
  - Rental: 30.3% of income
  - Property Expenses: 3.6%
- **Interpretation:**
  - A high-income finance professional with substantial rental and property-related deductions.

## Key Findings
- **All users are included** in the recommendations, regardless of gap size.
- **Confidence levels** (Very Weak to Very Strong) indicate the strength of each recommendation.
- **Cluster profiles** reveal distinct peer groups based on occupation, income, and expense patterns.
- **Personalized recommendations** are generated for every user and category, even if the gap is small or negative.

## How to Interpret the Results
- **Recommendation Text:** Explains how your spending compares to your peer group for each category.
- **Gap %:** Positive means you spend less than your peers; negative means you spend more.
- **Confidence Level:** Higher confidence means stronger evidence for the recommendation.
- **Cluster Context:** Your peer group is defined by similar income, occupation, and expense patterns.

## Limitations
- Small clusters (with 1–2 users) may yield less reliable recommendations.
- Recommendations for categories with very low peer usage may be less actionable.
- The analysis is based on available data and may not capture all relevant user circumstances.

## Usage Recommendations
- Focus on recommendations with Moderate, Strong, or Very Strong confidence.
- Use cluster profiles to understand your peer group context.
- Treat Very Weak recommendations as exploratory insights.

## Technical Details
- **Clustering:** Hierarchical (Ward's method), PCA for dimensionality reduction
- **Features:** Expense ratios, occupation, age, family status, region
- **Outputs:**
  - `hierarchical_tax_recommendations_with_confidence.csv`: All user-category recommendations
  - `hierarchical_cluster_profiles.txt`: Cluster summaries
  - `enriched_tax_data.csv`: Original data with cluster assignments

---
For questions or further analysis, see the cluster profiles or contact the data science team.

# German Tax Deduction Classifier Analysis

## Overview
The German Tax Deduction Classifier is a sophisticated system that analyzes tax combinations based on occupation, family status, and expense categories to determine tax deductibility under German tax law. The system uses LLM-based classification with OpenAI models via LiteLLM, incorporating structured outputs and validation through Pydantic.

## Process Flow

### Data Input
Takes unique tax combinations including:
- Occupation category
- Family status
- Subcategory
- Description
- Income band

### Classification Process
- Uses Jinja templating for tax rule context
- Leverages LiteLLM's built-in prompt caching
- Processes in batches with error handling
- Generates structured outputs with confidence scores

## Output Structure
Each classification result includes:
- `is_deductible`: Boolean indicating deductibility
- `confidence_score`: Score between 0.0 and 1.0
- `category`: Tax category (e.g., Werbungskosten, Sonderausgaben)
- `max_deduction_percentage`: Maximum percentage deductible
- `max_deduction_amount`: Maximum amount in EUR
- `reasons`: Legal references and explanations

## Results Analysis

### Key Categories

#### Werbungskosten (Income-related Expenses)
- Most common category
- Includes work equipment, training costs, commuting expenses
- Often 100% deductible with specific limits

#### Sonderausgaben (Special Expenses)
- Includes child-care costs (80% up to €4,800)
- Donations (up to 20% of income)

#### Betriebsausgaben (Business Expenses)
- Primarily for self-employed individuals
- 100% deductible for business-related expenses

#### Außergewöhnliche Belastungen (Extraordinary Burdens)
- Medical expenses
- Subject to income-based thresholds

### Notable Patterns
- High confidence scores (0.8–0.95) for most classifications
- Clear distinction between employee and self-employed deductions
- Specific limits for certain categories (e.g., €4,500 for commuting)
- Comprehensive legal references in explanations

## Technical Implementation
- Uses Pydantic for data validation
- Implements batch processing for efficiency
- Includes error handling and fallback mechanisms
- Maintains detailed logging of processing steps

## Usage
The classifier can be run with various parameters:
- Batch size control
- Model selection
- Start/end index specification
- Custom input/output file paths

## Limitations
- Requires OpenAI API key
- Subject to API rate limits
- Confidence scores may vary based on input complexity
- Some edge cases may require manual review
