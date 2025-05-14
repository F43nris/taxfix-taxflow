# taxfix-taxflow

## this will be a separate analysis readme


## A) Business Goal: 
"Help every Taxfix user uncover actionable tax-optimisation opportunities, at scale."

## B) Data Understanding, Validation and Cleaning

**Data Validator Summary:**
**Input:**
- Takes 3 input CSV files from casestudy/data/:
    - transactions_csv.txt
    - users_csv.txt
    - tax_filings_csv.txt

**Output:**
- Clean transaction data (transactions_csv_clean.txt)

**Transaction Structure & Volume**
- Comprehensive transaction dataset with categorized expenses (~1000+ transactions)
- Data spans approximately 2022-2023 based on monthly transaction chart
- Mean transaction amount: €339.95, median: €95.00 - indicating right-skewed distribution -> Most transactions are small, but a few large ones pull the mean up.
- Std: €413.19
- Min/Max: €35.00 / €2,200.00
- Quartiles: 25% = €65.00, 75% = €555.00

Interpretation:
- The distribution is right-skewed (long tail to the right), typical for expense data: most transactions are small, but a few large ones (e.g., rent, self-employment) pull the mean up.
- This has implications for modeling: median and quantile-based analytics are more robust than mean-based for user-facing features.

![alt text](notebooks/charts/amount_distribution.png "Amount Distribution") 
![alt text](notebooks/charts/monthly_transactions.png "Monthly Transactions")

**Key Categories by Volume**
- Transportation: 345 transactions (highest volume)
- Child Care: 200 transactions
- Medical: 114 transactions
- Business Meals: 96 transactions
- Self-Employment income: 71 transactions

![alt text](notebooks/charts/category_distribution.png)

**Financial Pattern Highlights**
- Self-Employment and Rental have highest transaction values
- Professional Development shows significant expenses 
- Transportation dominates by volume but has lower per-transaction value
- Small recurring expenses dominate (insurance, transportation, business meals)

![alt text](notebooks/charts/category_amount_comparison.png)

**Vendor Analysis**
- BVG dominates (143 transactions) suggesting Berlin residency
- Multiple transport companies (MVG, HVV, RMV) indicate travel within German cities
- Several childcare providers suggest family tax situation
- Multiple restaurant transactions suggest business entertainment expenses

Implication:
- The data is rich for both classification (categorizing new transactions) and recommendation (suggesting deductions).

![alt text](notebooks/charts/top_vendors.png)

## German Tax Implications
1. Self-Employment Deductions
- Significant business expense opportunities: transportation, business meals, work equipment
- Professional development expenses (€819 average) fully deductible for self-employed
- Home office expenses present but modest (€270 average)
2. Family-Related Deductions
- Child care expenses substantial (€460 average) - eligible for Kinderbetreuungskosten tax benefits
- Medical expenses may qualify for extraordinary burden deductions (außergewöhnliche Belastungen)
3. Transportation Tax Benefits
- High volume of transportation suggests potential for Entfernungspauschale (commuting allowance)
- Mix of public transport vendors indicates organized receipts needed for tax documentation
- Rental Income
- Property expenses properly categorized - important for Werbungskosten deductions against rental income


## Data & AI Engineering Implications
1. Data Modeling:
- Hierarchical category/subcategory structure and vendor normalization should be reflected in the data model.
2. ML/AI Opportunities:
- Recommendation: Suggest missing deductions or flag unusual transactions for review.
- Forecasting: Predict future deductible expenses for cash flow and tax planning. Research did not provide any input on whether actions within one year causally influence taxes positively, making time series forecasting a challenging but potentially valuable area of exploration.
- Personalization: Tailor tax advice based on user-specific transaction patterns.


# EDA - Tax Optimization Insights Summary

## Key Demographic Patterns
1. Occupation-Based Optimization
- Self-employed individuals achieve 21.8% higher deduction ratios than manufacturing workers (4.31% vs 3.54%)
- Self-employed users represent 27% of transactions but 38.6% of spending
- Finance professionals experience highest synergistic deduction effects (8.1 ratio)
- Healthcare workers show lowest synergistic benefits (2.7 ratio)
2. Family Status Impact
- Single parents achieve highest deduction ratio (4.10%)
- Married with children represent 41% of transactions and 45% of spending but have lowest deduction ratio (3.88%)
- Child Care is second-highest expense category (26% of spending)
- Married couples without children experience strongest synergistic effects (7.8 ratio)
3. Age-Related Patterns
- Younger taxpayers (25-35) benefit most from synergistic deduction combinations (7.0 ratio)
- Benefits decrease with age (50-60 group shows lowest ratio at 2.5)
4. Regional Variations
Hamburg appears frequently in under-claimed deduction analysis
Berlin and Frankfurt residents achieve stronger synergistic effects (6.2 and 6.5 ratios)

## Category Performance & Correlations
1. Highest-Impact Categories
- Business-related expenses show strongest positive impact
- Client Payment (+0.4069) and Client Meeting (+0.3975) have highest correlations with tax outcomes
- Travel-Business Trip shows highest impact (+0.0056)
- Medical expenses, particularly prescriptions and therapy, show moderate positive correlations
2. Surprising Underperformers
- Work Equipment (0.003), Child Care (0.04) show unexpectedly weak correlations
- Transportation subcategories show divergent effects: public transit positive, car expenses negative
- Property and rental expenses consistently correlate negatively with deduction ratios
3. Category Synergies
- Medical × Business Meals (18.17x multiplier effect)
- Medical × Self-Employment (16.62x)
- Work Equipment × Medical (13.85x)
- These combinations produce deduction effects far beyond their individual contributions

## Under-Claimed Opportunities
1. Self-Employment & Business Meals
- 3 users identified with high spending but below-median deduction ratios
- Average Self-Employment spending: €15,567
- Profile: Primarily single parents or married without children
- Locations: Hamburg and Munich
2. Medical Expenses
- 8 users across diverse demographics under-claiming
- Average spending: €501
- Most prevalent among married with children in Hamburg
3. Year-Over-Year Improvements
- Average deductions increased by 7.9% (2022-2023)
- Average refunds increased by 10%
- Average transaction amount increased by 8.9%

## Strategic Recommendations
1. Category Combination Strategy
- Pair medical expenses with business meals or self-employment expenses for maximum impact
- Combine work equipment with insurance or medical expenses for professionals
- For threshold-crossing effects, strategically time expenses to maximize deduction potential
2. Demographic-Targeted Approaches
- Self-employed: Focus on business meals and client meeting documentation
- IT professionals: Prioritize work equipment and insurance combinations
- Families: Improve child-related expense optimization
- Young taxpayers: Maximize deductions through medical and work equipment pairings
3. Filing Optimization (Warning! Correlation is not Causation)
- Data suggests March filing yields better outcomes than January or April
- Early filings (January) consistently show lower refunds
- Avoid April rush which correlates with poorer outcomes
4. Regional Focus (Warning! Correlation is not Causation)
- Hamburg residents need targeted medical and transportation deduction guidance
- Berlin residents should prioritize work equipment deductions
- Frankfurt taxpayers benefit from professional development combinations



