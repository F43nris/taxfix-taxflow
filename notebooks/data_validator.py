import csv
import re
import pathlib
import os
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


class DataValidator:
    """
    A class for validating and analyzing financial transaction data.
    
    This class provides methods to:
    - Check data structure and format
    - Clean problematic data
    - Perform exploratory data analysis
    - Generate visualizations
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the DataValidator with paths to data files.
        Args:
            base_dir: Optional base directory containing the data files. If None, use current working directory.
        """
        if base_dir is not None:
            self.data_dir = pathlib.Path(base_dir)
        else:
            self.data_dir = pathlib.Path.cwd()

        # Define data file paths (relative to data_dir)
        self.tx_path = self.data_dir / "transactions.csv"
        self.user_path = self.data_dir / "users.csv"
        self.filings_path = self.data_dir / "tax_filings.csv"
        # Define processed data directory inside notebooks
        self.processed_dir = pathlib.Path.cwd() / "notebooks" / "processed_data"
        self.processed_dir.mkdir(exist_ok=True)
        self.clean_tx_path = self.processed_dir / "transactions_cleaned.csv"

        # Create charts directory at the project root (not under data_dir)
        self.charts_dir = pathlib.Path.cwd() / "charts"
        self.charts_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('ggplot')
        sns.set_palette('Set2')
        
        # Storage for data
        self.tx_data = None
        self.reports = {}
        
        # Check if data files exist
        self._verify_data_files()
    
    def _verify_data_files(self) -> None:
        """Verify that required data files exist."""
        assert self.tx_path.exists(), f"{self.tx_path} missing"
        assert self.user_path.exists(), f"{self.user_path} missing"
        assert self.filings_path.exists(), f"{self.filings_path} missing"
    
    def headline(self, text: str) -> None:
        """Print formatted headline text."""
        print(f"\n\033[1m{text}\033[0m")
    
    def check_row_length(self, file_path: pathlib.Path, expected_length: int = 8) -> List[Tuple[int, list]]:
        """
        Check if all rows in a CSV file have the expected number of columns.
        
        Args:
            file_path: Path to the CSV file to check
            expected_length: Expected number of columns
            
        Returns:
            List of tuples containing line number and row content for bad rows
        """
        self.headline(f"Row length check (expecting {expected_length} columns)")
        
        bad_rows = []
        with file_path.open(newline="") as fh:
            rdr = csv.reader(fh)
            for ln, row in enumerate(rdr, 1):
                if len(row) != expected_length:
                    bad_rows.append((ln, row))
        
        if bad_rows:
            print(f"❌ {len(bad_rows)} rows do not have {expected_length} columns (showing first 5):")
            for ln, row in bad_rows[:5]:
                print(f"  line {ln}: {row}")
        else:
            print("✅ PASS")
        
        self.reports["row_length"] = {
            "passed": len(bad_rows) == 0,
            "bad_rows": bad_rows,
            "expected_length": expected_length
        }
        
        return bad_rows
    
    def remove_problematic_rows(self, line_numbers: List[int]) -> None:
        """
        Remove specific rows from the transactions file and save to clean file.
        
        Args:
            line_numbers: List of line numbers to remove (1-indexed)
        """
        self.headline(f"Removing problematic rows: {line_numbers}")
        
        with self.tx_path.open('r') as file:
            lines = file.readlines()
        
        # Remove problematic lines
        cleaned_lines = [line for i, line in enumerate(lines, 1) if i not in line_numbers]
        
        # Write the cleaned data to the clean file path
        with self.clean_tx_path.open('w') as file:
            file.writelines(cleaned_lines)
        
        print(f"Removed {len(lines) - len(cleaned_lines)} rows. Original: {len(lines)} rows, Cleaned: {len(cleaned_lines)} rows")
    
    def load_transaction_data(self) -> pd.DataFrame:
        """
        Load transaction data into a pandas DataFrame.
        
        Returns:
            DataFrame containing transaction data
        """
        # Define column names
        tx_cols = [
            "transaction_id", "user_id", "transaction_date", "amount", 
            "category", "subcategory", "description", "vendor"
        ]
        
        # Read data
        tx = pd.read_csv(self.clean_tx_path, names=tx_cols, header=None, dtype=str)
        print(f"Loaded {len(tx):,} transaction rows")
        
        # Store data
        self.tx_data = tx
        return tx
    
    def check_uniqueness(self, column: str) -> bool:
        """
        Check if values in a column are unique.
        
        Args:
            column: Column to check for uniqueness
            
        Returns:
            True if all values are unique, False otherwise
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        self.headline(f"TRANSACTIONS – {column} uniqueness")
        dupes = self.tx_data[column].duplicated(keep=False)
        
        if dupes.any():
            print(f"❌ Duplicate {column} values (first 10):")
            print(self.tx_data.loc[dupes, column].unique()[:10])
            passed = False
        else:
            print("✅ PASS")
            passed = True
        
        self.reports[f"{column}_uniqueness"] = {
            "passed": passed,
            "duplicate_count": dupes.sum() if dupes.any() else 0
        }
        
        return passed
    
    def check_date_format(self) -> bool:
        """
        Check if dates are in the correct format and can be parsed.
        
        Returns:
            True if all dates are valid, False otherwise
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        self.headline("TRANSACTIONS – date parseability")
        parsed_dates = pd.to_datetime(self.tx_data["transaction_date"], errors="coerce", format="%Y-%m-%d")
        bad_dates = self.tx_data[parsed_dates.isna()]
        
        if not bad_dates.empty:
            print(f"❌ {len(bad_dates)} un-parseable dates (first 5 shown)")
            display(bad_dates.head())
            passed = False
        else:
            print("✅ PASS")
            passed = True
        
        # Add parsed dates to the DataFrame
        self.tx_data["date"] = parsed_dates
        
        self.reports["date_format"] = {
            "passed": passed,
            "bad_dates_count": len(bad_dates) if not bad_dates.empty else 0
        }
        
        return passed
    
    def check_amount_format(self) -> Tuple[bool, bool]:
        """
        Check transaction amounts for format issues and negative values.
        
        Returns:
            Tuple of (format_check_passed, no_negatives_check_passed)
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        self.headline("TRANSACTIONS – amount format")
        
        # Check amount format
        def good_amount(x: str) -> bool:
            """Check if amount is properly formatted."""
            try:
                float(x)
                return re.match(r"^[+-]?\d+\.\d{1,2}$", x) is not None
            except Exception:
                return False
        
        fmt_issues = self.tx_data[~self.tx_data["amount"].apply(good_amount)]
        negatives = self.tx_data[self.tx_data["amount"].astype(float) < 0]
        
        # Report format issues
        if not fmt_issues.empty:
            print(f"❌ {len(fmt_issues)} rows with bad amount formatting")
            display(fmt_issues.head())
            format_passed = False
        else:
            print("✅ formatting PASS")
            format_passed = True
        
        # Report negative amounts
        if not negatives.empty:
            print(f"⚠️  {len(negatives)} rows have negative amounts (verify refunds)")
            negatives_passed = False
        else:
            print("✅ no negative amounts")
            negatives_passed = True
        
        self.reports["amount_format"] = {
            "format_passed": format_passed,
            "format_issues_count": len(fmt_issues) if not fmt_issues.empty else 0,
            "negatives_passed": negatives_passed,
            "negatives_count": len(negatives) if not negatives.empty else 0
        }
        
        return (format_passed, negatives_passed)
    
    def check_id_formats(self) -> Dict[str, bool]:
        """
        Check that IDs follow the expected format.
        
        Returns:
            Dictionary with check results
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        results = {}
        
        # Check user_id format
        self.headline("TRANSACTIONS - user_id format check")
        def valid_user_id(user_id: str) -> bool:
            """Check if user_id follows the expected format."""
            return bool(re.match(r'^U\d+$', user_id))
        
        invalid_user_ids = self.tx_data[~self.tx_data["user_id"].apply(valid_user_id)]
        
        if not invalid_user_ids.empty:
            print(f"❌ {len(invalid_user_ids)} rows with invalid user_id format")
            display(invalid_user_ids.head())
            results["user_id_passed"] = False
        else:
            print("✅ All user_ids have valid format")
            results["user_id_passed"] = True
        
        # Check transaction_id format
        self.headline("TRANSACTIONS - transaction_id format check")
        def valid_transaction_id(tx_id: str) -> bool:
            """Check if transaction_id follows the expected format."""
            return bool(re.match(r'^T\d+$', tx_id))
        
        invalid_tx_ids = self.tx_data[~self.tx_data["transaction_id"].apply(valid_transaction_id)]
        
        if not invalid_tx_ids.empty:
            print(f"❌ {len(invalid_tx_ids)} rows with invalid transaction_id format")
            display(invalid_tx_ids.head())
            results["transaction_id_passed"] = False
        else:
            print("✅ All transaction_ids have valid format")
            results["transaction_id_passed"] = True
        
        self.reports["id_formats"] = results
        return results
    
    def export_clean_data(self) -> None:
        """
        Export cleaned data with headers to a CSV file (transactions_cleaned.csv).
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        output_path = self.clean_tx_path
        # Always include headers in the output
        self.tx_data.to_csv(output_path, index=False, header=True)
        print(f"Exported cleaned data to {output_path}")
    
    def prepare_data_for_analysis(self) -> None:
        """Prepare transaction data for analysis by converting data types."""
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        # Convert data types for analysis
        self.tx_data['amount'] = self.tx_data['amount'].astype(float)
        self.tx_data['date'] = pd.to_datetime(self.tx_data['transaction_date'])
        self.tx_data['year'] = self.tx_data['date'].dt.year
        self.tx_data['month'] = self.tx_data['date'].dt.month
    
    def analyze_transaction_amounts(self) -> pd.Series:
        """
        Calculate basic statistics for transaction amounts.
        
        Returns:
            Series containing statistical summary of transaction amounts
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        # Ensure data types are appropriate
        if not pd.api.types.is_float_dtype(self.tx_data['amount']):
            self.prepare_data_for_analysis()
        
        # Calculate statistics
        self.headline("Basic Statistics - Transaction Amounts")
        amount_stats = self.tx_data['amount'].describe()
        print(amount_stats)
        
        return amount_stats
    
    def analyze_transaction_categories(self) -> Tuple[pd.Series, pd.Series]:
        """
        Analyze transaction categories and subcategories.
        
        Returns:
            Tuple of (category_counts, subcategory_counts)
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        # Transaction Count by Category
        self.headline("Transaction Count by Category")
        category_counts = self.tx_data['category'].value_counts()
        print(category_counts)
        
        # Transaction Count by Subcategory
        self.headline("Transaction Count by Subcategory")
        subcategory_counts = self.tx_data['subcategory'].value_counts()
        print(subcategory_counts)
        
        return category_counts, subcategory_counts
    
    def analyze_transaction_vendors(self) -> pd.Series:
        """
        Analyze transaction vendors.
        
        Returns:
            Series containing vendor counts
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        # Transaction Count by Vendor
        self.headline("Transaction Count by Vendor")
        vendor_counts = self.tx_data['vendor'].value_counts()
        print(vendor_counts.to_string())
        
        return vendor_counts
    
    def analyze_avg_amount_by_category(self) -> pd.Series:
        """
        Calculate average transaction amount by category.
        
        Returns:
            Series containing average amounts by category
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        # Ensure data types are appropriate
        if not pd.api.types.is_float_dtype(self.tx_data['amount']):
            self.prepare_data_for_analysis()
        
        # Average Transaction Amount by Category
        self.headline("Average Transaction Amount by Category")
        category_avg = self.tx_data.groupby('category')['amount'].mean().sort_values(ascending=False)
        print(category_avg)
        
        return category_avg
    
    def analyze_monthly_transactions(self) -> pd.DataFrame:
        """
        Analyze transaction counts by month.
        
        Returns:
            DataFrame containing monthly transaction counts
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        # Ensure data types are appropriate
        if 'year' not in self.tx_data.columns or 'month' not in self.tx_data.columns:
            self.prepare_data_for_analysis()
        
        # Transactions Over Time (Monthly)
        self.headline("Transactions Over Time (Monthly)")
        
        # Group by year and month, count transactions
        monthly_counts = self.tx_data.groupby(['year', 'month']).size()
        monthly_counts = monthly_counts.reset_index(name='count')
        monthly_counts['date'] = pd.to_datetime(monthly_counts[['year', 'month']].assign(day=1))
        
        return monthly_counts
    
    def plot_monthly_transactions(self, monthly_counts: Optional[pd.DataFrame] = None,
                                file_name: str = "monthly_transactions.png") -> None:
        """
        Plot monthly transaction counts.
        
        Args:
            monthly_counts: DataFrame containing monthly transaction counts. If None, will be calculated.
            file_name: Name of the output file
        """
        if monthly_counts is None:
            monthly_counts = self.analyze_monthly_transactions()
        
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_counts['date'], monthly_counts['count'], marker='o', color='#1f77b4', 
                 linewidth=2, markersize=8, label='Transaction Count')
        plt.title('Number of Transactions per Month', fontsize=16)
        plt.ylabel('Transaction Count', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add data labels
        for x, y in zip(monthly_counts['date'], monthly_counts['count']):
            plt.annotate(str(y), (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        save_path = self.charts_dir / file_name
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")
    
    def plot_amount_distribution(self, file_name: str = "amount_distribution.png") -> None:
        """
        Plot the distribution of transaction amounts.
        
        Args:
            file_name: Name of the output file
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        # Ensure data types are appropriate
        if not pd.api.types.is_float_dtype(self.tx_data['amount']):
            self.prepare_data_for_analysis()
        
        self.headline("Distribution of Transaction Amounts")
        
        plt.figure(figsize=(12, 6))
        # Use histplot without kde, then add KDE separately
        sns.histplot(data=self.tx_data, x='amount', bins=30, color='#1f77b4')
        
        # Add KDE plot separately
        sns.kdeplot(data=self.tx_data, x='amount', color='red', linewidth=2, label='Density')
        
        plt.title('Distribution of Transaction Amounts', fontsize=16)
        plt.xlabel('Amount (€)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        
        # Add mean and median lines
        mean = self.tx_data['amount'].mean()
        median = self.tx_data['amount'].median()
        plt.axvline(mean, color='green', linestyle='--', linewidth=2, label=f'Mean: €{mean:.2f}')
        plt.axvline(median, color='orange', linestyle='-.', linewidth=2, label=f'Median: €{median:.2f}')
        
        plt.legend(fontsize=12)
        plt.tight_layout()
        save_path = self.charts_dir / file_name
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")
    
    def plot_category_distribution(self, file_name: str = "category_distribution.png") -> None:
        """
        Plot the distribution of transaction categories.
        
        Args:
            file_name: Name of the output file
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        self.headline("Distribution of Transaction Categories")
        
        # Get category counts
        category_counts = self.tx_data['category'].value_counts()
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
        plt.title('Transaction Count by Category', fontsize=16)
        plt.xlabel('Category', fontsize=14)
        plt.ylabel('Number of Transactions', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add data labels
        for i, v in enumerate(category_counts.values):
            ax.text(i, v + 5, str(v), ha='center', fontsize=10)
        
        plt.tight_layout()
        save_path = self.charts_dir / file_name
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")
    
    def plot_category_amount_comparison(self, file_name: str = "category_amount_comparison.png") -> None:
        """
        Plot a comparison of average amount vs. transaction count by category.
        
        Args:
            file_name: Name of the output file
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        # Ensure data types are appropriate
        if not pd.api.types.is_float_dtype(self.tx_data['amount']):
            self.prepare_data_for_analysis()
        
        self.headline("Category Amount vs. Count Comparison")
        
        # Calculate metrics by category
        category_stats = self.tx_data.groupby('category').agg(
            avg_amount=('amount', 'mean'),
            count=('transaction_id', 'count')
        ).reset_index()
        
        plt.figure(figsize=(14, 8))
        
        # Create scatter plot
        scatter = plt.scatter(
            x=category_stats['count'], 
            y=category_stats['avg_amount'],
            s=category_stats['count'] * 2,  # Size based on count
            c=range(len(category_stats)),   # Color by index
            cmap='viridis',
            alpha=0.7
        )
        
        # Add category labels
        for i, row in category_stats.iterrows():
            plt.annotate(
                row['category'], 
                (row['count'], row['avg_amount']),
                xytext=(7, 0),
                textcoords='offset points',
                fontsize=9
            )
        
        plt.title('Average Amount vs. Transaction Count by Category', fontsize=16)
        plt.xlabel('Number of Transactions', fontsize=14)
        plt.ylabel('Average Amount (€)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add reference lines for overall average
        overall_avg = self.tx_data['amount'].mean()
        plt.axhline(y=overall_avg, color='red', linestyle='--', 
                   label=f'Overall Average: €{overall_avg:.2f}')
        
        plt.legend()
        plt.tight_layout()
        save_path = self.charts_dir / file_name
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")
    
    def plot_top_vendors(self, top_n: int = 15, file_name: str = "top_vendors.png") -> None:
        """
        Plot the top vendors by transaction count.
        
        Args:
            top_n: Number of top vendors to display
            file_name: Name of the output file
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        self.headline(f"Top {top_n} Vendors by Transaction Count")
        
        # Get vendor counts
        vendor_counts = self.tx_data['vendor'].value_counts().head(top_n)
        
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x=vendor_counts.index, y=vendor_counts.values, palette='Blues_d')
        plt.title(f'Top {top_n} Vendors by Transaction Count', fontsize=16)
        plt.xlabel('Vendor', fontsize=14)
        plt.ylabel('Number of Transactions', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add data labels
        for i, v in enumerate(vendor_counts.values):
            ax.text(i, v + 1, str(v), ha='center', fontsize=9)
        
        plt.tight_layout()
        save_path = self.charts_dir / file_name
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")
    
    def create_visualization_dashboard(self, file_name: str = "transaction_dashboard.html") -> None:
        """
        Create an HTML dashboard with all visualizations.
        
        Args:
            file_name: Name of the output HTML file
        """
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        
        self.headline("Creating Visualization Dashboard")
        
        # Get list of all PNG files in the charts directory
        chart_files = list(self.charts_dir.glob("*.png"))
        
        if not chart_files:
            print("No charts found. Run visualization methods first.")
            return
        
        # Create HTML content
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>Transaction Data Visualization Dashboard</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        h1 { color: #333366; }",
            "        .chart-container { margin-bottom: 30px; }",
            "        .chart-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }",
            "        img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>Transaction Data Visualization Dashboard</h1>",
            "    <p>Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "</p>"
        ]
        
        # Add each chart to the HTML
        for chart_file in chart_files:
            chart_name = chart_file.stem.replace('_', ' ').title()
            html_content.extend([
                f"    <div class='chart-container'>",
                f"        <div class='chart-title'>{chart_name}</div>",
                f"        <img src='{chart_file.name}' alt='{chart_name}'>",
                f"    </div>"
            ])
        
        # Close HTML
        html_content.extend([
            "</body>",
            "</html>"
        ])
        
        # Write HTML file
        dashboard_path = self.charts_dir / file_name
        with open(dashboard_path, 'w') as f:
            f.write('\n'.join(html_content))
        
        print(f"Dashboard created at {dashboard_path}")
    
    def run_visualization_suite(self) -> None:
        """Run all visualization methods and save the plots."""
        if self.tx_data is None:
            raise ValueError("No transaction data loaded. Call load_transaction_data() first.")
        # Ensure data is prepared for analysis
        self.prepare_data_for_analysis()
        # Generate all visualizations
        self.headline("Generating Visualization Suite")
        # Basic visualizations
        monthly_data = self.analyze_monthly_transactions()
        self.plot_monthly_transactions(monthly_data)
        self.plot_amount_distribution()
        # Additional visualizations
        self.plot_category_distribution()
        self.plot_category_amount_comparison()
        self.plot_top_vendors()
        # Create dashboard HTML
        self.create_visualization_dashboard()
        print("\n✅ All visualizations generated successfully!")
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate a summary report of all validation checks.
        
        Returns:
            Dictionary containing validation results
        """
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_files": {
                "transactions": str(self.tx_path),
                "clean_transactions": str(self.clean_tx_path)
            },
            "record_count": len(self.tx_data) if self.tx_data is not None else None,
            "validation_results": self.reports
        }
    
    def run_validation_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete validation pipeline.
        
        Returns:
            Dictionary with validation report
        """
        # Step 1: Check raw data structure
        bad_rows = self.check_row_length(self.tx_path)
        
        # Step 2: Clean data if needed
        if bad_rows:
            print("Found problematic rows, cleaning data...")
            self.remove_problematic_rows([row[0] for row in bad_rows])
            # Verify cleaned data
            self.check_row_length(self.clean_tx_path)
        else:
            # If no bad rows, just copy the original file to the clean path
            with self.tx_path.open('r') as file:
                lines = file.readlines()
            with self.clean_tx_path.open('w') as file:
                file.writelines(lines)
            print("No problematic rows found, created clean copy of original file.")
        
        # Step 3: Load data into DataFrame
        self.load_transaction_data()
        
        # Step 4: Run validation checks
        self.check_uniqueness("transaction_id")
        self.check_date_format()
        self.check_amount_format()
        self.check_id_formats()
        
        # Step 5: Export cleaned data
        self.export_clean_data()
        
        # Step 6: Prepare data for analysis
        self.prepare_data_for_analysis()
        
        # Step 7: Run analyses
        self.analyze_transaction_amounts()
        self.analyze_transaction_categories()
        self.analyze_transaction_vendors()
        self.analyze_avg_amount_by_category()
        
        # Step 8: Generate all visualizations
        self.run_visualization_suite()
        
        # Return validation report
        return self.generate_validation_report()

def main():
    """Main function to run the validation pipeline."""
    validator = DataValidator(base_dir="notebooks/unprocessed_data")
    report = validator.run_validation_pipeline()
    
    print("\n" + "="*50)
    print("VALIDATION COMPLETE")
    print("="*50)
    print(f"Processed {report['record_count']} records")
    
    # Summarize validation results
    all_passed = all(result.get('passed', False) for result in report['validation_results'].values() 
                     if isinstance(result, dict) and 'passed' in result)
    
    if all_passed:
        print("✅ All validation checks passed!")
    else:
        print("⚠️  Some validation checks failed. See detailed report above.")
    
    print("="*50)
    print("Visualizations saved as PNG files.")

if __name__ == "__main__":
    main() 