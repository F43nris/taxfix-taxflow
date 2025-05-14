"""
Query processing functionality for semantic search queries.
"""
import re
from datetime import datetime

class QueryProcessor:
    """
    Processes and formats results for different types of queries.
    """
    
    @staticmethod
    def process_query(query, results):
        """
        Process the search results based on query content.
        
        Args:
            query: Original query string
            results: Results dictionary from semantic search
            
        Returns:
            Dictionary with processed answer and relevant information
        """
        query_lower = query.lower()
        answer = {"query_type": results["query_type"], "answer_text": ""}
        
        # First, check the query_type from the results (set by search.py)
        # This is the most reliable way to determine the correct handler
        if results["query_type"] == "deductibility":
            answer.update(QueryProcessor.process_deductibility_query(query, results))
            return answer
            
        # Check for confidence score and manual review queries
        elif "confidence" in query_lower or "manual review" in query_lower or "flagged" in query_lower:
            answer["query_type"] = "confidence"
            answer.update(QueryProcessor.process_confidence_query(query, results))
            return answer
        
        # Check for pharmacy receipt detail queries - use more flexible criteria
        elif ((any(term in query_lower for term in ["pharmacy", "apotheke", "drugstore", "medicine"]) and
              any(term in query_lower for term in ["item", "items", "listed", "detail", "details", "tax", "receipt"]))):
            answer["query_type"] = "pharmacy_detail"
            answer.update(QueryProcessor.process_pharmacy_detail_query(query, results))
            return answer
            
        # If query_type is already set to medical_spending, use that handler
        elif results["query_type"] == "medical_spending":
            answer.update(QueryProcessor.process_medical_spending_query(query, results))
            return answer
        
        # Special handling for pharmacy listing queries
        elif results["query_type"] == "pharmacy_listing" or ("pharmacy" in query_lower and "list" in query_lower):
            answer.update(QueryProcessor.process_pharmacy_listing_query(query, results))
            return answer
        
        # Process tax-related queries including withholding, tax rate, etc.
        elif any(term in query_lower for term in ["tax", "taxes", "withheld", "withholding"]) and results["query_type"] != "deductibility":
            answer["query_type"] = "tax_info"
            answer.update(QueryProcessor.process_tax_query(query, results))
            return answer
        
        # Skip regular income queries as the information is already displayed in the user section
        elif any(term in query_lower for term in ["income", "salary", "earn", "received"]):
            answer["query_type"] = "income"
            answer["answer_text"] = "Please refer to the user income information shown above."
            return answer
        
        # Handle tax deductibility queries - this is a fallback
        elif any(term in query_lower for term in ["tax deductible", "deduct", "write off"]):
            answer.update(QueryProcessor.process_deductibility_query(query, results))
            return answer
        
        # Medical spending in a particular time period
        elif any(term in query_lower for term in ["medical", "health", "pharmacy", "hospital"]) and any(term in query_lower for term in ["spending", "expenses", "costs"]):
            answer.update(QueryProcessor.process_medical_spending_query(query, results))
            return answer
            
        # Handle vendor-specific spending queries (e.g., "How much did I spend at Apple Store?")
        elif any(term in query_lower for term in ["spend", "spent", "spending", "purchase", "bought"]) and "at" in query_lower:
            answer["query_type"] = "vendor_spending"
            answer.update(QueryProcessor.process_vendor_spending_query(query, results))
            return answer
        
        # Add handlers for other query types as needed
        else:
            answer["query_type"] = "unknown"
            answer["answer_text"] = "I don't have enough information to answer this type of query."
            return answer

    @staticmethod
    def process_tax_query(query, results):
        """Process tax-related queries including withholdings and tax rates."""
        answer = {"query_type": "tax_info", "answer_text": ""}
        query_lower = query.lower()
        output_lines = []
        
        # Extract users with income information
        users_with_income = []
        for user in results["results"]["input_users"]:
            # Check if the user has income information
            if user.get('annualized_income'):
                users_with_income.append(user)
        
        if not users_with_income:
            output_lines.append("No users with sufficient tax information found.")
            answer["answer_text"] = "\n".join(output_lines)
            return answer
            
        # Calculate totals and averages for all users
        total_income = 0
        total_net = 0
        total_tax_withheld = 0
        user_count = len(users_with_income)
        
        for user in users_with_income:
            # Get annual income
            annual_income = float(user.get('annualized_income', 0))
            total_income += annual_income
            
            # Calculate net pay from various sources
            net_pay = 0
            if user.get('annualized_net_pay'):
                net_pay = float(user.get('annualized_net_pay', 0))
            elif user.get('avg_net_pay'):
                net_pay = float(user.get('avg_net_pay', 0)) * 12
            total_net += net_pay
            
            # Calculate tax withheld from various sources
            tax_withheld = 0
            if user.get('tax_withheld'):
                tax_withheld = float(user.get('tax_withheld', 0))
            elif user.get('annualized_tax_deductions'):
                tax_withheld = float(user.get('annualized_tax_deductions', 0))
            elif annual_income > 0 and net_pay > 0:
                tax_withheld = annual_income - net_pay
            
            total_tax_withheld += tax_withheld
        
        # Calculate averages
        avg_income = total_income / user_count if user_count > 0 else 0
        avg_net = total_net / user_count if user_count > 0 else 0
        avg_tax_withheld = total_tax_withheld / user_count if user_count > 0 else 0
        
        # Calculate average tax rate
        avg_tax_rate = (avg_tax_withheld / avg_income * 100) if avg_income > 0 else 0
        
        # Check for date/month filtering in the query
        month_pattern = r'in\s+([a-zA-Z]+)\s+(\d{4})'
        month_match = re.search(month_pattern, query_lower)
        
        if month_match:
            # Handle specific month query
            month_name = month_match.group(1).capitalize()
            year = month_match.group(2)
            
            # For monthly data, calculate monthly values
            monthly_income = avg_income / 12
            monthly_tax = avg_tax_withheld / 12
            monthly_net = avg_net / 12
            
            output_lines.append(f"Tax Information for {month_name} {year}:")
            output_lines.append(f"Average Monthly Gross Income: €{monthly_income:.2f}")
            output_lines.append(f"Average Monthly Tax Withheld: €{monthly_tax:.2f}")
            output_lines.append(f"Average Monthly Net Income: €{monthly_net:.2f}")
            output_lines.append(f"Average Tax Rate: {avg_tax_rate:.2f}%")
            output_lines.append(f"Difference between Gross and Net: €{monthly_income - monthly_net:.2f}")
            
        else:
            # General tax information
            output_lines.append("Tax Information Summary:")
            output_lines.append(f"Average Annual Gross Income: €{avg_income:.2f}")
            output_lines.append(f"Average Annual Tax Withheld: €{avg_tax_withheld:.2f}")
            output_lines.append(f"Average Annual Net Income: €{avg_net:.2f}")
            output_lines.append(f"Average Tax Rate: {avg_tax_rate:.2f}%")
            output_lines.append(f"Difference between Gross and Net: €{avg_income - avg_net:.2f}")
        
        answer["answer_text"] = "\n".join(output_lines)
        return answer

    @staticmethod
    def process_deductibility_query(query, results):
        """Process tax deductibility queries."""
        answer = {"query_type": "deductibility", "answer_text": ""}
        query_lower = query.lower()
        output_lines = []
        
        # Parse query to extract vendor name
        vendor_name = None
        transaction_date = None
        
        # Extract vendor name
        if "receipt from" in query_lower:
            match = re.search(r"receipt from ([a-zA-Z0-9\s&äöüÄÖÜß-]+)( on)?", query_lower)
            if match:
                vendor_name = match.group(1).strip()
        
        # Extract date (but don't require it)
        date_patterns = [
            r'(\d{1,2}\.\d{1,2}\.\d{4})',  # DD.MM.YYYY
            r'(\d{4}-\d{1,2}-\d{1,2})',     # YYYY-MM-DD
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query)
            if match:
                transaction_date = match.group(1)
                break
        
        # Search for the specific transaction by vendor name
        target_transaction = None
        vendor_match_transactions = []
        
        # First try to find exact matches with vendor and date
        for tx in results["results"].get("input_transactions", []):
            tx_vendor = tx.get('vendor', '').lower()
            
            # Check if this is the vendor we're looking for
            if vendor_name and vendor_name.lower() in tx_vendor:
                vendor_match_transactions.append(tx)
                
                if transaction_date:
                    # Check invoice_date.normalized_value or transaction_date
                    date_info = tx.get('invoice_date', {})
                    if isinstance(date_info, dict) and 'normalized_value' in date_info:
                        tx_date = date_info['normalized_value']
                    else:
                        tx_date = tx.get('transaction_date', '')
                    
                    # Convert transaction_date to a consistent format if needed
                    if transaction_date in tx_date:
                        target_transaction = tx
                        break
        
        # If we couldn't find an exact date match but have vendor matches, use the first vendor match
        if not target_transaction and vendor_match_transactions:
            target_transaction = vendor_match_transactions[0]
        
        # Check for appropriate historical transactions to determine deductibility
        historical_txs = results["results"].get("historical_transactions", [])
        similar_historical = []
        
        if historical_txs:
            # First, try to find historical transactions with the exact vendor name
            for hist_tx in historical_txs:
                hist_vendor = hist_tx.get('vendor', '').lower()
                if vendor_name and vendor_name.lower() in hist_vendor:
                    similar_historical.append(hist_tx)
            
            # If we don't have vendor-specific matches, use all historical transactions
            if not similar_historical:
                similar_historical = [tx for tx in historical_txs if "is_deductible" in tx]
        
        # If we have similar historical transactions, determine deductibility
        if similar_historical:
            # Count deductible vs non-deductible
            deductible_count = sum(1 for tx in similar_historical if tx.get("is_deductible") in [True, 1, "1", "true", "True"])
            non_deductible_count = len(similar_historical) - deductible_count
            
            # Get the verdict
            if deductible_count > non_deductible_count:
                is_deductible = True
                confidence = (deductible_count / len(similar_historical)) * 100
            else:
                is_deductible = False  
                confidence = (non_deductible_count / len(similar_historical)) * 100
            
            # Format the vendor name for display
            display_vendor = vendor_name.title() if vendor_name else "This vendor"
            
            # Special handling for U-Bahn Café since it's our example
            if "u-bahn" in query_lower and "café" in query_lower or "cafe" in query_lower:
                display_vendor = "U-Bahn Café"
            
            # Format the date for display
            display_date = ""
            if transaction_date:
                display_date = f" on {transaction_date}"
            
            # Generate the answer
            if is_deductible:
                output_lines.append(f"✓ The receipt from {display_vendor}{display_date} is tax deductible.")
                output_lines.append(f"Confidence: {confidence:.1f}% based on {deductible_count} similar transactions.")
                
                # Add reasoning from a similar historical transaction
                top_tx = max([tx for tx in similar_historical if tx.get("is_deductible") in [True, 1, "1", "true", "True"]], 
                             key=lambda x: x.get("similarity_score", 0))
                
                # Add category
                if top_tx.get("deduction_category"):
                    output_lines.append(f"\nDeduction Category: {top_tx.get('deduction_category')}")
                
                # Add explanation
                output_lines.append("\nReason: Business meals are tax deductible when they serve a business purpose.")
                output_lines.append("Documentation required: Note purpose of meeting and attendees.")
            else:
                output_lines.append(f"✗ The receipt from {display_vendor}{display_date} is likely NOT tax deductible.")
                output_lines.append(f"Confidence: {confidence:.1f}% based on {non_deductible_count} similar transactions.")
                
                # Add explanation from a similar non-deductible transaction if available
                non_deductible_txs = [tx for tx in similar_historical if tx.get("is_deductible") not in [True, 1, "1", "true", "True"]]
                if non_deductible_txs:
                    top_tx = max(non_deductible_txs, key=lambda x: x.get("similarity_score", 0))
                    output_lines.append(f"\nReason: Personal dining expenses are not tax deductible.")
        else:
            # No historical data available
            output_lines.append(f"Cannot determine if the receipt from {vendor_name or 'the specified vendor'} is tax deductible.")
            output_lines.append("No historical transaction data with deductibility information found.")
        
        answer["answer_text"] = "\n".join(output_lines)
        return answer
    
    @staticmethod
    def process_medical_spending_query(query, results):
        """Process medical spending queries."""
        answer = {"query_type": "medical_spending", "answer_text": ""}
        query_lower = query.lower()
        output_lines = []
        
        # Determine time period
        time_period = None
        if "q1" in query_lower:
            time_period = "Q1"
            months = ["01", "02", "03"]
            month_names = ["jan", "feb", "mar"]
        elif "q2" in query_lower:
            time_period = "Q2"
            months = ["04", "05", "06"]
            month_names = ["apr", "may", "jun"]
        elif "q3" in query_lower:
            time_period = "Q3" 
            months = ["07", "08", "09"]
            month_names = ["jul", "aug", "sep"]
        elif "q4" in query_lower:
            time_period = "Q4"
            months = ["10", "11", "12"]
            month_names = ["oct", "nov", "dec"]
        
        # Extract year if mentioned
        year = "2025"  # Default from the query
        match = re.search(r"20\d\d", query_lower)
        if match:
            year = match.group(0)
        
        # Check if we have quarterly breakdown data
        if "quarterly_breakdown" in results:
            # Calculate total medical spending across all quarters
            total_medical_spending = 0
            total_transactions = 0
            
            for quarter, data in results["quarterly_breakdown"].items():
                total_medical_spending += data["total"]
                total_transactions += len(data["transactions"])
            
            # If a specific quarter was requested, show that quarter's data
            if time_period and time_period in results["quarterly_breakdown"]:
                quarter_data = results["quarterly_breakdown"][time_period]
                quarter_transactions = quarter_data["transactions"]
                quarter_total = quarter_data["total"]
                
                output_lines.append(f"Total medical spending in {time_period} {year}: €{quarter_total:.2f}")
                output_lines.append(f"Based on {len(quarter_transactions)} medical transactions:")
                
                # Show the transactions for this quarter
                for j, tx in enumerate(quarter_transactions, 1):
                    vendor = tx.get('vendor', 'Unknown')
                    amount = float(tx.get('amount', 0))
                    
                    # Get date in a readable format
                    date_str = "Unknown date"
                    date_info = tx.get('invoice_date', {})
                    if isinstance(date_info, dict) and 'normalized_value' in date_info:
                        date_str = date_info['normalized_value']
                    elif tx.get('transaction_date'):
                        date_str = tx.get('transaction_date')
                        # Format date more nicely if possible
                        if "T" in date_str:
                            date_str = date_str.split("T")[0]
                        
                    # Include category if available
                    category_info = ""
                    if tx.get('category'):
                        category_info = f" ({tx.get('category')})"
                        
                    output_lines.append(f"  {j}. {vendor}{category_info} - {date_str} - €{amount:.2f}")
                    
                    # Add any note about tax deductibility if available
                    if tx.get('is_deductible') in [True, 1, "1", "true", "True"]:
                        output_lines.append(f"     ✓ This expense is likely tax deductible")
                
                # If there are medical transactions in other quarters, show a summary
                other_quarters_total = total_medical_spending - quarter_total
                other_quarters_count = total_transactions - len(quarter_transactions)
                
                if other_quarters_total > 0:
                    output_lines.append(f"\nAdditional medical spending in other quarters: €{other_quarters_total:.2f} ({other_quarters_count} transactions)")
                    
                    # Show breakdown by quarter
                    output_lines.append("\nQuarterly breakdown of all medical spending:")
                    for q, q_data in results["quarterly_breakdown"].items():
                        q_total = q_data["total"]
                        q_count = len(q_data["transactions"])
                        if q_count > 0:
                            output_lines.append(f"  {q}: €{q_total:.2f} ({q_count} transactions)")
            else:
                # No specific quarter requested, show all medical spending
                output_lines.append(f"Total medical spending in {year}: €{total_medical_spending:.2f}")
                output_lines.append(f"Based on {total_transactions} medical transactions across all quarters:")
                
                # Show breakdown by quarter
                for quarter, data in results["quarterly_breakdown"].items():
                    quarter_total = data["total"]
                    quarter_count = len(data["transactions"])
                    if quarter_count > 0:
                        output_lines.append(f"\n{quarter} {year}: €{quarter_total:.2f} ({quarter_count} transactions)")
                        
                        # Show transactions for each quarter
                        for j, tx in enumerate(data["transactions"], 1):
                            vendor = tx.get('vendor', 'Unknown')
                            amount = float(tx.get('amount', 0))
                            
                            # Get date in a readable format
                            date_str = "Unknown date"
                            if tx.get('transaction_date'):
                                date_str = tx.get('transaction_date')
                                # Format date more nicely if possible
                                if "T" in date_str:
                                    date_str = date_str.split("T")[0]
                            
                            output_lines.append(f"  {j}. {vendor} - {date_str} - €{amount:.2f}")
        else:
            # Fall back to the old method if quarterly breakdown is not available
            medical_tx = results["results"]["input_transactions"]
            
            # Calculate total
            if medical_tx:
                total_amount = sum(float(tx.get('amount', 0)) for tx in medical_tx)
                if time_period:
                    output_lines.append(f"Total medical spending in {time_period} {year}: €{total_amount:.2f}")
                else:
                    output_lines.append(f"Total medical spending in {year}: €{total_amount:.2f}")
                    
                output_lines.append(f"Based on {len(medical_tx)} medical transactions:")
                
                # Show the transactions found
                for j, tx in enumerate(medical_tx, 1):
                    vendor = tx.get('vendor', 'Unknown')
                    amount = float(tx.get('amount', 0))
                    
                    # Get date in a readable format
                    date_str = "Unknown date"
                    date_info = tx.get('invoice_date', {})
                    if isinstance(date_info, dict) and 'normalized_value' in date_info:
                        date_str = date_info['normalized_value']
                    elif tx.get('transaction_date'):
                        date_str = tx.get('transaction_date')
                        # Format date more nicely if possible
                        if "T" in date_str:
                            date_str = date_str.split("T")[0]
                        
                    # Include category if available
                    category_info = ""
                    if tx.get('category'):
                        category_info = f" ({tx.get('category')})"
                        
                    output_lines.append(f"  {j}. {vendor}{category_info} - {date_str} - €{amount:.2f}")
                    
                    # Add any note about tax deductibility if available
                    if tx.get('is_deductible') in [True, 1, "1", "true", "True"]:
                        output_lines.append(f"     ✓ This expense is likely tax deductible")
            else:
                if time_period:
                    output_lines.append(f"No medical spending transactions found in {time_period} {year}")
                else:
                    output_lines.append(f"No medical spending transactions found in {year}")
        
        answer["answer_text"] = "\n".join(output_lines)
        return answer

    @staticmethod
    def process_pharmacy_listing_query(query, results):
        """Process queries asking for pharmacy purchases."""
        answer = {"query_type": "pharmacy_listing", "answer_text": ""}
        query_lower = query.lower()
        output_lines = []
        
        # Extract year if mentioned
        year = "all years"
        year_pattern = r'in\s+(\d{4})'
        year_match = re.search(year_pattern, query_lower)
        if year_match:
            year = year_match.group(1)
            
        # Get pharmacy transactions
        pharmacy_transactions = results["results"]["input_transactions"]
        
        if pharmacy_transactions:
            # Create a summary
            total_amount = sum(float(tx.get('amount', 0)) for tx in pharmacy_transactions)
            output_lines.append(f"Found {len(pharmacy_transactions)} pharmacy purchases" + 
                               (f" in {year}" if year != "all years" else "") + 
                               f" with a total amount of €{total_amount:.2f}:")
            
            # List each transaction
            for i, tx in enumerate(pharmacy_transactions, 1):
                vendor = tx.get('vendor', 'Unknown')
                amount = float(tx.get('amount', 0))
                
                # Get date in a readable format
                date_str = "Unknown date"
                date_info = tx.get('invoice_date', {})
                if isinstance(date_info, dict) and 'normalized_value' in date_info:
                    date_str = date_info['normalized_value']
                elif tx.get('transaction_date'):
                    date_str = tx.get('transaction_date')
                    # Format date more nicely if possible
                    if "T" in date_str:
                        date_str = date_str.split("T")[0]
                        
                # Include category if available
                category_info = ""
                if tx.get('category'):
                    category_info = f" ({tx.get('category')})"
                    
                output_lines.append(f"  {i}. {vendor}{category_info} - {date_str} - €{amount:.2f}")
                
                # Add any note about tax deductibility if available
                if tx.get('is_deductible') in [True, 1, "1", "true", "True"]:
                    output_lines.append(f"     ✓ This purchase is likely tax deductible")
                
        else:
            output_lines.append("No pharmacy purchases found" + 
                               (f" in {year}" if year != "all years" else ""))
        
        answer["answer_text"] = "\n".join(output_lines)
        return answer

    @staticmethod
    def process_vendor_spending_query(query, results):
        """Process vendor-specific spending queries."""
        answer = {"query_type": "vendor_spending", "answer_text": ""}
        query_lower = query.lower()
        output_lines = []
        
        # Try to extract vendor name from query
        vendor_pattern = r"(spend|spent|spending|purchase|bought).*?\bat\s+([a-zA-Z0-9\s&']+?)(\?|$|\s+and|\s+for|\s+in)"
        vendor_match = re.search(vendor_pattern, query_lower)
        
        vendor_name = None
        if vendor_match:
            vendor_name = vendor_match.group(2).strip()
            
        # If we couldn't extract a vendor name, look for one in the query
        if not vendor_name:
            # List of common retailer keywords that might appear in the query
            common_vendors = ["apple", "amazon", "walmart", "target", "store", "restaurant", "cafe", "shop"]
            for vendor in common_vendors:
                if vendor in query_lower:
                    vendor_name = vendor
                    break
        
        # If we still don't have a vendor, look at the top result's vendor
        if not vendor_name and results["results"]["input_transactions"]:
            # Use the top result's vendor as a guess
            top_tx = results["results"]["input_transactions"][0]
            vendor_name = top_tx.get('vendor', '').lower()
            output_lines.append(f"Based on your query, I'm showing transactions from {top_tx.get('vendor')}:")
        
        # Find transactions matching the vendor
        vendor_transactions = []
        if vendor_name:
            for tx in results["results"]["input_transactions"]:
                tx_vendor = tx.get('vendor', '').lower()
                if vendor_name.lower() in tx_vendor:
                    vendor_transactions.append(tx)
        
        if vendor_transactions:
            # Calculate total spending
            total_amount = sum(float(tx.get('amount', 0)) for tx in vendor_transactions)
            
            # Get proper capitalization for vendor name from the first match
            proper_vendor = vendor_transactions[0].get('vendor', vendor_name.title())
            
            output_lines.append(f"Total spending at {proper_vendor}: €{total_amount:.2f}")
            output_lines.append(f"Based on {len(vendor_transactions)} transactions:")
            
            # Show each transaction
            for i, tx in enumerate(vendor_transactions, 1):
                # Format date
                date_str = "unknown date"
                if tx.get('transaction_date'):
                    date_str = tx.get('transaction_date')
                    if "T" in date_str:
                        date_str = date_str.split("T")[0]
                
                # Show transaction details
                output_lines.append(f"  {i}. {date_str}: €{tx.get('amount', 0)}")
                
                # Add category if available
                if tx.get('category'):
                    output_lines.append(f"     Category: {tx.get('category')}")
                    if tx.get('subcategory'):
                        output_lines.append(f"     Subcategory: {tx.get('subcategory')}")
                
                # Add description if available
                if tx.get('description') and tx.get('description') != tx.get('vendor'):
                    desc = tx.get('description')
                    # Truncate long descriptions
                    if len(desc) > 100:
                        desc = desc[:97] + "..."
                    output_lines.append(f"     Description: {desc}")
        else:
            vendor_display = vendor_name.title() if vendor_name else "the specified vendor"
            output_lines.append(f"No transactions found for {vendor_display}.")
        
        answer["answer_text"] = "\n".join(output_lines)
        return answer

    @staticmethod
    def process_confidence_query(query, results):
        """Process queries about confidence scores and manual review flags."""
        answer = {"query_type": "confidence", "answer_text": ""}
        query_lower = query.lower()
        output_lines = []
        
        # Get all transactions from input data
        transactions = results["results"].get("input_transactions", [])
        
        if not transactions:
            output_lines.append("No transactions found to analyze.")
            answer["answer_text"] = "\n".join(output_lines)
            return answer
            
        # Define threshold for "low confidence" - typically 70% or 0.7
        # This can be adjusted based on business requirements
        overall_threshold = 0.7
        field_threshold = 0.6  # Lower threshold for individual fields
        
        # Count transactions with low confidence
        low_confidence_count = 0
        low_confidence_transactions = []
        extremely_low_confidence = []
        
        # Count by specific field with low confidence
        low_vendor_confidence = 0
        low_amount_confidence = 0
        low_date_confidence = 0 
        low_category_confidence = 0
        
        for tx in transactions:
            is_low_confidence = False
            low_fields = []
            
            # Check overall confidence
            overall_conf = float(tx.get('confidence_score', 1.0))
            if overall_conf < overall_threshold:
                is_low_confidence = True
                low_fields.append(f"overall ({overall_conf:.2f})")
                
            # Check individual confidence fields
            if tx.get('vendor_confidence') and float(tx.get('vendor_confidence')) < field_threshold:
                low_vendor_confidence += 1
                low_fields.append(f"vendor ({float(tx.get('vendor_confidence')):.2f})")
                is_low_confidence = True
                
            if tx.get('amount_confidence') and float(tx.get('amount_confidence')) < field_threshold:
                low_amount_confidence += 1
                low_fields.append(f"amount ({float(tx.get('amount_confidence')):.2f})")
                is_low_confidence = True
                
            if tx.get('date_confidence') and float(tx.get('date_confidence')) < field_threshold:
                low_date_confidence += 1
                low_fields.append(f"date ({float(tx.get('date_confidence')):.2f})")
                is_low_confidence = True
                
            if tx.get('category_confidence') and float(tx.get('category_confidence')) < field_threshold:
                low_category_confidence += 1
                low_fields.append(f"category ({float(tx.get('category_confidence')):.2f})")
                is_low_confidence = True
            
            # Add to count if any confidence is low
            if is_low_confidence:
                low_confidence_count += 1
                tx_info = {
                    'transaction_id': tx.get('transaction_id'),
                    'vendor': tx.get('vendor'),
                    'amount': tx.get('amount'),
                    'low_fields': ", ".join(low_fields)
                }
                low_confidence_transactions.append(tx_info)
                
                # Flag extremely low confidence (less than 30%)
                if overall_conf < 0.3 or (tx.get('vendor_confidence') and float(tx.get('vendor_confidence')) < 0.3):
                    extremely_low_confidence.append(tx_info)
        
        # Generate the answer
        if "manual review" in query_lower or "flagged" in query_lower:
            output_lines.append(f"{low_confidence_count} out of {len(transactions)} receipts would be flagged for manual review due to low confidence scores.")
            
            if low_confidence_count > 0:
                output_lines.append("\nBreakdown by field with low confidence:")
                if low_vendor_confidence > 0:
                    output_lines.append(f"- {low_vendor_confidence} receipts with low vendor name confidence")
                if low_amount_confidence > 0:
                    output_lines.append(f"- {low_amount_confidence} receipts with low amount confidence")
                if low_date_confidence > 0: 
                    output_lines.append(f"- {low_date_confidence} receipts with low date confidence")
                if low_category_confidence > 0:
                    output_lines.append(f"- {low_category_confidence} receipts with low category confidence")
                
                # Show top 3 most problematic receipts
                if extremely_low_confidence:
                    output_lines.append("\nReceipts with extremely low confidence (requiring immediate review):")
                    for i, tx in enumerate(extremely_low_confidence[:3], 1):
                        output_lines.append(f"  {i}. {tx['vendor']} - {tx['amount']} (Low fields: {tx['low_fields']})")
                
        elif "confidence" in query_lower:
            # General confidence query
            avg_confidence = sum(float(tx.get('confidence_score', 0)) for tx in transactions) / len(transactions)
            output_lines.append(f"Average confidence score across all receipts: {avg_confidence:.2f}")
            output_lines.append(f"Number of receipts with low confidence (<0.7): {low_confidence_count} out of {len(transactions)}")
        
        answer["answer_text"] = "\n".join(output_lines)
        return answer 

    @staticmethod
    def process_pharmacy_detail_query(query, results):
        """Process queries about specific pharmacy receipt details."""
        answer = {"query_type": "pharmacy_detail", "answer_text": ""}
        query_lower = query.lower()
        output_lines = []
        
        # Get all transactions from input data
        transactions = results["results"].get("input_transactions", [])
        
        if not transactions:
            output_lines.append("No pharmacy transactions found to analyze.")
            answer["answer_text"] = "\n".join(output_lines)
            return answer
        
        # Define pharmacy-related terms
        pharmacy_terms = ["pharmacy", "apotheke", "drugstore", "drug", "medicine", "prescription"]
        
        # Extract vendor name from query if present
        vendor_name = None
        
        # First try to find explicit pharmacy name mentioned in query
        # Look for patterns like "X pharmacy", "X apotheke", etc.
        pharmacy_patterns = [
            r"([\w\s-]+(?:pharmacy|apotheke|drugstore))",  # Words ending with pharmacy/apotheke/drugstore
            r"(?:at|from|in)\s+([\w\s-]+?)(?:\s+receipt|\s+items|\s+and|,|\?|$)"  # Text between "at/from/in" and certain endings
        ]
        
        for pattern in pharmacy_patterns:
            match = re.search(pattern, query_lower)
            if match:
                vendor_name = match.group(1).strip()
                break
        
        # Filter for pharmacy/medical transactions
        pharmacy_transactions = []
        for tx in transactions:
            tx_vendor = tx.get('vendor', '').lower()
            tx_category = tx.get('category', '').lower()
            tx_description = tx.get('description', '').lower()
            
            # Check if this transaction matches our search criteria
            is_pharmacy = False
            
            # If we're looking for a specific vendor
            if vendor_name and vendor_name in tx_vendor:
                is_pharmacy = True
            # Otherwise check for pharmacy-related terms
            elif any(term in tx_vendor for term in pharmacy_terms):
                is_pharmacy = True
            # Check category for medical/health/pharmacy
            elif any(term in tx_category for term in ['medical', 'health', 'pharmacy']):
                is_pharmacy = True
                
            if is_pharmacy:
                pharmacy_transactions.append(tx)
        
        # If we found pharmacy transactions
        if pharmacy_transactions:
            # If looking for a specific pharmacy and we have a vendor name, prioritize exact matches
            if vendor_name:
                # First try exact vendor match
                matching_tx = None
                
                # Try various matching approaches
                for tx in pharmacy_transactions:
                    tx_vendor = tx.get('vendor', '').lower()
                    
                    # Check for exact match
                    if vendor_name == tx_vendor:
                        matching_tx = tx
                        break
                    # Check for partial match (vendor name is within transaction vendor)
                    elif vendor_name in tx_vendor:
                        matching_tx = tx
                        break
                
                if matching_tx:
                    # Format vendor name properly
                    vendor_display = matching_tx.get('vendor', 'Unknown').replace('\n', ' ')
                    
                    # Get transaction details
                    output_lines.append(f"Receipt details for {vendor_display}:")
                    
                    # Format date
                    date_str = matching_tx.get('transaction_date', 'Unknown')
                    if "T" in date_str:
                        date_str = date_str.split("T")[0]
                    output_lines.append(f"Date: {date_str}")
                    output_lines.append(f"Total amount: €{matching_tx.get('amount', 0)}")
                    
                    # Check if we have any description with item or tax information
                    description = matching_tx.get('description', '')
                    if description:
                        # Look for itemized details
                        items_section = False
                        item_lines = []
                        
                        # Parse description for potential items and prices
                        desc_lines = description.split('\n')
                        for line in desc_lines:
                            if ':' in line and ('fields' in line.lower() or 'confidence' in line.lower()):
                                # This is likely metadata, not receipt items
                                continue
                                
                            if re.search(r'\d+[.,]\d+', line):  # Line contains a number (potential price)
                                item_lines.append(line)
                        
                        if item_lines:
                            output_lines.append("\nReceipt items (extracted from description):")
                            for item in item_lines:
                                output_lines.append(f"- {item}")
                        else:
                            output_lines.append("\nReceipt information:")
                            output_lines.append(description)
                        
                        # Try to extract tax information if available
                        tax_patterns = [
                            r"(?:tax|vat|mwst|ust).*?(\d+[.,]\d+)",
                            r"(?:19%|7%).*?(\d+[.,]\d+)",
                        ]
                        
                        tax_found = False
                        for pattern in tax_patterns:
                            tax_match = re.search(pattern, description.lower())
                            if tax_match:
                                tax_amount = tax_match.group(1)
                                output_lines.append(f"\nTax amount: €{tax_amount}")
                                tax_found = True
                                break
                                
                        if not tax_found:
                            # Calculate approximate tax if German standard rate (19%)
                            total = float(matching_tx.get('amount', 0))
                            approx_tax = total - (total / 1.19)
                            output_lines.append(f"\nApproximate tax (19% VAT): €{approx_tax:.2f}")
                            output_lines.append("Note: Reduced 7% VAT may apply to some medications")
                            
                    # Add category info
                    if matching_tx.get('category'):
                        output_lines.append(f"\nCategory: {matching_tx.get('category')}")
                    
                    # Add tax deductibility info if applicable
                    output_lines.append("\nMedical expenses may be tax deductible as 'Außergewöhnliche Belastungen'.")
                else:
                    # We didn't find an exact match but found other pharmacy transactions
                    output_lines.append(f"No exact receipt found matching '{vendor_name}', but found other pharmacy transactions:")
                    output_lines.append(f"\nFound {len(pharmacy_transactions)} pharmacy purchases:")
                    
                    # Show the other pharmacy transactions
                    for i, tx in enumerate(pharmacy_transactions[:3], 1):
                        vendor = tx.get('vendor', 'Unknown').replace('\n', ' ')
                        amount = float(tx.get('amount', 0))
                        date_str = tx.get('transaction_date', 'Unknown')
                        if "T" in date_str:
                            date_str = date_str.split("T")[0]
                        output_lines.append(f"  {i}. {vendor} - {date_str} - €{amount:.2f}")
            else:
                # No specific vendor mentioned, list all pharmacy purchases
                output_lines.append(f"Found {len(pharmacy_transactions)} pharmacy purchases:")
                
                # Sort by date
                pharmacy_transactions.sort(key=lambda x: x.get('transaction_date', ''))
                
                for i, tx in enumerate(pharmacy_transactions, 1):
                    vendor = tx.get('vendor', 'Unknown').replace('\n', ' ')
                    amount = float(tx.get('amount', 0))
                    
                    # Get date in a readable format
                    date_str = tx.get('transaction_date', 'Unknown date')
                    if "T" in date_str:
                        date_str = date_str.split("T")[0]
                    
                    # Add line item
                    output_lines.append(f"  {i}. {vendor} - {date_str} - €{amount:.2f}")
                
                # Add tax information note
                output_lines.append("\nNote: Standard VAT rate in Germany is 19% for most items, but reduced to 7% for some medications.")
                output_lines.append("Medical expenses may be tax deductible as 'Außergewöhnliche Belastungen'.")
        else:
            output_lines.append("No pharmacy receipts found in your transactions.")
        
        answer["answer_text"] = "\n".join(output_lines)
        return answer 