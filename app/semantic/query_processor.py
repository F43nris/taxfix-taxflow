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
        
        # If query_type is already set to medical_spending, use that handler
        if results["query_type"] == "medical_spending":
            answer.update(QueryProcessor.process_medical_spending_query(query, results))
            return answer
        
        # Special handling for pharmacy listing queries
        if results["query_type"] == "pharmacy_listing" or ("pharmacy" in query_lower and "list" in query_lower):
            answer.update(QueryProcessor.process_pharmacy_listing_query(query, results))
            return answer
        
        # Process tax-related queries including withholding, tax rate, etc.
        if any(term in query_lower for term in ["tax", "taxes", "withheld", "withholding"]):
            answer["query_type"] = "tax_info"
            answer.update(QueryProcessor.process_tax_query(query, results))
            return answer
        
        # Skip regular income queries as the information is already displayed in the user section
        elif any(term in query_lower for term in ["income", "salary", "earn", "received"]):
            answer["query_type"] = "income"
            answer["answer_text"] = "Please refer to the user income information shown above."
            return answer
        
        # Handle tax deductibility queries
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
            match = re.search(r"receipt from ([a-zA-Z0-9\s-]+)( on)?", query_lower)
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
        for tx in results["results"]["input_transactions"]:
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
            output_lines.append(f"Note: Couldn't find receipt from {vendor_name} on exact date {transaction_date}, using available receipt instead.")
        
        # If we found a matching transaction or at least a vendor match
        if target_transaction or vendor_name:
            if target_transaction:
                tx_date = "unknown date"
                date_info = target_transaction.get('invoice_date', {})
                if isinstance(date_info, dict) and 'normalized_value' in date_info:
                    tx_date = date_info['normalized_value']
                elif target_transaction.get('transaction_date'):
                    tx_date = target_transaction.get('transaction_date')
                    
                output_lines.append(f"Found receipt from {target_transaction.get('vendor')} on {tx_date}")
                output_lines.append(f"Amount: €{target_transaction.get('amount', 'N/A')}")
                output_lines.append(f"Category: {target_transaction.get('category', 'N/A')}")
            
            # Look for similar historical transactions to determine deductibility
            similar_historical = []
            for hist_tx in results["results"]["historical_transactions"]:
                # Check if the transaction has deductibility info
                if "is_deductible" in hist_tx:
                    similar_historical.append(hist_tx)
            
            # If we found similar historical transactions with deductibility info
            if similar_historical:
                # Calculate the weighted vote for deductibility based on similarity scores
                deductible_votes = 0
                non_deductible_votes = 0
                total_weight = 0
                total_votes = 0
                
                for hist_tx in similar_historical:
                    weight = hist_tx.get("similarity_score", 0.5)  # Default to 0.5 if no similarity score
                    total_weight += weight
                    total_votes += 1
                    
                    if hist_tx.get("is_deductible") in [True, 1, "1", "true", "True"]:
                        deductible_votes += weight
                    else:
                        non_deductible_votes += weight
                
                # Show summary of historical data
                output_lines.append(f"\nBased on {total_votes} similar historical transactions:")
                
                # Calculate confidence percentage
                confidence_percentage = 0
                if total_weight > 0:
                    if deductible_votes > non_deductible_votes:
                        confidence_percentage = (deductible_votes / total_weight) * 100
                    else:
                        confidence_percentage = (non_deductible_votes / total_weight) * 100
                
                # Determine the verdict
                if deductible_votes > non_deductible_votes:
                    output_lines.append(f"This receipt is LIKELY TAX DEDUCTIBLE (Confidence: {confidence_percentage:.2f}%)")
                    
                    # Show the most similar deductible transaction as an example
                    deductible_txs = [tx for tx in similar_historical if tx.get("is_deductible") in [True, 1, "1", "true", "True"]]
                    if deductible_txs:
                        top_tx = max(deductible_txs, key=lambda x: x.get("similarity_score", 0))
                        output_lines.append(f"\nLegal Information:")
                        
                        # Process deduction recommendation to extract just the explanation
                        if top_tx.get("deduction_recommendation"):
                            deduction_rec = top_tx.get("deduction_recommendation")
                            # Check if it's a list of dicts with law_reference and explanation
                            if isinstance(deduction_rec, list) and deduction_rec:
                                for rec_item in deduction_rec:
                                    if isinstance(rec_item, dict) and "explanation" in rec_item:
                                        output_lines.append(f"  {rec_item['explanation']}")
                                        # Include law reference if available
                                        if "law_reference" in rec_item:
                                            output_lines.append(f"  (Reference: {rec_item['law_reference']})")
                                        break
                            else:
                                # Just show the recommendation as is if it's not in expected format
                                output_lines.append(f"  {deduction_rec}")
                        elif top_tx.get("deduction_category"):
                            output_lines.append(f"  Deduction Category: {top_tx.get('deduction_category')}")
                        
                        # Show any other top deductible transactions with different explanations
                        if len(deductible_txs) > 1:
                            unique_explanations = set()
                            # Get explanation from first transaction
                            first_explanation = ""
                            if top_tx.get("deduction_recommendation") and isinstance(top_tx.get("deduction_recommendation"), list):
                                for rec_item in top_tx.get("deduction_recommendation"):
                                    if isinstance(rec_item, dict) and "explanation" in rec_item:
                                        first_explanation = rec_item.get("explanation")
                                        unique_explanations.add(first_explanation)
                                        break
                            
                            # Look for other transactions with different explanations
                            for tx in sorted(deductible_txs[1:3], key=lambda x: x.get("similarity_score", 0), reverse=True):
                                if tx.get("deduction_recommendation") and isinstance(tx.get("deduction_recommendation"), list):
                                    for rec_item in tx.get("deduction_recommendation"):
                                        if isinstance(rec_item, dict) and "explanation" in rec_item:
                                            explanation = rec_item.get("explanation")
                                            if explanation and explanation not in unique_explanations:
                                                output_lines.append(f"\n  Additional information:")
                                                output_lines.append(f"  {explanation}")
                                                if "law_reference" in rec_item:
                                                    output_lines.append(f"  (Reference: {rec_item['law_reference']})")
                                                unique_explanations.add(explanation)
                                                break
                else:
                    output_lines.append(f"This receipt is LIKELY NOT TAX DEDUCTIBLE (Confidence: {confidence_percentage:.2f}%)")
                    
                    # Show the most similar non-deductible transaction as an example
                    non_deductible_txs = [tx for tx in similar_historical if tx.get("is_deductible") not in [True, 1, "1", "true", "True"]]
                    if non_deductible_txs:
                        top_tx = max(non_deductible_txs, key=lambda x: x.get("similarity_score", 0))
                        output_lines.append(f"\nSimilar non-deductible transaction example:")
                        output_lines.append(f"  Vendor: {top_tx.get('vendor', 'N/A')}")
                        output_lines.append(f"  Category: {top_tx.get('category', 'N/A')}")
            else:
                output_lines.append("\nCouldn't find sufficient historical transaction data with deductibility information.")
                
                # Make a best guess based on category
                if target_transaction:
                    category = target_transaction.get('category', '').lower()
                    if any(biz_term in category for biz_term in ['business', 'work', 'office']):
                        output_lines.append("However, the category suggests it might be deductible as a business expense")
                    else:
                        output_lines.append("Based on the category 'Food & Dining', this appears to be a personal expense, which is generally not deductible unless it was for a business meal.")
                        output_lines.append("To be deductible, you would need to document that it was a business-related expense.")
        else:
            output_lines.append(f"Could not find a receipt from {vendor_name or 'the specified vendor'}")
            output_lines.append("However, based on historical transactions:")
            
            # Get info from historical data about cafe receipts
            cafe_historical = []
            for hist_tx in results["results"]["historical_transactions"]:
                if "cafe" in hist_tx.get('vendor', '').lower() and "is_deductible" in hist_tx:
                    cafe_historical.append(hist_tx)
            
            if cafe_historical:
                deductible_count = sum(1 for tx in cafe_historical if tx.get("is_deductible") in [True, 1, "1", "true", "True"])
                if deductible_count > len(cafe_historical) / 2:
                    output_lines.append("Cafe receipts are typically tax deductible when they are for business meetings.")
                    # Find a deductible cafe receipt with explanation
                    for tx in cafe_historical:
                        if tx.get("is_deductible") in [True, 1, "1", "true", "True"] and tx.get("deduction_recommendation"):
                            # Process recommendation to extract explanation
                            deduction_rec = tx.get("deduction_recommendation")
                            if isinstance(deduction_rec, list) and deduction_rec:
                                for rec_item in deduction_rec:
                                    if isinstance(rec_item, dict) and "explanation" in rec_item:
                                        output_lines.append(f"Legal information: {rec_item['explanation']}")
                                        if "law_reference" in rec_item:
                                            output_lines.append(f"(Reference: {rec_item['law_reference']})")
                                        break
                                else:
                                    # Fallback to showing the raw recommendation
                                    output_lines.append(f"Example reason: {deduction_rec}")
                            else:
                                output_lines.append(f"Example reason: {deduction_rec}")
                            break
                else:
                    output_lines.append("Cafe receipts are typically NOT tax deductible unless they are for business purposes.")
            else:
                output_lines.append("No information available about cafe receipts in historical data.")
        
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