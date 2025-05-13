def handle_example_queries():
    """
    Run the example queries from the case study to demonstrate the system.
    """
    print("\n" + "="*80)
    print("RUNNING EXAMPLE QUERIES")
    print("="*80)
    
    # Connect once for all queries
    client = get_weaviate_client()
    if not client:
        print("Failed to connect to Weaviate!")
        return
    
    try:
        example_queries = [
            "What was my total medical spending in Q1 2025?",
            "How much income did I receive from Company X in March 2025?",
            "Is the receipt from U-Bahn Café on 10.03.2025 tax deductible?",
            "List all purchases made at pharmacies in 2025.",
            "What was the total amount of taxes withheld in February 2025?",
            "How much did I spend at the Apple Store, and for what items?",
            "Show all receipts categorized under business expenses.",
            "Compare my gross and net income for January and February 2025.",
            "How many receipts were flagged for manual review due to low classification confidence?",
            "What were the items listed in the Berlin Apotheke receipt, and how much tax was paid on them?"
        ]
        
        for i, query in enumerate(example_queries, 1):
            print(f"\n{'#'*40}")
            print(f"QUERY {i}/10: {query}")
            print(f"{'#'*40}")
            
            # Run semantic search
            results = semantic_search(query, client=client)
            
            # Display summarized results
            if "historical_users" in results["results"] and results["results"]["historical_users"]:
                print(f"\nHistorical Users ({len(results['results']['historical_users'])} results):")
                for j, user in enumerate(results["results"]["historical_users"][:3], 1):
                    print(f"  {j}. User ID: {user.get('user_id')}")
                    print(f"     Occupation: {user.get('occupation_category', 'N/A')}")
                    print(f"     Income: {user.get('total_income', 'N/A')}")
                    print(f"     Similarity: {user.get('similarity_score', 'N/A'):.4f}")
            
            if "input_users" in results["results"] and results["results"]["input_users"]:
                print(f"\nInput Users ({len(results['results']['input_users'])} results):")
                for j, user in enumerate(results["results"]["input_users"][:3], 1):
                    print(f"  {j}. User ID: {user.get('user_id')}")
                    print(f"     Occupation: {user.get('occupation_category', 'N/A')}")
                    print(f"     Income: {user.get('annualized_income', 'N/A')}")
            
            if "historical_transactions" in results["results"] and results["results"]["historical_transactions"]:
                print(f"\nHistorical Transactions ({len(results['results']['historical_transactions'])} results):")
                for j, tx in enumerate(results["results"]["historical_transactions"][:3], 1):
                    print(f"  {j}. Transaction ID: {tx.get('transaction_id')}")
                    print(f"     Vendor: {tx.get('vendor', 'N/A')}")
                    print(f"     Category: {tx.get('category', 'N/A')}")
                    print(f"     Amount: {tx.get('amount', 'N/A')}")
                    print(f"     Deductible: {tx.get('is_deductible', 'N/A')}")
                    print(f"     Similarity: {tx.get('similarity_score', 'N/A'):.4f}")
            
            if "input_transactions" in results["results"] and results["results"]["input_transactions"]:
                print(f"\nInput Transactions ({len(results['results']['input_transactions'])} results):")
                for j, tx in enumerate(results["results"]["input_transactions"][:3], 1):
                    print(f"  {j}. Transaction ID: {tx.get('transaction_id')}")
                    print(f"     Vendor: {tx.get('vendor', 'N/A')}")
                    print(f"     Category: {tx.get('category', 'N/A')}")
                    print(f"     Amount: {tx.get('amount', 'N/A')}")
            
            # Process results based on query type
            print("\nPROCESSED ANSWER:")
            
            if "What was my total medical spending in Q1 2025" in query:
                # Filter transactions related to medical spending in Q1 2025
                medical_tx = []
                for tx_list in [results["results"]["historical_transactions"], results["results"]["input_transactions"]]:
                    for tx in tx_list:
                        # Check if it's medical-related
                        is_medical = False
                        category = tx.get('category', '').lower()
                        vendor = tx.get('vendor', '').lower()
                        description = tx.get('description', '').lower()
                        
                        medical_terms = ['medical', 'health', 'doctor', 'hospital', 'pharmacy', 'apotheke', 'medicine']
                        if any(term in category or term in vendor or term in description for term in medical_terms):
                            is_medical = True
                        
                        # Check if it's in Q1 2025
                        in_q1 = False
                        date_str = tx.get('transaction_date', '')
                        if date_str and '2025' in date_str:
                            # Check for month (rough approximation)
                            if any(month in date_str.lower() for month in ['01', '02', '03', 'jan', 'feb', 'mar']):
                                in_q1 = True
                        
                        if is_medical and in_q1:
                            medical_tx.append(tx)
                
                # Calculate total
                total_amount = sum(tx.get('amount', 0) for tx in medical_tx)
                print(f"Total medical spending in Q1 2025: €{total_amount:.2f}")
                print(f"Based on {len(medical_tx)} medical transactions")
                
            elif "How much income did I receive from Company X in March 2025" in query:
                # Filter income statements from Company X in March 2025
                company_x_income = 0
                income_count = 0
                
                for user_list in [results["results"]["historical_users"], results["results"]["input_users"]]:
                    for user in user_list:
                        # Filter for Company X in description or employer fields
                        is_company_x = False
                        for field in ['occupation_category', 'employer_name']:
                            if field in user and user[field] and 'company x' in str(user[field]).lower():
                                is_company_x = True
                                break
                        
                        # Check if it's March 2025
                        is_march = False
                        date_str = user.get('filing_date', '')
                        if date_str and '2025' in date_str:
                            if any(month in date_str.lower() for month in ['03', 'mar']):
                                is_march = True
                        
                        if is_company_x and is_march:
                            # Get the income amount
                            income = user.get('annualized_income') or user.get('total_income')
                            if income:
                                income_count += 1
                                company_x_income += income / 12  # Monthly income (approximation)
                
                print(f"Income from Company X in March 2025: €{company_x_income:.2f}")
                print(f"Based on {income_count} income records")
                
            elif "Is the receipt from U-Bahn Café" in query:
                # Find the specific transaction from U-Bahn Café
                ubahn_tx = None
                for tx_list in [results["results"]["input_transactions"], results["results"]["historical_transactions"]]:
                    for tx in tx_list:
                        vendor = tx.get('vendor', '').lower()
                        date_str = tx.get('transaction_date', '').lower()
                        
                        if 'u-bahn café' in vendor or 'u-bahn cafe' in vendor:
                            if '10.03.2025' in date_str or '2025-03-10' in date_str:
                                ubahn_tx = tx
                                break
                    if ubahn_tx:
                        break
                
                if ubahn_tx:
                    # Check if it's deductible based on historical data or explicit flag
                    is_deductible = ubahn_tx.get('is_deductible')
                    deduction_reason = ubahn_tx.get('deduction_recommendation', '')
                    
                    # If not explicitly marked, make a determination based on category
                    if is_deductible is None:
                        category = ubahn_tx.get('category', '').lower()
                        if 'business' in category or 'work' in category:
                            is_deductible = True
                            deduction_reason = "Business expense category"
                        else:
                            is_deductible = False
                            deduction_reason = "Personal expense category"
                    
                    if is_deductible:
                        print(f"Yes, the receipt from U-Bahn Café on 10.03.2025 is tax deductible.")
                        if deduction_reason:
                            print(f"Reason: {deduction_reason}")
                    else:
                        print(f"No, the receipt from U-Bahn Café on 10.03.2025 is not tax deductible.")
                        if deduction_reason:
                            print(f"Reason: {deduction_reason}")
                else:
                    print("No receipt from U-Bahn Café on 10.03.2025 was found.")
                
            elif "List all purchases made at pharmacies in 2025" in query:
                # Filter transactions from pharmacies in 2025
                pharmacy_tx = []
                for tx_list in [results["results"]["historical_transactions"], results["results"]["input_transactions"]]:
                    for tx in tx_list:
                        # Check if it's pharmacy-related
                        is_pharmacy = False
                        vendor = tx.get('vendor', '').lower()
                        category = tx.get('category', '').lower()
                        
                        pharmacy_terms = ['pharmacy', 'apotheke', 'drugstore', 'medicine']
                        if any(term in vendor or term in category for term in pharmacy_terms):
                            is_pharmacy = True
                        
                        # Check if it's in 2025
                        in_2025 = False
                        date_str = tx.get('transaction_date', '')
                        if date_str and '2025' in date_str:
                            in_2025 = True
                        
                        if is_pharmacy and in_2025:
                            pharmacy_tx.append(tx)
                
                # Display the transactions
                if pharmacy_tx:
                    print(f"Found {len(pharmacy_tx)} pharmacy purchases in 2025:")
                    for i, tx in enumerate(pharmacy_tx, 1):
                        vendor = tx.get('vendor', 'Unknown Pharmacy')
                        date = tx.get('transaction_date', 'Unknown Date')
                        amount = tx.get('amount', 0)
                        print(f"  {i}. {vendor} - {date} - €{amount:.2f}")
                else:
                    print("No pharmacy purchases found in 2025.")
                
            elif "total amount of taxes withheld in February 2025" in query:
                # Filter income statements from February 2025
                feb_tax_withheld = 0
                tax_records = 0
                
                for user_list in [results["results"]["historical_users"], results["results"]["input_users"]]:
                    for user in user_list:
                        # Check if it's February 2025
                        date_str = user.get('filing_date', '')
                        in_feb = False
                        if date_str and '2025' in date_str:
                            if any(month in date_str.lower() for month in ['02', 'feb']):
                                in_feb = True
                        
                        if in_feb:
                            # Calculate tax withheld (simplified as difference between gross and net)
                            gross = user.get('annualized_income') or user.get('total_income')
                            net = user.get('annualized_net_pay') or user.get('net_pay')
                            
                            if gross is not None and net is not None:
                                tax_withheld = (gross - net) / 12  # Monthly amount
                                feb_tax_withheld += tax_withheld
                                tax_records += 1
                            
                            # Alternative: use tax deductions directly
                            tax_deductions = user.get('annualized_tax_deductions') or user.get('total_deductions')
                            if tax_deductions is not None:
                                monthly_tax = tax_deductions / 12
                                # Only count if we didn't already count from gross-net
                                if not (gross is not None and net is not None):
                                    feb_tax_withheld += monthly_tax
                                    tax_records += 1
                
                print(f"Total taxes withheld in February 2025: €{feb_tax_withheld:.2f}")
                print(f"Based on {tax_records} tax records")
                
            elif "Apple Store" in query:
                # Filter transactions from Apple Store
                apple_tx = []
                for tx_list in [results["results"]["historical_transactions"], results["results"]["input_transactions"]]:
                    for tx in tx_list:
                        vendor = tx.get('vendor', '').lower()
                        if 'apple' in vendor and ('store' in vendor or 'shop' in vendor):
                            apple_tx.append(tx)
                
                # Calculate total and extract items
                total_spent = sum(tx.get('amount', 0) for tx in apple_tx)
                items = []
                for tx in apple_tx:
                    description = tx.get('description', '')
                    if description:
                        # Extract items from description (simplified)
                        lines = description.split('\n')
                        for line in lines:
                            if line.strip():
                                items.append(line.strip())
                
                if apple_tx:
                    print(f"Total spent at Apple Store: €{total_spent:.2f}")
                    print(f"Based on {len(apple_tx)} transactions")
                    
                    if items:
                        print("Items purchased:")
                        for i, item in enumerate(items, 1):
                            print(f"  {i}. {item}")
                    else:
                        print("No specific items could be extracted from the receipts.")
                else:
                    print("No transactions from Apple Store found.")
                
            elif "business expenses" in query:
                # Filter transactions categorized as business expenses
                business_tx = []
                for tx_list in [results["results"]["historical_transactions"], results["results"]["input_transactions"]]:
                    for tx in tx_list:
                        category = tx.get('category', '').lower()
                        subcategory = tx.get('subcategory', '').lower()
                        
                        if 'business' in category or 'work' in subcategory or tx.get('is_deductible') is True:
                            business_tx.append(tx)
                
                if business_tx:
                    print(f"Found {len(business_tx)} business expense receipts:")
                    for i, tx in enumerate(business_tx, 1):
                        vendor = tx.get('vendor', 'Unknown Vendor')
                        date = tx.get('transaction_date', 'Unknown Date')
                        amount = tx.get('amount', 0)
                        print(f"  {i}. {vendor} - {date} - €{amount:.2f}")
                else:
                    print("No business expense receipts found.")
                
            elif "Compare my gross and net income for January and February 2025" in query:
                # Filter income statements from January and February 2025
                jan_gross = 0
                jan_net = 0
                feb_gross = 0
                feb_net = 0
                
                jan_count = 0
                feb_count = 0
                
                for user_list in [results["results"]["historical_users"], results["results"]["input_users"]]:
                    for user in user_list:
                        date_str = user.get('filing_date', '').lower()
                        
                        # Extract monthly income
                        annual_gross = user.get('annualized_income') or user.get('total_income')
                        annual_net = user.get('annualized_net_pay') or user.get('net_pay')
                        
                        if annual_gross and annual_net:
                            monthly_gross = annual_gross / 12
                            monthly_net = annual_net / 12
                            
                            # Check which month
                            if '2025' in date_str:
                                if any(month in date_str for month in ['01', 'jan']):
                                    jan_gross += monthly_gross
                                    jan_net += monthly_net
                                    jan_count += 1
                                elif any(month in date_str for month in ['02', 'feb']):
                                    feb_gross += monthly_gross
                                    feb_net += monthly_net
                                    feb_count += 1
                
                # Calculate differences
                gross_diff = feb_gross - jan_gross
                net_diff = feb_net - jan_net
                
                print("Income Comparison for January and February 2025:")
                print(f"January Gross Income: €{jan_gross:.2f} (based on {jan_count} records)")
                print(f"January Net Income: €{jan_net:.2f}")
                print(f"January Tax Withholding: €{(jan_gross - jan_net):.2f}")
                print("")
                print(f"February Gross Income: €{feb_gross:.2f} (based on {feb_count} records)")
                print(f"February Net Income: €{feb_net:.2f}")
                print(f"February Tax Withholding: €{(feb_gross - feb_net):.2f}")
                print("")
                print(f"Change in Gross Income: €{gross_diff:.2f} ({gross_diff/jan_gross*100:.1f}% change)")
                print(f"Change in Net Income: €{net_diff:.2f} ({net_diff/jan_net*100:.1f}% change)")
                
            elif "flagged for manual review due to low classification confidence" in query:
                # Filter transactions with low confidence scores
                low_confidence_threshold = 0.5  # 50% confidence or lower
                low_confidence_tx = []
                
                for tx_list in [results["results"]["historical_transactions"], results["results"]["input_transactions"]]:
                    for tx in tx_list:
                        # Check overall confidence
                        confidence = tx.get('confidence_score') or 0
                        
                        # Also check individual field confidences
                        fields_confidence = [
                            tx.get('amount_confidence') or 0,
                            tx.get('date_confidence') or 0,
                            tx.get('vendor_confidence') or 0,
                            tx.get('category_confidence') or 0
                        ]
                        
                        # If any confidence is below threshold
                        if confidence < low_confidence_threshold or any(fc < low_confidence_threshold for fc in fields_confidence if fc > 0):
                            low_confidence_tx.append(tx)
                
                print(f"Found {len(low_confidence_tx)} receipts flagged for manual review due to low confidence scores:")
                for i, tx in enumerate(low_confidence_tx[:5], 1):  # Show first 5
                    vendor = tx.get('vendor', 'Unknown Vendor')
                    confidence = tx.get('confidence_score') or 0
                    print(f"  {i}. {vendor} - Overall Confidence: {confidence:.1%}")
                    
                    # Show specific low confidence fields
                    low_fields = []
                    if (tx.get('amount_confidence') or 1) < low_confidence_threshold:
                        low_fields.append(f"Amount: {tx.get('amount_confidence', 0):.1%}")
                    if (tx.get('date_confidence') or 1) < low_confidence_threshold:
                        low_fields.append(f"Date: {tx.get('date_confidence', 0):.1%}")
                    if (tx.get('vendor_confidence') or 1) < low_confidence_threshold:
                        low_fields.append(f"Vendor: {tx.get('vendor_confidence', 0):.1%}")
                    if (tx.get('category_confidence') or 1) < low_confidence_threshold:
                        low_fields.append(f"Category: {tx.get('category_confidence', 0):.1%}")
                    
                    if low_fields:
                        print(f"     Low confidence fields: {', '.join(low_fields)}")
                
            elif "Berlin Apotheke receipt" in query:
                # Find the specific Berlin Apotheke transaction
                apotheke_tx = None
                for tx_list in [results["results"]["input_transactions"], results["results"]["historical_transactions"]]:
                    for tx in tx_list:
                        vendor = tx.get('vendor', '').lower()
                        if 'berlin' in vendor and 'apotheke' in vendor:
                            apotheke_tx = tx
                            break
                    if apotheke_tx:
                        break
                
                if apotheke_tx:
                    # Extract items from description or details
                    description = apotheke_tx.get('description', '')
                    items = []
                    tax_amount = 0
                    
                    if description:
                        # Parse description for items and tax info
                        lines = description.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line:
                                # Look for tax information
                                if 'tax' in line.lower() or 'vat' in line.lower() or 'mwst' in line.lower():
                                    # Try to extract tax amount
                                    import re
                                    tax_matches = re.findall(r'(\d+[.,]\d+)', line)
                                    if tax_matches:
                                        try:
                                            tax_amount = float(tax_matches[0].replace(',', '.'))
                                        except:
                                            pass
                                else:
                                    # Assume it's an item
                                    items.append(line)
                    
                    # If no items found, create a generic item
                    if not items:
                        items = ["Pharmacy items"]
                    
                    # If no tax found, estimate it (19% VAT in Germany)
                    if tax_amount == 0:
                        tax_amount = apotheke_tx.get('amount', 0) * 0.19 / 1.19  # Extract VAT from gross amount
                    
                    print(f"Berlin Apotheke Receipt Details:")
                    print(f"Date: {apotheke_tx.get('transaction_date', 'N/A')}")
                    print(f"Total Amount: €{apotheke_tx.get('amount', 0):.2f}")
                    print(f"Tax Amount: €{tax_amount:.2f}")
                    print(f"Items ({len(items)}):")
                    for i, item in enumerate(items, 1):
                        print(f"  {i}. {item}")
                else:
                    print("No receipt from Berlin Apotheke found.")
            
            print("\n" + "-"*60)
    
    finally:
        if client:
            client.close()
            
    print("\n" + "="*80)
    print("EXAMPLE QUERIES COMPLETED")
    print("="*80)