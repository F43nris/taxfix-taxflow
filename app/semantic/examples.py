"""
Example semantic query runner for demonstration purposes.
"""
from app.vector.client import get_weaviate_client
from app.semantic.search import search
from app.semantic.query_processor import QueryProcessor

def run_example_semantic_queries():
    """
    Run the example semantic queries from the case study to demonstrate the system.
    """
    print("\n" + "="*80)
    print("RUNNING EXAMPLE SEMANTIC QUERIES")
    print("="*80)
    
    # Connect once for all queries
    client = get_weaviate_client()
    if not client:
        print("Failed to connect to Weaviate!")
        return
    
    try:
        example_queries = [
            "How much income did I receive from Company X in March 2025?"
        ]
        
        for i, query in enumerate(example_queries, 1):
            print(f"\n{'#'*40}")
            print(f"QUERY {i}/{len(example_queries)}: {query}")
            print(f"{'#'*40}")
            
            # Run semantic search
            results = search(query, client=client)
            
            # Show if this was detected as a personal query
            print(f"Query type: {results['query_type']}")
            print(f"Personal query: {'Yes' if results['is_personal_query'] else 'No'}")
            print(f"Requires historical data: {'Yes' if results['requires_historical_data'] else 'No'}")
            print()
            
            # Display summarized results
            if "input_users" in results["results"] and results["results"]["input_users"]:
                print(f"\nInput Users ({len(results['results']['input_users'])} results):")
                for j, user in enumerate(results['results']['input_users'][:3], 1):
                    print(f"  {j}. User ID: {user.get('user_id')}")
                    print(f"     Occupation: {user.get('occupation_category', 'N/A')}")
                    
                    # Show income information
                    income = user.get('annualized_income', 'N/A')
                    if income != 'N/A':
                        monthly_gross = float(income) / 12
                        print(f"     Monthly Gross Pay: €{monthly_gross:.2f}")
                        print(f"     Annual Income: {income}")
                    else:
                        print(f"     Income: {income}")
                    
                    # Add tax information
                    if user.get('tax_withheld') or user.get('tax_rate'):
                        print(f"     Tax Information:")
                        if user.get('tax_withheld'):
                            print(f"       Tax Withheld: €{user.get('tax_withheld')}")
                        if user.get('tax_rate'):
                            print(f"       Tax Rate: {user.get('tax_rate')}%")
                        if user.get('annualized_tax_deductions'):
                            print(f"       Annual Tax Deductions: €{user.get('annualized_tax_deductions')}")
                    
                    if "similarity_score" in user:
                        print(f"     Similarity: {user.get('similarity_score', 'N/A'):.4f}")
            
            # Special handling for medical spending queries with quarterly breakdown
            if results["query_type"] == "medical_spending" and "quarterly_breakdown" in results:
                # Calculate total medical spending across all quarters
                total_medical_spending = 0
                total_transactions = 0
                
                print("\nMEDICAL SPENDING BREAKDOWN BY QUARTER:")
                for quarter, data in results["quarterly_breakdown"].items():
                    tx_count = len(data["transactions"])
                    total = data["total"]
                    total_medical_spending += total
                    total_transactions += tx_count
                    
                    if tx_count > 0:
                        print(f"  {quarter}: {tx_count} transactions, total: €{total:.2f}")
                        for tx in data["transactions"]:
                            print(f"    - {tx.get('vendor', 'unknown')}: €{tx.get('amount', '0')} ({tx.get('transaction_date', 'unknown')})")
                
                print(f"\nTOTAL MEDICAL SPENDING: €{total_medical_spending:.2f} ({total_transactions} transactions)")
                
                # Get transactions from the specific quarter in the query, if any
                quarter_pattern = r'q[1-4]'
                import re
                quarter_match = re.search(quarter_pattern, query.lower())
                if quarter_match:
                    quarter = f"Q{quarter_match.group(0)[1]}"
                    print(f"\nSHOWING DETAILS FOR {quarter}:")
            
            if "input_transactions" in results["results"] and results["results"]["input_transactions"]:
                print(f"\nInput Transactions ({len(results['results']['input_transactions'])} results):")
                for j, tx in enumerate(results["results"]["input_transactions"][:5], 1):
                    print(f"  {j}. Transaction ID: {tx.get('transaction_id')}")
                    print(f"     Vendor: {tx.get('vendor', 'N/A')}")
                    print(f"     Category: {tx.get('category', 'N/A')}")
                    print(f"     Amount: {tx.get('amount', 'N/A')}")
                    
                    # Show date information if available
                    date_info = tx.get('invoice_date', {})
                    if isinstance(date_info, dict) and 'normalized_value' in date_info:
                        print(f"     Date: {date_info['normalized_value']}")
                    elif tx.get('transaction_date'):
                        print(f"     Date: {tx.get('transaction_date')}")
                        
                    if "similarity_score" in tx:
                        print(f"     Similarity: {tx.get('similarity_score', 'N/A'):.4f}")
                    
                    # If it's a pharmacy query, highlight if vendor contains pharmacy terms
                    if any(term in query.lower() for term in ['pharmacy', 'apotheke', 'medicine', 'medical']):
                        vendor = tx.get('vendor', '').lower()
                        if any(term in vendor for term in ['pharmacy', 'apotheke', 'medicine', 'medical', 'drug', 'hospital']):
                            print(f"     ** Medical match in vendor name **")
            
            # Only show historical transaction data if required (no historical users)
            if results["requires_historical_data"] and "historical_transactions" in results["results"] and results["results"]["historical_transactions"]:
                print(f"\nHistorical Transactions ({len(results['results']['historical_transactions'])} results):")
                for j, tx in enumerate(results["results"]["historical_transactions"][:3], 1):
                    print(f"  {j}. Transaction ID: {tx.get('transaction_id')}")
                    print(f"     Vendor: {tx.get('vendor', 'N/A')}")
                    print(f"     Category: {tx.get('category', 'N/A')}")
                    print(f"     Amount: {tx.get('amount', 'N/A')}")
                    print(f"     Deductible: {tx.get('is_deductible', 'N/A')}")
                    print(f"     Similarity: {tx.get('similarity_score', 'N/A'):.4f}")
                    if tx.get("deduction_category"):
                        print(f"     Deduction Category: {tx.get('deduction_category')}")
            
            # Process the answer based on query content
            print("\nPROCESSED ANSWER:")
            processed = QueryProcessor.process_query(query, results)
            print(processed["answer_text"])
            
            print("\n" + "-"*60)
    
    finally:
        # Close connection when we're done
        if client:
            client.close()
            
    print("\n" + "="*80)
    print("EXAMPLE QUERIES COMPLETED")
    print("="*80) 