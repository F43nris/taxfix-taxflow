#!/usr/bin/env python3
"""
Debug script for semantic search medical transactions.
"""
import sys
import sqlite3
from app.semantic.search import search

def print_all_transactions():
    """Print all transactions in the database for debugging"""
    print("\n==== ALL TRANSACTIONS IN DATABASE ====")
    conn = sqlite3.connect("app/data/db/transactions.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM transactions")
    transactions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    for tx in transactions:
        print(f"ID: {tx.get('transaction_id', 'unknown')}")
        print(f"  Vendor: {tx.get('vendor', 'unknown')}")
        print(f"  Category: {tx.get('category', 'unknown')}")
        print(f"  Subcategory: {tx.get('subcategory', 'unknown')}")
        print(f"  Description: {tx.get('description', 'unknown')}")
        print(f"  Amount: {tx.get('amount', 'unknown')}")
        print(f"  Date: {tx.get('transaction_date', 'unknown')}")
        print()

def main():
    """Run the search and print results"""
    query = "What was my total medical spending in Q1 2025?"
    print(f"Running search for: '{query}'")
    
    # Print all transactions first
    print_all_transactions()
    
    # Run the search
    result = search(query)
    
    # Print the results
    print("\n==== SEARCH RESULTS ====")
    print(f"Query type: {result['query_type']}")
    print(f"Is personal: {result['is_personal_query']}")
    print(f"Requires historical data: {result['requires_historical_data']}")
    print()
    
    # Print filtered transactions based on query
    print(f"Found {len(result['results']['input_transactions'])} filtered medical transactions:")
    for tx in result['results']['input_transactions']:
        print(f"ID: {tx.get('transaction_id', 'unknown')}")
        print(f"  Vendor: {tx.get('vendor', 'unknown')}")
        print(f"  Category: {tx.get('category', 'unknown')}")
        print(f"  Subcategory: {tx.get('subcategory', 'unknown')}")
        print(f"  Description: {tx.get('description', 'unknown')}")
        print(f"  Amount: {tx.get('amount', 'unknown')}")
        print(f"  Date: {tx.get('transaction_date', 'unknown')}")
        print()
    
    # Print all medical transactions
    if "all_medical_transactions" in result and result["all_medical_transactions"]:
        print("\n==== ALL MEDICAL TRANSACTIONS ====")
        all_medical = result["all_medical_transactions"]
        if isinstance(all_medical, list):
            for tx in all_medical:
                print(f"ID: {tx.get('transaction_id', 'unknown')}")
                print(f"  Vendor: {tx.get('vendor', 'unknown')}")
                print(f"  Category: {tx.get('category', 'unknown')}")
                print(f"  Amount: {tx.get('amount', 'unknown')}")
                print(f"  Date: {tx.get('transaction_date', 'unknown')}")
                print()
    
    # Print quarterly breakdown
    if "quarterly_breakdown" in result:
        print("\n==== QUARTERLY BREAKDOWN ====")
        for quarter, data in result["quarterly_breakdown"].items():
            tx_count = len(data["transactions"])
            total = data["total"]
            if tx_count > 0:
                print(f"{quarter}: {tx_count} transactions, total: €{total:.2f}")
                for tx in data["transactions"]:
                    print(f"  - {tx.get('vendor', 'unknown')}: €{tx.get('amount', '0')}")

if __name__ == "__main__":
    main() 