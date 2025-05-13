#!/usr/bin/env python3
"""
Command-line interface for the tax insights search API.
"""
import argparse
import json
import sys
import os
from typing import Dict, Any, List, Optional

from app.vector.search_api import TaxInsightSearchAPI

def print_json(data: Any):
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2, default=str))

def print_user(user: Dict[str, Any]):
    """Print user information in a readable format."""
    print(f"User ID: {user.get('user_id')}")
    print(f"Name: {user.get('employee_name')}")
    print(f"Employer: {user.get('employer_name')}")
    print(f"Occupation: {user.get('occupation_category')}")
    print(f"Income Band: {user.get('income_band')}")
    print(f"Annual Income: {user.get('annualized_income')}")
    
    # Print similarity score if available
    if "similarity_score" in user:
        print(f"Similarity Score: {user.get('similarity_score'):.4f}")
    
    # Print enriched data if available
    if user.get("cluster_recommendation"):
        print(f"\nRecommendation: {user.get('cluster_recommendation')}")
    
    if user.get("uplift_message"):
        print(f"Uplift Message: {user.get('uplift_message')}")

def print_transaction(tx: Dict[str, Any]):
    """Print transaction information in a readable format."""
    print(f"Transaction ID: {tx.get('transaction_id')}")
    print(f"User ID: {tx.get('user_id')}")
    print(f"Date: {tx.get('transaction_date')}")
    print(f"Amount: {tx.get('amount')}")
    print(f"Category: {tx.get('category')}")
    print(f"Vendor: {tx.get('vendor')}")
    print(f"Description: {tx.get('description')}")
    
    # Print similarity score if available
    if "similarity_score" in tx:
        print(f"Similarity Score: {tx.get('similarity_score'):.4f}")
    
    # Print enriched data if available
    if "is_deductible" in tx:
        print(f"Deductible: {tx.get('is_deductible')}")
        if tx.get("deduction_recommendation"):
            print(f"Deduction Recommendation: {tx.get('deduction_recommendation')}")
        if tx.get("deduction_category"):
            print(f"Deduction Category: {tx.get('deduction_category')}")

def find_similar_users(api: TaxInsightSearchAPI, args):
    """Find similar users based on the provided arguments."""
    # Determine search parameters
    if args.user_id:
        results = api.find_similar_users(user_id=args.user_id, limit=args.limit)
    elif args.query:
        results = api.find_similar_users(query_text=args.query, limit=args.limit)
    else:
        print("Error: Either --user-id or --query must be provided")
        return 1
    
    # Print results
    print(f"\nFound {len(results)} similar users:\n")
    for i, user in enumerate(results):
        print(f"--- Result {i+1} ---")
        print_user(user)
        print()
    
    return 0

def find_similar_transactions(api: TaxInsightSearchAPI, args):
    """Find similar transactions based on the provided arguments."""
    # Determine search parameters
    if args.transaction_id:
        results = api.find_similar_transactions(transaction_id=args.transaction_id, limit=args.limit)
    elif args.query:
        results = api.find_similar_transactions(query_text=args.query, limit=args.limit)
    else:
        print("Error: Either --transaction-id or --query must be provided")
        return 1
    
    # Print results
    print(f"\nFound {len(results)} similar transactions:\n")
    for i, tx in enumerate(results):
        print(f"--- Result {i+1} ---")
        print_transaction(tx)
        print()
    
    return 0

def find_transactions_for_user(api: TaxInsightSearchAPI, args):
    """Find transactions for a user based on the provided arguments."""
    if not args.user_id:
        print("Error: --user-id must be provided")
        return 1
    
    results = api.find_transactions_for_user(user_id=args.user_id, limit=args.limit)
    
    # Print results
    print(f"\nFound {len(results)} transactions for user {args.user_id}:\n")
    for i, tx in enumerate(results):
        print(f"--- Result {i+1} ---")
        print_transaction(tx)
        print()
    
    return 0

def find_users_for_transaction(api: TaxInsightSearchAPI, args):
    """Find users for a transaction based on the provided arguments."""
    if not args.transaction_id:
        print("Error: --transaction-id must be provided")
        return 1
    
    results = api.find_users_for_transaction(transaction_id=args.transaction_id, limit=args.limit)
    
    # Print results
    print(f"\nFound {len(results)} users for transaction {args.transaction_id}:\n")
    for i, user in enumerate(results):
        print(f"--- Result {i+1} ---")
        print_user(user)
        print()
    
    return 0

def find_deductible_transactions(api: TaxInsightSearchAPI, args):
    """Find tax-deductible transactions based on the provided arguments."""
    results = api.find_deductible_transactions(user_id=args.user_id, limit=args.limit)
    
    # Print results
    print(f"\nFound {len(results)} tax-deductible transactions:\n")
    for i, tx in enumerate(results):
        print(f"--- Result {i+1} ---")
        print_transaction(tx)
        print()
    
    return 0

def get_tax_recommendations(api: TaxInsightSearchAPI, args):
    """Get tax recommendations for a user based on the provided arguments."""
    if not args.user_id:
        print("Error: --user-id must be provided")
        return 1
    
    recommendations = api.get_tax_recommendations(user_id=args.user_id)
    
    if "error" in recommendations:
        print(f"Error: {recommendations['error']}")
        return 1
    
    # Print user profile
    print("\nUser Profile:")
    print_user(recommendations["user_profile"])
    
    # Print transaction count
    print(f"\nTransaction Count: {recommendations['transaction_count']}")
    
    # Print similar users
    print(f"\nSimilar Users ({len(recommendations['similar_users'])}):")
    for i, user in enumerate(recommendations['similar_users'][:3]):  # Show top 3
        print(f"  {i+1}. {user.get('employee_name')} ({user.get('occupation_category')}) - {user.get('similarity_score'):.4f}")
    
    # Print potential deductions
    print(f"\nPotential Deductions ({len(recommendations['potential_deductions'])}):")
    for i, tx in enumerate(recommendations['potential_deductions'][:3]):  # Show top 3
        print(f"  {i+1}. {tx.get('category')} - {tx.get('amount')} ({tx.get('similarity_score'):.4f})")
    
    # Print recommendations
    if recommendations.get("cluster_recommendation"):
        print(f"\nRecommendation: {recommendations.get('cluster_recommendation')}")
    
    if recommendations.get("uplift_message"):
        print(f"\nUplift Message: {recommendations.get('uplift_message')}")
    
    return 0

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Tax Insights Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results to return")
    common_parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    # Similar users command
    similar_users_parser = subparsers.add_parser("similar-users", parents=[common_parser], help="Find similar users")
    similar_users_parser.add_argument("--user-id", help="User ID to search for")
    similar_users_parser.add_argument("--query", help="Text query to search for")
    
    # Similar transactions command
    similar_txs_parser = subparsers.add_parser("similar-transactions", parents=[common_parser], help="Find similar transactions")
    similar_txs_parser.add_argument("--transaction-id", help="Transaction ID to search for")
    similar_txs_parser.add_argument("--query", help="Text query to search for")
    
    # Transactions for user command
    txs_for_user_parser = subparsers.add_parser("transactions-for-user", parents=[common_parser], help="Find transactions for a user")
    txs_for_user_parser.add_argument("--user-id", required=True, help="User ID to search for")
    
    # Users for transaction command
    users_for_tx_parser = subparsers.add_parser("users-for-transaction", parents=[common_parser], help="Find users for a transaction")
    users_for_tx_parser.add_argument("--transaction-id", required=True, help="Transaction ID to search for")
    
    # Deductible transactions command
    deductible_txs_parser = subparsers.add_parser("deductible-transactions", parents=[common_parser], help="Find tax-deductible transactions")
    deductible_txs_parser.add_argument("--user-id", help="User ID to filter transactions")
    
    # Tax recommendations command
    tax_recs_parser = subparsers.add_parser("tax-recommendations", parents=[common_parser], help="Get tax recommendations for a user")
    tax_recs_parser.add_argument("--user-id", required=True, help="User ID to get recommendations for")
    
    args = parser.parse_args()
    
    # If no command is provided, show help
    if not args.command:
        parser.print_help()
        return 1
    
    # Create API instance
    with TaxInsightSearchAPI() as api:
        # Run the appropriate command
        if args.command == "similar-users":
            return find_similar_users(api, args)
        elif args.command == "similar-transactions":
            return find_similar_transactions(api, args)
        elif args.command == "transactions-for-user":
            return find_transactions_for_user(api, args)
        elif args.command == "users-for-transaction":
            return find_users_for_transaction(api, args)
        elif args.command == "deductible-transactions":
            return find_deductible_transactions(api, args)
        elif args.command == "tax-recommendations":
            return get_tax_recommendations(api, args)
        else:
            print(f"Error: Unknown command '{args.command}'")
            return 1

if __name__ == "__main__":
    sys.exit(main()) 