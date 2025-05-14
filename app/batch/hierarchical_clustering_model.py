import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy import stats

def tax_peer_group_hierarchical(data_path, results_dir='.', income_quantiles=5, distance_threshold=None, n_clusters=None):
    """
    Perform peer-group clustering on tax data using hierarchical clustering.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing tax data
    results_dir : str
        Directory to save visualization results
    income_quantiles : int
        Number of income bands to create (using quantiles)
    distance_threshold : float, optional
        Distance threshold for cutting the dendrogram (alternative to n_clusters)
    n_clusters : int, optional
        Number of clusters to form (if None, uses distance_threshold instead)
        
    Returns:
    --------
    pandas.DataFrame
        Original data enriched with cluster_id and category gap scores
    """
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Raw data shape: {df.shape}")
    print(f"Number of unique users: {df['user_id'].nunique()}")
    print(f"Years in data: {sorted(df['year'].unique())}")
    print(f"Column names in dataset: {df.columns.tolist()}")
    
    # Step 1: Aggregate data by user (across all years)
    print("Aggregating data by user (across all years)")
    
    if 'category' in df.columns and 'amount' in df.columns:
        # First, create user-year-category pivot
        pivot_df = df.pivot_table(
                index=['user_id', 'year'], 
                columns='category', 
                values='amount',
                aggfunc='sum',
                fill_value=0
            )
        
        # Then, aggregate across years for each user (mean)
        user_features = pivot_df.groupby(level='user_id').mean()
        print(f"After pivoting and aggregating: {user_features.shape}")
    else:
        # If data is already in wide format with one row per user-year
        user_features = df.groupby('user_id').mean()
        print(f"After aggregating pre-pivoted data: {user_features.shape}")
    
    # Detect income column
    income_col = next((col for col in ['total_income', 'income', 'annual_income', 'salary'] 
                      if col in df.columns), None)

    if income_col:
        print(f"Using '{income_col}' as income column")
        income_by_user = df.groupby('user_id')[income_col].mean()
        income_by_user = income_by_user.clip(lower=1000)  # Avoid division issues
        
        # Create income bands
        income_bands = pd.qcut(income_by_user, income_quantiles, 
                              labels=[f'Band {i+1}' for i in range(income_quantiles)])
        income_band_map = pd.DataFrame({'income_band': income_bands}, index=income_by_user.index)
        
        # Save income band information
        income_ranges = pd.DataFrame({
            'band': [f'Band {i+1}' for i in range(income_quantiles)],
            'min': [income_by_user[income_bands == band].min() for band in income_bands.cat.categories],
            'max': [income_by_user[income_bands == band].max() for band in income_bands.cat.categories],
            'count': [sum(income_bands == band) for band in income_bands.cat.categories]
        })
        income_ranges.to_csv(os.path.join(results_dir, 'income_bands.csv'), index=False)
    else:
        print("WARNING: No income column found. Using default value of 50000")
        income_by_user = pd.Series(50000, index=user_features.index)
        income_band_map = pd.DataFrame({'income_band': 'Unknown'}, index=user_features.index)
    
    # Step 2: Prepare features - normalize by income to get expense ratios
    expense_cols = [col for col in user_features.columns 
                   if col not in ['user_id', 'year', income_col]]
    
    # Calculate expense ratios as percentage of income
    expense_ratios = user_features[expense_cols].div(income_by_user, axis=0) * 100
    expense_ratios = expense_ratios.clip(upper=100)  # Cap at 100% of income
    print(f"Created expense ratios with shape: {expense_ratios.shape}")
    
    # Add demographic features to the clustering data
    demographic_features = []

    # Check and add occupation_category if available
    if 'occupation_category' in df.columns:
        print("Adding occupation_category to clustering features")
        occupations = df.groupby('user_id')['occupation_category'].agg(lambda x: x.mode()[0])
        occupation_dummies = pd.get_dummies(occupations, prefix='job')
        demographic_features.append(occupation_dummies)

    # Combine expense ratios with demographic features
    if demographic_features:
        # Combine all demographic features with expense ratios
        all_features = pd.concat([expense_ratios] + demographic_features, axis=1)
        print(f"Combined feature matrix shape: {all_features.shape}")
    else:
        all_features = expense_ratios
        print("No demographic features found in dataset")

    # Step 3: Scale the features and apply PCA for dimensionality reduction
    # First scale the data
    scaled_data = StandardScaler().fit_transform(all_features.fillna(0))
    
    # Apply PCA - use sqrt of sample count as rule of thumb for components
    n_features = all_features.shape[1]
    n_components = min(int(np.sqrt(len(all_features))), n_features)
    print(f"Reducing from {n_features} features to {n_components} principal components")

    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(scaled_data)

    # Report explained variance
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA explains {explained_variance:.2%} of total variance")
    
    # Save feature importance information
    if n_components <= 10:  # Only show details if we have a reasonable number of components
        feature_importance = pd.DataFrame({
            'feature': all_features.columns,
            'importance': np.abs(pca.components_[0])  # First component loadings
        }).sort_values('importance', ascending=False)
        print("Top features by PCA loading (first component):")
        print(feature_importance.head(5))
        
        # Save PCA loadings for interpretation
        pd.DataFrame(
            pca.components_,
            columns=all_features.columns
        ).to_csv(os.path.join(results_dir, 'pca_loadings.csv'))
    
    # Step 4: Perform hierarchical clustering
    print("Performing hierarchical clustering...")
    
    # Compute the linkage matrix
    # Ward's method tends to create more balanced clusters
    Z = linkage(reduced_features, method='ward')
    
    # Create dendrogram visualization
    plt.figure(figsize=(16, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    
    # Plot the dendrogram with truncation for better visibility
    dendrogram(
        Z,
        truncate_mode='lastp',  # Show only the last p merged clusters
        p=30,  # Show the 30 most recent merges
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True  # Show contracted nodes as a triangle
    )
    plt.savefig(os.path.join(results_dir, 'hierarchical_dendrogram.png'))
    
    # Create more detailed dendrogram with sample labels for inspection
    if len(reduced_features) <= 50:  # Only for small datasets
        plt.figure(figsize=(20, 12))
        plt.title('Complete Hierarchical Clustering Dendrogram with Labels')
        dendrogram(
            Z,
            leaf_rotation=90.,
            leaf_font_size=10.,
            labels=user_features.index.tolist()  # Use user IDs as labels
        )
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'hierarchical_dendrogram_labeled.png'))
    
    # Step 5: Determine optimal number of clusters using silhouette scores
    if n_clusters is None and distance_threshold is None:
        # Evaluate different cluster counts
        silhouette_scores = []
        k_range = range(2, min(11, len(reduced_features) // 2))  # 2 to 10 clusters
        
        for k in k_range:
            # Cut the tree to get k clusters
            labels = fcluster(Z, k, criterion='maxclust') - 1  # 0-based indexing
            
            # Calculate silhouette score
            score = silhouette_score(reduced_features, labels)
            silhouette_scores.append(score)
            print(f"  {k} clusters: silhouette score = {score:.4f}")
            
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(list(k_range), silhouette_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis for Optimal Clusters')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'hierarchical_silhouette.png'))
        
        # Find optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        
        # Cut tree to get optimal clusters
        cluster_labels = fcluster(Z, optimal_k, criterion='maxclust') - 1  # 0-based indexing
    elif n_clusters is not None:
        # Use specified number of clusters
        cluster_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 0-based indexing
        print(f"Using specified number of clusters: {n_clusters}")
    else:
        # Use distance threshold
        cluster_labels = fcluster(Z, distance_threshold, criterion='distance') - 1  # 0-based indexing
        print(f"Cut dendrogram at distance {distance_threshold}, resulting in {len(np.unique(cluster_labels))} clusters")
    
    # Step 6: Create user-to-cluster mapping
    user_clusters = pd.DataFrame({
        'user_id': expense_ratios.index,
        'cluster_id': cluster_labels
    })
    
    # Create combined DataFrame with income bands
    combined_data = pd.DataFrame({
        'user_id': expense_ratios.index,
        'cluster_id': cluster_labels,
        'income_band': income_bands  # Direct access to the income bands Series
    })
    
    # Step 7: Compute cluster profiles
    cluster_profiles = []
    
    for cluster_id in sorted(np.unique(cluster_labels)):
        # Get users in this cluster
        users_in_cluster = combined_data[combined_data['cluster_id'] == cluster_id]
        
        # Get income distribution
        income_dist = users_in_cluster['income_band'].value_counts(normalize=True).to_dict()
        
        # Get demographic distributions if available
        demographics = {}
        
        # Add occupation distribution if available
        if 'occupation_category' in df.columns:
            # Get most frequent occupation for each user
            user_occupations = df.groupby('user_id')['occupation_category'].agg(lambda x: x.mode()[0])
            # Filter to just users in this cluster
            cluster_occupations = user_occupations[user_occupations.index.isin(users_in_cluster['user_id'])]
            # Calculate distribution
            occupation_dist = cluster_occupations.value_counts(normalize=True).to_dict()
            demographics['occupation'] = occupation_dist
        
        # Get average expense ratios for this cluster
        user_ids = users_in_cluster['user_id']
        cluster_means = expense_ratios.loc[user_ids].mean()
        top_expenses = cluster_means.nlargest(3).to_dict()
        
        # Create profile
        profile = {
            'cluster_id': cluster_id,
            'size': len(users_in_cluster),
            'income_distribution': income_dist,
            'demographics': demographics,
            'top_expenses': top_expenses
        }
        cluster_profiles.append(profile)
        
        print(f"Cluster {cluster_id}: {profile['size']} users")
        print(f"  Users: {users_in_cluster['user_id'].tolist()[:5]}{' ...' if len(users_in_cluster) > 5 else ''}")
    
    # Save cluster profiles
    with open(os.path.join(results_dir, 'hierarchical_cluster_profiles.txt'), 'w') as f:
        f.write(f"Hierarchical Clustering Analysis: {len(cluster_profiles)} clusters\n")
        f.write(f"Number of users analyzed: {len(user_clusters)}\n\n")
        
        for profile in cluster_profiles:
            f.write(f"Cluster {profile['cluster_id']} ({profile['size']} members)\n")
            
            f.write("Income Distribution:\n")
            if profile['income_distribution']:
                for band, pct in profile['income_distribution'].items():
                    pct_str = f"{pct*100:.1f}%" if pd.notna(pct) else "0.0%"
                    f.write(f"  - {band}: {pct_str}\n")
            else:
                f.write("  - No income data available\n")
            
            # Add demographic information
            if 'demographics' in profile and profile['demographics']:
                for demo_type, distribution in profile['demographics'].items():
                    f.write(f"\n{demo_type.title()} Distribution:\n")
                    for category, pct in distribution.items():
                        pct_str = f"{pct*100:.1f}%" if pd.notna(pct) else "0.0%"
                        f.write(f"  - {category}: {pct_str}\n")
            
            f.write("\nTop Expense Categories:\n")
            for cat, val in profile['top_expenses'].items():
                f.write(f"  - {cat}: {val:.2f}%\n")
            f.write("\n")
    
    # Step 8: Calculate optimization gaps and generate recommendations
    gap_data = []
    recommendations = []

    for _, row in user_clusters.iterrows():
        user_id = row['user_id']
        cluster_id = row['cluster_id']
        
        # Get user's expense ratios
        user_expenses = expense_ratios.loc[user_id]
        
        # Get cluster average expense ratios
        cluster_user_ids = user_clusters[user_clusters['cluster_id'] == cluster_id]['user_id']
        cluster_avg = expense_ratios.loc[cluster_user_ids].mean()
        
        # Calculate gaps (positive means user spends less than cluster average)
        gaps = cluster_avg - user_expenses
        
        # Find significant gaps (potential tax optimization)
        for category, gap in gaps.items():
            if gap > 1.0:  # At least 1% of income difference
                # Calculate confidence metrics for this recommendation
                confidence_metrics = calculate_hierarchical_confidence(
                    user_id, cluster_id, category, gap, 
                    user_clusters, expense_ratios, 
                    reduced_features, Z  # Pass linkage matrix instead of kmeans
                )
                
                # Store basic gap data
                gap_data.append({
                    'user_id': user_id,
                    'cluster_id': cluster_id,
                    'category': category,
                    'user_pct': user_expenses[category],
                    'cluster_avg_pct': cluster_avg[category],
                    'gap_pct': gap
                })
                
                # Format recommendation with confidence level
                if confidence_metrics['overall_confidence'] == "Very Strong":
                    prefix = "⭐⭐⭐ HIGHLY RECOMMENDED:"
                elif confidence_metrics['overall_confidence'] == "Strong":
                    prefix = "⭐⭐ RECOMMENDED:"
                elif confidence_metrics['overall_confidence'] == "Moderate":
                    prefix = "⭐ CONSIDER:"
                elif confidence_metrics['overall_confidence'] == "Weak":
                    prefix = "💡 SUGGESTION:"
                else:
                    prefix = "🔍 EXPLORE:"
                
                # Get peer context
                if 'occupation_category' in df.columns:
                    peer_occupation = df[df['user_id'].isin(cluster_user_ids)]['occupation_category'].mode()[0]
                    peer_context = f"Other {peer_occupation}"
                else:
                    peer_context = "Similar taxpayers"
                
                # Create enhanced recommendation text
                if user_expenses[category] == 0:
                    rec_text = (f"{prefix} {peer_context} claim {cluster_avg[category]:.1f}% of income "
                               f"for {category}; you currently claim none. "
                               f"{confidence_metrics['consistency_ratio']*100:.0f}% of similar taxpayers "
                               f"benefit from this deduction.")
                else:
                    rec_text = (f"{prefix} {peer_context} claim {cluster_avg[category]:.1f}% of income "
                               f"for {category}; you're at {user_expenses[category]:.1f}%. "
                               f"Based on {confidence_metrics['sample_size']} similar taxpayers.")
                
                # Add statistical significance note if applicable
                if confidence_metrics['statistically_significant']:
                    rec_text += " This difference is statistically significant."
                
                # Create comprehensive recommendation with all metrics
                recommendation = {
                    'user_id': user_id,
                    'category': category,
                    'recommendation': rec_text,
                    'gap_pct': gap,
                    'cluster_avg_pct': cluster_avg[category],
                    'user_pct': user_expenses[category],
                    'confidence_level': confidence_metrics['overall_confidence'],
                    'weighted_score': confidence_metrics['weighted_score'],
                    'sample_size': confidence_metrics['sample_size'],
                    'cluster_compactness': confidence_metrics['cluster_compactness'],
                    'user_distance': confidence_metrics['user_distance'],
                    'consistency_ratio': confidence_metrics['consistency_ratio'],
                    'effect_size': confidence_metrics['effect_size'],
                    'statistically_significant': confidence_metrics['statistically_significant'],
                    'p_value': confidence_metrics['p_value']
                }
                
                recommendations.append(recommendation)

    # Create DataFrames from collected data
    gaps_df = pd.DataFrame(gap_data)
    recommendations_df = pd.DataFrame(recommendations)

    # Save gap analysis and enhanced recommendations
    if not gaps_df.empty:
        gaps_df.to_csv(os.path.join(results_dir, 'hierarchical_optimization_gaps.csv'), index=False)
        
    if not recommendations_df.empty:
        recommendations_df.to_csv(os.path.join(results_dir, 'hierarchical_tax_recommendations_with_confidence.csv'), index=False)
        
        # Also create a simplified version with just user_id and recommendation text
        simple_recommendations = recommendations_df[['user_id', 'recommendation']]
        simple_recommendations.to_csv(os.path.join(results_dir, 'hierarchical_tax_recommendations.csv'), index=False)
    
    # Step 9: Join cluster information back to original data
    # Convert user_clusters to dict for faster lookup
    cluster_dict = user_clusters.set_index('user_id')['cluster_id'].to_dict()
    
    # Add cluster ID to original data
    enriched_df = df.copy()
    enriched_df['cluster_id'] = enriched_df['user_id'].map(cluster_dict)
    
    # Create a cluster tree visualization that shows the hierarchical relationship
    create_cluster_relationship_viz(Z, cluster_labels, results_dir)
    
    return enriched_df, gaps_df, recommendations_df, Z, cluster_labels

def calculate_hierarchical_confidence(user_id, cluster_id, category, gap_pct, user_clusters, expense_ratios, reduced_features, linkage_matrix):
    """Calculate confidence metrics for recommendation using hierarchical clustering metrics"""
    
    # 1. Sample size confidence (base metric)
    cluster_size = len(user_clusters[user_clusters['cluster_id'] == cluster_id])
    if cluster_size >= 10:
        sample_confidence = "High"
    elif cluster_size >= 5:
        sample_confidence = "Medium"
    elif cluster_size >= 3:
        sample_confidence = "Low"
    else:
        sample_confidence = "Very Low"
    
    # 2. Cluster compactness (how tight is this cluster?)
    # Get indices of users in this cluster
    user_indices = np.where(user_clusters['cluster_id'] == cluster_id)[0]
    
    # Get the points for this cluster
    cluster_points = reduced_features[user_indices]
    
    # Calculate within-cluster distances
    within_distances = []
    for i in range(len(cluster_points)):
        for j in range(i+1, len(cluster_points)):
            within_distances.append(euclidean(cluster_points[i], cluster_points[j]))
    
    if within_distances:
        avg_within_distance = np.mean(within_distances)
    else:
        avg_within_distance = 0
    
    # Calculate between-cluster distances (for context)
    between_distances = []
    other_indices = np.where(user_clusters['cluster_id'] != cluster_id)[0]
    
    # Sample some points from other clusters (for efficiency)
    if len(other_indices) > 100:
        other_indices = np.random.choice(other_indices, 100, replace=False)
    
    # Sample some points from this cluster (for efficiency)
    if len(user_indices) > 20:
        sample_indices = np.random.choice(user_indices, 20, replace=False)
    else:
        sample_indices = user_indices
        
    # Calculate distances
    for i in sample_indices:
        for j in other_indices:
            between_distances.append(euclidean(reduced_features[i], reduced_features[j]))
    
    if between_distances and within_distances:
        avg_between_distance = np.mean(between_distances)
        distance_ratio = avg_within_distance / avg_between_distance
    else:
        distance_ratio = 1.0
    
    # Evaluate cluster compactness
    if distance_ratio < 0.3:
        compactness_confidence = "High"     # Very compact cluster
    elif distance_ratio < 0.5:
        compactness_confidence = "Medium"   # Reasonably compact
    elif distance_ratio < 0.7:
        compactness_confidence = "Low"      # Somewhat spread out
    else:
        compactness_confidence = "Very Low" # Not very distinct
    
    # 3. User distance from cluster (how typical is this user?)
    user_idx = user_clusters.index[user_clusters['user_id'] == user_id].tolist()[0]
    user_point = reduced_features[user_idx]
    
    # Get mean point (centroid) of the cluster
    centroid = np.mean(cluster_points, axis=0)
    
    # Calculate user distance from centroid
    user_centroid_distance = euclidean(user_point, centroid)
    
    # Normalize by average distance from centroid
    all_centroid_distances = [euclidean(point, centroid) for point in cluster_points]
    avg_centroid_distance = np.mean(all_centroid_distances) if all_centroid_distances else 1.0
    
    user_distance_ratio = user_centroid_distance / avg_centroid_distance
    
    if user_distance_ratio < 0.7:
        typicality_confidence = "High"     # Very typical cluster member
    elif user_distance_ratio < 1.0:
        typicality_confidence = "Medium"   # Fairly typical
    elif user_distance_ratio < 1.3:
        typicality_confidence = "Low"      # Somewhat atypical
    else:
        typicality_confidence = "Very Low" # Outlier in their cluster
    
    # 4. Category consistency (how common is this expense category in the cluster?)
    users_in_cluster = user_clusters[user_clusters['cluster_id'] == cluster_id]['user_id']
    users_with_category = sum(expense_ratios.loc[users_in_cluster][category] > 0)
    consistency_ratio = users_with_category / cluster_size if cluster_size > 0 else 0
    
    if consistency_ratio >= 0.7:
        consistency_confidence = "High"
    elif consistency_ratio >= 0.5:
        consistency_confidence = "Medium"
    elif consistency_ratio >= 0.3:
        consistency_confidence = "Low"
    else:
        consistency_confidence = "Very Low"
    
    # 5. Category variance (how consistent is the amount claimed?)
    category_values = expense_ratios.loc[users_in_cluster][category]
    category_mean = category_values.mean()
    category_std = category_values.std()
    coefficient_of_variation = (category_std / category_mean) if category_mean > 0 else float('inf')
    
    if coefficient_of_variation < 0.3:
        variance_confidence = "High"      # Very consistent amounts
    elif coefficient_of_variation < 0.6:
        variance_confidence = "Medium"    # Moderately consistent
    elif coefficient_of_variation < 1.0:
        variance_confidence = "Low"       # Highly variable
    else:
        variance_confidence = "Very Low"  # Extremely variable
    
    # 6. Effect size confidence
    if gap_pct >= 5.0:
        effect_confidence = "High"
    elif gap_pct >= 2.0:
        effect_confidence = "Medium"
    elif gap_pct >= 1.0:
        effect_confidence = "Low"
    else:
        effect_confidence = "Very Low"
    
    # 7. Statistical significance (using appropriate tests based on sample size)
    user_value = expense_ratios.loc[user_id, category]
    cluster_values = expense_ratios.loc[users_in_cluster, category]
    
    if len(cluster_values) >= 20:
        # Enough data for Wilcoxon test
        try:
            _, p_value = stats.wilcoxon(cluster_values - user_value)
            stat_sig = p_value < 0.1 and np.median(cluster_values) > user_value
            test_used = "Wilcoxon test"
        except:
            # Fall back to t-test if Wilcoxon fails
            _, p_value = stats.ttest_1samp(cluster_values, user_value)
            stat_sig = p_value < 0.1 and np.mean(cluster_values) > user_value
            test_used = "t-test (fallback)"
        
    elif len(cluster_values) >= 8:
        # For moderate samples, use sign test
        signs = np.sign(cluster_values - user_value)
        pos_count = sum(signs > 0)
        n = len(signs)
        # Simple sign test (binomial test)
        _, p_value = stats.binom_test(pos_count, n, p=0.5, alternative='greater')
        stat_sig = p_value < 0.1
        test_used = "Sign test"
        
    elif len(cluster_values) >= 3:
        # For very small samples, use simple percentile approach
        percentile_75 = np.percentile(cluster_values, 75)
        stat_sig = user_value < percentile_75
        p_value = 0.5  # Placeholder p-value
        test_used = "75th percentile comparison"
        
    else:
        # Too few samples for any statistical test
        stat_sig = False
        p_value = 1.0
        test_used = "Insufficient data"
    
    # 8. Overall confidence (weighted combination of metrics)
    confidence_scores = {
        "High": 3,
        "Medium": 2,
        "Low": 1,
        "Very Low": 0
    }
    
    # Calculate weighted score
    weights = {
        "sample_confidence": 0.25,         # Sample size is critical
        "compactness_confidence": 0.15,    # Cluster quality matters
        "typicality_confidence": 0.15,     # User fit to cluster is important
        "consistency_confidence": 0.15,    # Category consistency in cluster
        "variance_confidence": 0.10,       # Variance in amounts
        "effect_confidence": 0.20          # Size of the gap
    }
    
    weighted_score = (
        weights["sample_confidence"] * confidence_scores[sample_confidence] +
        weights["compactness_confidence"] * confidence_scores[compactness_confidence] +
        weights["typicality_confidence"] * confidence_scores[typicality_confidence] +
        weights["consistency_confidence"] * confidence_scores[consistency_confidence] +
        weights["variance_confidence"] * confidence_scores[variance_confidence] +
        weights["effect_confidence"] * confidence_scores[effect_confidence]
    )
    
    # Final rating based on weighted score and statistical significance
    if weighted_score >= 2.5 and stat_sig:
        overall_confidence = "Very Strong"
    elif weighted_score >= 2.0 or (weighted_score >= 1.8 and stat_sig):
        overall_confidence = "Strong"
    elif weighted_score >= 1.5 or (weighted_score >= 1.3 and stat_sig):
        overall_confidence = "Moderate"
    elif weighted_score >= 1.0:
        overall_confidence = "Weak"
    else:
        overall_confidence = "Very Weak"
    
    # Return all metrics for transparency
    return {
        "sample_size": cluster_size,
        "sample_confidence": sample_confidence,
        "cluster_compactness": distance_ratio,
        "compactness_confidence": compactness_confidence,
        "user_distance": user_distance_ratio,
        "typicality_confidence": typicality_confidence,
        "consistency_ratio": consistency_ratio,
        "consistency_confidence": consistency_confidence,
        "coefficient_of_variation": coefficient_of_variation,
        "variance_confidence": variance_confidence,
        "effect_size": gap_pct,
        "effect_confidence": effect_confidence,
        "statistically_significant": stat_sig,
        "statistical_test": test_used,
        "p_value": p_value,
        "weighted_score": weighted_score,
        "overall_confidence": overall_confidence
    }

def create_cluster_relationship_viz(Z, cluster_labels, results_dir):
    """Create a visualization showing the hierarchical relationship between clusters"""
    
    # Get the number of clusters
    n_clusters = len(np.unique(cluster_labels))
    
    # Create a figure with cluster relationship visualization
    plt.figure(figsize=(12, 8))
    
    # Plot dendrogram with cluster coloring
    plt.title(f'Hierarchical Cluster Relationships ({n_clusters} clusters)')
    
    # Use different colors for each cluster
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
    
    # Draw dendrogram with labeled clusters
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=40,  # Show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        link_color_func=lambda k: colors[cluster_labels[k]] if k < len(cluster_labels) else 'gray'
    )
    
    # Add a legend mapping colors to cluster IDs
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=f'Cluster {i}') 
                      for i in range(n_clusters)]
    plt.legend(handles=legend_elements, title='Clusters', loc='upper right')
    
    plt.savefig(os.path.join(results_dir, 'cluster_relationships.png'))
    plt.close()

if __name__ == "__main__":
    # Dynamically construct data path and results directory
    data_path = os.path.join(os.getenv('HOME'), 'taxfix', 'taxfix-taxflow', 'notebooks', 'processed_data', 'full_joined_data.csv')
    results_dir = os.path.join(os.getenv('HOME'), 'taxfix', 'taxfix-taxflow', 'app', 'data', 'processed', 'tax_hierarchical_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run hierarchical clustering analysis
    enriched_data, gaps_df, recommendations_df, linkage_matrix, cluster_labels = tax_peer_group_hierarchical(
        data_path=data_path,
        results_dir=results_dir,
        income_quantiles=5
    )
    
    # Save results
    enriched_data.to_csv(os.path.join(results_dir, "enriched_tax_data.csv"), index=False)
    
    print(f"Analysis complete. Results saved to '{results_dir}' folder.")
    print(f"Generated {len(np.unique(cluster_labels))} clusters with hierarchical clustering.")