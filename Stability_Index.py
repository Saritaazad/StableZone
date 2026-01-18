import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# Configure matplotlib for professional appearance
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 16,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 14
})

# Create output folder
OUTPUT_FOLDER = 'Images_StabilityIndex'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"📁 Created output folder: {OUTPUT_FOLDER}/")

def load_and_prepare_data(csv_file):
    """Load and prepare the NWH data"""
    print("📖 Loading Northwestern Himalayas data...")
    
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    
    if 'elevation' in df.columns:
        df.rename(columns={'elevation': 'Elevation'}, inplace=True)
    
    print(f"✅ Loaded {len(df):,} records")
    
    # Clean data
    required_columns = ['Longitude', 'Latitude', 'Elevation', 'Prec', 'T2M', 'QV2M', 'WS10M']
    df = df.dropna(subset=required_columns)
    print(f"📊 After cleaning: {len(df):,} complete records")
    
    return df

def calculate_cv_profiles(df, n_bins=15):
    """
    Calculate Coefficient of Variation (CV) profiles for each parameter by elevation
    CV = Standard Deviation / Mean
    """
    print("\n📈 Calculating CV profiles for all parameters...")
    
    # Create elevation bins
    df['elev_bin'] = pd.cut(df['Elevation'], bins=n_bins, labels=False)
    
    # Parameters to analyze
    parameters = ['Prec', 'T2M', 'QV2M', 'WS10M']
    param_names = {
        'Prec': 'Rainfall',
        'T2M': 'Temperature',
        'QV2M': 'Humidity',
        'WS10M': 'Wind Speed'
    }
    
    # Calculate CV for each elevation bin and parameter
    cv_profiles = df.groupby('elev_bin').agg({
        'Elevation': 'mean',
        'Prec': ['mean', 'std'],
        'T2M': ['mean', 'std'],
        'QV2M': ['mean', 'std'],
        'WS10M': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    cv_profiles.columns = ['elev_bin', 'elevation', 
                           'Prec_mean', 'Prec_std',
                           'T2M_mean', 'T2M_std',
                           'QV2M_mean', 'QV2M_std',
                           'WS10M_mean', 'WS10M_std']
    
    # Calculate CV for each parameter (handle division by zero)
    for param in parameters:
        mean_col = f'{param}_mean'
        std_col = f'{param}_std'
        cv_col = f'{param}_CV'
        
        # Use absolute value of mean to handle negative temperatures
        cv_profiles[cv_col] = cv_profiles[std_col] / (np.abs(cv_profiles[mean_col]) + 0.001)
    
    # Sort by elevation
    cv_profiles = cv_profiles.sort_values('elevation').reset_index(drop=True)
    
    print(f"   Created CV profiles for {len(cv_profiles)} elevation bins")
    
    return cv_profiles, parameters, param_names

def calculate_stability_index(cv_profiles, parameters):
    """
    Calculate Stability Index (SI) using the formula:
    
    SI = (1/P) * Σ[(CV_p - CV_p,min) / (CV_p,max - CV_p,min)]
    
    Where:
    - P is the number of parameters
    - CV_p is the coefficient of variation for parameter p
    - CV_p,min is the minimum CV across all elevation bands for parameter p
    - CV_p,max is the maximum CV across all elevation bands for parameter p
    
    SI ranges from 0 (most stable) to 1 (least stable)
    """
    print("\n📊 Calculating Stability Index (SI)...")
    
    P = len(parameters)  # Number of parameters
    
    # Store normalized CV for each parameter
    normalized_cvs = {}
    
    print(f"\n   Number of parameters (P): {P}")
    print("\n   Min-Max Normalization for each parameter:")
    print("   " + "-"*50)
    
    for param in parameters:
        cv_col = f'{param}_CV'
        
        CV_min = cv_profiles[cv_col].min()
        CV_max = cv_profiles[cv_col].max()
        
        # Min-Max Normalization: (CV - CV_min) / (CV_max - CV_min)
        if CV_max - CV_min > 0:
            normalized_cv = (cv_profiles[cv_col] - CV_min) / (CV_max - CV_min)
        else:
            normalized_cv = pd.Series([0.5] * len(cv_profiles))  # If no variation
        
        normalized_cvs[param] = normalized_cv
        cv_profiles[f'{param}_CV_normalized'] = normalized_cv
        
        print(f"   {param}: CV_min = {CV_min:.4f}, CV_max = {CV_max:.4f}")
    
    # Calculate Stability Index: SI = (1/P) * Σ(normalized CV)
    si_values = np.zeros(len(cv_profiles))
    
    for param in parameters:
        si_values += normalized_cvs[param].values
    
    cv_profiles['Stability_Index'] = si_values / P
    
    print("\n   " + "-"*50)
    print(f"   SI = (1/{P}) × Σ(Normalized CV)")
    print(f"   SI range: {cv_profiles['Stability_Index'].min():.4f} (most stable) to {cv_profiles['Stability_Index'].max():.4f} (least stable)")
    
    return cv_profiles

def identify_stability_zones(cv_profiles):
    """Identify and report stability zones based on SI values"""
    print("\n🎯 Identifying Stability Zones...")
    
    # Find elevation with minimum SI (most stable)
    min_si_idx = cv_profiles['Stability_Index'].idxmin()
    most_stable_elev = cv_profiles.loc[min_si_idx, 'elevation']
    most_stable_si = cv_profiles.loc[min_si_idx, 'Stability_Index']
    
    # Find elevation with maximum SI (least stable)
    max_si_idx = cv_profiles['Stability_Index'].idxmax()
    least_stable_elev = cv_profiles.loc[max_si_idx, 'elevation']
    least_stable_si = cv_profiles.loc[max_si_idx, 'Stability_Index']
    
    print(f"\n   Most Stable Elevation: {most_stable_elev:.0f}m (SI = {most_stable_si:.4f})")
    print(f"   Least Stable Elevation: {least_stable_elev:.0f}m (SI = {least_stable_si:.4f})")
    
    # Analyze 3500-4000m zone
    stability_zone = cv_profiles[
        (cv_profiles['elevation'] >= 3500) & 
        (cv_profiles['elevation'] <= 4000)
    ]
    
    if len(stability_zone) > 0:
        zone_mean_si = stability_zone['Stability_Index'].mean()
        zone_min_si = stability_zone['Stability_Index'].min()
        zone_max_si = stability_zone['Stability_Index'].max()
        
        print(f"\n   📍 3500-4000m Zone Analysis:")
        print(f"      Mean SI: {zone_mean_si:.4f}")
        print(f"      Min SI: {zone_min_si:.4f}")
        print(f"      Max SI: {zone_max_si:.4f}")
        print(f"      SI Range: {zone_max_si - zone_min_si:.4f}")
    
    # Compare with overall statistics
    overall_mean_si = cv_profiles['Stability_Index'].mean()
    print(f"\n   Overall Mean SI: {overall_mean_si:.4f}")
    
    if len(stability_zone) > 0:
        if zone_mean_si < overall_mean_si:
            print(f"   ✅ 3500-4000m zone is MORE STABLE than average (by {((overall_mean_si - zone_mean_si)/overall_mean_si)*100:.1f}%)")
        else:
            print(f"   ⚠️ 3500-4000m zone is LESS STABLE than average (by {((zone_mean_si - overall_mean_si)/overall_mean_si)*100:.1f}%)")
    
    return cv_profiles

def plot_stability_index(cv_profiles, parameters, param_names):
    """Create comprehensive Stability Index plots"""
    print("\n🎨 Creating Stability Index plots...")
    
    # Plot 1: Main Stability Index Plot
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    
    ax.plot(cv_profiles['elevation'], cv_profiles['Stability_Index'], 
            'k-o', linewidth=3, markersize=10, markerfacecolor='red',
            markeredgecolor='black', markeredgewidth=1, label='Stability Index (SI)')
    
    # Add trend line
    z = np.polyfit(cv_profiles['elevation'], cv_profiles['Stability_Index'], 2)
    p = np.poly1d(z)
    ax.plot(cv_profiles['elevation'], p(cv_profiles['elevation']), 
            'r--', linewidth=2, alpha=0.7, label='Quadratic Trend')
    
    # Highlight 3500-4000m zone
    from matplotlib.patches import Rectangle
    y_min, y_max = ax.get_ylim()
    rect = Rectangle((3500, 0), 500, 1.1, 
                     linewidth=2, edgecolor='blue', 
                     facecolor='cyan', alpha=0.2, linestyle='--',
                     label='Stability Zone (3500-4000m)')
    ax.add_patch(rect)
    
    # Add horizontal reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='SI = 0.5 (Reference)')
    
    # Labels
    ax.set_xlabel('Elevation (m)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Stability Index (SI)', fontsize=16, fontweight='bold')
    ax.set_title('Multi-Parameter Stability Index vs Elevation\nNorthwestern Himalayas', 
                 fontsize=16, fontweight='bold')
    
    # Add annotation
    ax.annotate('SI = 0: Most Stable\nSI = 1: Least Stable', 
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Legend
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    
    # Style
    ax.set_facecolor('white')
    ax.tick_params(colors='black', labelsize=14)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, 'stability_index_main.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {filepath}")
    plt.close()
    
    # Plot 2: SI with Individual Normalized CVs
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), facecolor='white')
    
    # Top plot: Individual normalized CVs
    ax1 = axes[0]
    colors = {'Prec': '#FF0000', 'T2M': '#0000FF', 'QV2M': '#00AA00', 'WS10M': '#800080'}
    markers = {'Prec': 'o', 'T2M': 's', 'QV2M': '^', 'WS10M': 'D'}
    
    for param in parameters:
        ax1.plot(cv_profiles['elevation'], cv_profiles[f'{param}_CV_normalized'], 
                marker=markers[param], linewidth=2.5, markersize=8, 
                color=colors[param], label=param_names[param],
                markeredgecolor='black', markeredgewidth=0.5)
    
    ax1.axvspan(3500, 4000, alpha=0.2, color='cyan')
    ax1.set_xlabel('Elevation (m)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Normalized CV', fontsize=14, fontweight='bold')
    ax1.set_title('Individual Parameter Normalized CV Profiles', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.05)
    
    # Bottom plot: Stability Index
    ax2 = axes[1]
    ax2.fill_between(cv_profiles['elevation'], 0, cv_profiles['Stability_Index'], 
                     alpha=0.3, color='red')
    ax2.plot(cv_profiles['elevation'], cv_profiles['Stability_Index'], 
            'r-o', linewidth=3, markersize=10, markeredgecolor='black',
            label='Stability Index (SI)')
    
    ax2.axvspan(3500, 4000, alpha=0.2, color='cyan', label='3500-4000m Zone')
    ax2.axhline(y=cv_profiles['Stability_Index'].mean(), color='green', 
                linestyle='--', linewidth=2, label=f'Mean SI = {cv_profiles["Stability_Index"].mean():.3f}')
    
    ax2.set_xlabel('Elevation (m)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Stability Index (SI)', fontsize=14, fontweight='bold')
    ax2.set_title('Aggregated Stability Index (SI = Average of Normalized CVs)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 1.05)
    
    plt.suptitle('Stability Index Analysis: Multi-Parameter Approach\nSI = (1/P) × Σ[(CV - CV_min)/(CV_max - CV_min)]',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, 'stability_index_detailed.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {filepath}")
    plt.close()
    
    # Plot 3: CV Comparison (Raw and Normalized)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
    
    # Left: Raw CVs
    ax1 = axes[0]
    for param in parameters:
        ax1.plot(cv_profiles['elevation'], cv_profiles[f'{param}_CV'], 
                marker=markers[param], linewidth=2.5, markersize=8, 
                color=colors[param], label=param_names[param],
                markeredgecolor='black', markeredgewidth=0.5)
    
    ax1.axvspan(3500, 4000, alpha=0.2, color='cyan')
    ax1.set_xlabel('Elevation (m)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Coefficient of Variation (CV)', fontsize=14, fontweight='bold')
    ax1.set_title('Raw CV Values', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right: Normalized CVs
    ax2 = axes[1]
    for param in parameters:
        ax2.plot(cv_profiles['elevation'], cv_profiles[f'{param}_CV_normalized'], 
                marker=markers[param], linewidth=2.5, markersize=8, 
                color=colors[param], label=param_names[param],
                markeredgecolor='black', markeredgewidth=0.5)
    
    ax2.axvspan(3500, 4000, alpha=0.2, color='cyan')
    ax2.set_xlabel('Elevation (m)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Normalized CV (0-1)', fontsize=14, fontweight='bold')
    ax2.set_title('Min-Max Normalized CV Values', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 1.05)
    
    plt.suptitle('CV Normalization for Stability Index Calculation',
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, 'cv_raw_vs_normalized.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {filepath}")
    plt.close()
    
    return fig

def plot_stability_index_by_zone(cv_profiles):
    """Create bar plot of mean SI by elevation zone"""
    print("\n🎨 Creating Stability Index by Zone plot...")
    
    # Define elevation zones
    zones = [
        (0, 2000, '<2000m'),
        (2000, 2500, '2000-2500m'),
        (2500, 3000, '2500-3000m'),
        (3000, 3500, '3000-3500m'),
        (3500, 4000, '3500-4000m'),
        (4000, 5000, '4000-5000m'),
        (5000, 7000, '>5000m')
    ]
    
    zone_names = []
    zone_mean_si = []
    zone_std_si = []
    
    for low, high, name in zones:
        zone_data = cv_profiles[
            (cv_profiles['elevation'] >= low) & 
            (cv_profiles['elevation'] < high)
        ]
        if len(zone_data) > 0:
            zone_names.append(name)
            zone_mean_si.append(zone_data['Stability_Index'].mean())
            zone_std_si.append(zone_data['Stability_Index'].std())
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    
    x = np.arange(len(zone_names))
    colors = ['steelblue'] * len(zone_names)
    
    # Highlight 3500-4000m zone
    if '3500-4000m' in zone_names:
        highlight_idx = zone_names.index('3500-4000m')
        colors[highlight_idx] = 'gold'
    
    bars = ax.bar(x, zone_mean_si, yerr=zone_std_si, capsize=5, 
                  color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, zone_mean_si)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + zone_std_si[i] + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add horizontal line for overall mean
    overall_mean = cv_profiles['Stability_Index'].mean()
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, 
               label=f'Overall Mean SI = {overall_mean:.3f}')
    
    ax.set_xlabel('Elevation Zone', fontsize=16, fontweight='bold')
    ax.set_ylabel('Mean Stability Index (SI)', fontsize=16, fontweight='bold')
    ax.set_title('Stability Index by Elevation Zone\nLower SI = More Stable', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(zone_names, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_FOLDER, 'stability_index_by_zone.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {filepath}")
    plt.close()
    
    return zone_names, zone_mean_si

def print_formula_explanation():
    """Print the Stability Index formula explanation"""
    print("\n" + "="*70)
    print("📐 STABILITY INDEX (SI) FORMULA")
    print("="*70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │                    1   P   CV_p - CV_p,min                      │
    │           SI  =   ─── Σ   ─────────────────                     │
    │                    P  p=1  CV_p,max - CV_p,min                  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Where:
    ─────────────────────────────────────────────────────────────────
    P           = Number of meteorological parameters (4)
    CV_p        = Coefficient of Variation for parameter p at given elevation
    CV_p,min    = Minimum CV across all elevations for parameter p
    CV_p,max    = Maximum CV across all elevations for parameter p
    
    Interpretation:
    ─────────────────────────────────────────────────────────────────
    SI = 0      → Most stable (all parameters at their minimum variability)
    SI = 1      → Least stable (all parameters at their maximum variability)
    SI = 0.5    → Average stability
    
    Parameters used:
    ─────────────────────────────────────────────────────────────────
    1. Rainfall (Prec)
    2. Temperature (T2M)
    3. Humidity (QV2M)
    4. Wind Speed (WS10M)
    """)

def save_results(cv_profiles, parameters):
    """Save results to CSV"""
    print("\n💾 Saving results to CSV...")
    
    # Save main results
    output_cols = ['elevation', 'Stability_Index']
    for param in parameters:
        output_cols.extend([f'{param}_CV', f'{param}_CV_normalized'])
    
    results_df = cv_profiles[output_cols].copy()
    filepath = os.path.join(OUTPUT_FOLDER, 'stability_index_results.csv')
    results_df.to_csv(filepath, index=False)
    print(f"   Saved: {filepath}")
    
    return results_df

def main(csv_file='AllDataAllFeature+Elevation.csv'):
    """Main analysis function"""
    print("="*70)
    print("🏔️ STABILITY INDEX (SI) ANALYSIS")
    print("Northwestern Himalayas Weather Data")
    print("="*70)
    
    # Print formula explanation
    print_formula_explanation()
    
    # Load and prepare data
    df = load_and_prepare_data(csv_file)
    
    # Calculate CV profiles
    cv_profiles, parameters, param_names = calculate_cv_profiles(df)
    
    # Calculate Stability Index
    cv_profiles = calculate_stability_index(cv_profiles, parameters)
    
    # Identify stability zones
    cv_profiles = identify_stability_zones(cv_profiles)
    
    # Create plots
    plot_stability_index(cv_profiles, parameters, param_names)
    plot_stability_index_by_zone(cv_profiles)
    
    # Save results
    results_df = save_results(cv_profiles, parameters)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📄 Generated files in '{OUTPUT_FOLDER}/' folder:")
    print("   • stability_index_main.png")
    print("   • stability_index_detailed.png")
    print("   • cv_raw_vs_normalized.png")
    print("   • stability_index_by_zone.png")
    print("   • stability_index_results.csv")
    
    return cv_profiles, results_df

if __name__ == "__main__":
    cv_profiles, results = main('AllDataAllFeature+Elevation.csv')