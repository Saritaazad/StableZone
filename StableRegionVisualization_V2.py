import pandas as pd
import numpy as np

def analyze_stable_region_events(main_data_file='AllDataAllFeature+Elevation.csv',
                                  events_file='extremeEventsSR.csv'):
    """
    For each extreme event in the stable region (from extremeEventsSR.csv),
    check if there are 95th or 99th percentile events outside the stable region
    on the same date.
    """
    
    print("=" * 70)
    print("STABLE REGION EXTREME EVENT PROPAGATION ANALYSIS")
    print("=" * 70)
    
    # Read main data
    print("\nReading main data file...")
    df = pd.read_csv(main_data_file)
    
    # Create date column
    df['FullDate'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=df['Date']))
    df['date_str'] = df['FullDate'].dt.strftime('%Y-%m-%d')
    
    # Calculate percentile thresholds
    p95 = df['Prec'].quantile(0.95)
    p99 = df['Prec'].quantile(0.99)
    
    print(f"95th percentile threshold: {p95:.4f} mm")
    print(f"99th percentile threshold: {p99:.4f} mm")
    
    # Read stable region events
    print("\nReading stable region events file...")
    events_df = pd.read_csv(events_file)
    events_df['Date'] = pd.to_datetime(events_df['Date'])
    events_df['date_str'] = events_df['Date'].dt.strftime('%Y-%m-%d')
    
    print(f"Total stable region events to analyze: {len(events_df)}")
    
    # Get unique dates from stable region events
    unique_dates = events_df['date_str'].unique()
    print(f"Unique dates: {len(unique_dates)}")
    
    print("\n" + "=" * 70)
    print("CHECKING CONCURRENT EXTREME EVENTS OUTSIDE STABLE REGION")
    print("=" * 70)
    
    results = []
    
    for date in unique_dates:
        print(f"\n{'─' * 50}")
        print(f"DATE: {date}")
        print(f"{'─' * 50}")
        
        # Get stable region events for this date
        stable_events = events_df[events_df['date_str'] == date]
        print(f"Stable region events (3500-4000m): {len(stable_events)}")
        
        for _, se in stable_events.iterrows():
            print(f"  • Lon: {se['Longitude']}, Lat: {se['Latitude']}, "
                  f"Elev: {se['Elevation']}m, Prec: {se['Precipitation']:.2f}mm ({se['Percentile']})")
        
        # Get all data for this date OUTSIDE stable region
        df_date = df[df['date_str'] == date]
        df_outside_stable = df_date[
            (df_date['elevation'] < 3500) | (df_date['elevation'] > 4000)
        ]
        
        # Find extreme events outside stable region
        extreme_outside_99 = df_outside_stable[df_outside_stable['Prec'] >= p99]
        extreme_outside_95 = df_outside_stable[
            (df_outside_stable['Prec'] >= p95) & (df_outside_stable['Prec'] < p99)
        ]
        
        total_outside_extreme = len(extreme_outside_99) + len(extreme_outside_95)
        
        print(f"\nExtreme events OUTSIDE stable region on {date}:")
        print(f"  • 99th percentile events: {len(extreme_outside_99)}")
        print(f"  • 95th percentile events: {len(extreme_outside_95)}")
        print(f"  • Total extreme events: {total_outside_extreme}")
        
        # Store results
        results.append({
            'Date': date,
            'Stable_Region_Events': len(stable_events),
            'Outside_99th_Events': len(extreme_outside_99),
            'Outside_95th_Events': len(extreme_outside_95),
            'Total_Outside_Extreme': total_outside_extreme,
            'Has_Concurrent_Extreme': total_outside_extreme > 0
        })
        
        # Print details of outside events
        if len(extreme_outside_99) > 0:
            print(f"\n  99th percentile locations outside stable region:")
            for _, row in extreme_outside_99.head(10).iterrows():
                print(f"    Lon: {row['Longitude']}, Lat: {row['Latitude']}, "
                      f"Elev: {row['elevation']:.0f}m, Prec: {row['Prec']:.2f}mm")
            if len(extreme_outside_99) > 10:
                print(f"    ... and {len(extreme_outside_99) - 10} more")
        
        if len(extreme_outside_95) > 0:
            print(f"\n  95th percentile locations outside stable region:")
            for _, row in extreme_outside_95.head(10).iterrows():
                print(f"    Lon: {row['Longitude']}, Lat: {row['Latitude']}, "
                      f"Elev: {row['elevation']:.0f}m, Prec: {row['Prec']:.2f}mm")
            if len(extreme_outside_95) > 10:
                print(f"    ... and {len(extreme_outside_95) - 10} more")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    
    total_dates = len(results_df)
    dates_with_concurrent = results_df['Has_Concurrent_Extreme'].sum()
    
    print(f"\nTotal dates analyzed: {total_dates}")
    print(f"Dates with concurrent extreme events outside stable region: {dates_with_concurrent}")
    print(f"Probability of concurrent extremes: {dates_with_concurrent/total_dates*100:.1f}%")
    
    print("\nResults by date:")
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('concurrent_extreme_events_analysis.csv', index=False)
    print("\nResults saved to: concurrent_extreme_events_analysis.csv")
    
    return results_df, df, events_df, p95, p99


def plot_events(main_data_file='AllDataAllFeature+Elevation.csv',
                events_file='extremeEventsSR.csv'):
    """
    Create maps showing stable region events and concurrent extreme events outside.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Run analysis first
    results_df, df, events_df, p95, p99 = analyze_stable_region_events(
        main_data_file, events_file
    )
    
    unique_dates = events_df['date_str'].unique()
    
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION MAPS")
    print("=" * 70)
    
    for idx, date in enumerate(unique_dates):
        # Get data for this date
        df_date = df[df['date_str'] == date]
        
        # Stable region events
        stable_events = events_df[events_df['date_str'] == date]
        
        # Outside stable region
        df_outside = df_date[(df_date['elevation'] < 3500) | (df_date['elevation'] > 4000)]
        extreme_outside_99 = df_outside[df_outside['Prec'] >= p99]
        extreme_outside_95 = df_outside[(df_outside['Prec'] >= p95) & (df_outside['Prec'] < p99)]
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Map extent
        lon_min, lon_max = df_date['Longitude'].min() - 0.5, df_date['Longitude'].max() + 0.5
        lat_min, lat_max = df_date['Latitude'].min() - 0.5, df_date['Latitude'].max() + 0.5
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Map features
        ax.add_feature(cfeature.LAND, alpha=0.3, color='lightgray')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='gray')
        ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5, color='gray')
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Plot all grid points
        ax.scatter(df_date['Longitude'], df_date['Latitude'],
                   c='black', s=2, alpha=0.3,
                   transform=ccrs.PlateCarree(),
                   label=f'All grid points ({len(df_date):,})')
        
        # Plot 95th percentile outside stable region (orange)
        if len(extreme_outside_95) > 0:
            ax.scatter(extreme_outside_95['Longitude'], extreme_outside_95['Latitude'],
                       c='orange', s=80, alpha=0.8,
                       edgecolors='darkorange', linewidth=1,
                       transform=ccrs.PlateCarree(),
                       label=f'95th percentile outside SR ({len(extreme_outside_95)})')
        
        # Plot 99th percentile outside stable region (yellow)
        if len(extreme_outside_99) > 0:
            ax.scatter(extreme_outside_99['Longitude'], extreme_outside_99['Latitude'],
                       c='yellow', s=100, alpha=0.9,
                       edgecolors='gold', linewidth=1.5,
                       transform=ccrs.PlateCarree(),
                       label=f'99th percentile outside SR ({len(extreme_outside_99)})')
        
        # Plot stable region events (red)
        ax.scatter(stable_events['Longitude'], stable_events['Latitude'],
                   c='red', s=150, alpha=0.9,
                   edgecolors='darkred', linewidth=2,
                   marker='*',
                   transform=ccrs.PlateCarree(),
                   label=f'Stable region events ({len(stable_events)})')
        
        # Title
        ax.set_title(f'Date: {date}\n'
                     f'Red★: Stable Region (3500-4000m) | Yellow: 99th% outside | Orange: 95th% outside\n'
                     f'95th threshold: {p95:.2f}mm | 99th threshold: {p99:.2f}mm',
                     fontsize=12, fontweight='bold', pad=15)
        
        ax.legend(loc='lower left', fontsize=9, framealpha=0.95)
        
        # Stats box
        stats = (f'Stable Region Events: {len(stable_events)}\n'
                 f'Outside 99th%: {len(extreme_outside_99)}\n'
                 f'Outside 95th%: {len(extreme_outside_95)}')
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        # Save
        output_file = f'stable_region_event_{idx+1}_{date}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        
        plt.close()
    
    print("\nAll plots saved.")


# Run analysis
results_df, df, events_df, p95, p99 = analyze_stable_region_events(
    main_data_file='AllDataAllFeature+Elevation.csv',
    events_file='extremeEventsSR.csv'
)

# Uncomment below to also generate plots
# plot_events(
#     main_data_file='AllDataAllFeature+Elevation.csv',
#     events_file='extremeEventsSR.csv'
# )