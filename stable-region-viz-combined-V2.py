import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import matplotlib.patches as mpatches

def plot_rainfall_events_combined(input_file='AllDataAllFeature+Elevation.csv', 
                                 search_radius_km=100,
                                 num_events_to_plot=10):
    """
    Create a single map showing the 10 latest events with:
    - Gray blocks: All grid points (background)
    - Stars: Stable region (3600-4000m) 95th percentile events
    - Dots: Other regions' 95th percentile events within 100km radius
    """
    
    # Function to calculate haversine distance between two points
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on Earth (in km)"""
        from math import radians, sin, cos, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r
    
    # Read the data
    print("="*70)
    print("95th PERCENTILE RAINFALL SPATIAL ANALYSIS - COMBINED PLOT")
    print("="*70)
    print("\nReading data file...")
    df = pd.read_csv(input_file)
    
    # Calculate the 95th percentile of rainfall
    percentile_95 = df['Prec'].quantile(0.95)
    print(f"95th percentile rainfall threshold: {percentile_95:.2f} mm")
    
    # Filter for 95th percentile events
    df_95 = df[df['Prec'] >= percentile_95].copy()
    
    # Create date string for matching
    df['date_str'] = df.apply(
        lambda row: f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Date']):02d}", 
        axis=1
    )
    df_95['date_str'] = df_95.apply(
        lambda row: f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Date']):02d}", 
        axis=1
    )
    
    # Categorize by elevation
    df_stable = df_95[(df_95['elevation'] >= 3600) & (df_95['elevation'] <= 4000)].copy()
    df_other = df_95[(df_95['elevation'] < 3600) | (df_95['elevation'] > 4000)].copy()
    
    print(f"\n95th percentile events by elevation:")
    print(f"  • Stable region (3600-4000m): {len(df_stable):,}")
    print(f"  • Other regions: {len(df_other):,}")
    
    # Find dates with both stable and other region events
    stable_dates = set(df_stable['date_str'].unique())
    other_dates = set(df_other['date_str'].unique())
    overlap_dates = stable_dates.intersection(other_dates)
    
    print(f"\nDates with events in both regions: {len(overlap_dates)}")
    
    if len(overlap_dates) == 0:
        print("No dates found with events in both regions!")
        return
    
    # Select the 10 LATEST dates with overlapping events
    selected_dates = sorted(list(overlap_dates), reverse=True)[:num_events_to_plot]
    selected_dates = list(reversed(selected_dates))  # Put in chronological order for display
    
    print(f"\nSelected {len(selected_dates)} latest events (out of {len(overlap_dates)} total):")
    for i, date in enumerate(selected_dates):
        print(f"  {i+1:2d}. {date}")
    
    # Define color palette for different events
    # Using 10 distinct colors for better visibility
    event_colors = ['blue', 'green', 'purple', 'red', 'orange', 
                    'brown', 'pink', 'olive', 'cyan', 'magenta']
    
    # Create single figure
    print("\nCreating combined plot...")
    print("="*70)
    
    fig = plt.figure(figsize=(18, 14))  # Increased size for 10 events
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Determine map extent based on all data points
    lon_min = df['Longitude'].min() - 0.5
    lon_max = df['Longitude'].max() + 0.5
    lat_min = df['Latitude'].min() - 0.5
    lat_max = df['Latitude'].max() + 0.5
    
    # Set map extent
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Set plain white background - no map features
    ax.set_facecolor('white')
    
    # Optional: Add only minimal features (comment out if you want completely plain)
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color='lightgray', alpha=0.3)
    # ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.3, color='lightgray', alpha=0.3)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--', color='gray')
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot all grid points as uniform blocks (background)
    df_all_single_date = df[df['date_str'] == selected_dates[0]]
    ax.scatter(df_all_single_date['Longitude'], df_all_single_date['Latitude'], 
              c='lightgray', s=15, alpha=0.5, marker='s',  # 's' for square markers
              transform=ccrs.PlateCarree(),
              label=f'All grid points ({len(df_all_single_date):,})')
    
    # Create legend handles list
    legend_handles = []
    legend_handles.append(plt.Line2D([0], [0], marker='s', color='w', 
                                    markerfacecolor='lightgray', markersize=6, alpha=0.5,
                                    label=f'All grid points'))
    
    # Plot each event with its own color
    event_summary = []
    
    for idx, (date, color) in enumerate(zip(selected_dates, event_colors[:len(selected_dates)])):
        # Get stable region events for this date
        df_stable_date = df_stable[df_stable['date_str'] == date]
        
        # For each stable region event, find other region events within 100km radius
        df_other_nearby = pd.DataFrame()
        
        for _, stable_event in df_stable_date.iterrows():
            stable_lat = stable_event['Latitude']
            stable_lon = stable_event['Longitude']
            
            # Get all other region events on the same date
            df_other_same_day = df_other[df_other['date_str'] == date]
            
            # Calculate distances and filter for events within radius
            if len(df_other_same_day) > 0:
                distances = df_other_same_day.apply(
                    lambda row: haversine_distance(stable_lat, stable_lon, 
                                                  row['Latitude'], row['Longitude']), 
                    axis=1
                )
                
                # Filter for events within the search radius
                nearby_mask = distances <= search_radius_km
                df_nearby_temp = df_other_same_day[nearby_mask]
                
                # Add to our collection of nearby events (avoiding duplicates)
                df_other_nearby = pd.concat([df_other_nearby, df_nearby_temp]).drop_duplicates()
        
        # Store summary info
        event_summary.append({
            'date': date,
            'color': color,
            'stable_count': len(df_stable_date),
            'other_nearby_count': len(df_other_nearby),
            'total': len(df_stable_date) + len(df_other_nearby)
        })
        
        # Print event details
        print(f"\n📍 EVENT {idx+1} - Date: {date} (Color: {color})")
        print(f"  Stable region (3600-4000m) events: {len(df_stable_date)}")
        print(f"  Other regions events within {search_radius_km}km: {len(df_other_nearby)}")
        
        # Plot other region events within radius (as dots)
        if len(df_other_nearby) > 0:
            ax.scatter(df_other_nearby['Longitude'], df_other_nearby['Latitude'], 
                      c=color, s=60, alpha=0.7, 
                      marker='o',  # Circle marker for other regions
                      edgecolors='black', linewidth=0.5,
                      transform=ccrs.PlateCarree())
        
        # Plot stable region events (as stars)
        if len(df_stable_date) > 0:
            ax.scatter(df_stable_date['Longitude'], df_stable_date['Latitude'], 
                      c=color, s=150, alpha=0.9, 
                      marker='*',  # Star marker for stable region
                      edgecolors='black', linewidth=1,
                      transform=ccrs.PlateCarree())
        
        # Add to legend
        legend_handles.append(
            plt.Line2D([0], [0], marker='*', color='w', 
                      markerfacecolor=color, markersize=12, alpha=0.9,
                      markeredgecolor='black', markeredgewidth=1,
                      label=f'{date} (S:{len(df_stable_date)}, O:{len(df_other_nearby)})')
        )
    
    # Add title
    ax.set_title(f'Latest {len(selected_dates)} Events with 95th Percentile Rainfall (≥ {percentile_95:.1f}mm)\n' +
                f'Stars: Stable Region (3600-4000m) | Dots: Other Elevations within {search_radius_km}km radius\n' +
                f'Each event shown in a different color (see legend)',
                fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9, 
             framealpha=0.95, title=f'Events (S:Stable, O:Within {search_radius_km}km)',
             ncol=1 if len(legend_handles) <= 6 else 2)  # Use 2 columns if more than 6 items
    
    # Add statistics box
    stats_text = 'Event Statistics:\n'
    stats_text += '═' * 25 + '\n'
    for i, event in enumerate(event_summary):
        stats_text += f"{i+1}. {event['date']} ({event['color']})\n"
        stats_text += f"   Stable: {event['stable_count']:3d} | Within {search_radius_km}km: {event['other_nearby_count']:3d}\n"
    
    total_stable = sum(e['stable_count'] for e in event_summary)
    total_other_nearby = sum(e['other_nearby_count'] for e in event_summary)
    stats_text += '─' * 25 + '\n'
    stats_text += f'Total: Stable: {total_stable} | Nearby: {total_other_nearby}'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=8,  # Reduced font size for 10 events
            verticalalignment='bottom', horizontalalignment='left', 
            bbox=props, family='monospace')
    
    # Add color coding explanation
    explanation_text = 'Visualization Key:\n'
    explanation_text += '• Light gray blocks: All grid points\n'
    explanation_text += '• Colored stars (★): Stable region (3600-4000m) 95th percentile\n'
    explanation_text += f'• Colored dots (●): Other elevations within {search_radius_km}km\n'
    explanation_text += '• 10 distinct colors for different event dates'
    
    props2 = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.02, explanation_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', 
            bbox=props2)
    
    plt.tight_layout()
    
    # Save the combined plot
    output_filename = f'rainfall_events_combined_latest_{num_events_to_plot}.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n💾 Combined plot saved as: {output_filename}")
    
    # Display the plot
    plt.show()
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nDisplayed {len(selected_dates)} latest events with 95th percentile rainfall")
    print(f"Showing stable region (3600-4000m) events and other elevation events")
    print(f"within {search_radius_km}km radius of stable region events")
    print(f"\nDate range: {selected_dates[0]} to {selected_dates[-1]}")
    
    return selected_dates

# Run the analysis
if __name__ == "__main__":
    dates = plot_rainfall_events_combined(
        input_file='AllDataAllFeature+Elevation.csv',
        search_radius_km=100,
        num_events_to_plot=10
    )