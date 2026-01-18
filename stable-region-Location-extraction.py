import pandas as pd
import numpy as np

def extract_stable_region_locations(input_file='AllDataAllFeature+Elevation.csv', 
                                   output_file='stable_region_3600_4000m_locations.csv'):
    """
    Extract all unique grid locations in the elevation range 3600-4000m
    and save them to a CSV file with Longitude and Latitude columns.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file containing elevation data
    output_file : str
        Path for the output CSV file with stable region locations
    
    Returns:
    --------
    df_stable_locations : DataFrame
        DataFrame containing the unique longitude and latitude of stable region
    """
    
    print("="*70)
    print("STABLE REGION (3600-4000m) GRID LOCATIONS EXTRACTION")
    print("="*70)
    
    # Read the data
    print(f"\nReading data file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Display basic info about the dataset
    print(f"Total records in dataset: {len(df):,}")
    print(f"Elevation range in dataset: {df['elevation'].min():.1f}m to {df['elevation'].max():.1f}m")
    
    # Filter for elevation range 3600-4000m
    print(f"\nFiltering for elevation range: 3600-4000m")
    df_stable = df[(df['elevation'] >= 3600) & (df['elevation'] <= 4000)].copy()
    
    print(f"Records in elevation range 3600-4000m: {len(df_stable):,}")
    
    # Get unique grid locations (unique combinations of Longitude and Latitude)
    df_unique_locations = df_stable[['Longitude', 'Latitude']].drop_duplicates().reset_index(drop=True)
    
    print(f"\nUnique grid locations in stable region: {len(df_unique_locations):,}")
    
    # Sort by Latitude (descending) then Longitude (ascending) for organized output
    df_unique_locations = df_unique_locations.sort_values(
        by=['Latitude', 'Longitude'], 
        ascending=[False, True]
    ).reset_index(drop=True)
    
    # Display sample of locations
    print("\nSample of stable region locations:")
    print(df_unique_locations.head(10).to_string(index=False))
    
    # Save to CSV
    df_unique_locations.to_csv(output_file, index=False)
    print(f"\n✅ CSV file saved: {output_file}")
    print(f"   Total locations saved: {len(df_unique_locations)}")
    
    # Display geographic extent of stable region
    lon_min = df_unique_locations['Longitude'].min()
    lon_max = df_unique_locations['Longitude'].max()
    lat_min = df_unique_locations['Latitude'].min()
    lat_max = df_unique_locations['Latitude'].max()
    
    print(f"\nGeographic extent of stable region (3600-4000m):")
    print(f"   Longitude range: {lon_min:.4f}° to {lon_max:.4f}°")
    print(f"   Latitude range:  {lat_min:.4f}° to {lat_max:.4f}°")
    
    # Optional: Display elevation statistics for these locations
    df_stable_with_elev = df[(df['elevation'] >= 3600) & (df['elevation'] <= 4000)].copy()
    df_elev_stats = df_stable_with_elev.groupby(['Longitude', 'Latitude'])['elevation'].agg(['mean', 'min', 'max']).reset_index()
    
    print(f"\nElevation statistics for stable region grid points:")
    print(f"   Mean elevation: {df_elev_stats['mean'].mean():.1f}m")
    print(f"   Std deviation:  {df_elev_stats['mean'].std():.1f}m")
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    
    return df_unique_locations


def extract_stable_region_with_elevation(input_file='AllDataAllFeature+Elevation.csv', 
                                        output_file='stable_region_3600_4000m_with_elevation.csv'):
    """
    Extract all unique grid locations in the elevation range 3600-4000m
    along with their elevation values and save to CSV.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file containing elevation data
    output_file : str
        Path for the output CSV file with stable region locations and elevations
    
    Returns:
    --------
    df_stable_locations : DataFrame
        DataFrame containing longitude, latitude, and elevation
    """
    
    print("="*70)
    print("STABLE REGION (3600-4000m) WITH ELEVATION DATA")
    print("="*70)
    
    # Read the data
    print(f"\nReading data file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Filter for elevation range 3600-4000m
    df_stable = df[(df['elevation'] >= 3600) & (df['elevation'] <= 4000)].copy()
    
    # Get unique grid locations with their elevation
    # Group by location and take mean elevation (in case of slight variations)
    df_locations_elev = df_stable.groupby(['Longitude', 'Latitude'])['elevation'].mean().reset_index()
    df_locations_elev.columns = ['Longitude', 'Latitude', 'Elevation']
    
    # Sort by Latitude (descending) then Longitude (ascending)
    df_locations_elev = df_locations_elev.sort_values(
        by=['Latitude', 'Longitude'], 
        ascending=[False, True]
    ).reset_index(drop=True)
    
    print(f"\nUnique grid locations in stable region: {len(df_locations_elev):,}")
    print("\nSample of locations with elevation:")
    print(df_locations_elev.head(10).to_string(index=False))
    
    # Save to CSV
    df_locations_elev.to_csv(output_file, index=False)
    print(f"\n✅ CSV file saved: {output_file}")
    print(f"   Columns: Longitude, Latitude, Elevation")
    print(f"   Total locations: {len(df_locations_elev)}")
    
    # Display elevation distribution
    print(f"\nElevation distribution in stable region:")
    print(f"   Minimum: {df_locations_elev['Elevation'].min():.1f}m")
    print(f"   Maximum: {df_locations_elev['Elevation'].max():.1f}m")
    print(f"   Mean:    {df_locations_elev['Elevation'].mean():.1f}m")
    print(f"   Median:  {df_locations_elev['Elevation'].median():.1f}m")
    
    return df_locations_elev


# Main execution
if __name__ == "__main__":
    # Option 1: Extract just Longitude and Latitude
    print("Option 1: Extracting longitude and latitude only...")
    df_basic = extract_stable_region_locations(
        input_file='AllDataAllFeature+Elevation.csv',
        output_file='stable_region_3600_4000m_locations.csv'
    )
    
    print("\n" + "="*70 + "\n")
    
    # Option 2: Extract Longitude, Latitude, and Elevation
    print("Option 2: Extracting longitude, latitude, and elevation...")
    df_with_elev = extract_stable_region_with_elevation(
        input_file='AllDataAllFeature+Elevation.csv',
        output_file='stable_region_3600_4000m_with_elevation.csv'
    )