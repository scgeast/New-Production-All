import pandas as pd
import numpy as np

class ProductionDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.schedule_data = None
        self.target_data = None
        
    def load_data(self):
        """Load data from Excel file"""
        try:
            # Load main production data
            self.raw_data = pd.read_excel(self.file_path, sheet_name='RAWD')
            
            # Load schedule vs actual data
            self.schedule_data = pd.read_excel(self.file_path, sheet_name='SCHEDULE X ACTUAL')
            
            # Load MTD vs Target data
            self.target_data = pd.read_excel(self.file_path, sheet_name='MTD X TARGET')
            
            print("Data loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def clean_production_data(self):
        """Clean and preprocess production data"""
        if self.raw_data is None:
            return None
            
        df = self.raw_data.copy()
        
        # Remove formula indicators and clean column names
        df.columns = [str(col).replace('=', '').strip() for col in df.columns]
        
        # Convert numeric columns
        numeric_columns = ['Ann Fm Target', 'Mtd Vol', 'Avg Vol.Day', 'Rmc Schedule']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate performance metrics if not present
        if '% of Target' not in df.columns:
            df['% of Target'] = df['Mtd Vol'] / df['Ann Fm Target']
        
        if '%Actual Suppply' not in df.columns:
            df['%Actual Suppply'] = df['Mtd Vol'] / df['Rmc Schedule']
        
        return df
    
    def get_summary_stats(self):
        """Get summary statistics for the data"""
        df = self.clean_production_data()
        if df is None:
            return None
            
        summary = {
            'total_plants': df['Plant Name'].nunique(),
            'total_areas': df['Area'].nunique(),
            'total_periods': df['Periode'].nunique(),
            'avg_target_achievement': df['% of Target'].mean(),
            'total_volume': df['Mtd Vol'].sum(),
            'total_target': df['Ann Fm Target'].sum()
        }
        
        return summary
