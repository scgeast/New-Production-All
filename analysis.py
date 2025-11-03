import pandas as pd
import numpy as np

class ProductionAnalyzer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.df = data_loader.clean_production_data()
    
    def get_performance_by_area(self):
        """Analyze performance by area"""
        if self.df is None:
            return None
            
        area_performance = self.df.groupby('Area').agg({
            'Mtd Vol': 'sum',
            'Ann Fm Target': 'sum',
            '% of Target': 'mean',
            '%Actual Suppply': 'mean'
        }).round(3)
        
        area_performance['Performance_Status'] = np.where(
            area_performance['% of Target'] >= 1, 'Above Target', 'Below Target'
        )
        
        return area_performance
    
    def get_top_performers(self, n=5):
        """Get top performing plants"""
        if self.df is None:
            return None
            
        # Filter plants with sufficient data
        valid_plants = self.df[self.df['Ann Fm Target'] > 0]
        
        top_performers = valid_plants.nlargest(n, '% of Target')[
            ['Plant Name', 'Area', 'Mtd Vol', 'Ann Fm Target', '% of Target']
        ]
        
        return top_performers
    
    def get_underperforming_plants(self, threshold=0.8):
        """Get plants performing below threshold"""
        if self.df is None:
            return None
            
        underperformers = self.df[
            (self.df['% of Target'] < threshold) & 
            (self.df['Ann Fm Target'] > 0)
        ][['Plant Name', 'Area', 'Mtd Vol', 'Ann Fm Target', '% of Target']]
        
        return underperformers.sort_values('% of Target')
    
    def get_monthly_trends(self):
        """Analyze monthly production trends"""
        if self.df is None:
            return None
            
        monthly_trends = self.df.groupby('Periode').agg({
            'Mtd Vol': 'sum',
            'Ann Fm Target': 'sum',
            'Plant Name': 'count'
        }).rename(columns={'Plant Name': 'Plant_Count'})
        
        monthly_trends['Achievement_Rate'] = (
            monthly_trends['Mtd Vol'] / monthly_trends['Ann Fm Target']
        ).round(3)
        
        return monthly_trends
    
    def get_schedule_analysis(self):
        """Analyze schedule vs actual performance"""
        if self.data_loader.schedule_data is None:
            return None
            
        schedule_df = self.data_loader.schedule_data.copy()
        # Skip metadata rows and use actual data
        schedule_df = schedule_df.iloc[3:].reset_index(drop=True)
        schedule_df.columns = ['Plant_Name', 'Rmc_Schedule', 'Mtd_Vol', '%_Actual_Supply']
        
        # Convert to numeric
        schedule_df['Rmc_Schedule'] = pd.to_numeric(schedule_df['Rmc_Schedule'], errors='coerce')
        schedule_df['Mtd_Vol'] = pd.to_numeric(schedule_df['Mtd_Vol'], errors='coerce')
        schedule_df['%_Actual_Supply'] = pd.to_numeric(schedule_df['%_Actual_Supply'], errors='coerce')
        
        return schedule_df
