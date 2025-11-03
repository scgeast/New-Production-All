import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd

class ProductionVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def create_performance_dashboard(self):
        """Create main performance dashboard"""
        area_performance = self.analyzer.get_performance_by_area()
        monthly_trends = self.analyzer.get_monthly_trends()
        
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Performance by Area', 
                'Monthly Production Trends',
                'Target Achievement Rate by Area',
                'Plant Count by Period'
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # Performance by Area - Volume
        fig.add_trace(
            go.Bar(x=area_performance.index, y=area_performance['Mtd Vol'], 
                   name='MTD Volume', marker_color='blue'),
            row=1, col=1
        )
        
        # Monthly Trends
        fig.add_trace(
            go.Scatter(x=monthly_trends.index, y=monthly_trends['Mtd Vol'],
                      mode='lines+markers', name='Monthly Volume', line=dict(color='green')),
            row=1, col=2
        )
        
        # Target Achievement
        fig.add_trace(
            go.Bar(x=area_performance.index, y=area_performance['% of Target'],
                   name='Target Achievement', marker_color='orange'),
            row=2, col=1
        )
        
        # Plant Count
        fig.add_trace(
            go.Bar(x=monthly_trends.index, y=monthly_trends['Plant_Count'],
                   name='Plant Count', marker_color='purple'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Production Performance Dashboard", showlegend=False)
        return fig
    
    def create_plant_performance_chart(self):
        """Create chart showing plant performance"""
        top_performers = self.analyzer.get_top_performers(10)
        underperformers = self.analyzer.get_underperforming_plants(0.8)
        
        fig = go.Figure()
        
        if top_performers is not None and not top_performers.empty:
            fig.add_trace(go.Bar(
                x=top_performers['Plant Name'],
                y=top_performers['% of Target'],
                name='Top Performers',
                marker_color='green'
            ))
        
        if underperformers is not None and not underperformers.empty:
            fig.add_trace(go.Bar(
                x=underperformers['Plant Name'],
                y=underperformers['% of Target'],
                name='Underperformers',
                marker_color='red'
            ))
        
        fig.update_layout(
            title='Plant Performance Analysis',
            xaxis_title='Plant Name',
            yaxis_title='Target Achievement Rate',
            barmode='group'
        )
        
        return fig
    
    def create_schedule_analysis_chart(self):
        """Create schedule vs actual analysis chart"""
        schedule_df = self.analyzer.get_schedule_analysis()
        
        if schedule_df is None or schedule_df.empty:
            return go.Figure()
        
        fig = go.Figure(data=[
            go.Bar(name='RMC Schedule', x=schedule_df['Plant_Name'], y=schedule_df['Rmc_Schedule']),
            go.Bar(name='MTD Volume', x=schedule_df['Plant_Name'], y=schedule_df['Mtd_Vol'])
        ])
        
        fig.update_layout(
            title='RMC Schedule vs Actual Volume',
            xaxis_title='Plant',
            yaxis_title='Volume',
            barmode='group'
        )
        
        return fig
