# streamlit_app.py - Complete Production Dashboard
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Production Performance Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #2ecc71;
        font-weight: bold;
    }
    .negative {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ProductionAnalyzer:
    def __init__(self, df):
        self.df = df
        self.clean_data()
    
    def clean_data(self):
        """Clean and preprocess the data"""
        # Make a copy
        df = self.df.copy()
        
        # Convert numeric columns, handling errors
        numeric_columns = ['Ann Fm Target', 'Mtd Vol', 'Avg Vol.Day', 'Rmc Schedule']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate performance metrics
        df['Achievement %'] = (df['Mtd Vol'] / df['Ann Fm Target'] * 100).round(2)
        df['Schedule Achievement %'] = (df['Mtd Vol'] / df['Rmc Schedule'] * 100).round(2)
        
        # Categorize performance
        df['Performance Category'] = pd.cut(
            df['Achievement %'],
            bins=[0, 80, 100, float('inf')],
            labels=['Needs Attention', 'On Track', 'Exceeding Target']
        )
        
        self.df_clean = df
    
    def get_summary_metrics(self):
        """Get overall summary metrics"""
        total_volume = self.df_clean['Mtd Vol'].sum()
        total_target = self.df_clean['Ann Fm Target'].sum()
        overall_achievement = (total_volume / total_target * 100) if total_target > 0 else 0
        
        return {
            'total_plants': self.df_clean['Plant Name'].nunique(),
            'total_areas': self.df_clean['Area'].nunique(),
            'total_periods': self.df_clean['Periode'].nunique(),
            'total_volume': total_volume,
            'total_target': total_target,
            'overall_achievement': overall_achievement,
            'avg_daily_volume': self.df_clean['Avg Vol.Day'].mean()
        }
    
    def get_area_performance(self):
        """Get performance by area"""
        area_perf = self.df_clean.groupby('Area').agg({
            'Mtd Vol': 'sum',
            'Ann Fm Target': 'sum',
            'Achievement %': 'mean',
            'Plant Name': 'count'
        }).round(2)
        
        area_perf['Achievement %'] = (area_perf['Mtd Vol'] / area_perf['Ann Fm Target'] * 100).round(2)
        area_perf = area_perf.rename(columns={'Plant Name': 'Plant Count'})
        
        return area_perf
    
    def get_top_performers(self, n=10):
        """Get top performing plants"""
        return self.df_clean.nlargest(n, 'Achievement %')[
            ['Plant Name', 'Area', 'Mtd Vol', 'Ann Fm Target', 'Achievement %', 'Performance Category']
        ]
    
    def get_underperformers(self, threshold=80):
        """Get underperforming plants"""
        underperformers = self.df_clean[self.df_clean['Achievement %'] < threshold]
        return underperformers.nsmallest(10, 'Achievement %')[
            ['Plant Name', 'Area', 'Mtd Vol', 'Ann Fm Target', 'Achievement %', 'Performance Category']
        ]
    
    def get_monthly_trends(self):
        """Get monthly production trends"""
        monthly = self.df_clean.groupby('Periode').agg({
            'Mtd Vol': 'sum',
            'Ann Fm Target': 'sum',
            'Plant Name': 'count'
        }).round(2)
        
        monthly['Achievement %'] = (monthly['Mtd Vol'] / monthly['Ann Fm Target'] * 100).round(2)
        monthly = monthly.rename(columns={'Plant Name': 'Plant Count'})
        
        return monthly

def main():
    st.markdown('<div class="main-header">🏭 Production Performance Dashboard 2025</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Production Excel File", 
        type=['xlsx'],
        help="Upload your PRODUCTION ALL AREA 2025.xlsx file"
    )
    
    # Performance threshold
    performance_threshold = st.sidebar.slider(
        "Underperformance Threshold (%)",
        min_value=50,
        max_value=90,
        value=80,
        help="Plants below this achievement percentage will be flagged"
    )
    # Working days assumption
    working_days = st.sidebar.number_input(
        "Average Working Days per Month",
        min_value=1,
        max_value=31,
        value=22,
        help="Number of working days used for Avg Vol.Day calculation"
    )
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_excel(uploaded_file, sheet_name='RAWD')
            analyzer = ProductionAnalyzer(df)
            
            # Display success message
            st.success(f"✅ Data loaded successfully! {len(df)} records found.")
            
            # Summary metrics
            st.header("📊 Executive Summary")
            metrics = analyzer.get_summary_metrics()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Plants", metrics['total_plants'])
            with col2:
                st.metric("Total Areas", metrics['total_areas'])
            with col3:
                st.metric("Total Volume", f"{metrics['total_volume']:,.0f}")
            with col4:
                achievement_color = "normal" if metrics['overall_achievement'] >= 100 else "off"
                st.metric(
                    "Overall Achievement", 
                    f"{metrics['overall_achievement']:.1f}%",
                    delta=f"{metrics['overall_achievement'] - 100:.1f}%" if metrics['overall_achievement'] != 100 else None,
                    delta_color=achievement_color
                )
            
            # Performance Overview
            st.header("🎯 Performance Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Area performance chart
                area_perf = analyzer.get_area_performance()
                fig_area = px.bar(
                    area_perf.reset_index(),
                    x='Area',
                    y='Achievement %',
                    title='Achievement Rate by Area',
                    color='Achievement %',
                    color_continuous_scale='RdYlGn'
                )
                fig_area.update_layout(height=400)
                st.plotly_chart(fig_area, use_container_width=True)
            
            with col2:
                # Monthly trends
                monthly_trends = analyzer.get_monthly_trends()
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=monthly_trends.index,
                    y=monthly_trends['Mtd Vol'],
                    mode='lines+markers',
                    name='Actual Volume',
                    line=dict(color='#1f77b4')
                ))
                fig_trend.add_trace(go.Scatter(
                    x=monthly_trends.index,
                    y=monthly_trends['Ann Fm Target'],
                    mode='lines+markers',
                    name='Target Volume',
                    line=dict(color='#ff7f0e', dash='dash')
                ))
                fig_trend.update_layout(
                    title='Monthly Production Trends',
                    height=400,
                    xaxis_title='Period',
                    yaxis_title='Volume'
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            # Plant Performance Analysis
            st.header("🏭 Plant Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Performers")
                top_performers = analyzer.get_top_performers(10)
                st.dataframe(
                    top_performers.style.format({
                        'Mtd Vol': '{:,.0f}',
                        'Ann Fm Target': '{:,.0f}',
                        'Achievement %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("⚠️ Plants Needing Attention")
                underperformers = analyzer.get_underperformers(performance_threshold)
                if not underperformers.empty:
                    st.dataframe(
                        underperformers.style.format({
                            'Mtd Vol': '{:,.0f}',
                            'Ann Fm Target': '{:,.0f}',
                            'Achievement %': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info(f"No plants below {performance_threshold}% achievement rate")
            
            # Detailed Analysis
            st.header("🔍 Detailed Analysis")
            
            tab1, tab2, tab3 = st.tabs(["Area Performance", "Monthly Trends", "Raw Data"])
            
            with tab1:
                st.subheader("Area Performance Details")
                area_perf_detailed = analyzer.get_area_performance()
                st.dataframe(
                    area_perf_detailed.style.format({
                        'Mtd Vol': '{:,.0f}',
                        'Ann Fm Target': '{:,.0f}',
                        'Achievement %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            
            with tab2:
                st.subheader("Monthly Trends Details")
                monthly_detailed = analyzer.get_monthly_trends()
                st.dataframe(
                    monthly_detailed.style.format({
                        'Mtd Vol': '{:,.0f}',
                        'Ann Fm Target': '{:,.0f}',
                        'Achievement %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            
            with tab3:
                st.subheader("Raw Production Data")
                st.dataframe(
                    analyzer.df_clean.style.format({
                        'Ann Fm Target': '{:,.0f}',
                        'Mtd Vol': '{:,.0f}',
                        'Avg Vol.Day': '{:.1f}',
                        'Rmc Schedule': '{:,.0f}',
                        'Achievement %': '{:.1f}%',
                        'Schedule Achievement %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            
            # Performance Distribution
            st.header("📈 Performance Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance histogram
                fig_hist = px.histogram(
                    analyzer.df_clean,
                    x='Achievement %',
                    nbins=20,
                    title='Distribution of Achievement Rates',
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.add_vline(x=performance_threshold, line_dash="dash", line_color="red",
                                 annotation_text=f"Threshold: {performance_threshold}%")
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Performance categories
                performance_counts = analyzer.df_clean['Performance Category'].value_counts()
                fig_pie = px.pie(
                    values=performance_counts.values,
                    names=performance_counts.index,
                    title='Plants by Performance Category',
                    color=performance_counts.index,
                    color_discrete_map={
                        'Needs Attention': '#e74c3c',
                        'On Track': '#f39c12', 
                        'Exceeding Target': '#2ecc71'
                    }
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Export section
            st.header("📥 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download processed data
                csv = analyzer.df_clean.to_csv(index=False)
                st.download_button(
                    label="Download Processed Data as CSV",
                    data=csv,
                    file_name="production_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Download performance summary
                summary_df = analyzer.get_area_performance().reset_index()
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Performance Summary",
                    data=summary_csv,
                    file_name="performance_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please make sure you're uploading the correct Excel file format with a 'RAWD' sheet")
    
    else:
        # Welcome screen
        st.info("👆 Please upload your production Excel file to begin analysis")
        
        with st.expander("ℹ️ About this dashboard"):
            st.write("""
            This Production Performance Dashboard provides:
            
            - **Executive Summary**: Key metrics and overall performance
            - **Performance Overview**: Visual analysis by area and time period
            - **Plant Analysis**: Identification of top performers and plants needing attention
            - **Detailed Reports**: Comprehensive area and monthly performance data
            - **Performance Distribution**: Statistical analysis of achievement rates
            
            **Expected Data Format:**
            Your Excel file should contain a sheet named 'RAWD' with production data including:
            - Periode, Area, Plant Name
            - Ann Fm Target, Mtd Vol, Avg Vol.Day, Rmc Schedule
            """)
        
        # Sample data preview
        with st.expander("📋 Sample Data Structure"):
            sample_data = {
                'Periode': ['August', 'August', 'September'],
                'Area': ['West1', 'West1', 'West2'],
                'Plant Name': ['Ciujung', 'Cilegon', 'Serpong'],
                'Ann Fm Target': [1613, 1670, 2020],
                'Mtd Vol': [1734.5, 1327.5, 1547],
                'Avg Vol.Day': [55.95, 42.82, 49.90],
                'Rmc Schedule': [2096, 1603.5, 1700]
            }
            st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()
