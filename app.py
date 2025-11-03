# streamlit_app.py - Complete Production Dashboard
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

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
    def __init__(self, df, working_days=22):
        self.df = df
        self.working_days = working_days
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
        
        # Calculate correct Avg Vol.Day based on working days
        df['Calculated Avg Vol.Day'] = (df['Mtd Vol'] / self.working_days).round(2)
        
        # Categorize performance
        df['Performance Category'] = pd.cut(
            df['Achievement %'],
            bins=[0, 80, 100, float('inf')],
            labels=['Needs Attention', 'On Track', 'Exceeding Target']
        )
        
        # Add efficiency indicator
        df['Avg Vol Efficiency'] = ((df['Calculated Avg Vol.Day'] / df['Avg Vol.Day']) * 100).round(2)
        
        self.df_clean = df
    
    def filter_by_months(self, selected_months):
        """Filter data by selected months"""
        if 'All' in selected_months or not selected_months:
            return self.df_clean
        else:
            return self.df_clean[self.df_clean['Periode'].isin(selected_months)]
    
    def get_summary_metrics(self, selected_months=None):
        """Get overall summary metrics"""
        if selected_months:
            df_filtered = self.filter_by_months(selected_months)
        else:
            df_filtered = self.df_clean
            
        total_volume = df_filtered['Mtd Vol'].sum()
        total_target = df_filtered['Ann Fm Target'].sum()
        overall_achievement = (total_volume / total_target * 100) if total_target > 0 else 0
        
        return {
            'total_plants': df_filtered['Plant Name'].nunique(),
            'total_areas': df_filtered['Area'].nunique(),
            'total_periods': df_filtered['Periode'].nunique(),
            'total_volume': total_volume,
            'total_target': total_target,
            'overall_achievement': overall_achievement,
            'avg_daily_volume': df_filtered['Calculated Avg Vol.Day'].mean(),
            'original_avg_daily_volume': df_filtered['Avg Vol.Day'].mean(),
            'working_days': self.working_days,
            'selected_months': selected_months
        }
    
    def get_area_performance(self, selected_months=None):
        """Get performance by area"""
        if selected_months:
            df_filtered = self.filter_by_months(selected_months)
        else:
            df_filtered = self.df_clean
            
        area_perf = df_filtered.groupby('Area').agg({
            'Mtd Vol': 'sum',
            'Ann Fm Target': 'sum',
            'Achievement %': 'mean',
            'Plant Name': 'count',
            'Calculated Avg Vol.Day': 'mean',
            'Avg Vol.Day': 'mean'
        }).round(2)
        
        area_perf['Achievement %'] = (area_perf['Mtd Vol'] / area_perf['Ann Fm Target'] * 100).round(2)
        area_perf = area_perf.rename(columns={'Plant Name': 'Plant Count'})
        
        return area_perf
    
    def get_top_performers(self, n=10, selected_months=None):
        """Get top performing plants"""
        if selected_months:
            df_filtered = self.filter_by_months(selected_months)
        else:
            df_filtered = self.df_clean
            
        return df_filtered.nlargest(n, 'Achievement %')[
            ['Plant Name', 'Area', 'Periode', 'Mtd Vol', 'Ann Fm Target', 'Achievement %', 
             'Calculated Avg Vol.Day', 'Avg Vol.Day', 'Avg Vol Efficiency', 'Performance Category']
        ]
    
    def get_underperformers(self, threshold=80, selected_months=None):
        """Get underperforming plants"""
        if selected_months:
            df_filtered = self.filter_by_months(selected_months)
        else:
            df_filtered = self.df_clean
            
        underperformers = df_filtered[df_filtered['Achievement %'] < threshold]
        return underperformers.nsmallest(10, 'Achievement %')[
            ['Plant Name', 'Area', 'Periode', 'Mtd Vol', 'Ann Fm Target', 'Achievement %', 
             'Calculated Avg Vol.Day', 'Avg Vol.Day', 'Avg Vol Efficiency', 'Performance Category']
        ]
    
    def get_monthly_trends(self):
        """Get monthly production trends"""
        monthly = self.df_clean.groupby('Periode').agg({
            'Mtd Vol': 'sum',
            'Ann Fm Target': 'sum',
            'Plant Name': 'count',
            'Calculated Avg Vol.Day': 'mean',
            'Avg Vol.Day': 'mean'
        }).round(2)
        
        monthly['Achievement %'] = (monthly['Mtd Vol'] / monthly['Ann Fm Target'] * 100).round(2)
        monthly['Avg Vol Trend'] = (monthly['Calculated Avg Vol.Day'] / monthly['Avg Vol.Day'] * 100).round(2)
        monthly = monthly.rename(columns={'Plant Name': 'Plant Count'})
        
        return monthly
    
    def get_avg_vol_trends(self, selected_months=None):
        """Get detailed average volume trends by period and area"""
        if selected_months:
            df_filtered = self.filter_by_months(selected_months)
        else:
            df_filtered = self.df_clean
            
        trends = df_filtered.groupby(['Periode', 'Area']).agg({
            'Calculated Avg Vol.Day': 'mean',
            'Avg Vol.Day': 'mean',
            'Mtd Vol': 'sum',
            'Plant Name': 'count'
        }).round(2).reset_index()
        
        trends['Avg Vol Variance %'] = ((trends['Calculated Avg Vol.Day'] - trends['Avg Vol.Day']) / trends['Avg Vol.Day'] * 100).round(2)
        
        return trends

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
    
    # Initialize selected_months
    selected_months = []
    
    if uploaded_file is not None:
        try:
            # Load data first to get available months
            df = pd.read_excel(uploaded_file, sheet_name='RAWD')
            available_months = sorted(df['Periode'].unique())
            
            # Month filter - BAGIAN FILTER BULAN
            st.sidebar.subheader("📅 Month Filter")
            selected_months = st.sidebar.multiselect(
                "Select Months to Analyze",
                options=['All'] + list(available_months),
                default=['All'],
                help="Select one month, multiple months, or 'All' for all months"
            )
            
            # Handle 'All' selection
            if 'All' in selected_months:
                selected_months = list(available_months)
            
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
            
            # Initialize analyzer
            analyzer = ProductionAnalyzer(df, working_days)
            
            # Display filter info
            if selected_months:
                months_display = ", ".join(selected_months) if len(selected_months) <= 3 else f"{len(selected_months)} months selected"
                st.info(f"📊 Analyzing data for: **{months_display}**")
            
            # Display success message
            st.success(f"✅ Data loaded successfully! {len(df)} records found. Using {working_days} working days for calculations.")
            
            # Summary metrics
            st.header("📊 Executive Summary")
            metrics = analyzer.get_summary_metrics(selected_months)
            
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
            
            # Additional metrics for Avg Vol.Day
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Calculated Avg Vol/Day", 
                    f"{metrics['avg_daily_volume']:.1f}",
                    help=f"Based on MTD Volume ÷ {working_days} Working Days"
                )
            with col2:
                st.metric(
                    "Original Avg Vol/Day", 
                    f"{metrics['original_avg_daily_volume']:.1f}",
                    help="From original data"
                )
            with col3:
                variance = ((metrics['avg_daily_volume'] - metrics['original_avg_daily_volume']) / metrics['original_avg_daily_volume'] * 100) if metrics['original_avg_daily_volume'] > 0 else 0
                st.metric(
                    "Variance %", 
                    f"{variance:.1f}%",
                    delta=f"{variance:.1f}%",
                    delta_color="normal" if variance >= 0 else "off"
                )
            with col4:
                efficiency = (metrics['avg_daily_volume'] / metrics['original_avg_daily_volume'] * 100) if metrics['original_avg_daily_volume'] > 0 else 0
                st.metric(
                    "Efficiency %", 
                    f"{efficiency:.1f}%",
                    delta_color="normal" if efficiency >= 100 else "off"
                )
            
            # Performance Overview
            st.header("🎯 Performance Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Area performance chart
                area_perf = analyzer.get_area_performance(selected_months)
                fig_area = px.bar(
                    area_perf.reset_index(),
                    x='Area',
                    y='Achievement %',
                    title=f'Achievement Rate by Area ({len(selected_months)} months)',
                    color='Achievement %',
                    color_continuous_scale='RdYlGn'
                )
                fig_area.update_layout(height=400)
                st.plotly_chart(fig_area, use_container_width=True)
            
            with col2:
                # Monthly trends (always show all months for trend analysis)
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
                
                # Highlight selected months if not all are selected
                if selected_months and len(selected_months) < len(available_months):
                    for month in selected_months:
                        if month in monthly_trends.index:
                            fig_trend.add_vrect(
                                x0=month, x1=month,
                                fillcolor="green", opacity=0.1,
                                layer="below", line_width=0,
                                annotation_text=f"Selected: {month}",
                                annotation_position="top left"
                            )
                
                fig_trend.update_layout(
                    title='Monthly Production Trends (All Months)',
                    height=400,
                    xaxis_title='Period',
                    yaxis_title='Volume'
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            # Average Volume Trends Section
            st.header("📈 Average Volume per Day Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Avg Vol.Day comparison by area
                area_avg_vol = analyzer.get_area_performance(selected_months).reset_index()
                fig_avg_vol = go.Figure()
                fig_avg_vol.add_trace(go.Bar(
                    name='Calculated Avg Vol/Day',
                    x=area_avg_vol['Area'],
                    y=area_avg_vol['Calculated Avg Vol.Day'],
                    marker_color='#1f77b4'
                ))
                fig_avg_vol.add_trace(go.Bar(
                    name='Original Avg Vol/Day',
                    x=area_avg_vol['Area'],
                    y=area_avg_vol['Avg Vol.Day'],
                    marker_color='#ff7f0e'
                ))
                fig_avg_vol.update_layout(
                    title=f'Average Volume per Day Comparison by Area ({len(selected_months)} months)',
                    height=400,
                    xaxis_title='Area',
                    yaxis_title='Average Volume per Day',
                    barmode='group'
                )
                st.plotly_chart(fig_avg_vol, use_container_width=True)
            
            with col2:
                # Monthly Avg Vol.Day trends (filtered by selection)
                if selected_months:
                    monthly_avg_filtered = analyzer.get_avg_vol_trends(selected_months).groupby('Periode').agg({
                        'Calculated Avg Vol.Day': 'mean',
                        'Avg Vol.Day': 'mean'
                    }).round(2)
                    
                    fig_monthly_avg = go.Figure()
                    fig_monthly_avg.add_trace(go.Scatter(
                        x=monthly_avg_filtered.index,
                        y=monthly_avg_filtered['Calculated Avg Vol.Day'],
                        mode='lines+markers',
                        name='Calculated Avg Vol/Day',
                        line=dict(color='#1f77b4')
                    ))
                    fig_monthly_avg.add_trace(go.Scatter(
                        x=monthly_avg_filtered.index,
                        y=monthly_avg_filtered['Avg Vol.Day'],
                        mode='lines+markers',
                        name='Original Avg Vol/Day',
                        line=dict(color='#ff7f0e', dash='dash')
                    ))
                    fig_monthly_avg.update_layout(
                        title=f'Average Volume per Day Trends ({len(selected_months)} months)',
                        height=400,
                        xaxis_title='Period',
                        yaxis_title='Average Volume per Day'
                    )
                    st.plotly_chart(fig_monthly_avg, use_container_width=True)
            
            # Plant Performance Analysis
            st.header("🏭 Plant Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Performers")
                top_performers = analyzer.get_top_performers(10, selected_months)
                st.dataframe(
                    top_performers.style.format({
                        'Mtd Vol': '{:,.0f}',
                        'Ann Fm Target': '{:,.0f}',
                        'Achievement %': '{:.1f}%',
                        'Calculated Avg Vol.Day': '{:.1f}',
                        'Avg Vol.Day': '{:.1f}',
                        'Avg Vol Efficiency': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("⚠️ Plants Needing Attention")
                underperformers = analyzer.get_underperformers(performance_threshold, selected_months)
                if not underperformers.empty:
                    st.dataframe(
                        underperformers.style.format({
                            'Mtd Vol': '{:,.0f}',
                            'Ann Fm Target': '{:,.0f}',
                            'Achievement %': '{:.1f}%',
                            'Calculated Avg Vol.Day': '{:.1f}',
                            'Avg Vol.Day': '{:.1f}',
                            'Avg Vol Efficiency': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info(f"No plants below {performance_threshold}% achievement rate")
            
            # Detailed Analysis
            st.header("🔍 Detailed Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Area Performance", "Monthly Trends", "Avg Vol Trends", "Raw Data"])
            
            with tab1:
                st.subheader("Area Performance Details")
                area_perf_detailed = analyzer.get_area_performance(selected_months)
                st.dataframe(
                    area_perf_detailed.style.format({
                        'Mtd Vol': '{:,.0f}',
                        'Ann Fm Target': '{:,.0f}',
                        'Achievement %': '{:.1f}%',
                        'Calculated Avg Vol.Day': '{:.1f}',
                        'Avg Vol.Day': '{:.1f}'
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
                        'Achievement %': '{:.1f}%',
                        'Calculated Avg Vol.Day': '{:.1f}',
                        'Avg Vol.Day': '{:.1f}',
                        'Avg Vol Trend': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            
            with tab3:
                st.subheader("Average Volume Trends by Period and Area")
                avg_vol_trends = analyzer.get_avg_vol_trends(selected_months)
                st.dataframe(
                    avg_vol_trends.style.format({
                        'Calculated Avg Vol.Day': '{:.1f}',
                        'Avg Vol.Day': '{:.1f}',
                        'Mtd Vol': '{:,.0f}',
                        'Avg Vol Variance %': '{:.1f}%'
                    }),
                    use_container_width=True
                )
                
                # Heatmap of Avg Vol Efficiency by Area and Period
                if not avg_vol_trends.empty:
                    st.subheader("Avg Vol Efficiency Heatmap")
                    heatmap_data = avg_vol_trends.pivot_table(
                        index='Area', 
                        columns='Periode', 
                        values='Avg Vol Variance %',
                        aggfunc='mean'
                    ).round(1)
                    
                    if not heatmap_data.empty:
                        fig_heatmap = px.imshow(
                            heatmap_data,
                            title=f'Avg Volume Variance % by Area and Period ({len(selected_months)} months)',
                            color_continuous_scale='RdYlGn',
                            aspect='auto'
                        )
                        fig_heatmap.update_layout(height=400)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with tab4:
                st.subheader("Raw Production Data")
                if selected_months:
                    df_filtered = analyzer.filter_by_months(selected_months)
                else:
                    df_filtered = analyzer.df_clean
                    
                st.dataframe(
                    df_filtered.style.format({
                        'Ann Fm Target': '{:,.0f}',
                        'Mtd Vol': '{:,.0f}',
                        'Avg Vol.Day': '{:.1f}',
                        'Calculated Avg Vol.Day': '{:.1f}',
                        'Rmc Schedule': '{:,.0f}',
                        'Achievement %': '{:.1f}%',
                        'Schedule Achievement %': '{:.1f}%',
                        'Avg Vol Efficiency': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            
            # Performance Distribution
            st.header("📊 Performance Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance histogram
                if selected_months:
                    df_filtered = analyzer.filter_by_months(selected_months)
                else:
                    df_filtered = analyzer.df_clean
                    
                fig_hist = px.histogram(
                    df_filtered,
                    x='Achievement %',
                    nbins=20,
                    title=f'Distribution of Achievement Rates ({len(selected_months)} months)',
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.add_vline(x=performance_threshold, line_dash="dash", line_color="red",
                                 annotation_text=f"Threshold: {performance_threshold}%")
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Performance categories
                if selected_months:
                    df_filtered = analyzer.filter_by_months(selected_months)
                else:
                    df_filtered = analyzer.df_clean
                    
                performance_counts = df_filtered['Performance Category'].value_counts()
                fig_pie = px.pie(
                    values=performance_counts.values,
                    names=performance_counts.index,
                    title=f'Plants by Performance Category ({len(selected_months)} months)',
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
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download processed data
                if selected_months:
                    df_filtered = analyzer.filter_by_months(selected_months)
                else:
                    df_filtered = analyzer.df_clean
                    
                csv = df_filtered.to_csv(index=False)
                st.download_button(
                    label="Download Processed Data as CSV",
                    data=csv,
                    file_name="production_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Download performance summary
                summary_df = analyzer.get_area_performance(selected_months).reset_index()
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Performance Summary",
                    data=summary_csv,
                    file_name="performance_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col3:
                # Download avg vol trends
                avg_vol_df = analyzer.get_avg_vol_trends(selected_months)
                avg_vol_csv = avg_vol_df.to_csv(index=False)
                st.download_button(
                    label="Download Avg Vol Trends",
                    data=avg_vol_csv,
                    file_name="avg_volume_trends.csv",
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
            - **Average Volume Analysis**: Corrected Avg Vol.Day calculations and trends
            - **Plant Analysis**: Identification of top performers and plants needing attention
            - **Detailed Reports**: Comprehensive area and monthly performance data
            - **Performance Distribution**: Statistical analysis of achievement rates
            
            **New Features:**
            - ✅ Month Filter: Select single month, multiple months, or all months
            - ✅ Configurable working days for Avg Vol.Day calculation
            - ✅ Corrected Avg Vol.Day calculation based on working days
            - ✅ Average Volume trends by area and period
            - ✅ Efficiency analysis comparing calculated vs original Avg Vol.Day
            - ✅ Heatmap visualization of volume variance
            
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
