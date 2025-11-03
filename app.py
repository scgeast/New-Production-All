# streamlit_app.py - Modern Web Dashboard
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Production Performance Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan modern
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
        border-left: 4px solid #3498db;
        padding-left: 10px;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #3498db;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.3rem;
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Status badges */
    .status-excellent { background-color: #2ecc71; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; }
    .status-good { background-color: #3498db; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; }
    .status-warning { background-color: #f39c12; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; }
    .status-critical { background-color: #e74c3c; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; }
    
    /* Data table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Progress bars */
    .progress-container {
        background: #ecf0f1;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    .progress-bar {
        height: 8px;
        border-radius: 10px;
        background: linear-gradient(90deg, #2ecc71, #3498db);
    }
</style>
""", unsafe_allow_html=True)

class ProductionAnalyzer:
    def __init__(self, df):
        self.df = df
        self.clean_data()
    
    def clean_data(self):
        """Clean and preprocess the data"""
        df = self.df.copy()
        
        # Convert numeric columns
        numeric_columns = ['Ann Fm Target', 'Mtd Vol', 'Avg Vol.Day', 'Rmc Schedule']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate performance metrics
        df['Achievement %'] = (df['Mtd Vol'] / df['Ann Fm Target'] * 100).round(2)
        df['Schedule Achievement %'] = (df['Mtd Vol'] / df['Rmc Schedule'] * 100).round(2)
        
        # Handle infinite values
        df['Achievement %'] = df['Achievement %'].replace([np.inf, -np.inf], 0)
        df['Schedule Achievement %'] = df['Schedule Achievement %'].replace([np.inf, -np.inf], 0)
        
        # Categorize performance
        conditions = [
            df['Achievement %'] >= 100,
            (df['Achievement %'] >= 80) & (df['Achievement %'] < 100),
            (df['Achievement %'] >= 60) & (df['Achievement %'] < 80),
            df['Achievement %'] < 60
        ]
        choices = ['Excellent', 'Good', 'Warning', 'Critical']
        df['Performance Status'] = np.select(conditions, choices, default='Unknown')
        
        self.df_clean = df
    
    def filter_by_month(self, selected_months):
        """Filter data by selected months"""
        if 'All' in selected_months or not selected_months:
            return self.df_clean
        else:
            return self.df_clean[self.df_clean['Periode'].isin(selected_months)]
    
    def get_summary_metrics(self, df_filtered, working_days):
        """Get overall summary metrics"""
        total_volume = df_filtered['Mtd Vol'].sum()
        total_target = df_filtered['Ann Fm Target'].sum()
        overall_achievement = (total_volume / total_target * 100) if total_target > 0 else 0
        
        # Calculate corrected daily average
        df_filtered['Avg Vol.Day Corrected'] = (df_filtered['Mtd Vol'] / working_days).round(1)
        
        return {
            'total_plants': df_filtered['Plant Name'].nunique(),
            'total_areas': df_filtered['Area'].nunique(),
            'total_periods': df_filtered['Periode'].nunique(),
            'total_volume': total_volume,
            'total_target': total_target,
            'overall_achievement': overall_achievement,
            'avg_daily_volume': df_filtered['Avg Vol.Day Corrected'].mean(),
            'performance_score': df_filtered['Achievement %'].mean()
        }
    
    def get_performance_distribution(self, df_filtered):
        """Get performance distribution"""
        return df_filtered['Performance Status'].value_counts()
    
    def get_area_performance(self, df_filtered):
        """Get performance by area"""
        area_perf = df_filtered.groupby('Area').agg({
            'Mtd Vol': 'sum',
            'Ann Fm Target': 'sum',
            'Achievement %': 'mean',
            'Plant Name': 'count'
        }).round(2)
        
        area_perf['Achievement %'] = (area_perf['Mtd Vol'] / area_perf['Ann Fm Target'] * 100).round(2)
        area_perf = area_perf.rename(columns={'Plant Name': 'Plant Count'})
        
        return area_perf
    
    def get_top_performers(self, df_filtered, n=10):
        """Get top performing plants"""
        return df_filtered.nlargest(n, 'Achievement %')[
            ['Plant Name', 'Area', 'Periode', 'Mtd Vol', 'Ann Fm Target', 'Achievement %', 'Performance Status']
        ]
    
    def get_underperformers(self, df_filtered, threshold=80):
        """Get underperforming plants"""
        underperformers = df_filtered[df_filtered['Achievement %'] < threshold]
        return underperformers.nsmallest(10, 'Achievement %')[
            ['Plant Name', 'Area', 'Periode', 'Mtd Vol', 'Ann Fm Target', 'Achievement %', 'Performance Status']
        ]

def create_metric_card(label, value, delta=None, delta_type="normal"):
    """Create a beautiful metric card"""
    delta_color = {
        "normal": "",
        "inverse": "color: #e74c3c" if delta and float(delta.strip('%')) < 0 else "color: #2ecc71"
    }
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div style="{delta_color.get(delta_type, '')}">{delta if delta else ''}</div>
    </div>
    """

def main():
    # Header Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header">🏭 Production Performance Dashboard</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: center; color: #7f8c8d; margin-bottom: 2rem;">Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙️ Dashboard Controls</div>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "📤 Upload Production Data", 
            type=['xlsx'],
            help="Upload your PRODUCTION ALL AREA 2025.xlsx file"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file, sheet_name='RAWD')
                analyzer = ProductionAnalyzer(df)
                
                # Get available months
                available_months = sorted(analyzer.df_clean['Periode'].unique())
                
                # Working days setting
                st.markdown("---")
                st.markdown('<div class="sidebar-header">📅 Filter Settings</div>', unsafe_allow_html=True)
                
                working_days = st.slider(
                    "Working Days per Month",
                    min_value=20,
                    max_value=31,
                    value=26,
                    help="Number of working days for daily average calculation"
                )
                
                # Month filter
                selected_months = st.multiselect(
                    "Select Months to Analyze:",
                    options=['All'] + available_months,
                    default=['All'],
                    help="Choose specific months for analysis"
                )
                
                if 'All' in selected_months:
                    selected_months = available_months
                
                # Performance threshold
                performance_threshold = st.slider(
                    "Performance Alert Threshold (%)",
                    min_value=50,
                    max_value=90,
                    value=75,
                    help="Plants below this percentage will be flagged"
                )
                
                # Filter data
                df_filtered = analyzer.filter_by_month(selected_months)
                
                # Summary metrics
                metrics = analyzer.get_summary_metrics(df_filtered, working_days)
                
                # Sidebar metrics
                st.markdown("---")
                st.markdown('<div class="sidebar-header">📊 Quick Stats</div>', unsafe_allow_html=True)
                
                st.metric("Plants", metrics['total_plants'])
                st.metric("Areas", metrics['total_areas'])
                st.metric("Total Volume", f"{metrics['total_volume']:,.0f}")
                st.metric("Achievement", f"{metrics['overall_achievement']:.1f}%")
                
                # Performance distribution
                perf_dist = analyzer.get_performance_distribution(df_filtered)
                st.markdown("---")
                st.markdown('<div class="sidebar-header">🎯 Performance Overview</div>', unsafe_allow_html=True)
                
                for status, count in perf_dist.items():
                    status_color = {
                        'Excellent': 'status-excellent',
                        'Good': 'status-good', 
                        'Warning': 'status-warning',
                        'Critical': 'status-critical'
                    }.get(status, 'status-warning')
                    
                    st.markdown(f'<span class="{status_color}">{status}: {count} plants</span>', unsafe_allow_html=True)
                
                return analyzer, df_filtered, metrics, working_days, selected_months, performance_threshold
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None, None, None, None, None, None
        else:
            st.info("Please upload an Excel file to begin analysis")
            return None, None, None, None, None, None

    # Main Content
    if uploaded_file is not None and analyzer is not None:
        # Executive Summary Section
        st.markdown('<div class="sub-header">📈 Executive Summary</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card(
                "Total Production Volume", 
                f"{metrics['total_volume']:,.0f}",
                f"Target: {metrics['total_target']:,.0f}"
            ), unsafe_allow_html=True)
        
        with col2:
            achievement_color = "inverse" if metrics['overall_achievement'] < 100 else "normal"
            st.markdown(create_metric_card(
                "Overall Achievement", 
                f"{metrics['overall_achievement']:.1f}%",
                f"{metrics['overall_achievement'] - 100:+.1f}% vs Target",
                achievement_color
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metric_card(
                "Average Daily Volume", 
                f"{metrics['avg_daily_volume']:,.1f}",
                f"Based on {working_days} days"
            ), unsafe_allow_html=True)
        
        with col4:
            performance_color = "normal" if metrics['performance_score'] >= 80 else "inverse"
            st.markdown(create_metric_card(
                "Performance Score", 
                f"{metrics['performance_score']:.1f}%",
                "Average plant achievement",
                performance_color
            ), unsafe_allow_html=True)
        
        # Performance Charts Section
        st.markdown('<div class="sub-header">📊 Performance Analytics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Area performance chart
            area_perf = analyzer.get_area_performance(df_filtered)
            if not area_perf.empty:
                fig_area = px.bar(
                    area_perf.reset_index(),
                    x='Area',
                    y='Achievement %',
                    title='<b>Achievement Rate by Area</b>',
                    color='Achievement %',
                    color_continuous_scale='RdYlGn',
                    text='Achievement %'
                )
                fig_area.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#2c3e50")
                )
                fig_area.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_area, use_container_width=True)
        
        with col2:
            # Performance distribution pie chart
            perf_dist = analyzer.get_performance_distribution(df_filtered)
            if not perf_dist.empty:
                colors = {'Excellent': '#2ecc71', 'Good': '#3498db', 'Warning': '#f39c12', 'Critical': '#e74c3c'}
                fig_pie = px.pie(
                    values=perf_dist.values,
                    names=perf_dist.index,
                    title='<b>Performance Distribution</b>',
                    color=perf_dist.index,
                    color_discrete_map=colors
                )
                fig_pie.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#2c3e50"),
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # Plant Performance Section
        st.markdown('<div class="sub-header">🏭 Plant Performance Analysis</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🏆 Top Performers", "⚠️ Needs Attention", "📋 All Plants"])
        
        with tab1:
            top_performers = analyzer.get_top_performers(df_filtered, 15)
            if not top_performers.empty:
                # Add progress bars for visual effect
                display_df = top_performers.copy()
                st.dataframe(
                    display_df.style.format({
                        'Mtd Vol': '{:,.0f}',
                        'Ann Fm Target': '{:,.0f}',
                        'Achievement %': '{:.1f}%'
                    }).applymap(lambda x: f"background-color: #2ecc71; color: white" if x == 'Excellent' else 
                               f"background-color: #3498db; color: white" if x == 'Good' else 
                               f"background-color: #f39c12; color: white" if x == 'Warning' else 
                               f"background-color: #e74c3c; color: white" if x == 'Critical' else '', 
                               subset=['Performance Status']),
                    use_container_width=True,
                    height=400
                )
        
        with tab2:
            underperformers = analyzer.get_underperformers(df_filtered, performance_threshold)
            if not underperformers.empty:
                st.dataframe(
                    underperformers.style.format({
                        'Mtd Vol': '{:,.0f}',
                        'Ann Fm Target': '{:,.0f}',
                        'Achievement %': '{:.1f}%'
                    }).applymap(lambda x: f"background-color: #e74c3c; color: white" if x == 'Critical' else 
                               f"background-color: #f39c12; color: white" if x == 'Warning' else '', 
                               subset=['Performance Status']),
                    use_container_width=True,
                    height=400
                )
            else:
                st.success(f"🎉 All plants are performing above {performance_threshold}%!")
        
        with tab3:
            # Detailed data view
            st.dataframe(
                df_filtered.style.format({
                    'Ann Fm Target': '{:,.0f}',
                    'Mtd Vol': '{:,.0f}',
                    'Achievement %': '{:.1f}%',
                    'Schedule Achievement %': '{:.1f}%'
                }).applymap(lambda x: f"background-color: #2ecc71; color: white" if x == 'Excellent' else 
                           f"background-color: #3498db; color: white" if x == 'Good' else 
                           f"background-color: #f39c12; color: white" if x == 'Warning' else 
                           f"background-color: #e74c3c; color: white" if x == 'Critical' else '', 
                           subset=['Performance Status']),
                use_container_width=True,
                height=500
            )
        
        # Export Section
        st.markdown('<div class="sub-header">📥 Export & Reports</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download Performance Report", use_container_width=True):
                csv = df_filtered.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download as CSV",
                    data=csv,
                    file_name="production_performance_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📈 Download Summary", use_container_width=True):
                summary_df = analyzer.get_area_performance(df_filtered).reset_index()
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Summary",
                    data=summary_csv,
                    file_name="performance_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            if st.button("🔄 Refresh Analysis", use_container_width=True):
                st.rerun()
    
    else:
        # Welcome screen
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            ## 🚀 Welcome to Production Dashboard
            
            **Get started in 3 easy steps:**
            
            1. **Upload** your production Excel file
            2. **Configure** analysis settings in sidebar  
            3. **Explore** interactive insights and reports
            
            ### 📋 Supported Features:
            - ✅ Monthly performance analysis
            - ✅ Area-wise comparison
            - ✅ Plant performance ranking
            - ✅ Achievement tracking
            - ✅ Export capabilities
            """)
        
        with col2:
            st.markdown("""
            ## 📊 Sample Data Structure
            
            Your Excel file should contain:
            
            ```csv
            Periode,Area,Plant Name,Ann Fm Target,Mtd Vol,...
            August,West1,Ciujung,1613,1734.5,...
            August,West1,Cilegon,1670,1327.5,...
            September,West1,Ciujung,1700,1944,...
            ```
            
            ### 🎯 Key Metrics Tracked:
            - Production Volume vs Target
            - Achievement Percentage
            - Performance Status
            - Area-wise Analysis
            """)

if __name__ == "__main__":
    main()
