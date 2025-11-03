# streamlit_app.py - Dengan toggle untuk info area
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
    /* Style untuk info area */
    .info-area {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
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
        
        # PERBAIKAN: Hitung ulang Avg Vol.Day yang benar
        # Asumsi: 22 hari kerja per bulan (bisa disesuaikan)
        working_days_per_month = 22
        df['Avg Vol.Day Corrected'] = (df['Mtd Vol'] / working_days_per_month).round(1)
        
        # Calculate performance metrics
        df['Achievement %'] = (df['Mtd Vol'] / df['Ann Fm Target'] * 100).round(2)
        df['Schedule Achievement %'] = (df['Mtd Vol'] / df['Rmc Schedule'] * 100).round(2)
        
        # Handle infinite values (ketika target = 0)
        df['Achievement %'] = df['Achievement %'].replace([np.inf, -np.inf], 0)
        df['Schedule Achievement %'] = df['Schedule Achievement %'].replace([np.inf, -np.inf], 0)
        
        # Categorize performance
        df['Performance Category'] = pd.cut(
            df['Achievement %'],
            bins=[0, 80, 100, float('inf')],
            labels=['Needs Attention', 'On Track', 'Exceeding Target']
        )
        
        self.df_clean = df
    
    def filter_by_month(self, selected_months):
        """Filter data by selected months"""
        if 'All' in selected_months or not selected_months:
            return self.df_clean
        else:
            return self.df_clean[self.df_clean['Periode'].isin(selected_months)]
    
    def get_summary_metrics(self, df_filtered):
        """Get overall summary metrics for filtered data"""
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
            'avg_daily_volume': df_filtered['Avg Vol.Day Corrected'].mean()
        }
    
    def get_area_performance(self, df_filtered):
        """Get performance by area for filtered data"""
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
        """Get top performing plants from filtered data"""
        return df_filtered.nlargest(n, 'Achievement %')[
            ['Plant Name', 'Area', 'Periode', 'Mtd Vol', 'Ann Fm Target', 'Achievement %', 'Performance Category', 'Avg Vol.Day Corrected']
        ]
    
    def get_underperformers(self, df_filtered, threshold=80):
        """Get underperforming plants from filtered data"""
        underperformers = df_filtered[df_filtered['Achievement %'] < threshold]
        return underperformers.nsmallest(10, 'Achievement %')[
            ['Plant Name', 'Area', 'Periode', 'Mtd Vol', 'Ann Fm Target', 'Achievement %', 'Performance Category', 'Avg Vol.Day Corrected']
        ]
    
    def get_monthly_trends(self, df_filtered):
        """Get monthly production trends for filtered data"""
        monthly = df_filtered.groupby('Periode').agg({
            'Mtd Vol': 'sum',
            'Ann Fm Target': 'sum',
            'Plant Name': 'count',
            'Avg Vol.Day Corrected': 'mean'
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
    
    # Working days setting
    st.sidebar.subheader("⚙️ Calculation Settings")
    working_days = st.sidebar.number_input(
        "Working Days per Month",
        min_value=1,
        max_value=31,
        value=22,
        help="Number of working days used for average daily volume calculation"
    )
    
    # TOGGLE untuk Info Area
    st.sidebar.subheader("ℹ️ Display Settings")
    show_info_area = st.sidebar.checkbox(
        "Show Data Information", 
        value=True,
        help="Toggle to show/hide data loading information"
    )
    
    # Initialize session state for months
    if 'available_months' not in st.session_state:
        st.session_state.available_months = []
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_excel(uploaded_file, sheet_name='RAWD')
            
            # Apply working days setting
            df_temp = df.copy()
            numeric_columns = ['Ann Fm Target', 'Mtd Vol', 'Avg Vol.Day', 'Rmc Schedule']
            for col in numeric_columns:
                if col in df_temp.columns:
                    df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
            
            # Calculate corrected average with user-defined working days
            df_temp['Avg Vol.Day Corrected'] = (df_temp['Mtd Vol'] / working_days).round(1)
            
            analyzer = ProductionAnalyzer(df)
            
            # Get available months from data
            available_months = sorted(analyzer.df_clean['Periode'].unique())
            st.session_state.available_months = available_months
            
            # Month filter - di sidebar
            st.sidebar.subheader("📅 Filter Bulan")
            selected_months = st.sidebar.multiselect(
                "Pilih Bulan untuk Ditampilkan:",
                options=['All'] + available_months,
                default=['All'],
                help="Pilih satu atau beberapa bulan untuk dianalisis"
            )
            
            # Handle 'All' selection
            if 'All' in selected_months:
                selected_months = available_months
            
            # Performance threshold
            performance_threshold = st.sidebar.slider(
                "Underperformance Threshold (%)",
                min_value=50,
                max_value=90,
                value=80,
                help="Plants below this achievement percentage will be flagged"
            )
            
            # Filter data berdasarkan bulan yang dipilih
            df_filtered = analyzer.filter_by_month(selected_months)
            
            # TOGGLE AREA INFO - Tampilkan hanya jika toggle aktif
            if show_info_area:
                st.markdown('<div class="info-area">', unsafe_allow_html=True)
                st.success(f"✅ Data loaded successfully! {len(df_filtered)} records found for {len(selected_months)} selected month(s).")
                st.info(f"📊 **Calculation Settings:** {working_days} working days per month")
                
                # Tampilkan bulan yang aktif
                if selected_months:
                    months_display = ", ".join(selected_months)
                    st.info(f"📅 **Currently viewing:** {months_display}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Tampilkan minimal info saja
                st.success("✅ Data loaded successfully!")
            
            # Summary metrics untuk data yang difilter
            st.header("📊 Executive Summary")
            metrics = analyzer.get_summary_metrics(df_filtered)
            
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
            
            # Daily Performance
            st.header("📈 Daily Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_daily_volume = df_filtered['Avg Vol.Day Corrected'].mean()
                st.metric("Average Daily Volume", f"{avg_daily_volume:,.1f}")
            
            with col2:
                total_daily_capacity = df_filtered['Avg Vol.Day Corrected'].sum()
                st.metric("Total Daily Capacity", f"{total_daily_capacity:,.1f}")
            
            with col3:
                daily_efficiency = (df_filtered['Mtd Vol'].sum() / (df_filtered['Avg Vol.Day Corrected'].sum() * working_days) * 100) if df_filtered['Avg Vol.Day Corrected'].sum() > 0 else 0
                st.metric("Daily Efficiency", f"{daily_efficiency:.1f}%")
            
            # Performance Overview
            st.header("🎯 Performance Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Area performance chart untuk data filtered
                area_perf = analyzer.get_area_performance(df_filtered)
                if not area_perf.empty:
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
                else:
                    st.info("No data available for selected filters")
            
            with col2:
                # Daily volume by area
                daily_by_area = df_filtered.groupby('Area')['Avg Vol.Day Corrected'].mean().reset_index()
                if not daily_by_area.empty:
                    fig_daily = px.bar(
                        daily_by_area,
                        x='Area',
                        y='Avg Vol.Day Corrected',
                        title='Average Daily Volume by Area',
                        color='Avg Vol.Day Corrected',
                        color_continuous_scale='Blues'
                    )
                    fig_daily.update_layout(height=400)
                    st.plotly_chart(fig_daily, use_container_width=True)
                else:
                    st.info("No daily volume data available")
            
            # Plant Performance Analysis
            st.header("🏭 Plant Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Performers")
                top_performers = analyzer.get_top_performers(df_filtered, 10)
                if not top_performers.empty:
                    st.dataframe(
                        top_performers.style.format({
                            'Mtd Vol': '{:,.0f}',
                            'Ann Fm Target': '{:,.0f}',
                            'Achievement %': '{:.1f}%',
                            'Avg Vol.Day Corrected': '{:.1f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No top performers data for selected filters")
            
            with col2:
                st.subheader("⚠️ Plants Needing Attention")
                underperformers = analyzer.get_underperformers(df_filtered, performance_threshold)
                if not underperformers.empty:
                    st.dataframe(
                        underperformers.style.format({
                            'Mtd Vol': '{:,.0f}',
                            'Ann Fm Target': '{:,.0f}',
                            'Achievement %': '{:.1f}%',
                            'Avg Vol.Day Corrected': '{:.1f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info(f"No plants below {performance_threshold}% achievement rate for selected filters")
            
            # Detailed Analysis
            st.header("🔍 Detailed Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Area Performance", "Monthly Trends", "Daily Performance", "Raw Data"])
            
            with tab1:
                st.subheader("Area Performance Details")
                area_perf_detailed = analyzer.get_area_performance(df_filtered)
                if not area_perf_detailed.empty:
                    st.dataframe(
                        area_perf_detailed.style.format({
                            'Mtd Vol': '{:,.0f}',
                            'Ann Fm Target': '{:,.0f}',
                            'Achievement %': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No area performance data for selected filters")
            
            with tab2:
                st.subheader("Monthly Trends Details")
                monthly_detailed = analyzer.get_monthly_trends(df_filtered)
                if not monthly_detailed.empty:
                    st.dataframe(
                        monthly_detailed.style.format({
                            'Mtd Vol': '{:,.0f}',
                            'Ann Fm Target': '{:,.0f}',
                            'Achievement %': '{:.1f}%',
                            'Avg Vol.Day Corrected': '{:.1f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No monthly trends data for selected filters")
            
            with tab3:
                st.subheader("Daily Performance Analysis")
                daily_analysis = df_filtered[['Plant Name', 'Area', 'Periode', 'Mtd Vol', 'Avg Vol.Day Corrected', 'Achievement %']].sort_values('Avg Vol.Day Corrected', ascending=False)
                if not daily_analysis.empty:
                    st.dataframe(
                        daily_analysis.style.format({
                            'Mtd Vol': '{:,.0f}',
                            'Avg Vol.Day Corrected': '{:.1f}',
                            'Achievement %': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                    
                    # Daily volume distribution
                    fig_daily_dist = px.histogram(
                        df_filtered,
                        x='Avg Vol.Day Corrected',
                        title='Distribution of Daily Volumes',
                        nbins=20,
                        color_discrete_sequence=['#3498db']
                    )
                    fig_daily_dist.update_layout(height=400)
                    st.plotly_chart(fig_daily_dist, use_container_width=True)
                else:
                    st.info("No daily performance data for selected filters")
            
            with tab4:
                st.subheader("Raw Production Data")
                if not df_filtered.empty:
                    # Tampilkan kolom yang relevan saja untuk menghindari overload
                    display_columns = ['Periode', 'Area', 'Plant Name', 'Ann Fm Target', 'Mtd Vol', 
                                     'Avg Vol.Day Corrected', 'Rmc Schedule', 'Achievement %', 
                                     'Schedule Achievement %', 'Performance Category']
                    
                    available_columns = [col for col in display_columns if col in df_filtered.columns]
                    
                    st.dataframe(
                        df_filtered[available_columns].style.format({
                            'Ann Fm Target': '{:,.0f}',
                            'Mtd Vol': '{:,.0f}',
                            'Avg Vol.Day Corrected': '{:.1f}',
                            'Rmc Schedule': '{:,.0f}',
                            'Achievement %': '{:.1f}%',
                            'Schedule Achievement %': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No raw data available for selected filters")
            
            # Export section
            st.header("📥 Export Results")
            
            if not df_filtered.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download processed data
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
                    summary_df = analyzer.get_area_performance(df_filtered).reset_index()
                    summary_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="Download Performance Summary",
                        data=summary_csv,
                        file_name="performance_summary.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("No data available for export with current filters")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please make sure you're uploading the correct Excel file format with a 'RAWD' sheet")
    
    else:
        # Welcome screen
        st.info("👆 Please upload your production Excel file to begin analysis")
        
        with st.expander("ℹ️ About this dashboard"):
            st.write("""
            This Production Performance Dashboard provides:
            
            - **Accurate Daily Calculations**: Corrected average daily volume based on working days
            - **Monthly Filter**: Analyze data by specific months
            - **Executive Summary**: Key metrics and overall performance
            - **Performance Overview**: Visual analysis by area and time period
            - **Plant Analysis**: Identification of top performers and plants needing attention
            - **Detailed Reports**: Comprehensive area and monthly performance data
            
            **Calculation Method:**
            - Avg Vol.Day = Mtd Vol / Working Days per Month
            - Default: 22 working days (configurable in sidebar)
            """)

if __name__ == "__main__":
    main()
