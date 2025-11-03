import streamlit as st
import pandas as pd
from data_loader import ProductionDataLoader
from analysis import ProductionAnalyzer
from visualization import ProductionVisualizer

# Page configuration
st.set_page_config(
    page_title="Production Dashboard",
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
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">🏭 Production Performance Dashboard 2025</div>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.data_loader = None
        st.session_state.analyzer = None
        st.session_state.visualizer = None
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Performance Dashboard", "Plant Analysis", "Schedule Analysis", "Data Explorer"]
    )
    
    # File upload
    st.sidebar.title("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Production Excel File", 
        type=['xlsx'],
        help="Upload PRODUCTION ALL AREA 2025.xlsx file"
    )
    
    if uploaded_file is not None:
        if not st.session_state.data_loaded:
            with st.spinner('Loading data...'):
                st.session_state.data_loader = ProductionDataLoader(uploaded_file)
                if st.session_state.data_loader.load_data():
                    st.session_state.analyzer = ProductionAnalyzer(st.session_state.data_loader)
                    st.session_state.visualizer = ProductionVisualizer(st.session_state.analyzer)
                    st.session_state.data_loaded = True
                    st.sidebar.success("Data loaded successfully!")
                else:
                    st.sidebar.error("Error loading data!")
    
    # Main content based on navigation
    if app_mode == "Home":
        show_home_page()
    elif app_mode == "Performance Dashboard":
        show_performance_dashboard()
    elif app_mode == "Plant Analysis":
        show_plant_analysis()
    elif app_mode == "Schedule Analysis":
        show_schedule_analysis()
    elif app_mode == "Data Explorer":
        show_data_explorer()

def show_home_page():
    st.header("Welcome to Production Analytics")
    
    if st.session_state.data_loaded:
        summary = st.session_state.data_loader.get_summary_stats()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Plants", summary['total_plants'])
        with col2:
            st.metric("Total Areas", summary['total_areas'])
        with col3:
            st.metric("Total Volume", f"{summary['total_volume']:,.0f}")
        with col4:
            achievement_rate = (summary['total_volume'] / summary['total_target']) * 100
            st.metric("Overall Achievement", f"{achievement_rate:.1f}%")
        
        # Quick insights
        st.subheader("Quick Insights")
        analyzer = st.session_state.analyzer
        
        top_performers = analyzer.get_top_performers(3)
        underperformers = analyzer.get_underperforming_plants(0.5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("🏆 **Top Performers**")
            if top_performers is not None and not top_performers.empty:
                for _, plant in top_performers.iterrows():
                    st.write(f"- {plant['Plant Name']}: {plant['% of Target']:.1%}")
        
        with col2:
            st.write("⚠️ **Need Attention**")
            if underperformers is not None and not underperformers.empty:
                for _, plant in underperformers.iterrows():
                    st.write(f"- {plant['Plant Name']}: {plant['% of Target']:.1%}")
    
    else:
        st.info("👈 Please upload your production data file to get started")

def show_performance_dashboard():
    st.header("Performance Dashboard")
    
    if st.session_state.data_loaded:
        visualizer = st.session_state.visualizer
        
        # Main dashboard
        fig = visualizer.create_performance_dashboard()
        st.plotly_chart(fig, use_container_width=True)
        
        # Area performance table
        st.subheader("Area Performance Details")
        area_performance = st.session_state.analyzer.get_performance_by_area()
        st.dataframe(area_performance.style.format({
            'Mtd Vol': '{:,.0f}',
            'Ann Fm Target': '{:,.0f}',
            '% of Target': '{:.1%}',
            '%Actual Suppply': '{:.1%}'
        }))
    
    else:
        st.warning("Please upload data first to view the dashboard")

def show_plant_analysis():
    st.header("Plant Performance Analysis")
    
    if st.session_state.data_loaded:
        analyzer = st.session_state.analyzer
        visualizer = st.session_state.visualizer
        
        # Performance chart
        fig = visualizer.create_plant_performance_chart()
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Performers")
            top_performers = analyzer.get_top_performers(10)
            if top_performers is not None:
                st.dataframe(top_performers.style.format({
                    'Mtd Vol': '{:,.0f}',
                    'Ann Fm Target': '{:,.0f}',
                    '% of Target': '{:.1%}'
                }))
        
        with col2:
            st.subheader("Plants Needing Attention")
            underperformers = analyzer.get_underperforming_plants(0.8)
            if underperformers is not None:
                st.dataframe(underperformers.style.format({
                    'Mtd Vol': '{:,.0f}',
                    'Ann Fm Target': '{:,.0f}',
                    '% of Target': '{:.1%}'
                }))

def show_schedule_analysis():
    st.header("Schedule vs Actual Analysis")
    
    if st.session_state.data_loaded:
        analyzer = st.session_state.analyzer
        visualizer = st.session_state.visualizer
        
        fig = visualizer.create_schedule_analysis_chart()
        st.plotly_chart(fig, use_container_width=True)
        
        schedule_df = analyzer.get_schedule_analysis()
        if schedule_df is not None:
            st.subheader("Schedule Performance Details")
            st.dataframe(schedule_df.style.format({
                'Rmc_Schedule': '{:,.0f}',
                'Mtd_Vol': '{:,.0f}',
                '%_Actual_Supply': '{:.1%}'
            }))

def show_data_explorer():
    st.header("Data Explorer")
    
    if st.session_state.data_loaded:
        df = st.session_state.data_loader.clean_production_data()
        
        st.subheader("Raw Production Data")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            areas = ['All'] + list(df['Area'].unique())
            selected_area = st.selectbox("Filter by Area", areas)
        
        with col2:
            plants = ['All'] + list(df['Plant Name'].unique())
            selected_plant = st.selectbox("Filter by Plant", plants)
        
        with col3:
            periods = ['All'] + list(df['Periode'].unique())
            selected_period = st.selectbox("Filter by Period", periods)
        
        # Apply filters
        filtered_df = df.copy()
        if selected_area != 'All':
            filtered_df = filtered_df[filtered_df['Area'] == selected_area]
        if selected_plant != 'All':
            filtered_df = filtered_df[filtered_df['Plant Name'] == selected_plant]
        if selected_period != 'All':
            filtered_df = filtered_df[filtered_df['Periode'] == selected_period]
        
        st.dataframe(filtered_df.style.format({
            'Ann Fm Target': '{:,.0f}',
            'Mtd Vol': '{:,.0f}',
            '% of Target': '{:.1%}',
            'Avg Vol.Day': '{:.1f}',
            'Rmc Schedule': '{:,.0f}',
            '%Actual Suppply': '{:.1%}'
        }))
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_production_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
