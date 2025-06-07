import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- INITIAL SETUP ---
st.set_page_config(page_title="KSJ Data 2025", layout="wide")

# --- UTILITY FUNCTIONS (DENGAN REVISI) ---
def check_login(username, password):
    """Memeriksa kredensial login dan me-refresh halaman."""
    if username == "Blitz" and password == "ksj2025":
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
        st.rerun()  # REVISI: Langsung refresh untuk masuk ke dashboard.
    else:
        st.session_state["logged_in"] = False
        st.session_state.pop("username", None)
        st.error("Incorrect Username or Password!")

def logout():
    """Membersihkan session state dan me-refresh halaman untuk logout."""
    # Simpan beberapa kunci yang tidak ingin dihapus (jika ada)
    keys_to_keep = [] 
    # Hapus semua state lain untuk memastikan reset total
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            st.session_state.pop(key)
    st.session_state["logged_in"] = False
    st.rerun() # REVISI: Langsung refresh untuk kembali ke halaman login.

def set_view(view_name):
    """Mengatur halaman yang akan ditampilkan dan me-refresh."""
    st.session_state["view"] = view_name
    st.rerun() # REVISI: Kunci untuk mengatasi masalah 'klik 2x' pada navigasi.

# --- DATA LOADING (TETAP SAMA) ---
@st.cache_data
def load_and_process_main_data():
    """Memuat dan membersihkan data dari Excel, hanya sekali."""
    df = pd.read_excel("KSJ Data 2025.xlsx")
    df.columns = [col.strip().replace(" ", "").replace("#", "").lower() for col in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day'] = df['date'].dt.day_name()
    if 'week' not in df.columns and 'date' in df.columns:
        df['week'] = df['date'].dt.isocalendar().week.astype(int)
    numeric_cols = ['selling', 'revenue', 'cups']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- PAGE DISPLAY FUNCTIONS ---
def display_main_menu():
    """Menampilkan menu utama dengan 3 pilihan kartu."""
    st.header("Main Menu")
    st.markdown("Select an analysis to view from the options below.")
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        with st.container(border=True):
            st.subheader("📊 All Data")
            # REVISI: Teks diseragamkan panjangnya untuk menyamakan tinggi kartu.
            st.markdown("View raw data with interactive filters, plus weekly productivity graphs and daily sales analysis.")
            if st.button("Open Report", key="grup1_button", use_container_width=True):
                set_view('grup_1')
    with col2:
        with st.container(border=True):
            st.subheader("🗓️ Business Dashboard")
            st.markdown("View a consolidated dashboard of business performance, position, and seller retention analysis.")
            if st.button("Open Report", key="grup2_button", use_container_width=True):
                set_view('grup_2')
    with col3:
        with st.container(border=True):
            st.subheader("📍 Area & Outlet Analysis")
            # REVISI: Teks diseragamkan panjangnya untuk menyamakan tinggi kartu.
            st.markdown("Drill down into area-specific performance, with detailed location and outlet-level breakdowns.")
            if st.button("Open Report", key="area_button", use_container_width=True):
                set_view('area_analysis')

def display_grup_1():
    """Menampilkan halaman 'All Data' dengan filter dan visualisasi."""
    if st.button("⬅️ Back to Menu"):
        set_view('main_menu')
        return

    st.markdown("---")
    df = st.session_state["main_df"]
    st.subheader("📄 All Data")

    # REVISI: Logika filter disederhanakan untuk mengatasi kelambatan dan 'nyangkut'.
    if 'grup1_filters' not in st.session_state:
        st.session_state.grup1_filters = {}
        for col in df.columns:
            if df[col].dtype != 'datetime64[ns]' and len(df[col].dropna().unique()) < 100:
                st.session_state.grup1_filters[col] = sorted(df[col].dropna().unique())

    with st.expander("🔎 Filter Data", expanded=False):
        st.write("Apply filters below. Data will update automatically.")
        filter_cols = st.columns(3)
        col_idx = 0
        for col, unique_vals in st.session_state.grup1_filters.items():
            with filter_cols[col_idx]:
                st.session_state.grup1_filters[col] = st.multiselect(
                    f"Select {col.title()}",
                    options=unique_vals,
                    default=st.session_state.grup1_filters.get(col, []),
                    key=f"multiselect_{col}_g1"
                )
            col_idx = (col_idx + 1) % 3

    filtered_df = df.copy()
    for col, selected_values in st.session_state.grup1_filters.items():
        if not selected_values:
            filtered_df = pd.DataFrame(columns=df.columns)
            break
        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
            
    if not filtered_df.empty:
        styled_df = filtered_df.copy()
        if 'date' in styled_df.columns:
            styled_df['date'] = styled_df['date'].dt.strftime('%d/%m/%Y')
        for col in ['selling', 'revenue']:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(lambda x: f"Rp {x:,.0f}".replace(",", ".") if pd.notnull(x) else "-")
        styled_df.columns = [col.title() for col in styled_df.columns]
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No data to display for the selected filters.")
        
    st.subheader("📈 Week-on-Week Productivity")
    if 'week' in filtered_df.columns and 'cups' in filtered_df.columns and 'revenue' in filtered_df.columns:
        if not filtered_df.empty:
            df_chart_cups = filtered_df.groupby('week')['cups'].sum().reset_index()
            df_chart_revenue = filtered_df.groupby('week')['revenue'].sum().reset_index()
            col1, col2 = st.columns(2)
            with col1:
                fig_cups = px.bar(df_chart_cups, x='week', y='cups', title='Total Cups per Week', labels={'cups': 'Cups Sold', 'week': 'Week'}, text_auto=True)
                st.plotly_chart(fig_cups, use_container_width=True)
            with col2:
                fig_revenue = px.bar(df_chart_revenue, x='week', y='revenue', title='Total Revenue per Week', labels={'revenue': 'Total Revenue (Rp)', 'week': 'Week'}, text_auto=True)
                fig_revenue.update_traces(texttemplate='Rp%{y:,.0f}', textposition='outside')
                fig_revenue.update_yaxes(title_text='Total Revenue (Rp)')
                st.plotly_chart(fig_revenue, use_container_width=True)
        else:
            st.info("No data available for Week-on-Week Productivity based on current filters.")
    else:
        st.warning("Column 'week', 'cups', or 'revenue' is not available for charting.")
        
    st.subheader("📅 Productivity by Day")
    if 'day' in filtered_df.columns and 'cups' in filtered_df.columns and 'ridername' in filtered_df.columns:
        if not filtered_df.empty:
            df_day = filtered_df.copy()
            daily_stats = df_day.groupby(['day', 'date']).agg(total_cups=('cups', 'sum'), unique_riders=('ridername', 'nunique')).reset_index()
            avg_sellers_day = daily_stats.groupby('day')['unique_riders'].mean().round(0).astype(int).reset_index(name='avg_sellers')
            total_cups_day = daily_stats.groupby('day')['total_cups'].sum().reset_index(name='total_cups')
            daily_stats['productivity'] = daily_stats.apply(lambda row: row['total_cups'] / row['unique_riders'] if row['unique_riders'] > 0 else 0, axis=1)
            avg_productivity_day = daily_stats.groupby('day')['productivity'].mean().round(0).astype(int).reset_index(name='avg_productivity')
            summary = total_cups_day.merge(avg_sellers_day, on='day').merge(avg_productivity_day, on='day')
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            summary['day'] = pd.Categorical(summary['day'], categories=day_order, ordered=True)
            summary = summary.sort_values('day')
            st.dataframe(summary.rename(columns={'day': 'Day','total_cups': 'Total Cups','avg_sellers': 'Avg. Sellers','avg_productivity': 'Avg. Cups Sold Per Day'}), use_container_width=True, hide_index=True)
            pie_fig = px.pie(summary, names='day', values='total_cups', title='Cups Sold by Day', category_orders={'day': day_order})
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("No data available for Productivity by Day based on current filters.")
    else:
        st.warning("Column 'day', 'ridername', or 'cups' is not available.")

def display_grup_2():
    """Menampilkan halaman 'Business Dashboard'."""
    if st.button("⬅️ Back to Menu"):
        set_view('main_menu')
        return

    st.markdown("---")
    st.header("🗓️ Business Dashboard")
    df = st.session_state["main_df"]
    
    st.subheader("Weekly Performance Summary")
    summary_df = df.groupby('week').agg(ksj_revenue=('selling', 'sum'),blitz_revenue=('revenue', 'sum'),active_sellers=('ridername', 'nunique'),total_cups=('cups', 'sum')).reset_index()
    display_summary_df = summary_df.rename(columns={'week': 'Week', 'ksj_revenue': "KSJ's Revenue", 'blitz_revenue': "Blitz's Revenue",'active_sellers': 'Active Sellers', 'total_cups': 'Total Cups'})
    st.dataframe(display_summary_df.style.format({"KSJ's Revenue": "Rp {:,.0f}","Blitz's Revenue": "Rp {:,.0f}"}),use_container_width=True, hide_index=True)
    st.markdown("---")

    st.subheader("Business Summary")
    total_ksj_revenue = df['selling'].sum()
    total_blitz_revenue = df['revenue'].sum()
    total_cups_sold = df['cups'].sum() 
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Client's Revenue (KSJ)", f"Rp {total_ksj_revenue:,.0f}")
    col2.metric("Total Blitz's Revenue", f"Rp {total_blitz_revenue:,.0f}")
    col3.metric("Total Product Sold", f"{total_cups_sold:,}")
    st.markdown("---")

    st.subheader("Business Position")
    if 'date' not in df.columns or 'week' not in df.columns:
        st.warning("Kolom 'date' atau 'week' tidak ditemukan untuk analisis posisi bisnis.")
        return 
    
    latest_month_period = df['date'].dt.to_period('M').max()
    previous_month_period = latest_month_period - 1
    
    cups_latest_month = df[df['date'].dt.to_period('M') == latest_month_period]['cups'].sum()
    cups_previous_month = df[df['date'].dt.to_period('M') == previous_month_period]['cups'].sum()
    cups_latest_month = 0 if pd.isna(cups_latest_month) else cups_latest_month
    cups_previous_month = 0 if pd.isna(cups_previous_month) else cups_previous_month

    revenue_latest_month = df[df['date'].dt.to_period('M') == latest_month_period]['revenue'].sum()
    revenue_previous_month = df[df['date'].dt.to_period('M') == previous_month_period]['revenue'].sum()
    revenue_latest_month = 0 if pd.isna(revenue_latest_month) else revenue_latest_month
    revenue_previous_month = 0 if pd.isna(revenue_previous_month) else revenue_previous_month

    latest_week = df['week'].max()
    previous_week = latest_week - 1
    
    cups_latest_week = df[df['week'] == latest_week]['cups'].sum()
    cups_previous_week = df[df['week'] == previous_week]['cups'].sum()
    cups_latest_week = 0 if pd.isna(cups_latest_week) else cups_latest_week
    cups_previous_week = 0 if pd.isna(cups_previous_week) else cups_previous_week

    revenue_latest_week = df[df['week'] == latest_week]['revenue'].sum()
    revenue_previous_week = df[df['week'] == previous_week]['revenue'].sum()
    revenue_latest_week = 0 if pd.isna(revenue_latest_week) else revenue_latest_week
    revenue_previous_week = 0 if pd.isna(revenue_previous_week) else revenue_previous_week

    delta_cups_month = int(cups_latest_month - cups_previous_month)
    delta_cups_week = int(cups_latest_week - cups_previous_week)
    delta_revenue_month = int(revenue_latest_month - revenue_previous_month)
    delta_revenue_week = int(revenue_latest_week - revenue_previous_week)

    col1, col2 = st.columns(2)
    col1.metric(label=f"Product Sold ({latest_month_period})", value=f"{cups_latest_month:,}", delta=delta_cups_month)
    col2.metric(label=f"Product Sold (Week {latest_week})", value=f"{cups_latest_week:,}", delta=delta_cups_week)
    
    col3, col4 = st.columns(2)
    delta_revenue_month_formatted = f"Rp {delta_revenue_month:,.0f} vs Prv. Month".replace(",", ".")
    color_month = "red" if delta_revenue_month < 0 else "green"
    col3.metric(label=f"Blitz's Revenue ({latest_month_period})", value=f"Rp {revenue_latest_month:,.0f}".replace(",", "."))
    col3.markdown(f"<span style='color:{color_month}; font-size: 0.8em;'>{'⬇️' if delta_revenue_month < 0 else '⬆️'} {delta_revenue_month_formatted}</span>", unsafe_allow_html=True)

    delta_revenue_week_formatted = f"Rp {delta_revenue_week:,.0f} vs Prv. Week".replace(",", ".")
    color_week = "red" if delta_revenue_week < 0 else "green"
    col4.metric(label=f"Blitz's Revenue (Week {latest_week})", value=f"Rp {revenue_latest_week:,.0f}".replace(",", "."))
    col4.markdown(f"<span style='color:{color_week}; font-size: 0.8em;'>{'⬇️' if delta_revenue_week < 0 else '⬆️'} {delta_revenue_week_formatted}</span>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Seller Retention Analysis")
    @st.cache_data
    def calculate_seller_retention(dataf):
        if 'week' not in dataf.columns or 'ridername' not in dataf.columns: return None
        weekly_sellers = dataf.groupby('week')['ridername'].unique().apply(set).sort_index()
        results = []
        if weekly_sellers.empty: return pd.DataFrame()
        first_week_num = weekly_sellers.index[0]
        first_week_sellers = weekly_sellers.iloc[0]
        results.append({"Week": first_week_num, "Total Sellers": len(first_week_sellers), "New Sellers": len(first_week_sellers), "Retained Sellers": 0, "Churned Sellers": 0, "Retention Rate (%)": 0.0})
        for i in range(1, len(weekly_sellers)):
            current_week_num, prev_week_num = weekly_sellers.index[i], weekly_sellers.index[i-1]
            current_sellers, prev_sellers = weekly_sellers.loc[current_week_num], weekly_sellers.loc[prev_week_num]
            retained_sellers_set = current_sellers.intersection(prev_sellers)
            new_sellers_set = current_sellers.difference(prev_sellers)
            churned_sellers_set = prev_sellers.difference(current_sellers)
            retention_rate = (len(retained_sellers_set) / len(prev_sellers)) * 100 if len(prev_sellers) > 0 else 0
            results.append({"Week": current_week_num, "Total Sellers": len(current_sellers), "New Sellers": len(new_sellers_set), "Retained Sellers": len(retained_sellers_set), "Churned Sellers": len(churned_sellers_set), "Retention Rate (%)": retention_rate})
        return pd.DataFrame(results)
    
    retention_df = calculate_seller_retention(df)
    if retention_df is not None and not retention_df.empty:
        st.dataframe(retention_df.style.format({"Retention Rate (%)": "{:.2f}%"}), use_container_width=True, hide_index=True)
        fig_retention = px.line(retention_df, x='Week', y='Retention Rate (%)', title='Seller Retention Rate Over Time', markers=True)
        fig_retention.update_layout(yaxis_ticksuffix="%")
        st.plotly_chart(fig_retention, use_container_width=True)
    else:
        st.warning("Cannot perform retention analysis. Required columns 'week' or 'ridername' not found or no data.")

def display_area_analysis():
    """Menampilkan halaman 'Area & Outlet Analysis'."""
    if st.button("⬅️ Back to Menu"):
        set_view('main_menu')
        return

    st.markdown("---")
    st.header("📍 Area & Outlet Analysis")
    df = st.session_state["main_df"]
    
    area_cols = ['area', 'city', 'district', 'outlet']
    if not all(col in df.columns for col in area_cols):
        st.error("Data 'Area', 'City', 'District', or 'Outlet' not found.")
        return

    st.subheader("Time & Location Filters")
    all_weeks = sorted(df['week'].unique())
    selected_weeks = st.multiselect("Select Week(s) to Analyze", options=all_weeks, default=all_weeks)
    
    if not selected_weeks:
        st.warning("Please select at least one week to continue.")
        return
        
    time_filtered_df = df[df['week'].isin(selected_weeks)]

    filter_cols = st.columns(4)
    with filter_cols[0]:
        area_list = ['All Areas'] + sorted(time_filtered_df['area'].unique().tolist())
        selected_area = st.selectbox("Select Area", area_list)
    df_filtered = time_filtered_df[time_filtered_df['area'] == selected_area] if selected_area != 'All Areas' else time_filtered_df
    with filter_cols[1]:
        city_list = ['All Cities'] + sorted(df_filtered['city'].unique().tolist())
        selected_city = st.selectbox("Select City", city_list)
    df_filtered = df_filtered[df_filtered['city'] == selected_city] if selected_city != 'All Cities' else df_filtered
    with filter_cols[2]:
        district_list = ['All Districts'] + sorted(df_filtered['district'].unique().tolist())
        selected_district = st.selectbox("Select District", district_list)
    df_filtered = df_filtered[df_filtered['district'] == selected_district] if selected_district != 'All Districts' else df_filtered
    with filter_cols[3]:
        outlet_list = ['All Outlets'] + sorted(df_filtered['outlet'].unique().tolist())
        selected_outlet = st.selectbox("Select Outlet", outlet_list)
    df_filtered = df_filtered[df_filtered['outlet'] == selected_outlet] if selected_outlet != 'All Outlets' else df_filtered
    st.markdown("---")
    
    if df_filtered.empty:
        st.warning("No data available for the current filter selection.")
        return

    st.subheader("Performance Summary for Selection")
    total_revenue = df_filtered['revenue'].sum()
    total_cups = df_filtered['cups'].sum()
    active_sellers = df_filtered['ridername'].nunique()
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Total Blitz's Revenue", f"Rp {total_revenue:,.0f}")
    kpi_cols[1].metric("Total Cups Sold", f"{total_cups:,}")
    kpi_cols[2].metric("Total Active Sellers", f"{active_sellers:,}")
    st.markdown("---")
    
    st.subheader("Performance Breakdown")
    chart_labels = {'revenue': 'Revenue', 'area': 'Area', 'city': 'City', 'district': 'District', 'outlet': 'Outlet', 'cups': 'Cups Sold'}
    col_revenue_chart, col_cups_chart = st.columns(2)

    # Logika untuk grafik revenue
    with col_revenue_chart:
        title_revenue = "Revenue Breakdown"
        if selected_outlet != 'All Outlets':
            data_to_plot_revenue = df_filtered.groupby('week')['revenue'].sum().reset_index()
            fig_revenue = px.line(data_to_plot_revenue, x='week', y='revenue', title=f"Weekly Revenue for {selected_outlet}", markers=True, labels=chart_labels)
        elif selected_district != 'All Districts':
            data_to_plot_revenue = df_filtered.groupby('outlet')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
            fig_revenue = px.bar(data_to_plot_revenue, x='outlet', y='revenue', title=f"Revenue by Outlet in {selected_district}", labels=chart_labels)
        elif selected_city != 'All Cities':
            data_to_plot_revenue = df_filtered.groupby('district')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
            fig_revenue = px.bar(data_to_plot_revenue, x='district', y='revenue', title=f"Revenue by District in {selected_city}", labels=chart_labels)
        elif selected_area != 'All Areas':
            data_to_plot_revenue = df_filtered.groupby('city')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
            fig_revenue = px.bar(data_to_plot_revenue, x='city', y='revenue', title=f"Revenue by City in {selected_area}", labels=chart_labels)
        else: # All Areas
            data_to_plot_revenue = df_filtered.groupby('area')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
            fig_revenue = px.bar(data_to_plot_revenue, x='area', y='revenue', title="Overall Revenue by Area", labels=chart_labels)
        st.plotly_chart(fig_revenue, use_container_width=True)

    # Logika untuk grafik cups
    with col_cups_chart:
        title_cups = "Cups Sold Breakdown"
        if selected_outlet != 'All Outlets':
            data_to_plot_cups = df_filtered.groupby('week')['cups'].sum().reset_index()
            fig_cups = px.line(data_to_plot_cups, x='week', y='cups', title=f"Weekly Cups Sold for {selected_outlet}", markers=True, labels=chart_labels)
        elif selected_district != 'All Districts':
            data_to_plot_cups = df_filtered.groupby('outlet')['cups'].sum().reset_index().sort_values('cups', ascending=False)
            fig_cups = px.bar(data_to_plot_cups, x='outlet', y='cups', title=f"Cups Sold by Outlet in {selected_district}", labels=chart_labels)
        elif selected_city != 'All Cities':
            data_to_plot_cups = df_filtered.groupby('district')['cups'].sum().reset_index().sort_values('cups', ascending=False)
            fig_cups = px.bar(data_to_plot_cups, x='district', y='cups', title=f"Cups Sold by District in {selected_city}", labels=chart_labels)
        elif selected_area != 'All Areas':
            data_to_plot_cups = df_filtered.groupby('city')['cups'].sum().reset_index().sort_values('cups', ascending=False)
            fig_cups = px.bar(data_to_plot_cups, x='city', y='cups', title=f"Cups Sold by City in {selected_area}", labels=chart_labels)
        else: # All Areas
            data_to_plot_cups = df_filtered.groupby('area')['cups'].sum().reset_index().sort_values('cups', ascending=False)
            fig_cups = px.bar(data_to_plot_cups, x='area', y='cups', title="Overall Cups Sold by Area", labels=chart_labels)
        st.plotly_chart(fig_cups, use_container_width=True)
    
    st.markdown("---")
    st.subheader("District Productivity Ranking")
    district_summary = df_filtered.groupby('district').agg(total_revenue=('revenue', 'sum'),total_cups=('cups', 'sum')).reset_index()
    if not district_summary.empty:
        rank_labels={'total_revenue': 'Total Revenue', 'total_cups': 'Total Cups', 'district': 'District'}
        st.markdown("#### Highest Productivity Districts")
        col_high_rev, col_high_cups = st.columns(2)
        with col_high_rev:
            top_10_revenue = district_summary.nlargest(10, 'total_revenue')
            fig_top_rev = px.bar(top_10_revenue.sort_values('total_revenue'), x='total_revenue', y='district', orientation='h', title="Top 10 by Blitz's Revenue", text_auto=True, labels=rank_labels)
            fig_top_rev.update_traces(texttemplate='Rp%{x:,.0f}')
            st.plotly_chart(fig_top_rev, use_container_width=True)
        with col_high_cups:
            top_10_cups = district_summary.nlargest(10, 'total_cups')
            fig_top_cups = px.bar(top_10_cups.sort_values('total_cups'), x='total_cups', y='district', orientation='h', title="Top 10 by Product Sold", text_auto=True, labels=rank_labels)
            st.plotly_chart(fig_top_cups, use_container_width=True)
            
        st.markdown("#### Lowest Productivity Districts")
        col_low_rev, col_low_cups = st.columns(2)
        with col_low_rev:
            bottom_10_revenue = district_summary.nsmallest(10, 'total_revenue')
            fig_low_rev = px.bar(bottom_10_revenue.sort_values('total_revenue', ascending=False), x='total_revenue', y='district', orientation='h', title="Bottom 10 by Blitz's Revenue", text_auto=True, labels=rank_labels)
            fig_low_rev.update_traces(texttemplate='Rp%{x:,.0f}')
            st.plotly_chart(fig_low_rev, use_container_width=True)
        with col_low_cups:
            bottom_10_cups = district_summary.nsmallest(10, 'total_cups')
            fig_low_cups = px.bar(bottom_10_cups.sort_values('total_cups', ascending=False), x='total_cups', y='district', orientation='h', title="Bottom 10 by Product Sold", text_auto=True, labels=rank_labels)
            st.plotly_chart(fig_low_cups, use_container_width=True)

    st.markdown("---")
    st.subheader("Outlet Supply vs. Demand Analysis")
    daily_sales = df_filtered.groupby(['outlet', 'date'])['cups'].sum().reset_index()
    demand_summary = daily_sales.groupby('outlet')['cups'].mean().reset_index().rename(columns={'cups': 'demand'})
    supply_summary = df_filtered.groupby('outlet')['ridername'].nunique().reset_index().rename(columns={'ridername': 'supply'})
    supply_demand_summary = pd.merge(demand_summary, supply_summary, on='outlet', how='left').fillna(0)
    supply_demand_summary['ratio'] = supply_demand_summary.apply(lambda row: row['supply'] / row['demand'] if row['demand'] > 0 else 0, axis=1)
    supply_demand_summary['status'] = supply_demand_summary['ratio'].apply(lambda x: "Productive" if (x > 0 and x <= 0.0222) else "Not Productive")

    status_options = ["Productive", "Not Productive"]
    selected_statuses = st.multiselect("Filter by Status", options=status_options, default=status_options)

    final_sd_df = supply_demand_summary[supply_demand_summary['status'].isin(selected_statuses)] if selected_statuses else supply_demand_summary

    if not final_sd_df.empty:
        st.dataframe(
            final_sd_df.rename(columns={'outlet': 'Outlet', 'demand': 'Avg Daily Demand', 'supply': 'Sellers', 'ratio': 'Ratio', 'status': 'Status'}),
            use_container_width=True, hide_index=True,
            column_config={"Avg Daily Demand": st.column_config.NumberColumn(format="%.2f"), "Ratio": st.column_config.NumberColumn(format="%.4f")}
        )
        fig_sd = px.scatter(
            final_sd_df, x="demand", y="supply", hover_name="outlet", color="status",
            title="Supply (Sellers) vs. Average Daily Demand (Cups) per Outlet",
            labels={"demand": "Average Daily Demand (Cups)", "supply": "Total Unique Sellers", "status": "Status"}
        )
        max_val = max(final_sd_df['demand'].max(), final_sd_df['supply'].max()) if not final_sd_df.empty else 1
        fig_sd.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color='Gray', dash='dash'))
        st.plotly_chart(fig_sd, use_container_width=True)
    else:
        st.info("No supply/demand data available for the current filter selection.")


# --- MAIN APPLICATION FLOW ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Ganti "BLITZ LOGO.png" dengan path yang benar jika logo tidak ada di folder yang sama
        try:
            st.image("BLITZ LOGO.png", width=150)
        except FileNotFoundError:
            st.warning("Logo file not found. Please place 'BLITZ LOGO.png' in the same folder.")
        st.title("Login - KSJ Data Dashboard")
        username_input = st.text_input("Username", placeholder="Enter your username")
        password_input = st.text_input("Password", type="password", placeholder="Enter your password")
        if st.button("Login", use_container_width=True):
            check_login(username_input, password_input)
else:
    if "main_df" not in st.session_state:
        with st.spinner("Processing Your Data. Please wait... :)"):
            st.session_state["main_df"] = load_and_process_main_data()
            
    col1, col2 = st.columns([1, 8])
    with col1:
        try:
            st.image("BLITZ LOGO.png", width=80)
        except FileNotFoundError:
            pass # Jangan tampilkan warning lagi jika sudah login
    with col2:
        st.title("📊 KSJ Data 2025")
        
    st.sidebar.title(f"Welcome, {st.session_state.get('username', 'User')}!")
    if st.sidebar.button("Logout"):
        logout()
    st.sidebar.markdown("---")
    
    current_view = st.session_state.get("view", "main_menu")

    if current_view == "main_menu":
        display_main_menu()
    elif current_view == "grup_1":
        display_grup_1()
    elif current_view == "grup_2":
        display_grup_2()
    elif current_view == "area_analysis":
        display_area_analysis()
