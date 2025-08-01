import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import gc 

# --- INITIAL SETUP & FUNCTIONS ---
st.set_page_config(
    page_title="KSJ Data 2025", 
    layout="wide",
    page_icon="rideblitz_logo.jpeg")

# --- UTILITY FUNCTIONS ---
def to_excel(df: pd.DataFrame):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def check_login(username, password):
    # Akses tabel 'users' dari secrets
    # st.secrets.users akan mengembalikan dictionary
    all_users = st.secrets["users"]

    # Username checking, pastikan sesuai dengan users di secrets
    if username in all_users:
        # Password checking
        if all_users[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            return True # Login berhasil
    
    # Jika username tidak ada atau password salah
    st.session_state["logged_in"] = False
    st.session_state.pop("username", None)
    st.error("Incorrect Username or Password!")
    return False # Login gagal

def logout():
    for key in list(st.session_state.keys()):
        st.session_state.pop(key)
    st.session_state["logged_in"] = False
    gc.collect() 
    st.rerun()

def set_view(view_name):
    st.session_state["view"] = view_name
    st.session_state.page_number = 0 
    st.rerun()

# --- DATA LOADING ---
@st.cache_data
def load_and_process_main_data():
    try:
        df = pd.read_parquet("KSJ_Data_2025.parquet")
    except FileNotFoundError:
        df = pd.read_excel("KSJ Data 2025.xlsx")

    df.columns = [col.strip().replace(" ", "").replace("#", "").lower() for col in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
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
    st.header("Main Menu")
    st.markdown("Select an analysis to view from the options below.")
    
    # Mengubah layout kembali menjadi 3 kolom per baris
    row1_col1, row1_col2, row1_col3 = st.columns(3, gap="large")

    with row1_col1:
        with st.container(border=True):
            st.subheader("📊 All Data")
            st.markdown("View all raw data with interactive filters, plus weekly productivity graphs and daily sales analysis.")
            if st.button("Open Module", key="grup1_button", use_container_width=True):
                # Logika untuk inisialisasi filter grup 1
                df = st.session_state["main_df"]
                st.session_state.grup1_filter_options = {}
                for col in df.columns:
                    if df[col].dtype != 'datetime64[ns]' and len(df[col].dropna().unique()) < 200:
                        st.session_state.grup1_filter_options[col] = sorted(df[col].dropna().unique())
                st.session_state.grup1_selections = {col: options for col, options in st.session_state.grup1_filter_options.items()}
                set_view('grup_1')
    
    with row1_col2:
        with st.container(border=True):
            st.subheader("🗓️ Business Dashboard")
            st.markdown("View a consolidated dashboard of business performance, position, and seller retention.")
            if st.button("Open Module", key="grup2_button", use_container_width=True):
                set_view('grup_2')

    with row1_col3:
        with st.container(border=True):
            st.subheader("📍 Area & Outlet Analysis")
            st.markdown("Drill down into area-specific performance, with detailed location and outlet-level breakdowns.")
            if st.button("Open Module", key="area_button", use_container_width=True):
                set_view('area_analysis')

    # Baris kedua untuk kartu selanjutnya
    row2_col1, row2_col2, row2_col3 = st.columns(3, gap="large")

    with row2_col1:
        with st.container(border=True):
            st.subheader("💸 Payroll Management")
            st.markdown("Manage and analyze seller payroll, deductions, and incentives (Under Development).")
            if st.button("Open Module", key="payroll_button", use_container_width=True):
                set_view('payroll_management')

    with row2_col2:
        with st.container(border=True):
            st.subheader("🛵 EV Management")
            st.markdown("Monitor and manage Electric Vehicle assets and revenue from EV Deduction (Under Development).")
            if st.button("Open Module", key="ev_button", use_container_width=True):
                set_view('ev_management')


def display_grup_1():
    if st.button("⬅️ Back to Menu"):
        set_view('main_menu')
        return

    st.markdown("---")
    df = st.session_state["main_df"]
    st.subheader("📄 All Data")

    with st.expander("🔎 Filter Data", expanded=True):
        with st.form(key='filter_form'):
            filter_cols = st.columns(3)
            col_idx = 0
            
            temp_selections = {}
            for col, options in st.session_state.grup1_filter_options.items():
                with filter_cols[col_idx]:
                    temp_selections[col] = st.multiselect(
                        f"Select {col.title()}",
                        options=options,
                        default=st.session_state.grup1_selections.get(col, [])
                    )
                col_idx = (col_idx + 1) % 3
            
            submitted = st.form_submit_button('Apply Filters', use_container_width=True)
            
            if submitted:
                st.session_state.grup1_selections = temp_selections
                st.session_state.page_number = 0
                st.rerun()

    filtered_df = df
    for col, selected_values in st.session_state.grup1_selections.items():
        if selected_values and col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

    if not filtered_df.empty:
        PAGE_SIZE = 1000 
        if 'page_number' not in st.session_state:
            st.session_state.page_number = 0
        
        total_rows = len(filtered_df)
        total_pages = max(1, (total_rows // PAGE_SIZE) + (1 if total_rows % PAGE_SIZE > 0 else 0))
        
        if st.session_state.page_number >= total_pages:
            st.session_state.page_number = 0

        prev_col, mid_col, next_col = st.columns([2, 3, 2])
        with prev_col:
            if st.button("⬅️ Previous Page", use_container_width=True, disabled=(st.session_state.page_number == 0)):
                st.session_state.page_number -= 1
                st.rerun()
        with mid_col:
            st.markdown(f"<div style='text-align: center; margin-top: 1rem;'>Page {st.session_state.page_number + 1} of {total_pages} ({total_rows} total rows)</div>", unsafe_allow_html=True)
        with next_col:
            if st.button("Next Page ➡️", use_container_width=True, disabled=(st.session_state.page_number >= total_pages - 1)):
                st.session_state.page_number += 1
                st.rerun()

        st.download_button(
             label="📥 Export All Data to Excel", data=to_excel(filtered_df), 
             file_name=f"all_data_export_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
             use_container_width=True
         )

        start_index = st.session_state.page_number * PAGE_SIZE
        end_index = start_index + PAGE_SIZE
        df_to_display = filtered_df.iloc[start_index:end_index]
        
        styled_df = df_to_display.copy()
        if 'date' in styled_df.columns:
            styled_df['date'] = styled_df['date'].dt.strftime('%d/%m/%Y')
        for col in ['selling', 'revenue']:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(lambda x: f"Rp {x:,.0f}".replace(",", ".") if pd.notnull(x) else "-")
        styled_df.columns = [col.title() for col in styled_df.columns]
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No data to display for the selected filters.")
    
    if not filtered_df.empty:
        agg_data = filtered_df.groupby(['week', 'day', 'date']).agg(
            total_cups=('cups', 'sum'),
            total_revenue=('revenue', 'sum'),
            unique_riders=('ridername', 'nunique')
        ).reset_index()

        st.subheader("📈 Week-on-Week Productivity")
        weekly_summary = agg_data.groupby('week').agg(cups=('total_cups', 'sum'), revenue=('total_revenue', 'sum')).reset_index()
        col1, col2 = st.columns(2)
        with col1:
            fig_cups = px.bar(weekly_summary, x='week', y='cups', title='Total Cups per Week', labels={'cups': 'Cups Sold', 'week': 'Week'}, text_auto=True)
            st.plotly_chart(fig_cups, use_container_width=True)
        with col2:
            fig_revenue = px.bar(weekly_summary, x='week', y='revenue', title='Total Revenue per Week', labels={'revenue': 'Total Revenue (Rp)', 'week': 'Week'}, text_auto=True)
            fig_revenue.update_traces(texttemplate='Rp%{y:,.0f}', textposition='outside')
            fig_revenue.update_yaxes(title_text='Total Revenue (Rp)')
            st.plotly_chart(fig_revenue, use_container_width=True)

        st.subheader("📅 Productivity by Day")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        total_cups_day = agg_data.groupby('day')['total_cups'].sum().reset_index()
        avg_sellers_day = agg_data.groupby('day')['unique_riders'].mean().round(0).astype(int).reset_index(name='avg_sellers')
        agg_data['productivity'] = agg_data.apply(lambda row: row['total_cups'] / row['unique_riders'] if row['unique_riders'] > 0 else 0, axis=1)
        avg_productivity_day = agg_data.groupby('day')['productivity'].mean().round(0).astype(int).reset_index(name='avg_productivity')
        day_summary = pd.merge(total_cups_day, avg_sellers_day, on='day').merge(avg_productivity_day, on='day')
        day_summary['day'] = pd.Categorical(day_summary['day'], categories=day_order, ordered=True)
        day_summary = day_summary.sort_values('day')
        st.dataframe(day_summary.rename(columns={'day': 'Day','total_cups': 'Total Cups','avg_sellers': 'Avg. Sellers','avg_productivity': 'Avg. Cups Sold Per Day'}), use_container_width=True, hide_index=True)
        pie_fig = px.pie(day_summary, names='day', values='total_cups', title='Cups Sold by Day', category_orders={'day': day_order})
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.subheader("📈 Week-on-Week Productivity")
        st.info("No data available for Week-on-Week Productivity based on current filters.")
        st.subheader("📅 Productivity by Day")
        st.info("No data available for Productivity by Day based on current filters.")

def display_grup_2():
    if st.button("⬅️ Back to Menu"):
        set_view('main_menu')
        return
    st.markdown("---")
    st.header("🗓️ Business Dashboard")
    df = st.session_state["main_df"]

    # --- PENAMBAHAN FILTER GEOGRAFIS BERJENJANG ---
    st.subheader("Geographic Filters")
    
    filter_cols = st.columns(3)
    with filter_cols[0]:
        area_list = ['All Areas'] + sorted(df['area'].unique().tolist())
        selected_area = st.selectbox("Select Area", area_list, key="grup2_area")
    
    # Filter data berdasarkan pilihan area
    df_filtered_area = df[df['area'] == selected_area] if selected_area != 'All Areas' else df

    with filter_cols[1]:
        city_list = ['All Cities'] + sorted(df_filtered_area['city'].unique().tolist())
        selected_city = st.selectbox("Select City", city_list, key="grup2_city")

    # Filter data turunan
    df_filtered_city = df_filtered_area[df_filtered_area['city'] == selected_city] if selected_city != 'All Cities' else df_filtered_area
    
    with filter_cols[2]:
        district_list = ['All Districts'] + sorted(df_filtered_city['district'].unique().tolist())
        selected_district = st.selectbox("Select District", district_list, key="grup2_district")

    # DataFrame final yang akan digunakan
    filtered_df = df_filtered_city[df_filtered_city['district'] == selected_district] if selected_district != 'All Districts' else df_filtered_city
    
    st.markdown("---")
    
    if filtered_df.empty:
        st.warning("No data available for the current filter selection.")
        return

    # --- SEMUA MODUL DI BAWAH INI MENGGUNAKAN 'filtered_df' ---
    
    st.subheader("Weekly Performance Summary")
    summary_df = filtered_df.groupby('week').agg(ksj_revenue=('selling', 'sum'),blitz_revenue=('revenue', 'sum'),active_sellers=('ridername', 'nunique'),total_cups=('cups', 'sum')).reset_index()
    display_summary_df = summary_df.rename(columns={'week': 'Week', 'ksj_revenue': "KSJ's Revenue", 'blitz_revenue': "Blitz's Revenue",'active_sellers': 'Active Sellers', 'total_cups': 'Total Cups'})
    st.dataframe(display_summary_df.style.format({"KSJ's Revenue": "Rp {:,.0f}","Blitz's Revenue": "Rp {:,.0f}"}),use_container_width=True, hide_index=True)
    st.download_button(
        label="📥 Export Weekly Performance to Excel",
        data=to_excel(display_summary_df),
        file_name="weekly_performance_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.markdown("---")

    st.subheader("Business Summary")
    total_ksj_revenue = filtered_df['selling'].sum()
    total_blitz_revenue = filtered_df['revenue'].sum()
    total_cups_sold = filtered_df['cups'].sum() 
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Client's Revenue (KSJ)", f"Rp {total_ksj_revenue:,.0f}")
    col2.metric("Total Blitz's Revenue", f"Rp {total_blitz_revenue:,.0f}")
    col3.metric("Total Product Sold", f"{total_cups_sold:,}")
    st.markdown("---")

    st.subheader("Business Position")
    # Kalkulasi sekarang berdasarkan filtered_df
    latest_month_period = filtered_df['date'].dt.to_period('M').max()
    previous_month_period = latest_month_period - 1
    cups_latest_month = filtered_df[filtered_df['date'].dt.to_period('M') == latest_month_period]['cups'].sum()
    cups_previous_month = filtered_df[filtered_df['date'].dt.to_period('M') == previous_month_period]['cups'].sum()
    cups_latest_month = 0 if pd.isna(cups_latest_month) else cups_latest_month
    cups_previous_month = 0 if pd.isna(cups_previous_month) else cups_previous_month
    revenue_latest_month = filtered_df[filtered_df['date'].dt.to_period('M') == latest_month_period]['revenue'].sum()
    revenue_previous_month = filtered_df[filtered_df['date'].dt.to_period('M') == previous_month_period]['revenue'].sum()
    revenue_latest_month = 0 if pd.isna(revenue_latest_month) else revenue_latest_month
    revenue_previous_month = 0 if pd.isna(revenue_previous_month) else revenue_previous_month
    latest_week = filtered_df['week'].max()
    previous_week = latest_week - 1
    cups_latest_week = filtered_df[filtered_df['week'] == latest_week]['cups'].sum()
    cups_previous_week = filtered_df[filtered_df['week'] == previous_week]['cups'].sum()
    cups_latest_week = 0 if pd.isna(cups_latest_week) else cups_latest_week
    cups_previous_week = 0 if pd.isna(cups_previous_week) else cups_previous_week
    revenue_latest_week = filtered_df[filtered_df['week'] == latest_week]['revenue'].sum()
    revenue_previous_week = filtered_df[filtered_df['week'] == previous_week]['revenue'].sum()
    revenue_latest_week = 0 if pd.isna(revenue_latest_week) else revenue_latest_week
    revenue_previous_week = 0 if pd.isna(revenue_previous_week) else revenue_previous_week
    delta_cups_month = int(cups_latest_month - cups_previous_month)
    delta_cups_week = int(cups_latest_week - cups_previous_week)
    delta_revenue_month = int(revenue_latest_month - revenue_previous_month)
    delta_revenue_week = int(revenue_latest_week - revenue_previous_week)
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col1.metric(label=f"Product Sold ({latest_month_period})", value=f"{cups_latest_month:,}", delta=delta_cups_month)
    col2.metric(label=f"Product Sold (Week {latest_week})", value=f"{cups_latest_week:,}", delta=delta_cups_week)
    col3.metric(label=f"Blitz's Revenue ({latest_month_period})", value=f"Rp {revenue_latest_month:,.0f}", delta=delta_revenue_month)
    col4.metric(label=f"Blitz's Revenue (Week {latest_week})", value=f"Rp {revenue_latest_week:,.0f}", delta=delta_revenue_week)
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
    
    # Analisis retensi sekarang berdasarkan filtered_df
    retention_df = calculate_seller_retention(filtered_df)
    if retention_df is not None and not retention_df.empty:
        st.dataframe(retention_df.style.format({"Retention Rate (%)": "{:.2f}%"}), use_container_width=True, hide_index=True)
        st.download_button(
            label="📥 Export Seller Retention to Excel",
            data=to_excel(retention_df),
            file_name="seller_retention_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        fig_retention = px.line(retention_df, x='Week', y='Retention Rate (%)', title='Seller Retention Rate Over Time', markers=True)
        fig_retention.update_layout(yaxis_ticksuffix="%")
        st.plotly_chart(fig_retention, use_container_width=True)
    else:
        st.warning("No seller retention data to display for the current selection.")

def display_area_analysis():
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
    if 'area_week_selection' not in st.session_state:
        st.session_state.area_week_selection = all_weeks
    def select_all_weeks_area():
        st.session_state.area_week_selection = all_weeks
    def unselect_all_weeks_area():
        st.session_state.area_week_selection = []
    c1, c2, c3 = st.columns([4, 1, 1])
    with c1:
        selected_weeks = st.multiselect("Select Week(s) to Analyze",options=all_weeks,default=st.session_state.area_week_selection,key="area_week_multiselect")
        st.session_state.area_week_selection = selected_weeks
    with c2:
        st.button("Select All", on_click=select_all_weeks_area, use_container_width=True)
    with c3:
        st.button("Unselect All", on_click=unselect_all_weeks_area, use_container_width=True)
    
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

    with col_revenue_chart:
        if selected_outlet != 'All Outlets':
            data_to_plot = df_filtered.groupby('week')['revenue'].sum().reset_index()
            fig = px.line(data_to_plot, x='week', y='revenue', title=f"Weekly Revenue for {selected_outlet}", markers=True, labels=chart_labels)
            fig.update_traces(texttemplate='Rp%{y:,.0f}', textposition='top_center')
        elif selected_district != 'All Districts':
            data_to_plot = df_filtered.groupby('outlet')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
            fig = px.bar(data_to_plot, x='outlet', y='revenue', title=f"Revenue by Outlet in {selected_district}", labels=chart_labels, text_auto=True)
            fig.update_traces(texttemplate='Rp%{y:,.0f}')
        elif selected_city != 'All Cities':
            data_to_plot = df_filtered.groupby('district')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
            fig = px.bar(data_to_plot, x='district', y='revenue', title=f"Revenue by District in {selected_city}", labels=chart_labels, text_auto=True)
            fig.update_traces(texttemplate='Rp%{y:,.0f}')
        elif selected_area != 'All Areas':
            data_to_plot = df_filtered.groupby('city')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
            fig = px.bar(data_to_plot, x='city', y='revenue', title=f"Revenue by City in {selected_area}", labels=chart_labels, text_auto=True)
            fig.update_traces(texttemplate='Rp%{y:,.0f}')
        else:
            data_to_plot = df_filtered.groupby('area')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
            fig = px.bar(data_to_plot, x='area', y='revenue', title="Overall Revenue by Area", labels=chart_labels, text_auto=True)
            fig.update_traces(texttemplate='Rp%{y:,.0f}')
        st.plotly_chart(fig, use_container_width=True)

    with col_cups_chart:
        if selected_outlet != 'All Outlets':
            data_to_plot = df_filtered.groupby('week')['cups'].sum().reset_index()
            fig = px.line(data_to_plot, x='week', y='cups', title=f"Weekly Cups Sold for {selected_outlet}", markers=True, labels=chart_labels)
        elif selected_district != 'All Districts':
            data_to_plot = df_filtered.groupby('outlet')['cups'].sum().reset_index().sort_values('cups', ascending=False)
            fig = px.bar(data_to_plot, x='outlet', y='cups', title=f"Cups Sold by Outlet in {selected_district}", labels=chart_labels)
        elif selected_city != 'All Cities':
            data_to_plot = df_filtered.groupby('district')['cups'].sum().reset_index().sort_values('cups', ascending=False)
            fig = px.bar(data_to_plot, x='district', y='cups', title=f"Cups Sold by District in {selected_city}", labels=chart_labels)
        elif selected_area != 'All Areas':
            data_to_plot = df_filtered.groupby('city')['cups'].sum().reset_index().sort_values('cups', ascending=False)
            fig = px.bar(data_to_plot, x='city', y='cups', title=f"Cups Sold by City in {selected_area}", labels=chart_labels)
        else:
            data_to_plot = df_filtered.groupby('area')['cups'].sum().reset_index().sort_values('cups', ascending=False)
            fig = px.bar(data_to_plot, x='area', y='cups', title="Overall Cups Sold by Area", labels=chart_labels)
        st.plotly_chart(fig, use_container_width=True)
    
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
    if not df_filtered.empty:
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
            st.dataframe(final_sd_df.rename(columns={'outlet': 'Outlet', 'demand': 'Avg Daily Demand', 'supply': 'Sellers', 'ratio': 'Ratio', 'status': 'Status'}), use_container_width=True, hide_index=True, column_config={"Avg Daily Demand": st.column_config.NumberColumn(format="%.2f"), "Ratio": st.column_config.NumberColumn(format="%.4f")})
            st.download_button(label="📥 Export Supply vs Demand to Excel", data=to_excel(final_sd_df), file_name="outlet_supply_demand_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            fig_sd = px.scatter(final_sd_df, x="demand", y="supply", hover_name="outlet", color="status", title="Supply (Sellers) vs. Average Daily Demand (Cups) per Outlet", labels={"demand": "Average Daily Demand (Cups)", "supply": "Total Unique Sellers", "status": "Status"})
            fig_sd.update_yaxes(range=[0, 150])
            max_val = max(final_sd_df['demand'].max(), final_sd_df['supply'].max()) if not final_sd_df.empty else 1
            fig_sd.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color='Gray', dash='dash'))
            st.plotly_chart(fig_sd, use_container_width=True)
        else:
            st.info("No supply/demand data available for the current filter selection.")

# ===================================================================
# --- BLOK KODE PAYROLL
# ===================================================================

# --- KALKULASI PAYROLL ---

def calculate_selling_incentive_v2(total_selling):
    """(BAGIAN A - NASIONAL) Menghitung insentif penjualan berdasarkan total selling mingguan."""
    if 1008000 <= total_selling <= 1440000: multiplier = 0.05
    elif 1440001 <= total_selling <= 1920000: multiplier = 0.08
    elif 1920001 <= total_selling <= 2400000: multiplier = 0.09
    elif 2400001 <= total_selling <= 2640000: multiplier = 0.10
    elif 2640001 <= total_selling <= 2880000: multiplier = 0.11
    elif 2880001 <= total_selling <= 3120000: multiplier = 0.1125
    elif 3120001 <= total_selling <= 3360000: multiplier = 0.115
    elif 3360001 <= total_selling <= 6240000: multiplier = 0.14
    elif 6240001 <= total_selling <= 7200000: multiplier = 0.1425
    elif 7200001 <= total_selling <= 8400000: multiplier = 0.145
    elif total_selling >= 8400001: multiplier = 0.1475
    else: multiplier = 0.0
    return total_selling * multiplier

def calculate_attendance_incentive_jkt_bdg(total_selling, active_days):
    """(BAGIAN B - JKT/BDG) Menghitung insentif kehadiran berdasarkan total selling mingguan."""
    daily_rate = 0
    if 0 <= total_selling <= 1440000: daily_rate = 25000
    elif 1440001 <= total_selling <= 1920000: daily_rate = 30000
    elif 1920001 <= total_selling <= 2400000: daily_rate = 42500
    elif 2400001 <= total_selling <= 3120000: daily_rate = 45000
    elif 3120001 <= total_selling <= 4320000: daily_rate = 50000
    elif 4320001 <= total_selling <= 6240000: daily_rate = 55000
    elif total_selling >= 6240001: daily_rate = 60000
    return daily_rate * active_days

def get_daily_incentive_sby_smg(daily_selling):
    """(Helper untuk SBY/SMG) Menentukan tarif insentif harian berdasarkan selling harian."""
    if 0 <= daily_selling <= 240000: return 25000
    elif 240001 <= daily_selling <= 320000: return 30000
    elif 320001 <= daily_selling <= 400000: return 42500
    elif 400001 <= daily_selling <= 520000: return 45000
    elif 520001 <= daily_selling <= 720000: return 50000
    elif 720001 <= daily_selling <= 1040000: return 55000
    elif daily_selling >= 1040001: return 60000
    return 0

# --- PAYROLL MANAGEMENT ---

def display_payroll_management():
    # State untuk mengontrol tampilan antara tabel utama dan detail slip gaji
    if 'viewing_payslip_data' not in st.session_state:
        st.session_state.viewing_payslip_data = None
        st.session_state.daily_details_for_payslip = None

    # DETAIL SLIP GAJI
    if st.session_state.viewing_payslip_data is not None:
        payslip_data = st.session_state.viewing_payslip_data
        daily_details_df = st.session_state.daily_details_for_payslip

        if st.button("⬅️ Back to Payroll Summary"):
            st.session_state.viewing_payslip_data = None
            st.session_state.daily_details_for_payslip = None
            st.rerun()
        
        st.markdown("---")
        # Layout Slip Gaji
        col1, col2 = st.columns([5, 1])
        with col1: st.subheader("WEEKLY PAYSLIP")
        with col2:
            try: st.image("BLITZ LOGO.png", width=60)
            except: pass
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("RIDER NAME", payslip_data['Rider Name'])
        c2.metric("WEEK PERIOD", f"Week {payslip_data['Week']}")
        c3.metric("AREA", payslip_data['Area'])
        st.markdown("---")
        st.markdown(f"#### Total Fee: **Rp {payslip_data['Accumulated Fee']:,}**")
        sc1, sc2 = st.columns(2)
        sc1.metric("Total Selling (Client's Revenue)", f"Rp {payslip_data['Total Selling']:,}")
        sc2.metric("Total Active Days", f"{payslip_data['Active Days']} Day(s)")
        st.markdown("---")
        st.subheader("Fee Breakdown")
        st.markdown(f"**Weekly Selling Incentive:** `Rp {payslip_data['Selling Incentive']:,}`")
        st.markdown(f"**Weekly Attendance Incentive:** `Rp {payslip_data['Attendance Incentive']:,}`")
        
        if daily_details_df is not None and not daily_details_df.empty:
            st.markdown("###### Daily Fee Breakdown:")
            # Format baru untuk tabel rincian
            st.dataframe(daily_details_df.style.format({
                "Daily Selling": "Rp {:,.0f}", 
                "Daily Selling Incentive": "Rp {:,.0f}",
                "Daily Attendance Incentive": "Rp {:,.0f}"
            }), use_container_width=True, hide_index=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        print_button_html = """<button onclick="window.print()" style="padding:10px 20px; font-size:16px; cursor:pointer; border-radius:8px; border:none; background-color:#4CAF50; color:white;">🖨️ PRINT</button>"""
        st.components.v1.html(print_button_html, height=50)

    # TABEL RINGKASAN PAYROLL
    else:
        if st.button("⬅️ Back to Menu"):
            set_view('main_menu')
            return
        st.markdown("---")
        st.header("💸 Payroll Management")
        st.subheader("Weekly Payroll")
        df = st.session_state["main_df"]
        
        selected_week = st.selectbox("Select a Week to Calculate Payroll", options=sorted(df['week'].unique()), index=None, placeholder="Choose a week")

        if selected_week:
            time_filtered_df = df[df['week'] == selected_week].copy()
            st.markdown("---")
            st.subheader("Geographic Filters (Optional)")
            
            # ... (Blok filter) ...
            area_options = sorted(time_filtered_df['area'].unique())
            selected_areas = st.multiselect("Select Area(s)", options=area_options, placeholder="Leave empty to select all")
            area_filtered_df = time_filtered_df[time_filtered_df['area'].isin(selected_areas)] if selected_areas else time_filtered_df
            city_options = sorted(area_filtered_df['city'].unique())
            selected_cities = st.multiselect("Select City/Cities", options=city_options, placeholder="Leave empty to select all")
            city_filtered_df = area_filtered_df[area_filtered_df['city'].isin(selected_cities)] if selected_cities else area_filtered_df
            district_options = sorted(city_filtered_df['district'].unique())
            selected_districts = st.multiselect("Select District(s)", options=district_options, placeholder="Leave empty to select all")
            final_filtered_df = city_filtered_df[city_filtered_df['district'].isin(selected_districts)] if selected_districts else city_filtered_df
            
            if final_filtered_df.empty:
                st.warning("No sales data found for the selected week and location filters.")
                return

            # ... (Blok perhitungan payroll) ...
            payroll_summary = final_filtered_df.groupby(['ridername', 'area']).agg(total_cups_sold=('cups', 'sum'), total_selling=('selling', 'sum'), active_days=('date', 'nunique')).reset_index()
            payroll_summary['selling_incentive'] = payroll_summary['total_selling'].apply(calculate_selling_incentive_v2)
            payroll_summary['attendance_incentive'] = 0.0
            jkt_bdg_mask = payroll_summary['area'].isin(['Jakarta', 'Bandung'])
            if jkt_bdg_mask.any():
                payroll_summary.loc[jkt_bdg_mask, 'attendance_incentive'] = payroll_summary[jkt_bdg_mask].apply(
                    lambda row: calculate_attendance_incentive_jkt_bdg(row['total_selling'], row['active_days']), axis=1)
            sby_smg_mask = payroll_summary['area'].isin(['Surabaya', 'Semarang'])
            if sby_smg_mask.any():
                for index, rider_summary_row in payroll_summary[sby_smg_mask].iterrows():
                    rider_name = rider_summary_row['ridername']
                    rider_area = rider_summary_row['area']
                    rider_transactions = final_filtered_df[(final_filtered_df['ridername'] == rider_name) & (final_filtered_df['area'] == rider_area)]
                    daily_sales = rider_transactions.groupby(rider_transactions['date'].dt.date)['selling'].sum()
                    total_weekly_incentive = sum(daily_sales.apply(get_daily_incentive_sby_smg))
                    payroll_summary.loc[index, 'attendance_incentive'] = total_weekly_incentive
            payroll_summary['accumulated_fee'] = payroll_summary['selling_incentive'] + payroll_summary['attendance_incentive']
            numeric_cols = ['total_selling', 'selling_incentive', 'attendance_incentive', 'accumulated_fee']
            payroll_summary[numeric_cols] = payroll_summary[numeric_cols].fillna(0).astype(int)
            display_df = payroll_summary.rename(columns={'ridername': 'Rider Name', 'area': 'Area', 'total_cups_sold': 'Total Cups Sold', 'total_selling': 'Total Selling', 'active_days': 'Active Days', 'selling_incentive': 'Selling Incentive', 'attendance_incentive': 'Attendance Incentive', 'accumulated_fee': 'Accumulated Fee'})
            display_df['Week'] = selected_week

            # ... (Tampilan tabel utama) ...
            st.markdown("---")
            st.subheader(f"Payroll Summary for Week {selected_week}")
            editor_df = display_df.copy()
            editor_df['Total Selling'] = editor_df['Total Selling'].apply(lambda x: f"Rp {x:,}")
            editor_df['Selling Incentive'] = editor_df['Selling Incentive'].apply(lambda x: f"Rp {x:,}")
            editor_df['Attendance Incentive'] = editor_df['Attendance Incentive'].apply(lambda x: f"Rp {x:,}")
            editor_df['Accumulated Fee'] = editor_df['Accumulated Fee'].apply(lambda x: f"Rp {x:,}")
            editor_df['Detail'] = False
            cols_to_display = ['Rider Name', 'Area', 'Total Cups Sold', 'Total Selling', 'Active Days', 'Selling Incentive', 'Attendance Incentive', 'Accumulated Fee', 'Detail']
            disabled_cols = [col for col in cols_to_display if col != 'Detail']
            edited_df_result = st.data_editor(editor_df[cols_to_display], use_container_width=True, hide_index=True, column_order=cols_to_display, disabled=disabled_cols, column_config={"Detail": st.column_config.CheckboxColumn(help="Check to see payslip detail", default=False)})

            selected_rows = edited_df_result[edited_df_result['Detail']]
            if not selected_rows.empty:
                selected_index = selected_rows.index[0]
                payslip_data = display_df.loc[selected_index]
                st.session_state.viewing_payslip_data = payslip_data.to_dict()
                
                # --- LOGIKA UNTUK RINCIAN DETAIL ---
                rider_transactions = final_filtered_df[(final_filtered_df['ridername'] == payslip_data['Rider Name']) & (final_filtered_df['area'] == payslip_data['Area'])].copy()
                daily_sales_detail = rider_transactions.groupby(rider_transactions['date'].dt.date)['selling'].sum().reset_index()
                
                # Hitung Daily Attendance Incentive
                if payslip_data['Area'] in ['Surabaya', 'Semarang']:
                    daily_sales_detail['Daily Attendance Incentive'] = daily_sales_detail['selling'].apply(get_daily_incentive_sby_smg)
                else: 
                    fixed_daily_rate = payslip_data['Attendance Incentive'] / payslip_data['Active Days'] if payslip_data['Active Days'] > 0 else 0
                    daily_sales_detail['Daily Attendance Incentive'] = fixed_daily_rate
                
                # Hitung Daily Selling Incentive (Prorate)
                avg_daily_selling_incentive = payslip_data['Selling Incentive'] / payslip_data['Active Days'] if payslip_data['Active Days'] > 0 else 0
                daily_sales_detail['Daily Selling Incentive'] = avg_daily_selling_incentive
                
                daily_details_df = daily_sales_detail.rename(columns={'date': 'Date', 'selling': 'Daily Selling'})
                daily_details_df['Date'] = pd.to_datetime(daily_details_df['Date']).dt.strftime('%d %B %Y')
                daily_details_df['Day'] = pd.to_datetime(daily_details_df['Date'], format='%d %B %Y').dt.day_name()
                
                # Susun ulang kolom
                final_detail_cols = ['Day', 'Date', 'Daily Selling', 'Daily Selling Incentive', 'Daily Attendance Incentive']
                st.session_state.daily_details_for_payslip = daily_details_df[final_detail_cols]
                
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(label="📥 Export Payroll Data to Excel", data=to_excel(display_df[cols_to_display[:-1]]), file_name=f"payroll_week_{selected_week}_filtered.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        
        else:
            st.info("Please select a week to view the payroll report.")

# --- KODE UNTUK SUBSECTION MONTHLY BONUS ---
        st.markdown("---")
        st.subheader("Monthly Bonus")
        st.info("This feature is under development.")
# -----------------------------------------

def display_ev_management():
    """Menampilkan halaman placeholder untuk EV Management."""
    if st.button("⬅️ Back to Menu"):
        set_view('main_menu')
        return
    st.markdown("---")
    st.header("🛵 EV Management")
    st.info("This feature is under development.")

# --- MAIN APPLICATION FLOW ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("BLITZ LOGO.png", width=150)
        except FileNotFoundError:
            st.warning("Logo file 'BLITZ LOGO.png' not found.")
        st.title("Login - KSJ Data Dashboard")
        username_input = st.text_input("Username", placeholder="Enter your username")
        password_input = st.text_input("Password", type="password", placeholder="Enter your password")
        if st.button("Login", use_container_width=True):
            check_login(username_input, password_input)
else:
    if "main_df" not in st.session_state:
        with st.spinner("Processing Your Data. Please wait... :)"):
            st.session_state["main_df"] = load_and_process_main_data()
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0
    col1, col2 = st.columns([1, 8])
    with col1:
        try:
            st.image("BLITZ LOGO.png", width=80)
        except: pass
    with col2:
        st.title("📊 KSJ Data 2025")
    st.markdown('<script>document.title = "KSJ Data 2025";</script>', unsafe_allow_html=True)
    st.sidebar.title(f"Welcome, {st.session_state.get('username', 'User')}!")
    if st.sidebar.button("Logout"):
        logout()
    st.sidebar.markdown("---")
    if "view" not in st.session_state:
        st.session_state.view = "main_menu"
    current_view = st.session_state.get("view", "main_menu")

    if current_view == "main_menu":
        display_main_menu()
    elif current_view == "grup_1":
        display_grup_1()
    elif current_view == "grup_2":
        display_grup_2()
    elif current_view == "area_analysis":
        display_area_analysis()
    elif current_view == "payroll_management":
        display_payroll_management()
    elif current_view == "ev_management":
        display_ev_management()
