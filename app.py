import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- INITIAL SETUP & FUNCTIONS ---
st.set_page_config(page_title="KSJ Data 2025", layout="wide")

# --- UTILITY FUNCTIONS ---
def check_login(username, password):
    if username == "Blitz" and password == "ksj2025":
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
    else:
        st.session_state["logged_in"] = False
        st.error("Incorrect Username or Password!")

def logout():
    st.session_state["logged_in"] = False
    st.session_state.pop("username", None)
    st.session_state.pop("view", None)
    st.session_state.pop("main_df", None)

def set_view(view_name):
    st.session_state["view"] = view_name

# --- SINGLE, EFFICIENT DATA LOADING FUNCTION ---
@st.cache_data
def load_and_process_main_data():
    """Loads and cleans the main data file ONCE."""
    df = pd.read_excel("KSJ Data 2025.xlsx")
    df.columns = [col.strip().replace(" ", "").replace("#", "").lower() for col in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day'] = df['date'].dt.day_name()
    if 'week' not in df.columns and 'date' in df.columns:
        df['week'] = df['date'].dt.isocalendar().week
    numeric_cols = ['selling', 'revenue', 'cups']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- FUNCTIONS TO DISPLAY CONTENT ---
def display_main_menu():
    st.header("Main Menu")
    st.markdown("Select an analysis to view from the options below.")
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        with st.container(border=True):
            st.subheader("📊 All Data")
            st.markdown("View all raw data, week-on-week productivity graphs, and daily analysis.")
            st.button("Open Report", on_click=set_view, args=['grup_1'], key="grup1_button", use_container_width=True)
    with col2:
        with st.container(border=True):
            st.subheader("🗓️ Business Dashboard")
            st.markdown("View a consolidated dashboard of business performance, position, and seller retention.")
            st.button("Open Report", on_click=set_view, args=['grup_2'], key="grup2_button", use_container_width=True)
    with col3:
        with st.container(border=True):
            st.subheader("📍 Area Analysis")
            st.markdown("In-depth analysis based on performance in each area.")
            st.button("Open Report", on_click=set_view, args=['area_analysis'], key="area_button", use_container_width=True)

def display_grup_1():
    # KODE DI BAGIAN INI TIDAK DIUBAH SAMA SEKALI
    if st.button("⬅️ Back to Menu"):
        set_view('main_menu')
    st.markdown("---")
    df = st.session_state["main_df"]
    st.subheader("📄 All Data")
    filtered_df = df.copy()
    with st.expander("🔎 Filter"):
        for col in df.columns:
            if df[col].dtype != 'datetime64[ns]':
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) < 100:
                    default_vals = sorted(unique_vals)
                    select_all = st.checkbox(f"Select All {col.title()}", value=True, key=f"all_{col}")
                    if select_all:
                        selected_vals = st.multiselect(f"Filter {col.title()}", options=default_vals, default=default_vals, key=col)
                    else:
                        selected_vals = st.multiselect(f"Filter {col.title()}", options=default_vals, default=[], key=col)
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
    if st.button("🔄 Show Data"):
        styled_df = filtered_df.copy()
        if 'date' in styled_df.columns:
             styled_df['date'] = styled_df['date'].dt.strftime('%d/%m/%Y')
        for col in ['selling', 'revenue']:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(lambda x: f"Rp {x:,.0f}".replace(",", ".") if pd.notnull(x) else "-")
        styled_df.columns = [col.title() for col in styled_df.columns]
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    st.subheader("📈 Week-on-Week Productivity")
    if 'week' in filtered_df.columns and 'cups' in filtered_df.columns and 'revenue' in filtered_df.columns:
        col1, col2 = st.columns(2)
        with col1:
            df_chart_cups = filtered_df.groupby('week')['cups'].sum().reset_index()
            fig_cups = px.bar(df_chart_cups, x='week', y='cups', title='Total Cups per Week', labels={'cups': 'Cups Sold', 'week': 'Week'}, text_auto=True)
            st.plotly_chart(fig_cups, use_container_width=True)
        with col2:
            df_chart_revenue = filtered_df.groupby('week')['revenue'].sum().reset_index()
            fig_revenue = px.bar(df_chart_revenue, x='week', y='revenue', title='Total Revenue per Week', labels={'revenue': 'Total Revenue (Rp)', 'week': 'Week'}, text_auto=True)
            fig_revenue.update_traces(texttemplate='Rp%{y:,.0f}', textposition='outside')
            fig_revenue.update_yaxes(title_text='Total Revenue (Rp)')
            st.plotly_chart(fig_revenue, use_container_width=True)
    else:
        st.warning("Column 'week', 'cups', or 'revenue' is not available for charting.")
    st.subheader("📅 Productivity by Day")
    if 'day' in filtered_df.columns and 'cups' in filtered_df.columns and 'ridername' in filtered_df.columns:
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
        st.warning("Column 'day', 'ridername', or 'cups' is not available.")

def display_grup_2():
    # KODE DI BAGIAN INI TIDAK DIUBAH SAMA SEKALI
    if st.button("⬅️ Back to Menu"):
        set_view('main_menu')
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
    col1, col2 = st.columns(2)
    col1.metric("Total Client's Revenue (KSJ)", f"Rp {total_ksj_revenue:,.0f}")
    col2.metric("Total Blitz's Revenue", f"Rp {total_blitz_revenue:,.0f}")
    st.markdown("---")
    st.subheader("Business Position")
    latest_month_period = df['date'].dt.to_period('M').max()
    previous_month_period = latest_month_period - 1
    cups_latest_month = df[df['date'].dt.to_period('M') == latest_month_period]['cups'].sum()
    cups_previous_month = df[df['date'].dt.to_period('M') == previous_month_period]['cups'].sum()
    latest_week = df['week'].max()
    previous_week = latest_week - 1
    cups_latest_week = df[df['week'] == latest_week]['cups'].sum()
    cups_previous_week = df[df['week'] == previous_week]['cups'].sum()
    col1, col2 = st.columns(2)
    col1.metric(f"Product Sold ({latest_month_period})", f"{cups_latest_month:,}", f"{cups_latest_month - cups_previous_month:,} vs Prv. Month")
    col2.metric(f"Product Sold (Week {latest_week})", f"{cups_latest_week:,}", f"{cups_latest_week - cups_previous_week:,} vs Prv. Week")
    st.markdown("---")
    st.subheader("Seller Retention Analysis")
    @st.cache_data
    def calculate_seller_retention(dataf):
        if 'week' not in dataf.columns or 'ridername' not in dataf.columns: return None
        weekly_sellers = dataf.groupby('week')['ridername'].unique().apply(set).sort_index()
        results = []
        if not weekly_sellers.empty:
            first_week_num = weekly_sellers.index[0]
            first_week_sellers = weekly_sellers.iloc[0]
            results.append({"Week": first_week_num, "Total Sellers": len(first_week_sellers), "New Sellers": len(first_week_sellers), "Retained Sellers": 0, "Churned Sellers": 0, "Retention Rate (%)": 0.0})
        for i in range(1, len(weekly_sellers)):
            current_week_num = weekly_sellers.index[i]
            current_sellers, prev_sellers = weekly_sellers.iloc[i], weekly_sellers.iloc[i-1]
            retained_sellers_set = current_sellers.intersection(prev_sellers)
            new_sellers_set = current_sellers.difference(prev_sellers)
            churned_sellers_set = prev_sellers.difference(current_sellers)
            retention_rate = (len(retained_sellers_set) / len(prev_sellers)) * 100 if len(prev_sellers) > 0 else 0
            results.append({"Week": current_week_num, "Total Sellers": len(current_sellers), "New Sellers": len(new_sellers_set), "Retained Sellers": len(retained_sellers_set), "Churned Sellers": len(churned_sellers_set), "Retention Rate (%)": retention_rate})
        return pd.DataFrame(results)
    retention_df = calculate_seller_retention(df)
    if retention_df is not None:
        st.dataframe(retention_df.style.format({"Retention Rate (%)": "{:.2f}%"}), use_container_width=True, hide_index=True)
        fig_retention = px.line(retention_df, x='Week', y='Retention Rate (%)', title='Seller Retention Rate Over Time', markers=True)
        fig_retention.update_layout(yaxis_ticksuffix="%")
        st.plotly_chart(fig_retention, use_container_width=True)
    else:
        st.warning("Cannot perform retention analysis. Required columns 'week' or 'ridername' not found.")

def display_area_analysis():
    if st.button("⬅️ Back to Menu"):
        set_view('main_menu')
    st.markdown("---")
    st.header("📍 Area Analysis")

    df = st.session_state["main_df"]
    
    area_cols = ['area', 'city', 'district', 'outlet']
    if not all(col in df.columns for col in area_cols):
        st.error("Data 'Area', 'City', 'District', or 'Outlet' not found in the main data file.")
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
    
    if selected_weeks:
        time_filtered_df = df[df['week'].isin(selected_weeks)]
    else:
        st.warning("Please select at least one week to continue.")
        return

    filter_cols = st.columns(4)
    with filter_cols[0]:
        area_list = ['All Areas'] + time_filtered_df['area'].unique().tolist()
        selected_area = st.selectbox("Select Area", area_list)
    df_filtered = time_filtered_df[time_filtered_df['area'] == selected_area] if selected_area != 'All Areas' else time_filtered_df
    with filter_cols[1]:
        city_list = ['All Cities'] + df_filtered['city'].unique().tolist()
        selected_city = st.selectbox("Select City", city_list)
    df_filtered = df_filtered[df_filtered['city'] == selected_city] if selected_city != 'All Cities' else df_filtered
    with filter_cols[2]:
        district_list = ['All Districts'] + df_filtered['district'].unique().tolist()
        selected_district = st.selectbox("Select District", district_list)
    df_filtered = df_filtered[df_filtered['district'] == selected_district] if selected_district != 'All Districts' else df_filtered
    with filter_cols[3]:
        outlet_list = ['All Outlets'] + df_filtered['outlet'].unique().tolist()
        selected_outlet = st.selectbox("Select Outlet", outlet_list)
    df_filtered = df_filtered[df_filtered['outlet'] == selected_outlet] if selected_outlet != 'All Outlets' else df_filtered
    st.markdown("---")
    
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
    chart_labels = {'revenue': 'Revenue', 'area': 'Area', 'city': 'City', 'district': 'District', 'outlet': 'Outlet'}
    if selected_outlet != 'All Outlets':
        st.info(f"Showing performance for outlet: **{selected_outlet}**")
        outlet_trend = df_filtered.groupby('week')['revenue'].sum().reset_index()
        fig = px.line(outlet_trend, x='week', y='revenue', title=f"Weekly Revenue Trend for {selected_outlet}", markers=True, labels=chart_labels)
        st.plotly_chart(fig, use_container_width=True)
    elif selected_district != 'All Districts':
        breakdown_data = df_filtered.groupby('outlet')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
        fig = px.bar(breakdown_data, x='outlet', y='revenue', title=f"Revenue Breakdown by Outlet in {selected_district}", labels=chart_labels)
        st.plotly_chart(fig, use_container_width=True)
    elif selected_city != 'All Cities':
        breakdown_data = df_filtered.groupby('district')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
        fig = px.bar(breakdown_data, x='district', y='revenue', title=f"Revenue Breakdown by District in {selected_city}", labels=chart_labels)
        st.plotly_chart(fig, use_container_width=True)
    elif selected_area != 'All Areas':
        breakdown_data = df_filtered.groupby('city')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
        fig = px.bar(breakdown_data, x='city', y='revenue', title=f"Revenue Breakdown by City in {selected_area}", labels=chart_labels)
        st.plotly_chart(fig, use_container_width=True)
    else:
        breakdown_data = df_filtered.groupby('area')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
        fig = px.bar(breakdown_data, x='area', y='revenue', title="Overall Revenue Breakdown by Area", labels=chart_labels)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("District Productivity Ranking")
    if not df_filtered.empty:
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
        
        # --- REVISI LOGIKA STATUS DI SINI ---
        supply_demand_summary['status'] = supply_demand_summary['ratio'].apply(lambda x: "Productive" if (x > 0 and x <= 0.0222) else "Not Productive")

        status_options = ["Productive", "Not Productive"]
        selected_statuses = st.multiselect("Filter by Status", options=status_options, default=status_options)

        if selected_statuses:
            final_sd_df = supply_demand_summary[supply_demand_summary['status'].isin(selected_statuses)]
        else:
            final_sd_df = supply_demand_summary.copy()

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
    else:
        st.info("No data available to calculate Supply vs Demand after filtering.")

# --- MAIN APPLICATION FLOW ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if not st.session_state["logged_in"]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("BLITZ LOGO.png", width=150)
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
        st.image("BLITZ LOGO.png", width=80)
    with col2:
        st.title("📊 KSJ Data 2025")
    st.sidebar.title(f"Welcome, {st.session_state['username']}!")
    st.sidebar.button("Logout", on_click=logout)
    st.sidebar.markdown("---")
    if "view" not in st.session_state:
        st.session_state["view"] = "main_menu"
    if st.session_state["view"] == "main_menu":
        display_main_menu()
    elif st.session_state["view"] == "grup_1":
        display_grup_1()
    elif st.session_state["view"] == "grup_2":
        display_grup_2()
    elif st.session_state["view"] == "area_analysis":
        display_area_analysis()