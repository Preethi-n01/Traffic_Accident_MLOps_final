import streamlit as st
import pandas as pd
import psycopg2

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="UK Traffic Accident Analytics",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 UK Traffic Accident Analytics Dashboard")
st.write(
    "Interactive analysis of UK road accidents using accident and weather data."
)

# -----------------------------
# PostgreSQL connection
# -----------------------------

DB_CONFIG = {
    "host": "172.24.144.1",
    "port": 8080,
    "database": "traffic_accidents",
    "user": "postgres",
    "password": "traffic",
}

@st.cache_data
def load_data():
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
        SELECT
            collision_severity,
            number_of_vehicles,
            number_of_casualties,
            date,
            time,
            road_type,
            speed_limit,
            weather_condition,
            light_conditions,
            road_surface_conditions,
            latitude,
            longitude,
            hour,
            month,
            weekday
        FROM accident_weather_mart
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df

# -----------------------------
# Load data
# -----------------------------

try:
    df = load_data()
except Exception as e:
    st.error("Could not connect to PostgreSQL.")
    st.error(str(e))
    st.stop()

# -----------------------------
# Sidebar filters
# -----------------------------

st.sidebar.header("Filters")

severity_options = sorted(
    df["collision_severity"].dropna().unique()
)

selected_severity = st.sidebar.multiselect(
    "Collision Severity",
    severity_options,
    default=severity_options
)

weather_options = sorted(
    df["weather_condition"].dropna().unique()
)

selected_weather = st.sidebar.multiselect(
    "Weather Condition",
    weather_options,
    default=weather_options
)

filtered_df = df[
    df["collision_severity"].isin(selected_severity)
    & df["weather_condition"].isin(selected_weather)
]

# -----------------------------
# Key indicators
# -----------------------------

st.subheader("Key Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Accidents",
    f"{len(filtered_df):,}"
)

col2.metric(
    "Total Casualties",
    f"{filtered_df['number_of_casualties'].sum():,}"
)

col3.metric(
    "Vehicles Involved",
    f"{filtered_df['number_of_vehicles'].sum():,}"
)

col4.metric(
    "Average Casualties / Accident",
    f"{filtered_df['number_of_casualties'].mean():.2f}"
)

# -----------------------------
# View 1: Severity distribution
# -----------------------------

st.subheader("1. Accident Severity Distribution")

severity_counts = (
    filtered_df["collision_severity"]
    .value_counts()
    .sort_index()
)

severity_counts.index = severity_counts.index.map({
    1: "Fatal",
    2: "Serious",
    3: "Slight"
})

st.bar_chart(severity_counts)

# -----------------------------
# View 2: Accidents by hour
# -----------------------------

st.subheader("2. Accidents by Hour")

hour_counts = (
    filtered_df["hour"]
    .value_counts()
    .sort_index()
)

st.line_chart(hour_counts)

# -----------------------------
# View 3: Accidents by weekday
# -----------------------------

st.subheader("3. Accidents by Weekday")

weekday_counts = (
    filtered_df["weekday"]
    .value_counts()
)

st.bar_chart(weekday_counts)

# -----------------------------
# View 4: Weather impact
# -----------------------------

st.subheader("4. Weather Condition Impact")

weather_counts = (
    filtered_df["weather_condition"]
    .value_counts()
)

st.bar_chart(weather_counts)

# -----------------------------
# View 5: Road type analysis
# -----------------------------

st.subheader("5. Accidents by Road Type")

road_counts = (
    filtered_df["road_type"]
    .value_counts()
    .sort_index()
)

st.bar_chart(road_counts)

# -----------------------------
# View 6: Monthly trend
# -----------------------------

st.subheader("6. Monthly Accident Trend")

monthly_counts = (
    filtered_df
    .groupby("month")
    .size()
    .sort_index()
)

st.line_chart(monthly_counts)

# -----------------------------
# View 7: Location hotspot map
# -----------------------------

st.subheader("7. Accident Location Hotspots")

map_data = filtered_df[
    ["latitude", "longitude"]
].dropna()

map_data = map_data[
    (map_data["latitude"].between(-90, 90))
    & (map_data["longitude"].between(-180, 180))
]

st.map(map_data)

# -----------------------------
# Data preview
# -----------------------------

st.subheader("Analytical Data Preview")

st.dataframe(
    filtered_df.head(100),
    use_container_width=True
)