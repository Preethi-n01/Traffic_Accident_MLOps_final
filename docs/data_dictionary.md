# Data Dictionary and Validation Rules

## 1. Data Dictionary

The analytical dataset `accident_weather_mart` contains accident, road, location, severity, and weather-related attributes.

| Field | Data Type | Description |
|---|---|---|
| collision_index | VARCHAR | Unique identifier for the collision |
| collision_year | INTEGER | Year in which the collision occurred |
| collision_ref_no | VARCHAR | Collision reference number |
| location_easting_osgr | DOUBLE | Easting coordinate of accident location |
| location_northing_osgr | DOUBLE | Northing coordinate of accident location |
| longitude | DOUBLE | Longitude of accident location |
| latitude | DOUBLE | Latitude of accident location |
| police_force | INTEGER | Police force responsible for the area |
| collision_severity | INTEGER | Severity classification of the collision |
| number_of_vehicles | INTEGER | Number of vehicles involved |
| number_of_casualties | INTEGER | Number of casualties in the collision |
| date | DATE | Date of the collision |
| day_of_week | INTEGER | Day of the week on which the collision occurred |
| time | TIME | Time of the collision |
| local_authority_district | INTEGER | Local authority district code |
| local_authority_ons_district | VARCHAR | ONS local authority district code |
| local_authority_highway | VARCHAR | Highway authority identifier |
| local_authority_highway_current | VARCHAR | Current highway authority identifier |
| first_road_class | INTEGER | Classification of the first road involved |
| first_road_number | INTEGER | Number of the first road |
| road_type | INTEGER | Type/classification of the road |
| speed_limit | INTEGER | Speed limit at the accident location |
| junction_detail_historic | INTEGER | Historical junction detail classification |
| junction_detail | INTEGER | Junction detail classification |
| junction_control | INTEGER | Type of junction control |
| second_road_class | INTEGER | Classification of the second road |
| second_road_number | INTEGER | Number of the second road |
| pedestrian_crossing_human_control_historic | INTEGER | Historical human-controlled pedestrian crossing information |
| pedestrian_crossing_physical_facilities_historic | INTEGER | Historical physical pedestrian crossing facilities |
| pedestrian_crossing | INTEGER | Pedestrian crossing information |
| light_conditions | INTEGER | Lighting conditions at the accident location |
| weather_conditions | INTEGER | Weather condition recorded for the collision |
| road_surface_conditions | INTEGER | Road surface condition |
| special_conditions_at_site | INTEGER | Special conditions at the accident site |
| carriageway_hazards_historic | INTEGER | Historical carriageway hazard information |
| carriageway_hazards | INTEGER | Current carriageway hazard information |
| urban_or_rural_area | INTEGER | Indicates whether the accident occurred in an urban or rural area |
| did_police_officer_attend_scene_of_accident | INTEGER | Indicates police attendance at the accident scene |
| trunk_road_flag | INTEGER | Indicates whether the location is a trunk road |
| lsoa_of_accident_location | VARCHAR | Lower Layer Super Output Area code |
| enhanced_severity_collision | INTEGER | Enhanced collision severity classification |
| collision_injury_based | INTEGER | Injury-based collision indicator |
| collision_adjusted_severity_serious | DOUBLE | Adjusted serious-severity value |
| collision_adjusted_severity_slight | DOUBLE | Adjusted slight-severity value |
| weather_code_api | DOUBLE | Weather code obtained from the weather API |
| temperature_mean | DOUBLE | Mean temperature associated with the accident |
| temperature_max | DOUBLE | Maximum temperature |
| temperature_min | DOUBLE | Minimum temperature |
| precipitation | DOUBLE | Precipitation amount |
| rain | DOUBLE | Rain amount |
| snowfall | DOUBLE | Snowfall amount |
| wind_speed_max | DOUBLE | Maximum wind speed |
| weather_available | INTEGER | Indicates whether weather information was available |
| weather_condition | VARCHAR | Categorized weather condition |
| hour | INTEGER | Hour extracted from accident time |
| year | INTEGER | Year extracted from accident date |
| month | INTEGER | Month extracted from accident date |
| day | INTEGER | Day extracted from accident date |
| weekday | VARCHAR | Weekday name |

## 2. Data Validation Rules

The following validation rules are applied during the ETL pipeline:

| Validation Rule | Description |
|---|---|
| Required collision identifier | Collision records must contain a valid collision identifier |
| Duplicate check | Duplicate accident records are identified and removed |
| Date validation | Accident dates must contain valid date values |
| Time validation | Accident times must follow a valid time format |
| Coordinate validation | Latitude and longitude values must be valid geographic coordinates |
| Latitude range | Latitude must be between -90 and 90 |
| Longitude range | Longitude must be between -180 and 180 |
| Vehicle count | Number of vehicles must be a non-negative value |
| Casualty count | Number of casualties must be a non-negative value |
| Severity validation | Collision severity must contain a valid coded value |
| Speed limit validation | Speed-limit values must be valid numeric values |
| Weather validation | Weather values are checked for missing or invalid API responses |
| Missing-value handling | Missing values are handled during transformation rather than manually editing the analytical dataset |
| Rejected records | Records failing validation are stored in the rejected data layer |
| Data type validation | Fields are converted to appropriate numeric, date, time, or categorical data types |
| Row-count validation | Record counts are checked during ingestion and transformation |
| ETL reproducibility | The pipeline can be executed repeatedly through Apache Airflow |

## 3. Data Quality Layers

The project maintains separate data layers:

- **Raw:** Original accident, vehicle, and casualty files are retained without manual modification.
- **Staging:** Intermediate accident and weather data used during processing and enrichment.
- **Cleaned:** Validated and standardized accident, vehicle, and casualty data.
- **Rejected:** Records that fail validation checks.
- **Analytical:** Final accident-weather dataset used for PostgreSQL storage and dashboard analysis.

## 4. Analytical Dataset

The final analytical dataset is stored as:

`data/analytical/accident_weather.csv`

It contains **101,523 accident records** after processing and weather enrichment.