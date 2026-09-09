-- 1. Accidents and casualties by weather condition
SELECT
    weather_condition,
    COUNT(*) AS accident_count,
    SUM(number_of_casualties) AS total_casualties
FROM accident_weather_mart
GROUP BY weather_condition
ORDER BY accident_count DESC;


-- 2. Accidents and casualties by collision severity
SELECT
    collision_severity,
    COUNT(*) AS accident_count,
    SUM(number_of_casualties) AS total_casualties
FROM accident_weather_mart
GROUP BY collision_severity
ORDER BY collision_severity;


-- 3. Monthly accident trends
SELECT
    year,
    month,
    COUNT(*) AS accident_count,
    SUM(number_of_casualties) AS total_casualties
FROM accident_weather_mart
GROUP BY year, month
ORDER BY year, month;