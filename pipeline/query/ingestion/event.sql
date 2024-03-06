SELECT 
	CUST_ID, 
	'{snap_dt}' AS SNAP_DT,
	EVENT_TS,
	first_value(EVENT_CHANNEL) AS EVENT_CHANNEL,
	first_value(EVENT_CD) AS EVENT_CD
FROM 
    {obj_storage}
WHERE 
	date_trunc('day', EVENT_TS) = toDate32('{snap_dt}')
-- need group by because CH has bug in loading s3 with where statement
GROUP BY
	CUST_ID,
	SNAP_DT,
	EVENT_TS;