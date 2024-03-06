SELECT 
	CUST_ID, 
	'{snap_dt}' AS SNAP_DT,
	PUR_TS,
	SUM(PUR_AMT) AS PUR_AMT,
	SUM(PUR_NUM) AS PUR_NUM
FROM 
    {obj_storage}
WHERE 
	date_trunc('day', PUR_TS) = toDate32('{snap_dt}')
-- need group by because CH has bug in loading s3 with where statement
GROUP BY
	CUST_ID,
	SNAP_DT,
	PUR_TS;