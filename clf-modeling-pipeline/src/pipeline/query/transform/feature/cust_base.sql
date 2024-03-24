SELECT 
	CUST_ID,
	SNAP_DT,
	GENDER,
	extract(ifNull(EMAIL, 'xx@NA'), '@(\w+)') AS EMAIL_DOMAIN,
	BLOOD_GRP,
	if(trim(lower(OFFICE)) IN ('new york', 'los angeles', 'chicago'), TRUE, FALSE) AS SUPER_CITY,
	if(trim(lower(OFFICE)) IN ('new york', 'seattle', 'los angeles'), TRUE, FALSE) AS COASTAL_CITY,
	if(trim(lower(TITLE)) IN ('manager', 'senior manager', 'vp'), TRUE, FALSE) AS MANAGER_FLG,
	if(trim(lower(ORG)) IN ('devops', 'internal tools', 'platform'), TRUE, FALSE) AS TECHNICAL_FLG,
	date_diff('year', BIRTH_DT, SNAP_DT) AS AGE,
	date_diff('year', SINCE_DT, SNAP_DT) AS TENURE,
	ifNull(SALARY, 0) + ifNull(BONUS, 0) AS TOTAL_COMP,
	ifNull(BONUS, 0) / TOTAL_COMP AS BONUS_RATIO
FROM
	RAW.CUSTOMER
WHERE
	SNAP_DT = toDate32('{snap_dt}')