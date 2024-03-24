WITH
toDate32('{snap_dt}') AS SPDT,
CATESIAN AS (
    SELECT
        DISTINCT CUST_ID,
        DT
    FROM
        FEATURE.ACCT_BASE
        ARRAY JOIN arrayMap(
            x -> date_sub(DAY, x, SPDT),
            range(7)
        ) AS DT
    WHERE
        SNAP_DT = SPDT
),
WINDOW AS (
    SELECT
        C.CUST_ID AS CUST_ID,
        C.DT AS SNAP_DT,

        ifNull(NUM_ACCTS_DEB, 0) AS NUM_ACCTS_DEB,
        ifNull(NUM_ACCTS_CRE, 0) AS NUM_ACCTS_CRE,
        ifNull(NUM_ACCTS_DEB, 0) + ifNull(NUM_ACCTS_CRE, 0) AS NUM_ACCTS,
        ifNull(END_BAL_DEB, 0) AS END_BAL_DEB,
        ifNull(END_BAL_CRE, 0) AS END_BAL_CRE,
        ifNull(DR_AMT_DEB, 0) AS DR_AMT_DEB,
        ifNull(DR_AMT_CRE, 0) AS DR_AMT_CRE,
        ifNull(CR_AMT_DEB, 0) AS CR_AMT_DEB,
        ifNull(CR_AMT_CRE, 0) AS CR_AMT_CRE,
        ifNull(DR_AMT_DEB, 0) - ifNull(CR_AMT_DEB, 0) AS TOTAL_DEB_FLOW,
        ifNull(CR_AMT_CRE, 0) -  ifNull(DR_AMT_CRE, 0) AS TOTAL_CRE_FLOW,
        TOTAL_DEB_FLOW + TOTAL_CRE_FLOW AS TOTAL_FLOW,
        ifNull(CR_LMT, 0) AS CR_LMT
    FROM
        CATESIAN AS C
        LEFT JOIN FEATURE.ACCT_BASE AS B ON
            C.CUST_ID = B.CUST_ID
            AND C.DT = B.SNAP_DT
    ORDER BY
        SNAP_DT DESC
    --SETTINGS join_use_nulls = 1
),
ARR AS (
    SELECT
        CUST_ID,
        groupArray(WINDOW.SNAP_DT)[1] AS FIRST_SNAP_DT,
        -- past 7d feature
        arrayMax(groupArray(7)(WINDOW.NUM_ACCTS)) AS MAX_NUM_ACCT_7D,
        arraySum(groupArray(7)(WINDOW.TOTAL_FLOW)) AS TOTAL_FLOW_7D,
        arraySum(groupArray(7)(WINDOW.TOTAL_DEB_FLOW)) AS TOTAL_DEB_FLOW_7D,
        arraySum(groupArray(7)(WINDOW.TOTAL_CRE_FLOW)) AS TOTAL_CRE_FLOW_7D,
        arrayAvg(groupArray(7)(WINDOW.END_BAL_CRE)) AS AVG_END_BAL_CRE_7D,
        arrayMax(groupArray(7)(WINDOW.END_BAL_CRE)) AS MAX_END_BAL_CRE_7D,
        arrayMax(groupArray(7)(WINDOW.CR_LMT)) AS MAX_CR_LMT_7D,
        -- past 3d feature
        arraySum(groupArray(3)(WINDOW.DR_AMT_DEB)) AS TOTAL_DR_AMT_DEB_3D,
        arraySum(groupArray(3)(WINDOW.CR_AMT_DEB)) AS TOTAL_CR_AMT_DEB_3D,
        arrayAvg(groupArray(3)(WINDOW.END_BAL_DEB)) AS AVG_END_BAL_DEB_3D,
        arraySum(groupArray(3)(WINDOW.DR_AMT_CRE)) AS TOTAL_DR_AMT_CRE_3D,
        arraySum(groupArray(3)(WINDOW.CR_AMT_CRE)) AS TOTAL_CR_AMT_CRE_3D,
        arrayAvg(groupArray(3)(WINDOW.END_BAL_CRE)) AS AVG_END_BAL_CRE_3D
    FROM
        WINDOW
    GROUP BY
        CUST_ID
)
SELECT
    CUST_ID,
    '{snap_dt}' AS SNAP_DT,
    MAX_NUM_ACCT_7D,
    TOTAL_FLOW_7D,
    ifNull(TOTAL_DEB_FLOW_7D / Nullif(TOTAL_CRE_FLOW_7D, 0), 0) AS DEB_CRE_FLOW_7D,
    ifNull(AVG_END_BAL_CRE_7D / Nullif(MAX_CR_LMT_7D, 0), 0) AS AVG_CR_UTIL_7D,
    ifNull(MAX_END_BAL_CRE_7D / Nullif(MAX_CR_LMT_7D, 0), 0) AS MAX_CR_UTIL_7D,
    ifNull(TOTAL_DR_AMT_DEB_3D / Nullif(AVG_END_BAL_DEB_3D, 0), 0) AS AVG_DR_DEB_3D,
    ifNull(TOTAL_CR_AMT_DEB_3D / Nullif(AVG_END_BAL_DEB_3D, 0), 0) AS AVG_CR_DEB_3D,
    ifNull(TOTAL_DR_AMT_CRE_3D / Nullif(AVG_END_BAL_CRE_3D, 0), 0) AS AVG_DR_CRE_3D,
    ifNull(TOTAL_CR_AMT_CRE_3D / Nullif(AVG_END_BAL_CRE_3D, 0), 0) AS AVG_CR_CRE_3D
FROM
    ARR
WHERE
    SNAP_DT = SPDT