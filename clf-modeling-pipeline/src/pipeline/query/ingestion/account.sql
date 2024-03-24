SELECT 
   ACCT_ID,
   CUST_ID,
   SNAP_DT,
   ACCT_TYPE_CD,
   DR_AMT,
   CR_AMT,
   END_BAL,
   CR_LMT
FROM 
    {obj_storage}
-- WHERE
--     SNAP_DT = '{snap_dt}'