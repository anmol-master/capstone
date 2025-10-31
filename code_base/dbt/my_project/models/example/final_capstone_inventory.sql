{{ config(materialized='table') }}

SELECT
  COALESCE(supply.delivery_date, transaction.transaction_date) AS date_,
  COALESCE(supply.sku_id, transaction.sku_id) AS sku_id,
  COALESCE(supply.incoming_quanity,0) AS incoming_quanity,
  COALESCE(transaction.outgoing_quantity,0) AS outgoing_quantity,
  COALESCE(supply.incoming_quanity,0) - COALESCE(transaction.outgoing_quantity,0) AS net_quantity_day,
  inv.reorder_level
  
FROM 
  (
  SELECT delivery_date,sku_id,sum(accepted_quantity) as incoming_quanity FROM  supply_chain_deliveries 
  group by 1,2
  ) supply
FULL JOIN 
  (select transaction_date, sku_id, sum(quantity_consumed) as outgoing_quantity FROM 
    (
    select * from transactions_2023_sheet1 
    UNION
    select * from transactions_2024_sheet1 
    ) transaction
  group by 1,2) transaction
ON supply.delivery_date = transaction.transaction_date
AND transaction.sku_id = supply.sku_id
LEFT JOIN inventory_sheet1 inv
ON inv.sku_id = COALESCE(supply.sku_id, transaction.sku_id)
ORDER BY 1