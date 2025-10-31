{{ config(materialized='table') }}

SELECT t.transaction_id, t.transaction_date, t.quantity_consumed, t.unit_cost, t.total_cost, 
t.transaction_type, t.formulary_adherent, t.adherence_impact_pct, t.bounced, t.bounce_reason, t.revenue_lost, t.patient_complexity_score,
t.urgency_level
,h.hospital_id
,h.hospital_name
,h.hospital_type
,h.country
,h.city
,h.bed_capacity
,h.annual_budget
,h.establishment_year

,d.dept_id
,d.dept_name
,d.dept_type
,d.bed_count
,d.monthly_budget
,d.head_physician_id

,p.physician_id
,p.primary_dept_id
,p.specialty
,p.experience_years
,p.nationality
,p.employment_type
,p.prescribing_preference
,p.formulary_adherence_score
,p.cost_consciousness
,p.prescription_volume

,pt.patient_id
,pt.age_group
,pt.patient_type
,pt.chronic_conditions
,pt.insurance_type
,pt.socioeconomic_level
,pt.admission_date
,pt.discharge_date
,pt.length_of_stay

,sku.*
FROM (select * from transactions_2023_sheet1 
UNION
select * from transactions_2024_sheet1 ) t

INNER JOIN hospital_dataset_hospital h
ON t.hospital_id = h.hospital_id

INNER JOIN hospital_dataset_departments d
ON t.dept_id = d.dept_id
AND t.hospital_id = d.hospital_id

INNER JOIN hospital_dataset_physicians p
ON t.physician_id = p.physician_id
AND t.hospital_id = p.hospital_id

INNER JOIN patients_sheet1 pt
ON t.patient_id = pt.patient_id
AND t.hospital_id = pt.hospital_id

INNER JOIN hospital_dataset_skus sku
ON t.sku_id = sku.sku_id
