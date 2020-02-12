require ( magrittr )
require ( data.table )

setwd ( "<PARENT FOLDER HERE>" )

Keep = c (
  "Race" = "race",
  "Gender" = "gender",
  "Age" = "age",
  "Source" = "admission_source_id",
  "Stay" = "time_in_hospital",
  "Labs_Count" = "num_lab_procedures",
  "Procedure_Count" = "num_procedures",
  "Medication_Count" = "num_medications",
  "Emergency_Count" = "number_emergency",
  "Prev_Admissions_Count" = "number_inpatient",
  "Diagnosis" = "diag_1",
  "Diagnosis_Count" = "number_diagnoses",
  "HbA1C" = "A1Cresult",
  "Metformin" = "metformin",
  "Repaglinide" = "repaglinide",
  "Nateglinide" = "nateglinide",
  "Chlorpropamide" = "chlorpropamide",
  "Glimepiride" = "glimepiride",
  "Acetohexamide" = "acetohexamide",
  "Glipizide" = "glipizide",
  "Glyburide" = "glyburide",
  "Tolbutamide" = "tolbutamide",
  "Pioglitazone" = "pioglitazone",
  "Rosiglitazone" = "rosiglitazone",
  "Acarbose" = "acarbose",
  "Miglitol" = "miglitol",
  "Troglitazone" = "troglitazone",
  "Tolazamide" = "tolazamide",
  "Examide" = "examide",
  "Citoglipton" = "citoglipton",
  "Insulin" = "insulin",
  "Glyburide_metformin" = "glyburide-metformin",
  "Glipizide_metformin" = "glipizide-metformin",
  "Glimepiride_pioglitazone" = "glimepiride-pioglitazone",
  "Metformin_rosiglitazone" = "metformin-rosiglitazone",
  "Metformin_pioglitazone" = "metformin-pioglitazone",
  "Readmission" = "readmitted")

dat <- fread ("diabetic_data.csv", select = Keep %>% unname,
  colClasses = c ( admission_source_id = "factor", diag_1 = "character" ),
  stringsAsFactors = TRUE )

colnames ( dat ) <- Keep %>% names

dat [ Race == "?", Race := NA ]
dat [ Gender == "Unknown/Invalid", Gender := NA ]
dat [ Source %in% c ( 9, 15, 17, 20, 21 ), Source := NA ]
