-- Create table for Dataset/students_mental_health_survey_with_burnout_final.csv
-- Run in Supabase SQL Editor.

create table if not exists public.students_mental_health_survey_with_burnout_final (
  "Age" integer,
  "Course" text,
  "Gender" text,
  "CGPA" double precision,
  "Stress_Level" integer,
  "Depression_Score" integer,
  "Anxiety_Score" integer,
  "Sleep_Quality" text,
  "Physical_Activity" text,
  "Diet_Quality" text,
  "Social_Support" text,
  "Relationship_Status" text,
  "Substance_Use" text,
  "Counseling_Service_Use" text,
  "Family_History" text,
  "Chronic_Illness" text,
  "Financial_Stress" integer,
  "Extracurricular_Involvement" text,
  "Semester_Credit_Load" integer,
  "Residence_Type" text,
  "burnout_composite_score" double precision,
  "burnout" integer,
  "burnout_raw_score" double precision,
  "method1_tertiles" integer,
  "method2_wider" integer,
  "method3_very_wide" integer,
  "method4_manual" integer,
  "method5_manual2" integer,
  "method6_kmeans" integer
);