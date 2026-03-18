# Source Metadata

## Dataset
- Source: Hugging Face
- Dataset Name: `lukebarousse/data_jobs`
- Split: `train`

## Raw File
- Local Path: `data/raw/jobs_raw.parquet`
- S3 Path: `s3://career-intelligence-data-platform/bronze/jobs_raw.parquet`

## Shape
- Rows: 785741
- Columns: 17

## Columns
- job_title_short
- job_title
- job_location
- job_via
- job_schedule_type
- job_work_from_home
- search_location
- job_posted_date
- job_no_degree_mention
- job_health_insurance
- job_country
- salary_rate
- salary_year_avg
- salary_hour_avg
- company_name
- job_skills
- job_type_skills

## Notes
- Raw source preserved without cleaning.
- This file serves as the Bronze layer input for downstream Silver transformations.