# Databricks notebook source
# Get the caller's team name, which was passed in as a parameter by the notebook that called me.
# We'll strip special characters from this field, then use it to define unique path names (local and dbfs) as well as a unique database name.
# This prevents name collisions in a multi-team flight school.

dbutils.widgets.text("team_name", "");

# COMMAND ----------

# import to enable removal on non-alphanumeric characters (re stands for regular expressions)
import re

# Get the email address entered by the user on the calling notebook
team_name = dbutils.widgets.get("team_name")
print(f"Data entered in team_name field: {team_name}")

# Strip out special characters and make it lower case
clean_team_name = re.sub('[^A-Za-z0-9]+', '', team_name).lower();
print(f"Team Name with special characters removed: {clean_team_name}");

# Construct the unique path to be used to store files on the local file system
local_data_path = f"flight-school-{clean_team_name}/"
print(f"Path to be used for Local Files: {local_data_path}")

# Construct the unique path to be used to store files on the DBFS file system
dbfs_data_path = f"flight-school-{clean_team_name}/"
print(f"Path to be used for DBFS Files: {dbfs_data_path}")

# Construct the unique database name
database_name = f"flight_school_{clean_team_name}"
print(f"Database Name: {database_name}")

# COMMAND ----------

spark.sql(f"DROP DATABASE IF EXISTS {database_name} CASCADE")

spark.sql(f"CREATE DATABASE {database_name}")

# COMMAND ----------

spark.sql(f"USE {database_name}")

# COMMAND ----------

# Enables running shell commands from Python
# I need to do this, rather than just the %sh magic command, because I want to pass in python variables

import subprocess

# COMMAND ----------

# Delete local directories that may be present from a previous run

process = subprocess.Popen(['rm', '-f', '-r', clean_team_name],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

stdout.decode('utf-8'), stderr.decode('utf-8')

# COMMAND ----------

# Create local directories used in the workshop

process = subprocess.Popen(['mkdir', '-p', clean_team_name],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

stdout.decode('utf-8'), stderr.decode('utf-8')

# COMMAND ----------

# Download Initial CSV file used in the workshop
local_data_path = f"{clean_team_name}/"

process = subprocess.Popen(['wget', '-P', local_data_path, 'https://www.dropbox.com/s/h2ytt76rtsi91op/assignment_1_ingest_v4.csv'],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

stdout.decode('utf-8'), stderr.decode('utf-8')

# COMMAND ----------

# Copy the downloaded data to DBFS

dbutils.fs.rm(f"dbfs:/FileStore/flight/{clean_team_name}/assignment_1_ingest.csv")
dbutils.fs.cp(f"file:/databricks/driver/{clean_team_name}/assignment_1_ingest_v4.csv", f"dbfs:/FileStore/flight/{team_name}/assignment_1_ingest.csv")

# COMMAND ----------

# Download Backfill CSV file used in the workshop
local_data_path = f"{clean_team_name}/"

process = subprocess.Popen(['wget', '-P', local_data_path, 'https://www.dropbox.com/s/2le0apie8bo2o6w/assignment_1_backfill_v4.csv'],
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

stdout.decode('utf-8'), stderr.decode('utf-8')

# COMMAND ----------

# Copy the downloaded backfill data to DBFS

dbutils.fs.rm(f"dbfs:/FileStore/flight/{clean_team_name}/assignment_1_backfill.csv")
dbutils.fs.cp(f"file:/databricks/driver/{clean_team_name}/assignment_1_backfill_v4.csv", f"dbfs:/FileStore/flight/{team_name}/assignment_1_backfill.csv")

# COMMAND ----------

# Return to the caller, passing the variables needed for file paths and database

response = local_data_path + " " + dbfs_data_path + " " + database_name

dbutils.notebook.exit(response)