# Databricks notebook source
# MAGIC %md
# MAGIC <img src="https://github.com/billkellett/flight-school-resources/blob/master/images/databricks icon.png?raw=true" width=100/> 
# MAGIC # Flight School Assignment 1
# MAGIC 
# MAGIC ## Build a demo
# MAGIC 
# MAGIC __*Welcome to Flight School!*__
# MAGIC 
# MAGIC In this assignment, you'll be working with Delta Lake in batch mode.  You will ingest data, cleanse it, and aggregate it, using patterns that are common in a Bronze-Silver-Gold data lake paradigm.
# MAGIC 
# MAGIC In this notebook, we have structured a demo for you.  Some cells are functional, while others require you to enter code.
# MAGIC 
# MAGIC __NOTE:__ Use a cluster running Databricks 7.3 ML or higher.

# COMMAND ----------

# This creates the "team_name" field displayed at the top of the notebook.

dbutils.widgets.text("team_name", "Enter your team's name");

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://github.com/billkellett/flight-school-resources/blob/master/images/stop.png?raw=true" width=100/> 
# MAGIC ### Before you begin...
# MAGIC 
# MAGIC There are two important bits of housekeeping that are required in order to run this workshop:
# MAGIC 
# MAGIC - __First, enter your Team Name__ in the "team_name" field at the top of the notebook.  We use this when naming resources, to ensure that your database name and file paths do not conflict with other Flight School teams. 
# MAGIC - __Second, run the cell below__ in order to set up your data sources and Delta database for this assignment.
# MAGIC   - To run a cell, click in the cell, then click the triangle on the upper right of the cell and choose "Run Cell" - alternatively, you can use SHIFT+ENTER.

# COMMAND ----------

# Note that we have factored out the setup processing into a different notebook, which we call here.
# As a flight school student, you will probably want to look at the setup notebook.  
# Even though you'll want to look at it, we separated it out in order to demonstrate a best practice... 
# ... you can use this technique to keep your demos shorter, and avoid boring your audience with housekeeping.
# In addition, you can save demo time by running this initial setup command before you begin your demo.

# This cell should run in a few minutes or less

team_name = dbutils.widgets.get("team_name")

setup_responses = dbutils.notebook.run("./includes/flight_school_assignment_1_setup", 0, {"team_name": team_name}).split()

local_data_path = setup_responses[0]
dbfs_data_path = setup_responses[1]
database_name = setup_responses[2]

print(f"Path to be used for Local Files: {local_data_path}")
print(f"Path to be used for DBFS Files: {dbfs_data_path}")
print(f"Database Name: {database_name}")

# COMMAND ----------

# Let's set the default database name so we don't have to specify it on every query

spark.sql(f"USE {database_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bringing data into a dataframe
# MAGIC 
# MAGIC The use case we will be demonstrating here illustrates the "Bronze-Silver-Gold" paradigm which is a best practice for data lakes.
# MAGIC 
# MAGIC - We ingest data as soon as we can into the lake, even though we know it may need cleansing or enrichment.  This gives us a baseline of the freshest possible data for exploration.  We call this the __Bronze__ version of the data.
# MAGIC 
# MAGIC - We then cleanse and enrich the Bronze data, creating a "single version of truth" that we call the __Silver__ version.
# MAGIC 
# MAGIC - From the Silver data, we can generate many __Gold__ versions of the data.  Gold versions are typically project-specific, and typically filter, aggregate, and re-format Silver data to make it easy to use in specific projects.
# MAGIC 
# MAGIC We'll read the raw data into a __Dataframe__.  The dataframe is a key structure in Apache Spark.  It is an in-memory data structure in a rows-and-columns format that is very similar to a relational database table.  In fact, we'll be creating SQL Views against the dataframes so that we can manipulate them using standard SQL.

# COMMAND ----------

# Read the downloaded historical data into a dataframe
# This is MegaCorp data regarding power plant device performance.  It pre-dates our new IOT effort, but we want to save this data and use it in queries.

dataPath = f"dbfs:/FileStore/flight/{team_name}/assignment_1_ingest.csv"

df = spark.read.option("header","true").option("inferSchema","true").csv(dataPath)
#
# TO DO - Read the data from dataPath into a dataframe
# NOTES:
# - The input data is in CSV format
# - The input data contains a header row with column names
# - You can let Spark infer the schema
#

#
# END OF TO DO
#

display(df)

# COMMAND ----------

# This API call is totally awesome, because it lets us use spark.sql() or run %SQL cells!

df.createOrReplaceTempView("tmp")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT -- TO DO... Select all the distinct values from the device_operational_status column
# MAGIC distinct device_operational_status
# MAGIC FROM tmp

# COMMAND ----------

# Read the downloaded backfill data into a dataframe
# This is some backfill data that we'll need to merge into the main historical data.  

dataPath = f"dbfs:/FileStore/flight/{team_name}/assignment_1_backfill.csv"

df_backfill = spark.read.option("header","true").option("inferSchema","true").csv(dataPath)


display(df_backfill)

# COMMAND ----------

# Create a temporary view on the dataframes to enable SQL

df.createOrReplaceTempView("historical_bronze_vw")
df_backfill.createOrReplaceTempView("historical_bronze_backfill_vw")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Bronze Historical Tables
# MAGIC 
# MAGIC __NOTE TO FLIGHT SCHOOL STUDENTS...__ We're adding this graphic-heavy markdown cell just to demonstrate the kinds of markdown you might want to use in your own demos.  It's nice to put these graphics into your demo notebook, because then you don't have to flip back and forth between your notebook and your PowerPoint slides during a demo.
# MAGIC 
# MAGIC #### Databricks' Delta Lake is the world's most advanced data lake technology.  
# MAGIC 
# MAGIC Delta Lake brings __*Performance*__ and __*Reliability*__ to Data Lakes
# MAGIC 
# MAGIC Why did Delta Lake have to be invented?  Let's take a look...
# MAGIC 
# MAGIC <img src="https://github.com/billkellett/flight-school-resources/blob/master/images/projects_failing.png?raw=true" width=1000/>
# MAGIC 
# MAGIC As the graphic above shows, Big Data Lake projects have a very high failure rate.  In fact, Gartner Group estimates that 85% of these projects fail (see https://www.infoworld.com/article/3393467/4-reasons-big-data-projects-failand-4-ways-to-succeed.html ).  *Why* is the failure rate so high?
# MAGIC 
# MAGIC <img src="https://github.com/billkellett/flight-school-resources/blob/master/images/projects_failing_reasons.png?raw=true" width=1000/>
# MAGIC 
# MAGIC The graphic above shows the main __*reliability*__ issues with data lakes.  Unlike relational databases, typical data lakes are not capable of transactional (ACID) behavior.  This leads to a number of reliability issues:
# MAGIC 
# MAGIC - When a job fails, incomplete work is not rolled back, as it would be in a relational database.  Data may be left in an inconsistent state.  This issue is extremely difficult to deal with in production.
# MAGIC 
# MAGIC - Data lakes typically cannot enforce schema.  This is often touted as a "feature" called "schema-on-read," because it allows flexibility at data ingest time.  However, when downstream jobs fail trying to read corrupt data, we have a very difficult recovery problem.  It is often difficult just to find the source application that caused the problem... which makes fixing the problem even harder!
# MAGIC 
# MAGIC - Relational databases allow multiple concurrent users, and ensure that each user gets a consistent view of data.  Half-completed transactions never show up in the result sets of other concurrent users.  This is not true in a typical data lake.  Therefore, it is almost impossible to have a concurrent mix of read jobs and write jobs.  This becomes an even bigger problem with streaming data, because streams typically don't pause to let other jobs run!
# MAGIC 
# MAGIC Next, let's look at the key __*performance issues*__ with data lakes...
# MAGIC 
# MAGIC <img src="https://github.com/billkellett/flight-school-resources/blob/master/images/projects_failing_reasons_1.png?raw=true" width=1000/>
# MAGIC 
# MAGIC - We have already noted that data lakes cannot provide a consistent view of data to concurrent users.  This is a reliability problem, but it is also a __*performance*__ problem because if we must run jobs one at a time, our production time window becomes extremely limited.
# MAGIC 
# MAGIC - Most data lake engineers have come face-to-face with the "small-file problem."  Data is typically ingested into a data lake in batches.  Each batch typically becomes a separate physical file in a directory that defines a table in the lake.  Over time, the number of physical files can grow to be very large.  When this happens, performance suffers because opening and closing these files is a time-consuming operation.  
# MAGIC 
# MAGIC - Experienced relational database architects may be surprised to learn that Big Data usually cannot be indexed in the same way as relational databases.  The indexes become too large to be manageable and performant.  Instead, we "partition" data by putting it into sub-directories.  Each partition can represent a column (or a composite set of columns) in the table.  This lets us avoid scanning the entire data set... *if* our queries are based on the partition column.  However, in the real world, analysts are running a wide range of queries which may or may not be based on the partition column.  In these scenarios, there is no benefit to partitioning.  In addition, partitioning breaks down if we choose a partition column with extremely high cardinality.
# MAGIC 
# MAGIC - Data lakes typically live in cloud storage (e.g., S3 on AWS, ADLS on Azure), and these storage devices are quite slow compared to SSD disk drives.  Most data lakes have no capability to cache data on faster devices, and this fact has a major impact on performance.
# MAGIC 
# MAGIC __*Delta Lake was built to solve these reliability and performance problems.*__  First, let's consider how Delta Lake addresses *reliability* issues...
# MAGIC 
# MAGIC <img src="https://github.com/billkellett/flight-school-resources/blob/master/images/delta_reliability.png?raw=true" width=1000/>
# MAGIC 
# MAGIC Note the Key Features in the graphic above.  We'll be diving into all of these capabilities as we go through the Workshop:
# MAGIC 
# MAGIC - __ACID Transactions:__ Delta Lake ACID compliance ensures that half-completed transactions are never persisted in the Lake, and concurrent users never see other users' in-flight transactions.
# MAGIC 
# MAGIC - __Mutations:__ Experienced relational database architects may be surprised to learn that most data lakes do not support updates and deletes.  These lakes concern themselves only with data ingest, which makes error correction and backfill very difficult.  In contrast, Delta Lake provides full support for Inserts, Updates, and Deletes.
# MAGIC 
# MAGIC - __Schema Enforcement:__ Delta Lake provides full support for schema enforcement at write time, greatly increasing data reliability.
# MAGIC 
# MAGIC - __Unified Batch and Streaming:__ Streaming data is becoming an essential capability for all enterprises.  We'll see how Delta Lake supports both batch and streaming modes, and in fact blurs the line between them, enabling architects to design systems that use both batch and streaming capabilities simultaneously.
# MAGIC 
# MAGIC - __Time Travel:__ unlike most data lakes, Delta Lake enables queries of data *as it existed* at a specific point in time.  This has important ramifications for reliability, error recovery, and synchronization with other systems, as we shall see later in this Workshop.
# MAGIC 
# MAGIC We have seen how Delta Lake enhances reliability.  Next, let's see how Delta Lake optimizes __*performance*__...
# MAGIC 
# MAGIC <img src="https://github.com/billkellett/flight-school-resources/blob/master/images/delta_performance.png?raw=true" width=1000/>
# MAGIC 
# MAGIC Again, we'll be diving into all these capabilities throughout the Workshop.  We'll be concentrating especially on features that are only available in Databricks' distribution of Delta Lake...
# MAGIC 
# MAGIC - __Compaction:__ Delta Lake provides sophisticated capabilities to solve the "small-file problem" by compacting small files into larger units.
# MAGIC 
# MAGIC - __Caching:__ Delta Lake transparently caches data on the SSD drives of worker nodes in a Spark cluster, greatly improving performance.
# MAGIC 
# MAGIC - __Data Skipping:__ this Delta Lake feature goes far beyond the limits of mere partitioning.
# MAGIC 
# MAGIC - __Z-Ordering:__ this is a brilliant alternative to traditional indexing, and further enhances Delta Lake performance.
# MAGIC 
# MAGIC Now that we have introduced the value proposition of Delta Lake, let's get a deeper understanding of the overall "Data Lake" concept.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a Delta Lake table for the main bronze table
# MAGIC 
# MAGIC DROP TABLE IF EXISTS sensor_readings_historical_bronze;
# MAGIC 
# MAGIC CREATE TABLE sensor_readings_historical_bronze USING DELTA 
# MAGIC as select * from historical_bronze_vw
# MAGIC --
# MAGIC -- TO DO... create a DELTA LAKE table from historical_bronze_vw.  Use the CREATE TABLE ... AS... syntax
# MAGIC --

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's take a peek at our new bronze table
# MAGIC 
# MAGIC SELECT * FROM sensor_readings_historical_bronze

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's count the records in the Bronze table
# MAGIC 
# MAGIC SELECT COUNT(*) FROM sensor_readings_historical_bronze

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's make a query that shows a meaningful graphical view of the table
# MAGIC -- How many rows exist for each operational status?
# MAGIC -- Experiment with different graphical views... be creative!
# MAGIC 
# MAGIC SELECT device_operational_status,count(*) as DeviceOpStatusCnt
# MAGIC FROM sensor_readings_historical_bronze
# MAGIC GROUP BY device_operational_status
# MAGIC ORDER BY device_operational_status

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Now let's make a query that accepts run-time parameters.
# MAGIC -- NOTE that we have set default values so that a default query will return results on this data
# MAGIC 
# MAGIC CREATE WIDGET DROPDOWN PARAM_END_SECOND 
# MAGIC   DEFAULT '57'
# MAGIC   CHOICES SELECT DISTINCT SECOND(reading_time) AS end_second FROM sensor_readings_historical_bronze ORDER BY end_second ASC;
# MAGIC CREATE WIDGET DROPDOWN PARAM_START_SECOND 
# MAGIC   DEFAULT '54'
# MAGIC   CHOICES SELECT DISTINCT SECOND(reading_time) AS start_second FROM sensor_readings_historical_bronze ORDER BY start_second ASC;
# MAGIC CREATE WIDGET DROPDOWN PARAM_MINUTE 
# MAGIC   DEFAULT '18'
# MAGIC   CHOICES SELECT DISTINCT MINUTE(reading_time) AS minute FROM sensor_readings_historical_bronze ORDER BY minute ASC;
# MAGIC CREATE WIDGET DROPDOWN PARAM_HOUR 
# MAGIC   DEFAULT '10'
# MAGIC   CHOICES SELECT DISTINCT HOUR(reading_time) AS hour FROM sensor_readings_historical_bronze ORDER BY hour ASC;
# MAGIC CREATE WIDGET DROPDOWN PARAM_DAY 
# MAGIC   DEFAULT '23'
# MAGIC   CHOICES SELECT DISTINCT DAY(reading_time) AS day FROM sensor_readings_historical_bronze ORDER BY day ASC;
# MAGIC CREATE WIDGET DROPDOWN PARAM_MONTH 
# MAGIC   DEFAULT '2'
# MAGIC   CHOICES SELECT DISTINCT MONTH(reading_time) AS month FROM sensor_readings_historical_bronze ORDER BY month ASC;
# MAGIC CREATE WIDGET DROPDOWN PARAM_YEAR 
# MAGIC   DEFAULT '2015'
# MAGIC   CHOICES SELECT DISTINCT YEAR(reading_time) AS year FROM sensor_readings_historical_bronze ORDER BY year ASC;
# MAGIC CREATE WIDGET DROPDOWN PARAM_DEVICE_ID 
# MAGIC   DEFAULT '7G007R'
# MAGIC   CHOICES SELECT DISTINCT device_id FROM sensor_readings_historical_bronze ORDER BY device_id ASC;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's make a query that shows another meaningful graphical view of the table
# MAGIC -- We'll parameterize this query so a Business Analyst can examine fine-grained device performance issues
# MAGIC -- Experiment with different graphical views
# MAGIC 
# MAGIC SELECT 
# MAGIC   reading_time,
# MAGIC   reading_1,
# MAGIC   reading_2,
# MAGIC   reading_3
# MAGIC FROM sensor_readings_historical_bronze
# MAGIC WHERE 
# MAGIC   device_id = getArgument("PARAM_DEVICE_ID") -- use the getArgument function to grab values from the appropriate widget we defined above
# MAGIC   AND
# MAGIC   YEAR(reading_time) = getArgument("PARAM_YEAR")-- use the getArgument function to grab values from the appropriate widget we defined above
# MAGIC   AND
# MAGIC   MONTH(reading_time) = getArgument("PARAM_MONTH")-- use the getArgument function to grab values from the appropriate widget we defined above
# MAGIC   AND
# MAGIC   DAY(reading_time) = getArgument("PARAM_DAY")-- use the getArgument function to grab values from the appropriate widget we defined above
# MAGIC   AND 
# MAGIC   HOUR(reading_time) = getArgument("PARAM_HOUR")-- use the getArgument function to grab values from the appropriate widget we defined above
# MAGIC   AND
# MAGIC   MINUTE(reading_time) = getArgument("PARAM_MINUTE")-- use the getArgument function to grab values from the appropriate widget we defined above
# MAGIC   AND 
# MAGIC   SECOND(reading_time) BETWEEN getArgument("PARAM_START_SECOND")-- use the getArgument function to grab values from the appropriate widget we defined above 
# MAGIC     AND getArgument("PARAM_END_SECOND")-- use the getArgument function to grab values from the appropriate widget we defined above
# MAGIC ORDER BY reading_time ASC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's clean up that messy collection of widgets!
# MAGIC 
# MAGIC REMOVE WIDGET PARAM_DEVICE_ID;
# MAGIC REMOVE WIDGET PARAM_YEAR;
# MAGIC REMOVE WIDGET PARAM_MONTH;
# MAGIC REMOVE WIDGET PARAM_DAY;
# MAGIC REMOVE WIDGET PARAM_HOUR;
# MAGIC REMOVE WIDGET PARAM_MINUTE;
# MAGIC REMOVE WIDGET PARAM_START_SECOND;
# MAGIC REMOVE WIDGET PARAM_END_SECOND;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's try to understand our backfill data
# MAGIC 
# MAGIC DESCRIBE TABLE historical_bronze_backfill_vw

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's take a peek at the backfill data
# MAGIC 
# MAGIC SELECT * FROM historical_bronze_backfill_vw limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's count the records in the backfill data
# MAGIC 
# MAGIC SELECT COUNT(*) FROM historical_bronze_backfill_vw

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Silver table
# MAGIC 
# MAGIC MegaCorp has informed us that the Bronze historical data has a few issues.  Let's deal with them and create a clean Silver table.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's create a Silver table.  We'll start with the Bronze data, then make several improvements
# MAGIC 
# MAGIC DROP TABLE IF EXISTS sensor_readings_historical_silver;
# MAGIC 
# MAGIC CREATE TABLE sensor_readings_historical_silver USING DELTA 
# MAGIC as select * from historical_bronze_vw;
# MAGIC --
# MAGIC -- TO DO... create a DELTA LAKE Silver table.  Use historical_bronze_vw as a starting point.  Use the CREATE TABLE ... AS... syntax
# MAGIC --

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's take a peek at our new Silver table
# MAGIC 
# MAGIC SELECT * FROM sensor_readings_historical_silver
# MAGIC ORDER BY reading_time ASC

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from sensor_readings_historical_silver s, historical_bronze_backfill_vw v
# MAGIC where s.id=v.id

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's merge in the Bronze backfill data
# MAGIC -- MERGE INTO is one of the most important differentiators for Delta Lake
# MAGIC -- The entire backfill batch will be treated as an atomic transaction,
# MAGIC -- and we can do both inserts and updates within a single batch.
# MAGIC 
# MAGIC MERGE INTO sensor_readings_historical_silver AS silver -- TO DO... "AS" what?
# MAGIC USING historical_bronze_backfill_vw AS bronze-- TO DO... "AS" what?
# MAGIC ON  silver.id=bronze.id
# MAGIC WHEN MATCHED THEN update set *
# MAGIC WHEN NOT MATCHED THEN insert *;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify that the upserts worked correctly.
# MAGIC -- Newly inserted records have dates of 2015-02-21 (and id value beginning with 'ZZZ')
# MAGIC -- Updated records have id's in the backfill data that do NOT begin with 'ZZZ'.  
# MAGIC -- Check a few of these, and make sure that a tiny value was added to reading_1.
# MAGIC -- In order to check, you might try something similar to...
# MAGIC -- %sql
# MAGIC select a.id, a.reading_1,b.reading_1
# MAGIC from sensor_readings_historical_silver a
# MAGIC inner join sensor_readings_historical_bronze b
# MAGIC on a.id = b.id
# MAGIC where a.reading_1 <> b.reading_1;
# MAGIC 
# MAGIC --SELECT * FROM sensor_readings_historical_silver
# MAGIC --ORDER BY reading_time ASC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- MegaCorp just informed us of some dirty data.  Occasionally they would receive garbled data.
# MAGIC -- In those cases, they would put 999.99 in the readings.
# MAGIC -- Let's find these records
# MAGIC 
# MAGIC SELECT * 
# MAGIC FROM sensor_readings_historical_silver
# MAGIC WHERE reading_1 = 999.99

# COMMAND ----------

# MAGIC %sql
# MAGIC -- We want to fix these bogus readings.  Here's the idea...
# MAGIC -- - Use a SQL window function to order the readings by time within each device
# MAGIC -- - Whenever there is a 999.99 reading, replace it with the AVERAGE of the PREVIOUS and FOLLOWING readings.
# MAGIC -- HINTS:
# MAGIC -- Window functions use an "OVER" clause... OVER (PARTITION BY ... ORDER BY )
# MAGIC -- Look up the doc for SQL functions LAG() and LEAD()
# MAGIC 
# MAGIC -- We'll create a table of these interpolated readings, then later we'll merge it into the Silver table.
# MAGIC 
# MAGIC DROP TABLE IF EXISTS sensor_readings_historical_interpolations;
# MAGIC 
# MAGIC CREATE TABLE sensor_readings_historical_interpolations AS (
# MAGIC   WITH lags_and_leads AS (
# MAGIC     SELECT
# MAGIC       id, 
# MAGIC       reading_time,
# MAGIC       device_type,
# MAGIC       device_id,
# MAGIC       device_operational_status,
# MAGIC       reading_1,
# MAGIC       LAG(reading_1, 1, 0)  OVER (PARTITION BY device_id ORDER BY reading_time ASC, id ASC) AS reading_1_lag,
# MAGIC       LEAD(reading_1, 1, 0) OVER (PARTITION BY device_id ORDER BY reading_time ASC, id ASC) AS reading_1_lead,
# MAGIC       reading_2,
# MAGIC       LAG(reading_2, 1, 0)  OVER (PARTITION BY device_id ORDER BY reading_time ASC, id ASC) AS reading_2_lag,
# MAGIC       LEAD(reading_2, 1, 0) OVER (PARTITION BY device_id ORDER BY reading_time ASC, id ASC) AS reading_2_lead,
# MAGIC       reading_3,
# MAGIC       LAG(reading_3, 1, 0)  OVER (PARTITION BY device_id ORDER BY reading_time ASC, id ASC) AS reading_3_lag,
# MAGIC       LEAD(reading_3, 1, 0) OVER (PARTITION BY device_id ORDER BY reading_time ASC, id ASC) AS reading_3_lead
# MAGIC     FROM sensor_readings_historical_silver
# MAGIC   )
# MAGIC   SELECT 
# MAGIC     id,
# MAGIC     reading_time,
# MAGIC     device_type,
# MAGIC     device_id,
# MAGIC     device_operational_status,
# MAGIC     ((reading_1_lag + reading_1_lead) / 2) AS reading_1,
# MAGIC     ((reading_2_lag + reading_2_lead) / 2) AS reading_2,
# MAGIC     ((reading_3_lag + reading_3_lead) / 2) AS reading_3
# MAGIC   FROM lags_and_leads
# MAGIC   WHERE reading_1 = 999.99
# MAGIC   ORDER BY id ASC
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's examine our interpolations to make sure they are correct
# MAGIC 
# MAGIC SELECT * FROM sensor_readings_historical_interpolations

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's see how many interpolations we have.  There should be 367 rows.
# MAGIC 
# MAGIC SELECT COUNT(*) FROM sensor_readings_historical_interpolations

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Now use MERGE INTO to update the historical table
# MAGIC 
# MAGIC -- TO DO... you've already worked with MERGE INTO, so you should be able to write this one from scratch!
# MAGIC 
# MAGIC merge into sensor_readings_historical_silver as S
# MAGIC using sensor_readings_historical_interpolations as I
# MAGIC on S.id=I.id
# MAGIC when matched then update set *
# MAGIC when not matched then insert *;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Now make sure we got rid of all the bogus readings.
# MAGIC -- Gee, this is fast.  Why?  What feature in Delta Lake is making this so speedy?
# MAGIC 
# MAGIC SELECT * 
# MAGIC FROM sensor_readings_historical_silver
# MAGIC WHERE reading_1 = 999.99
# MAGIC 
# MAGIC 
# MAGIC -- Becauase Delta Lake stores statistics, e.g. min,max of column values in the Delta transaction log, querying using column value predicates are super fast!

# COMMAND ----------

# MAGIC %md
# MAGIC Now we've lost visibility into which readings were initially faulty... use Time Travel to recover this information.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- List all the versions of the table that are available to us
# MAGIC 
# MAGIC DESCRIBE HISTORY sensor_readings_historical_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Ah, version 1 should have the 999.99 values
# MAGIC 
# MAGIC SELECT * 
# MAGIC FROM sensor_readings_historical_silver -- TO DO... how can you specify the time travel?
# MAGIC version as of 1
# MAGIC WHERE reading_1 = 999.99

# COMMAND ----------

# MAGIC %md
# MAGIC How could we __*partition*__ the Silver data for faster access?  Suggest a method, and be ready to discuss its pros and cons.  Feel free to imagine the query patterns MegaCorp will be using.
# MAGIC 
# MAGIC To get started, let's examine how partitioning works.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- DESCRIBE EXTENDED will give us some partition information, and will also tell us the location of the data
# MAGIC -- Hmmm, looks like we are not partitioned.  What does that mean?
# MAGIC 
# MAGIC DESCRIBE EXTENDED sensor_readings_historical_silver

# COMMAND ----------

# Let's look at the physical file layout in a non-partitioned table

dbutils.fs.ls(f"dbfs:/user/hive/warehouse/{database_name}.db/sensor_readings_historical_silver")

# As you can see, the data is just broken into a set of files, without regard to the meaning of the data

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Maybe Date would be a good way to partition the data
# MAGIC 
# MAGIC SELECT DISTINCT DATE(reading_time) FROM sensor_readings_historical_silver
# MAGIC 
# MAGIC -- Hmmm, there are only three dates, so maybe that's not the best choice.

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct(device_id) from sensor_readings_historical_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's create a Silver table partitioned by Device. 
# MAGIC -- Create a new table, so we can compare new and old
# MAGIC 
# MAGIC DROP TABLE IF EXISTS sensor_readings_historical_silver_by_device;
# MAGIC 
# MAGIC CREATE TABLE sensor_readings_historical_silver_by_device 
# MAGIC partitioned by (device_id)
# MAGIC as select * from sensor_readings_historical_silver;
# MAGIC -- Sangbae : without using Delta, Delta table is default. Databricks Runtime 8.0 changes the default format to delta to make it simpler to create a Delta table. When you create a table using SQL commands
# MAGIC 
# MAGIC 
# MAGIC -- TO DO... Create a new Delta Lake table, partitioned by device_id

# COMMAND ----------

# MAGIC %sql
# MAGIC -- We can see partition information
# MAGIC 
# MAGIC DESCRIBE EXTENDED sensor_readings_historical_silver_by_device;

# COMMAND ----------

# Now we have subdirectories for each device, with physical files inside them
# Will that speed up queries?

dbutils.fs.ls(f"dbfs:/user/hive/warehouse/{database_name}.db/sensor_readings_historical_silver_by_device")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's create a Silver table partitioned by BOTH Date AND Hour. 
# MAGIC -- Note that Delta cannot partition by expressions, so I have to explicitly create the partition columns
# MAGIC -- HINT: Use the DATE() function to extract date from a timestamp, and use the HOUR() function to extract hour from a timestamp
# MAGIC 
# MAGIC DROP TABLE IF EXISTS sensor_readings_historical_silver_by_hour;
# MAGIC 
# MAGIC CREATE TABLE sensor_readings_historical_silver_by_hour 
# MAGIC USING DELTA
# MAGIC partitioned by (date_reading_time, hour_reading_time)
# MAGIC as select  *, DATE(reading_time) as date_reading_time, HOUR(reading_time) as hour_reading_time  from sensor_readings_historical_silver;
# MAGIC -- TO DO... create the table with correct partitioning

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select  *, DATE(reading_time) as dateOf_reading_time, HOUR(reading_time) as hourOf_reading_time  from sensor_readings_historical_silver limit 10;

# COMMAND ----------

# NOTE how the hour directories are nested within the date directories

dbutils.fs.ls(f"dbfs:/user/hive/warehouse/{database_name}.db/sensor_readings_historical_silver_by_hour/date_reading_time=2015-02-24/")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's create a Silver table partitioned by Date AND Hour AND Minute. 
# MAGIC -- Note that Delta cannot partition by expressions, so I have to explicitly create the partition columns
# MAGIC 
# MAGIC DROP TABLE IF EXISTS sensor_readings_historical_silver_by_hour_and_minute;
# MAGIC 
# MAGIC CREATE TABLE sensor_readings_historical_silver_by_hour_and_minute 
# MAGIC USING DELTA
# MAGIC partitioned by (date_reading_time,hour_reading_time,minute_reading_time)
# MAGIC as select *,  DATE(reading_time) as date_reading_time, HOUR(reading_time) as hour_reading_time, MINUTE(reading_time) as minute_reading_time from sensor_readings_historical_silver;
# MAGIC -- TO DO... create the table

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's take a peek at our minute-partitioned table
# MAGIC 
# MAGIC SELECT * 
# MAGIC FROM sensor_readings_historical_silver_by_hour_and_minute
# MAGIC LIMIT 100

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Now let's take some timings that compare our partitioned Silver tables against the unpartitioned Silver table
# MAGIC -- Here is an example "baseline" query against the unpartitioned Silver table
# MAGIC -- (run these queries several times to get a rough average)
# MAGIC 
# MAGIC SELECT * 
# MAGIC FROM sensor_readings_historical_silver
# MAGIC WHERE 
# MAGIC   DATE(reading_time) = '2015-02-24' AND
# MAGIC   HOUR(reading_time) = '14' AND
# MAGIC   MINUTE(reading_time) = '2'

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Now compare the time for the same query against a partitioned table
# MAGIC -- Think and discuss... Did both data skipping and partitioning play a part here?  How could you combine data skipping and partitioning to make queries even more performant?
# MAGIC 
# MAGIC SELECT * 
# MAGIC FROM sensor_readings_historical_silver_by_hour_and_minute
# MAGIC WHERE 
# MAGIC   date_reading_time = '2015-02-24' AND
# MAGIC   hour_reading_time = '14' AND
# MAGIC   minute_reading_time = '2'

# COMMAND ----------

# MAGIC %md
# MAGIC Imagine one or more Gold tables that Analysts and Data Scientists might want.  Create a few examples, and be ready to discuss your choices.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Here is an example of a Gold table
# MAGIC 
# MAGIC -- TO DO... create one or more Gold tables.  Focus on aggregations and filters that users might appreciate.
# MAGIC 
# MAGIC drop table if exists sensor_readings_historical_gold_by_reading_1_daily_avg_min_max;
# MAGIC 
# MAGIC create table sensor_readings_historical_gold_by_daily_avg_min_max
# MAGIC using delta
# MAGIC as select device_type,date_reading_time,avg(reading_1) as reading_1_daily_avg,min(reading_1) as reading1_daily_min, max(reading_1) as reading1_daily_max
# MAGIC from sensor_readings_historical_silver_by_hour_and_minute
# MAGIC group by device_type,date_reading_time;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sensor_readings_historical_gold_by_daily_avg_min_max
# MAGIC order by device_type, date_reading_time

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC drop table if exists sensor_readings_historical_gold_by_Rectifier_In_FailureHighIdle_AvgMinMax;
# MAGIC create table sensor_readings_historical_gold_by_Rectifier_In_FailureHighIdle_AvgMinMax using delta as
# MAGIC select
# MAGIC   device_type,
# MAGIC   device_operational_status,
# MAGIC   avg(reading_1) as reading_1_daily_avg,
# MAGIC   min(reading_1) as reading1_daily_min,
# MAGIC   max(reading_1) as reading1_daily_max
# MAGIC from
# MAGIC   sensor_readings_historical_silver_by_hour_and_minute
# MAGIC where
# MAGIC   device_type = 'RECTIFIER'
# MAGIC   and device_operational_status in ('FAILURE', 'HIGH', 'IDLE')
# MAGIC group by
# MAGIC   device_type,
# MAGIC   device_operational_status
# MAGIC order by
# MAGIC   device_type,
# MAGIC   device_operational_status; 

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from sensor_readings_historical_gold_by_Rectifier_In_FailureHighIdle_AvgMinMax;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Take it to the next level!
# MAGIC 
# MAGIC We haven't been able to touch on every feature of Delta Lake.  If time permits, demonstrate another important capability... like Schema Enforcement, or ACID behavior, or Z-Ordering, etc.

# COMMAND ----------

import os, binascii, random, datetime
from pyspark.sql.functions import * 
from pyspark.sql.types import *

from delta.tables import *


# Function to stop all streaming queries 
def stop_all_streams():
  # Stop all the streams
  print("Stopping all streams")
  for s in spark.streams.active:
    s.stop()
  print("Stopped all streams")
  print("Deleting checkpoints")  
  dbutils.fs.rm("/Users/ivan.tang@databricks.com/chkpt/", True)
  print("Deleted checkpoints")


def generate_hex(char_count):
  return binascii.b2a_hex(os.urandom(char_count)).decode('ASCII')

@udf(returnType=StringType())
def generate_id():
  id_arr = [generate_hex(8),generate_hex(4),generate_hex(4),generate_hex(12)]
  return '-'.join(id_arr)

device_types = ['RECTIFIER','TRANSFORMER']
device_operational_statuses = ["DESCENDING",
                              "RESETTING",
                              "IDLE",
                              "HIGH",
                              "NOMINAL",
                              "RISING",
                              "FAILURE"]

device_ids = ["8H008R",
              "8H008T",
              "9I009R",
              "9I009T",
              "4D004T",
              "6F006R"]

@udf(returnType=StringType())
def random_col_value(choices):
    str(random.choice(choices))
    
@udf(returnType=StringType())
def random_device_type():
  return random_col_value(device_types)

@udf(returnType=StringType())
def random_device_id():
  return random_col_value(device_ids)

@udf(returnType=StringType())
def random_operational_status():
  return random_col_value(device_operational_statuses)

@udf(returnType=DoubleType())
def random_sensor_reading1():
  return float(random.uniform(20, 30))

@udf(returnType=DoubleType())
def random_sensor_reading2():
  return float(random.uniform(-0.36, 0.06))

@udf(returnType=DoubleType())
def random_sensor_reading3():
  return float(random.uniform(-0.17, 0.05))
    
def random_checkpoint_dir(): 
  return "/dbfs/user/ivan.tang@databricks.com/chkpt/%s" % str(random.randint(0, 10000))


    
# Using Rate format for generating mock sensor readings

def generate_and_append_readings(table_format, table_path):
  stream_data = spark.readStream.format("rate").option("rowsPerSecond", 10).load() \
    .withColumn("id", generate_id()) \
    .withColumn("reading_time", col('timestamp')) \
    .withColumn("device_type", random_device_type()) \
    .withColumn("device_id", random_device_id()) \
    .withColumn("device_operational_status", random_operational_status()) \
    .withColumn("reading_1", random_sensor_reading1()) \
    .withColumn("reading_2", random_sensor_reading2()) \
    .withColumn("reading_3", random_sensor_reading3()) 
  
  query = stream_data.writeStream \
    .format(table_format) \
    .option("checkpointLocation", random_checkpoint_dir()) \
    .trigger(processingTime = "10 seconds") \
    .start(table_path)

# COMMAND ----------

table_path = "/user/hive/warehouse/flight_school_teamapj.db/sensor_readings_historical_bronze"
generate_and_append_readings("delta", table_path)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <br><br><br><br>
# MAGIC 
# MAGIC #### 🌱 Schema Enforcement is enabled out-of-the-box. 
# MAGIC 
# MAGIC ![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Fitang%2Fnc4Q9y_JFp.png?alt=media&token=b7bc8ef8-1e74-4ba6-818d-7d15eb84ead4)
# MAGIC 
# MAGIC * Notice that the above cell fails with `schema mismatch` error message. 
# MAGIC 
# MAGIC * This is due to the default behavior of preventing writes with a different schema corrupting the existing Delta table as there are two new columns, `timestamp` and `value`.
# MAGIC 
# MAGIC * Note that you can alllow schema evolution with `spark.sql("SET spark.databricks.delta.schema.autoMerge.enabled = true")`. 
# MAGIC 
# MAGIC * Setting `spark.databricks.delta.schema.autoMerge.enabled`=`true` will update the existing schema with additonal columns, `timestamp` and `value`.

# COMMAND ----------

spark.sql("SET spark.databricks.delta.schema.autoMerge.enabled = true")

# COMMAND ----------

deltaTable = DeltaTable.forPath(spark, table_path)
display(deltaTable.history())

# COMMAND ----------

# MAGIC %md
# MAGIC <br><br><br><br>
# MAGIC 
# MAGIC #### 🕰 Time Travel
# MAGIC 
# MAGIC ![](https://www.nme.com/wp-content/uploads/2019/05/RYM972-696x442.jpg)

# COMMAND ----------

spark.sql("RESTORE TABLE delta.`%s` VERSION AS OF 0" %(table_path)).show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 🎉  **Notice that `timestamp` and `value` columns are now gone! We have successfully rollback back to version 0!**

# COMMAND ----------

stop_all_streams()

# COMMAND ----------

# MAGIC %md 
# MAGIC <br><br><br><br>
# MAGIC 
# MAGIC #### 🚅 Optimization (bin-packing)

# COMMAND ----------

# MAGIC %sh 
# MAGIC ls -lh /dbfs/user/hive/warehouse/flight_school_teamapj.db/sensor_readings_historical_bronze

# COMMAND ----------

spark.sql("OPTIMIZE delta.`%s`" %(table_path)).show()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ⚠️ **Be very careful when running `VACUUM` with retain 0 hours. This will remove tombstoned data files, effectively rendering it impossible to time travel.**

# COMMAND ----------

spark.sql("set spark.databricks.delta.retentionDurationCheck.enabled = false").show()
spark.sql("VACUUM delta.`%s` RETAIN 0 HOURS" %(table_path)).show()

# COMMAND ----------

ls -lh /dbfs/user/hive/warehouse/flight_school_teamapj.db/sensor_readings_historical_bronze

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 🎉 Notice that our we have mitigate the "many small files" problem and this will likely improve query performance because you will have lesser file scans!

# COMMAND ----------

