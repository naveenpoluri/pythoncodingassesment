"""
Purpose: This file contains the user defined functions for reusability.
"""

## Function to read the data from file location
def readFile(spark, file_type,infer_schema,first_row_is_header,delimiter,file_location):
  """
  This Function is to read the data from a file location.
  :param file_type:-            Specifing the file type as csv,avro,parquet
  :param infer_schema:-         Will automatically go through the file and infer the schema of each column.
  :param first_row_is_header:-  Specifing weather first row of a file is header or not.
  :param delimiter:-            Specifing the delimiter if applicable.
  :param file_location:-        Location of file from where the data needs to read.
  :return:                      Dataframe object.
  """
  return spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)


## Function to write the data from file location
def writeFile(dataframe,format,mode,path):
  """
  This Function is to write the data to a filesystem from a Dataframe.
  :param dataframe:-Dataframe object which needs to write.
  :param df_format:-Specifying the format in which the data should be written as csv,avro,parquet.
  :param mode:-     Specifying the mode as overwrite or append.
  :param path:-    Location to store the result data.
  """
  dataframe.write\
  .format(format)\
  .mode(mode)\
  .save(path)
