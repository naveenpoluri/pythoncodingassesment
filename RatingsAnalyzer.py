from util.Utilities import readFile, writeFile
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
import sys

outputPath = sys.argv[1]

spark = SparkSession.builder.master("local[*]").appName("RatingsAnalyzer").getOrCreate()

moviesRawDf = readFile(spark, "csv", "True", "True", ",", "../ml-latest-small/movies.csv")
ratingsRawDf = readFile(spark, "csv", "True", "True", ",", "../ml-latest-small/ratings.csv")


#Validate, clean & Transform the data

#Remove imvalid data

moviesCleanDf = moviesRawDf.filter(F.col("movieId").isNotNull() & F.col("title").isNotNull() & F.col("genres").isNotNull())
ratingsCleanDf = ratingsRawDf.filter(F.col("userId").isNotNull() & F.col("movieId").isNotNull() & F.col("rating").isNotNull() & F.col("timestamp").isNotNull())

#Transform the data as per the requirement

#Extract date from timestamp
ratingsWithDate = ratingsCleanDf.withColumn("date", F.date_format(F.col("timestamp").cast(dataType=T.TimestampType()), "yyyy-MM-dd"))\
    .withColumn("year", F.date_format(F.col("date"), "yyyy")).drop("timestamp")

#Genres wise Movies
moviesGenresWise = moviesRawDf.withColumn("genresList", F.split(F.col("genres"), "\\|"))\
    .withColumn("genres", F.explode(F.col("genresList"))).drop("genresList") \
    .withColumn("year", F.regexp_extract(F.col("title"), '\((\d\d\d\d)\)', 1)) \
    .withColumn("decadeFloor", (F.floor(F.col("year")/10)*10).cast("int")) \
    .withColumn("decade", F.concat(F.col("decadeFloor").cast("string"),  F.lit("-"),  (F.col("decadeFloor") + F.lit(10)))).drop("decadeFloor")

moviesWithRatings = moviesGenresWise.join(ratingsWithDate, on = "movieId", how = "inner").cache()


#Popular movies
#Here popularity is measured as the movies which are rated more than 200 times and have average rating above 4

popularMovies = moviesWithRatings.groupBy(F.col("title"))\
    .agg(F.round(F.avg(F.col("rating")), 2).alias("avg_rating"), F.count(F.lit(1)).alias("numberOfRatings")) \
    .filter((F.col("avg_rating") > F.lit(4)) & (F.col("numberOfRatings") > F.lit(200))) \
    .orderBy(F.desc("numberOfRatings"), F.desc("avg_rating")).cache()

popularMovies.show(truncate=0)
writeFile(popularMovies,"parquet", "overwrite", outputPath + "/PopularMovies")


#Top 5 worst movies
#Here worst movies are measured as the movies which are rated more than 10 times and have average rating below 1.5
worstMovies = moviesWithRatings.groupBy(F.col("title"))\
    .agg(F.round(F.avg(F.col("rating")), 2).alias("avg_rating"), F.count(F.lit(1)).alias("numberOfRatings")) \
    .filter((F.col("avg_rating") <= F.lit(1.5)) & (F.col("numberOfRatings") > F.lit(10))) \
    .orderBy(F.col("avg_rating"), F.desc("numberOfRatings")).limit(5).cache()

worstMovies.show(truncate=0)
writeFile(worstMovies,"parquet", "overwrite", outputPath + "/WorstMovies")


#List of top 10 users who rated most of the movies
topTenUsers = moviesWithRatings.groupBy("userId").agg(F.count(F.lit(1)).alias("numberOfRatings")) \
    .orderBy(F.desc("numberOfRatings")).limit(10).cache()

topTenUsers.show(truncate=0)
writeFile(popularMovies,"parquet", "overwrite", outputPath + "/topTenRatedGenres")


#Top movies in each decade
aggregatedDecadeRating = moviesWithRatings.groupBy(F.col("decade"), F.col("title")) \
    .agg(F.round(F.avg(F.col("rating")), 2).alias("avg_rating"), F.count(F.lit(1)).alias("numberOfRatings")) \
    .filter((F.col("avg_rating") > F.lit(4)) & (F.col("numberOfRatings") > F.lit(100)))
windowDecade = Window.partitionBy("decade").orderBy(F.desc(F.col("numberOfRatings")), F.desc(F.col("avg_rating")))
topMovieInEachDecade = aggregatedDecadeRating.withColumn("rank", F.dense_rank().over(windowDecade)) \
    .filter(F.col("rank") == F.lit(1)).drop("rank").cache()
topMovieInEachDecade.show(truncate=0)
writeFile(topMovieInEachDecade,"parquet", "overwrite", outputPath + "/TopMovieInEachDecade")


#Remove invalid Genres data
cleanedGenres = moviesWithRatings.filter(F.col("genres") != F.lit("(no genres listed)"))

#Top 10 Genres which are rated most of the times and their average rating
topTenRatedGenres = cleanedGenres.groupBy("genres")\
    .agg(F.round(F.avg(F.col("rating")), 2).alias("avg_rating"), F.count(F.lit(1)).alias("numberOfRatings")) \
    .orderBy(F.desc("numberOfRatings")).limit(10).cache()

topTenRatedGenres.show(truncate=0)
writeFile(topTenRatedGenres,"parquet", "overwrite", outputPath + "/TopTenRatedGenres")


#Top Movies in each Genres
aggregatedGenresRating = cleanedGenres.groupBy(F.col("genres"), F.col("title")) \
    .agg(F.round(F.avg(F.col("rating")), 2).alias("avg_rating"), F.count(F.lit(1)).alias("numberOfRatings")) \
    .filter((F.col("avg_rating") > F.lit(4)) & (F.col("numberOfRatings") > F.lit(50)))
window = Window.partitionBy("genres").orderBy(F.desc(F.col("numberOfRatings")), F.desc(F.col("avg_rating")))
topMovieInEachGenres = aggregatedGenresRating.withColumn("rank", F.dense_rank().over(window)) \
    .filter(F.col("rank") == F.lit(1)).drop("rank").cache()

topMovieInEachGenres.show(truncate=0)
writeFile(topMovieInEachGenres,"parquet", "overwrite", outputPath + "/TopMovieInEachGenres")


#Top 10 Years in which most of the movies were released
mostReleases = moviesGenresWise.groupBy(F.col("year")).agg(F.count(F.lit(1)).alias("numberOfReleases"))\
    .orderBy(F.desc(F.col("numberOfReleases"))).limit(10).cache()
mostReleases.show(truncate=0)
writeFile(mostReleases,"parquet", "overwrite", outputPath + "/MostReleases")


#Movies which are rated most of the time in the released year
moviesRatedInSameYear = moviesGenresWise.join(ratingsWithDate, ["movieId", "year"], "inner") \
    .groupBy(F.col("title")) \
    .agg(F.round(F.avg(F.col("rating")), 2).alias("avg_rating"), F.count(F.lit(1)).alias("numberOfRatings")) \
    .orderBy(F.desc(F.col("numberOfRatings"))).limit(10).cache()

moviesRatedInSameYear.show(truncate=0)
writeFile(moviesRatedInSameYear,"parquet", "overwrite", outputPath + "/MoviesRatedInSameYear")
