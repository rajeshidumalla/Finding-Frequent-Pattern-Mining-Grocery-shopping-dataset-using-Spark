# Finding Frequent Pattern Mining (Grocery shopping dataset) using Spark

## Setup
Let's setup Spark on your Colab environment. Run the cell below!

```python
!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
```
Now we authenticate a Google Drive client to download the file we will be processing in our Spark job.

**Make sure to follow the interactive instructions.**
```python
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
```
```python
id='1dhi1F78ssqR8gE6U-AgB80ZW7V_9snX4'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('products.csv')

id='1KZBNEaIyMTcsRV817us6uLZgm-Mii8oU'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('order_products__train.csv')
```
If you executed the cells above, you should be able to see the dataset we will need for this Colab under the "Files" tab on the left panel.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
```
Let's initialize the Spark context.

```python
# create the session
conf = SparkConf().set("spark.ui.port", "4050")

# create the context
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
```
You can easily check the current version and get the link of the web interface. In the Spark UI, you can monitor the progress of your job and debug the performance bottlenecks (if your Colab is running with a **local runtime**).

```python
Spark
```
```python
SparkSession - in-memory
SparkContext
Spark UI
Version
v3.1.2
Master
local[*]
AppName
pyspark-shell
```

If you are running this Colab on the Google hosted runtime, the cell below will create a ngrok tunnel which will allow you to still check the Spark UI.

```python
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
get_ipython().system_raw('./ngrok http 4050 &')
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

If you run successfully the setup stage, you are ready to work with the 3 Million Instacart Orders dataset. In case you want to read more about it, check the official Instacart blog post about it, a concise schema description of the dataset, and the download page.
In this Colab, I will be working only with a small training dataset (~131K orders) to perform fast Frequent Pattern Mining with the FP-Growth algorithm.

```python
products = spark.read.csv('products.csv', header=True, inferSchema=True)orders = spark.read.csv('order_products__train.csv', header=True, 
inferSchema=True)
```

```python
#Let see the Schema of Product Data Frame
products.printSchema()
```

 ```
 root
 |-- product_id: integer (nullable = true)
 |-- product_name: string (nullable = true)
 |-- aisle_id: string (nullable = true)
 |-- department_id: string (nullable = true)
 ```
```pyhton

# Let see the Schema of Orders Data Frame
orders.printSchema()
```

 ```python
 root
 |-- order_id: integer (nullable = true)
 |-- product_id: integer (nullable = true)
 |-- add_to_cart_order: integer (nullable = true)
 |-- reordered: integer (nullable = true)
 ```
Using the Spark Dataframe API to join 'products' and 'orders', so that I will be able to see the product names in each transaction (and not only their ids). Then, group by the orders by 'order_id' to obtain one row per basket (i.e., set of products purchased together by one customer).

```python
transaction = orders.join(products, on='product_id').groupby('order_id').agg(collect_list('product_name').alias('products'))
transaction.take(5)
```
```python
[Row(order_id=1342, products=['Bag of Organic Bananas', 'Seedless Cucumbers', 'Organic Mandarins', 'Organic Strawberries', 'Versatile Stain Remover', 'Pink Lady Apples', 'Chicken Apple Sausage', 'Raw Shrimp']),
 Row(order_id=1591, products=['Cracked Wheat', 'Organic Greek Whole Milk Blended Vanilla Bean Yogurt', 'Navel Oranges', 'Spinach', 'Original Patties (100965) 12 Oz Breakfast', 'Cinnamon Multigrain Cereal', 'Chewy 25% Low Sugar Chocolate Chip Granola', 'Uncured Genoa Salami', 'Natural Vanilla Ice Cream', 'Lemon Yogurt', 'Strawberry Rhubarb Yoghurt', 'Garlic', 'Pure Vanilla Extract', 'Lower Sugar Instant Oatmeal  Variety', 'Organic Bunny Fruit Snacks Berry Patch', 'Buttermilk Waffles', 'Granny Smith Apples', 'Medium Scarlet Raspberries', 'Banana', 'Strawberry Banana Smoothie', 'Green Machine Juice Smoothie', 'Ultra Thin Sliced Provolone Cheese', 'Oven Roasted Turkey Breast', 'Original Turkey Burgers Smoke Flavor Added', 'Original Whole Grain Chips', 'Goldfish Pretzel Baked Snack Crackers', 'Twisted Tropical Tango Organic Juice Drink', 'Goodness Grapeness Organic Juice Drink', 'Nutty Bars', 'Honey Graham Snacks', 'Coconut Dreams Cookies']),
 Row(order_id=4519, products=['Beet Apple Carrot Lemon Ginger Organic Cold Pressed Juice Beverage']),
 Row(order_id=4935, products=['Vodka']),
 Row(order_id=6357, products=['Fresh Mozzarella Ball', 'Grated Parmesan', 'Organic Basil', 'Provolone', 'Gala Apples', 'Panko Bread Crumbs', 'Italian Pasta Sauce Basilico Tomato, Basil & Garlic', 'Globe Eggplant', 'Banana'])]
 ```
 ```python
 
# Let see the Schema of Transaction Data Frame
transaction.printSchema()
```
```python
 root
 |-- order_id: integer (nullable = true)
 |-- products: array (nullable = false)
 |    |-- element: string (containsNull = false)
 ```
 In this Colab I will explore MLlib, Apache Spark's scalable machine learning library. Specifically, I can use its implementation of the FP-Growth algorithm to perform efficiently Frequent Pattern Mining in Spark. I am using the Python example in the Spark documentation, and I am going to train a model with
minSupport=0.01 and minConfidence=0.5

```python
# Importing the Frequent pattern minning library
from pyspark.ml.fpm import FPGrowth

fpGrowth = FPGrowth(itemsCol="products", minSupport=0.01, minConfidence=0.5)
# Creating the FPM model
model = fpGrowth.fit(transaction)
```
Computing how many frequent itemsets and association rules were generated by running FP-growth.

```python
model.freqItemsets.show()
model.associationRules.show()
```
```python
+--------------------+-----+
|               items| freq|
+--------------------+-----+
|            [Banana]|18726|
|[Bag of Organic B...|15480|
|[Organic Strawber...|10894|
|[Organic Strawber...| 3074|
|[Organic Strawber...| 2174|
|[Organic Baby Spi...| 9784|
|[Organic Baby Spi...| 1639|
|[Organic Baby Spi...| 2236|
|[Organic Baby Spi...| 2000|
|       [Large Lemon]| 8135|
|[Large Lemon, Ban...| 2158|
|   [Organic Avocado]| 7409|
|[Organic Avocado,...| 1349|
|[Organic Avocado,...| 1402|
|[Organic Avocado,...| 2216|
|[Organic Hass Avo...| 7293|
|[Organic Hass Avo...| 1539|
|[Organic Hass Avo...| 2420|
|      [Strawberries]| 6494|
|[Strawberries, Ba...| 1948|
+--------------------+-----+
only showing top 20 rows

+----------+----------+----------+----+-------+
|antecedent|consequent|confidence|lift|support|
+----------+----------+----------+----+-------+
+----------+----------+----------+----+-------+
```
Now I am going to retrain the FP-growth model changing only minsupport=0.001 and compute how many frequent itemsets and association rules were generated.

```python
fpGrowth = FPGrowth(itemsCol="products", minSupport=0.001, minConfidence=0.5)
model = fpGrowth.fit(transaction)
model.freqItemsets.show()
model.associationRules.show()
```
```python
+--------------------+-----+
|               items| freq|
+--------------------+-----+
|[Organic Tomato B...|  772|
|[Organic Tomato B...|  175|
|[Organic Tomato B...|  144|
|[Organic Tomato B...|  179|
|[Organic Spinach ...|  475|
|[Whole Milk Ricot...|  347|
| [Medium Salsa Roja]|  275|
|    [Ground Buffalo]|  231|
|       [Tonic Water]|  194|
|[Original Coconut...|  173|
|[Low-Fat Strawber...|  152|
|[Organic SprouTof...|  137|
|            [Banana]|18726|
|[Fruit Punch Spor...|  275|
|[Kitchen Cheese E...|  230|
|[Country White Br...|  194|
|[Soft & Smooth Wh...|  173|
|[Natural Liquid L...|  152|
|[Bag of Organic B...|15480|
|[Organic Large Gr...|  769|
+--------------------+-----+
only showing top 20 rows

+--------------------+--------------------+------------------+------------------+--------------------+
|          antecedent|          consequent|        confidence|              lift|             support|
+--------------------+--------------------+------------------+------------------+--------------------+
|[Organic Kiwi, Or...|[Bag of Organic B...|0.5459770114942529| 4.627719489738336|0.001448071397541327|
|[Organic Raspberr...|[Bag of Organic B...|0.5984251968503937| 5.072272070642333|0.001737685677049...|
|[Organic Broccoli...|[Bag of Organic B...|0.5048231511254019| 4.278897986822536|0.001196564260073623|
|[Organic Unsweete...|[Bag of Organic B...|0.5141065830721003| 4.357584667849303|0.001249914258930...|
|[Yellow Onions, S...|            [Banana]|0.5357142857142857|3.7536332219526702|0.001143214261216...|
|[Organic Cucumber...|[Bag of Organic B...|          0.546875| 4.635330870478036|0.001066999977135...|
|[Organic Navel Or...|[Bag of Organic B...|0.5283018867924528| 4.477904539027839|0.001493799967990...|
|[Organic Raspberr...|[Bag of Organic B...| 0.521099116781158| 4.416853618458589|0.004046978484707604|
|[Organic D'Anjou ...|[Bag of Organic B...|0.5170454545454546|4.3824946411792345|0.001387099970276...|
|[Organic Navel Or...|[Bag of Organic B...|0.5412186379928315| 4.587387356098284|0.001150835689624...|
|[Organic Whole St...|[Bag of Organic B...|0.5314685314685315| 4.504745125675359|0.001158457118033...|
+--------------------+--------------------+------------------+------------------+--------------------+

```
```python

# stoping Spark Environment
sc.stop()
```

To conclude, I can report this results to a supermarket business owner to order more frequent products in order to maintain stable stock in the shelves.
