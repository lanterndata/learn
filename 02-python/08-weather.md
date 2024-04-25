# Weather Prediction

Here, we will use vector search on time series data to predict the weather in the next hour. Our data will come from the [Jena Climate dataset](https://www.kaggle.com/stytch16/jena-climate-2009-2016). This dataset is made up of quantities such as air temperature, atmospheric pressure, humidity, etc. that were recorded every 10 minutes over the course of several years.

We can treat each column as a feature vector identified by the time stamp associated with them. These will constitute the vectors that we perform our search with, and the nearest neighbors will give us an idea of what the weather will be for that hour. Although this is a very simple embedding process, we want to demonstrate how effective even a simple vector search with Lantern can be.

A Jupyter notebook with this code can be found [here](https://github.com/lanterndata/examples/blob/main/jupyter-notebooks/Weather_Prediction_with_Vector_Search_Lantern_and_psycopg2.ipynb).

## Loading Data and Preprocessing

Let's first import our dependencies

```python
import matplotlib as mpl
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List
import itertools
import os

mpl.rcParams['figure.figsize'] = (20, 16)
mpl.rcParams['axes.grid'] = False
```

### Load dataset

```python
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
```

### Preprocessing

Let's load the hourly data into a datafame, and then separate it into data that we will insert into Lantern ("train data") and data that we will use to query ("test data")

Note that the number of features is 14, and so our vectors will be of dimension 14.

```python
original_data_for_insert = pd.read_csv(csv_path)
original_data_for_insert = original_data_for_insert[5::6]

original_data_for_insert['Date Time'] = pd.to_datetime(original_data_for_insert['Date Time'], format='%d.%m.%Y %H:%M:%S')

n = len(original_data_for_insert)
train_data = original_data_for_insert[:int(n*0.9)]
test_data = original_data_for_insert[int(n*0.9):]

train_data.head()
```

## Create Postgres Table

Now let's set up `psycopg2` with postgres, and enable the lantern extension

```python
import psycopg2

# We use the dbname, user, and password that we specified above
conn = psycopg2.connect(
    dbname="ourdb",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
```

Now let's create the table that we will use to store these weather data vectors. We'll call the table `weather`, and it will have a primary key `id`, a text string `datetime` for the date-time string, and the weather embedding `vector`. Note that we make `vector` of type real array (`real[]`). We can add a dimension, like `real[14]`, but note that this dimension specified here is just syntactic sugar in postgres, and is not enforced.

```python
# Create the table
cursor = conn.cursor()

TABLE_NAME = "weather_data"

create_table_query = f"CREATE TABLE {TABLE_NAME} (id serial PRIMARY key, datetime text, vector real[]);"

cursor.execute(create_table_query)

conn.commit()
cursor.close()
```

## Inserting embeddings into our database

Now that we have a table created, let's insert our weather embeddings that we prepared earlier.

```python
from tqdm.auto import tqdm

cursor = conn.cursor()

values_tolist = train_data.values.tolist()

for i in tqdm(range(0, len(values_tolist))):
  row = values_tolist[i]
  datetimestring = str(row[0])
  vector = [float(x) for x in row[1:]]
  # Insert this vector and datetime into our db
  cursor.execute(f"INSERT INTO {TABLE_NAME} (datetime, vector) VALUES (%s, %s);", (datetimestring, vector))

conn.commit()
cursor.close()
```

## Creating an Index

Now that we have inserted the embeddings into our database, we need to construct an index in postgres using lantern. This is important because the index will tell allow postgres to use lantern when performing vector search.

Note that we specify cosine distance as the distance metric, because these weather embeddings are not normalized, and the cosine distance is concerned more about capturing the angle of deviation and is less concerned about magnitude.

Also, as a good practice, we specify the dimension of the index (although lantern can infer it from the vectors we've already inserted).

```python
cursor = conn.cursor()

cursor.execute(f"CREATE INDEX ON {TABLE_NAME} USING lantern_hnsw (vector dist_cos_ops) WITH (dim=14);")

conn.commit()
cursor.close()
```

## Predictions With Vector Search

Now that we have embedded our weather vectors, we can now perform vector search against other queries. Let's get the top candidate in the search for a range of queries from our test data we prepared earlier.

We'll prepare this query data first:

```python
# Prepare data that we will query
query_dates = []
query_data = []
for row in test_data.values.tolist():
    query_dates.append(str(row[0]))
    query_data.append(row[1:])
```

Now we can perform queries for each of the queries that we've prepared. We will get the top candidate for each query (including its vector column), specified by `LIMIT 1` in the query (this means we will get the 1 nearest neighbor returned by lantern to the query)

```python
query_results = []

cursor = conn.cursor()

# We only need to set this at the beginning of a session
cursor.execute("SET enable_seqscan = false;")

for query_raw in tqdm(query_data):
  query_vector = str([float(x) for x in query_raw])
  cursor.execute(f"SELECT vector, datetime FROM {TABLE_NAME} ORDER BY vector <-> ARRAY{query_vector} LIMIT 1;")

  record = cursor.fetchone()
  query_results.append(record)

cursor.close()
```

Now let's write a function to help us get the predicted and true values for a particular feature (one of our weather embedding dimensions). We use our vectors to find the most similar vector in the database and then reading the hour after that.

We also write helper functions to help us visualize our results.

```python
def get_predictions(feature):

    true_values = []
    predicted_values = []

    for test_date, qr in zip(query_dates, query_results):
        similar_date = qr[1]
        hour_from_original = datetime.strptime(str(test_date), '%Y-%m-%d %H:%M:%S') + timedelta(hours=1)
        hour_from_similar = datetime.strptime(similar_date, '%Y-%m-%d %H:%M:%S') + timedelta(hours=1)

        original_temperature = original_data_for_insert.loc[original_data_for_insert['Date Time'] == hour_from_original][feature].tolist()
        similar_temperature = original_data_for_insert.loc[original_data_for_insert['Date Time'] == hour_from_similar][feature].tolist()

        if original_temperature and similar_temperature:
            true_values.append(original_temperature[0])
            predicted_values.append(similar_temperature[0])
    return true_values, predicted_values



def plot_results(predicted_values: List, true_values: List):
    x_list = range(0, len(predicted_values))
    plt.plot(x_list[:200], predicted_values[:200], label='forecast')
    plt.plot(x_list[:200], true_values[:200], label='true')
    plt.legend()
    plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error

def print_results(true_values: List, predicted_values: List):
    print(f'MSE: {mean_squared_error(true_values, predicted_values)}')
    print(f'RMSE: {mean_squared_error(true_values, predicted_values, squared=False)}')
    print(f'MAE: {mean_absolute_error(true_values, predicted_values)}')
```

## Results

Let's plot the predicted and true values for all 14 features/dimensions in our data:

```python
for feature in original_data_for_insert.columns[1:]:
    print(f'Analyzing predictions for {feature}')
    true_values, predicted_values = get_predictions(feature)
    plot_results(true_values, predicted_values)
    print_results(true_values, predicted_values)
```

## Conclusion

From the plots above we can see that this vector search method is able to predict some features pretty accurately (VPdef, VPmax, rh(%) etc.), some pretty accurately (H20C, rho), and other features are not predicted that well (wd, max.vv, wv).

And that's how powerful even such a straightforward search can be!

To cleanup, close the Postgres connection.

```python
# Close the postgres connection
conn.close()
```
