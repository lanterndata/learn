# Quickstart

This is a basic Python example that sets up a PostgreSQL database with Lantern, and uses the `psycopg2` library to interact with the database and perform vector searches.

A Jupyter notebook with this code can be found [here](https://github.com/lanterndata/examples/blob/main/jupyter-notebooks/Lantern_Quickstart_with_psycopg2.ipynb).

## Connect to the database

Here, we use the `psycopg2` library to interact with Postgres from Python. The first step is to obtain a `conn` object to the postgres instance on our machine. We use the `connect` function and specify the user, password, and database name we used from earlier. The `host` and `port` parameters also let `psycopg2` know how to connect to postgres, and are the default values if you're running postgres locally.

NOTE: if at any time you encounter an error while executing a query, you should call `conn.rollback()` to restore the database to the most recent transaction.

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

## Creating a Table

Let's create a simple table called `small_world` with three columns: an `id` column of type `INTEGER`, and an array of real numbers (of the type `REAL[]`).

Note that although we specify "3" in the type of vector (by writing REAL[3]), this is actually only just syntactic sugar in postgres, and postgres will NOT enforce this length! This is done in postgres by design.

```python
cursor = conn.cursor()

create_table_query = "CREATE TABLE small_world (id integer, vector real[3]);"

cursor.execute(create_table_query)

conn.commit()
cursor.close()
```

## Inserting Data

Let's insert some data! We insert a few vectors into our table using the `INSERT` statement. As pointed out earlier, just because we specified `REAL[3]` during the creation of our table does not mean that inserting a vector with length other than 3 will fail here.

```python

cursor = conn.cursor()

# Let's insert a vector [0,0,0] with id 0 (note that postgres uses {} braces)
cursor.execute("INSERT INTO small_world (id, vector) VALUES (0, '{0, 0, 0}');")

# Now let's insert some more vectors
v1 = [0, 0, 1]
v2 = [0, 1, 1]
v3 = [1, 1, 1]
v4 = [2, 0, 1]

cursor.execute("INSERT INTO small_world (id, vector) VALUES (%s, %s), (%s, %s), (%s, %s), (%s, %s);", (1, v1, 2, v2, 3, v3, 4, v4))

conn.commit()
cursor.close()
```

## Creating an Index

In order to perform queries, we need to specify an index. In postgres, an index is a specialized way to store data that speeds up and allows for new ways to interact with your data. The `hnsw` index is from lantern, and it allows for blazingly fast vector search.

Note that we can specify options and parameters to our index creation. For example, we can specify the distance method that is used, which is how we calculate the distance between two vectors when we ultimately search for a vector's nearest neighbors. The default, as used below, is `l2sq`, which is the squared L-2 distance (which is the squared "Euclidean" distance, that you might be familiar with).

```python
cursor = conn.cursor()

cursor.execute("CREATE INDEX ON small_world USING hnsw (vector);")

# We can also specify additional parameters to the index like this:
"""CREATE INDEX ON small_world USING hnsw (vector dist_l2sq_ops)
WITH
(M=2, ef_construction=10, ef=4, dim=3);"""

conn.commit()
cursor.close()
```

## Vector Search

Now that we have created an index, we can start doing a nearest-neightbor vector search!

However, we first need to set `enable_seqscan` to false. The details of this can be elaborated upon elsewhere, but the gist of it is that we need postgres to use the index that we created above when performing queries (like with `SELECT`). By disabling this postgres runtime variable, we make sure that postgres always uses our index, which allows us to perform vector search using lantern.

Then, we do a search for the 3 nearest neighbors from our table to the vector [0,0,0]. Note that this "target" vector ([0,0,0]) does not need to be in our index. It is simply the vector from which we compute the distance from to find its nearest neightbors.

Since our index was built to use the L2-squared distance (squared Euclidean distance), and so that is the distance that is used during the search below. Note that `l2sq_dist` found in the first part of the statement only recomputes the distance with the neighbors to show up in the query for our convenience! The actual search, which occurs in the second half of the SQL statement, performs search in the index which was configured to use L2-squared distance when we built it above. Hence, we see this distance being reflected in the print statements below.

```python
cursor = conn.cursor()

# We only need to set this at the beginning of a session
cursor.execute("SET enable_seqscan = false;")
cursor.execute("SELECT id, l2sq_dist(vector, ARRAY[0,0,0]) AS dist, vector FROM small_world ORDER BY vector <-> ARRAY[0,0,0] LIMIT 3;")

record = cursor.fetchone()
while record:
    print(f"Vector {record[2]} with ID {record[0]} has a L2-squared distance of {record[1]} from [0,0,0]")
    record = cursor.fetchone()

cursor.close()
```

## Conclusion

To cleanup, simply close the Postgres connection.

```python
# Close the postgres connection
conn.close()
```

That's how you get up and running using Lantern and `psycopg2`! Feel free to explore more of our tutorials and demos.
