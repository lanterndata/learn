# Quickstart

This is a basic Node.js Javascript example that sets up a PostgreSQL database with Lantern, and uses the `pg` library to interact with the database and perform vector searches.

## Connect to the database

Here, we use the `pg` library to interact with Postgres from Node.js. The first step is to obtain a pool object to the postgres instance on our machine. We use the Pool function and specify the user, password, and database name we used from earlier. The host and port parameters also let `pg` know how to connect to postgres, and are the default values if you're running postgres locally.

```typescript
import { Pool } from "pg";

// We use the dbname, user, and password that we specified above
const pool = new Pool({
  database: "ourdb",
  user: "postgres",
  password: "postgres",
  host: "localhost",
  port: 5432,
});
```

## Create a table

Let's create a simple table called `small_world` with three columns: an `id` column of type `INTEGER`, and an array of real numbers (of the type `REAL[]`).

Note that although we specify "3" in the type of vector (by writing `REAL[3]`), this is actually only just syntactic sugar in Postgres, and Postgres will NOT enforce this length! This is done in Postgres by design

```typescript
const createTableQuery =
  "CREATE TABLE small_world (id integer, vector real[3]);";

pool
  .query(createTableQuery)
  .then((res) => console.log(res))
  .catch((err) => console.error(err));
```

## Insert Data

Let's insert some data! We insert a few vectors into our table using the `INSERT` statement. As pointed out earlier, just because we specified `REAL[3]` during the creation of our table does not mean that inserting a vector with length other than 3 will fail here.

```typescript
// Let's insert a vector [0,0,0] with id 0 (note that postgres uses {} braces)
pool
  .query("INSERT INTO small_world (id, vector) VALUES (0, '{0, 0, 0}');")
  .then((res) => console.log(res))
  .catch((err) => console.error(err));

// Now let's insert some more vectors
const v1 = [0, 0, 1];
const v2 = [0, 1, 1];
const v3 = [1, 1, 1];
const v4 = [2, 0, 1];

pool
  .query(
    "INSERT INTO small_world (id, vector) VALUES ($1, $2), ($3, $4), ($5, $6), ($7, $8);",
    [1, v1, 2, v2, 3, v3, 4, v4]
  )
  .then((res) => console.log(res))
  .catch((err) => console.error(err));
```

## Create an Index

In order to perform queries, we need to specify an index. In postgres, an index is a specialized way to store data that speeds up and allows for new ways to interact with your data. The `hnsw` index is from lantern, and it allows for blazingly fast vector search.

Note that we can specify options and parameters to our index creation. For example, we can specify the distance method that is used, which is how we calculate the distance between two vectors when we ultimately search for a vector's nearest neighbors. The default, as used below, is `l2sq`, which is the squared L-2 distance (which is the squared "Euclidean" distance, that you might be familiar with).

```typescript
pool
  .query("CREATE INDEX ON small_world USING lantern_hnsw (vector);")
  .then((res) => console.log(res))
  .catch((err) => console.error(err));

// We can also specify additional parameters to the index like this:
pool
  .query(
    `CREATE INDEX ON small_world USING lantern_hnsw (vector dist_l2sq_ops)
WITH
(M=2, ef_construction=10, ef=4, dim=3);`
  )
  .then((res) => console.log(res))
  .catch((err) => console.error(err));
```

## Vector Search

Now that we have created an index, we can start doing a nearest-neightbor vector search!

However, we first need to set `enable_seqscan` to false. The details of this can be elaborated upon elsewhere, but the gist of it is that we need postgres to use the index that we created above when performing queries (like with `SELECT`). By disabling this postgres runtime variable, we make sure that postgres always uses our index, which allows us to perform vector search using lantern.

We do a search for the 3 nearest neighbors from our table to the vector. Note that this "target" `vector` does not need to be in our index. It is simply the vector from which we compute the distance from to find its nearest neightbors.

Since our index was built to use the L2-squared distance (squared Euclidean distance), and so that is the distance that is used during the search below. Note that `l2sq_dist` found in the first part of the statement only recomputes the distance with the neighbors to show up in the query for our convenience! The actual search, which occurs in the second half of the SQL statement, performs search in the index which was configured to use L2-squared distance when we built it above. Hence, we see this distance being reflected in the print statements below.

```typescript
// We only need to set this at the beginning of a session
pool
  .query("SET enable_seqscan = false;")
  .then((res) => console.log(res))
  .catch((err) => console.error(err));

pool
  .query(
    "SELECT id, l2sq_dist(vector, ARRAY[0,0,0]) AS dist, vector FROM small_world ORDER BY vector <-> ARRAY[0,0,0] LIMIT 3;"
  )
  .then((res) => {
    res.rows.forEach((row) => {
      console.log(
        `Vector ${row.vector} with ID ${row.id} has a L2-squared distance of ${row.dist} from [0,0,0]`
      );
    });
  })
  .catch((err) => console.error(err));
```

## Conclusion

To cleanup, simply close the Postgres connection.

```typescript
# Close the postgres connection
pool.end();
```

That's how you get up and running using Lantern and `pg`! Feel free to explore more of our tutorials and demos.
