# Quickstart

This is a basic Ruby example that sets up a PostgreSQL database with Lantern, and uses the `pg` library to interact with the database and perform vector searches.

## Code

```ruby
require 'pg'

# Connect to the database
conn = PG.connect(
  dbname: "ourdb",
  user: "postgres",
  password: "postgres",
  host: "localhost",
  port: "5432"
)

# Create a table
conn.exec("CREATE TABLE small_world (id integer, vector real[3]);")

# Insert data
conn.exec("INSERT INTO small_world (id, vector) VALUES (0, '{0, 0, 0}');")

v1 = [0, 0, 1]
v2 = [0, 1, 1]
v3 = [1, 1, 1]
v4 = [2, 0, 1]

conn.exec_params("INSERT INTO small_world (id, vector) VALUES ($1, $2), ($3, $4), ($5, $6), ($7, $8);", [1, v1, 2, v2, 3, v3, 4, v4])

# Create an index
conn.exec("CREATE INDEX ON small_world USING hnsw (vector);")

# Vector search
conn.exec("SET enable_seqscan = false;")
result = conn.exec("SELECT id, l2sq_dist(vector, ARRAY[0,0,0]) AS dist, vector FROM small_world ORDER BY vector <-> ARRAY[0,0,0] LIMIT 3;")

result.each do |record|
  puts "Vector #{record['vector']} with ID #{record['id']} has a L2-squared distance of #{record['dist']} from [0,0,0]"
end

# Close the connection
conn.close
```
