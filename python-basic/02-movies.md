# Movie Recommender System

Here, we will use Lantern to implement a movie recommendation system. We will be able to search for movies similar to ones that a user has enjoyed so that we can show relevant recommendations.

We will use movie data from the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/).

A Jupyter notebook with this code can be found [here](https://github.com/lanterndata/examples/blob/main/jupyter-notebooks/Movie_Recommender_System_Lantern_and_psycopg2.ipynb).

## Gathering Movie Data

As we mentioned earlier, we will use the MovieLens 1M dataset. This dataset contains over 1 million anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users.

We use the following files:

- `movies.dat`: Contains movie information.
- `movie_vectors.txt`: Contains movie vectors that can be imported to Milvus easily.

```bash
# Download movie information
wget -P movie_recommender https://paddlerec.bj.bcebos.com/aistudio/movies.dat --no-check-certificate
# Download movie vectors
wget -P movie_recommender https://paddlerec.bj.bcebos.com/aistudio/movie_vectors.txt --no-check-certificate
```

## Create Postgres Table

Now that we have our movie data, let's set up `psycopg2` with postgres, and enable the lantern extension

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

We will make a table called `movies`, and it will have 4 columns: an id, the title of the film, a string denoting the genres of the film, and a vector that will be the embedding for the movie.

```python
# Create the table
cursor = conn.cursor()

TABLE_NAME = "movies"

create_table_query = f"CREATE TABLE {TABLE_NAME} (id integer, title text, genres text, vector real[]);"

cursor.execute(create_table_query)

conn.commit()
cursor.close()
```

## Inserting Movie Data

Now that we have our table, let's insert our movie data into our database.

Let's first get our movie embeddings/vectors from `movie_vectors.txt` and set up a dictionary to easily get the embedding for a movie from its id.

Note that the dimensionality of these embeddings is 32

```python
import json
import codecs

def get_vectors():
    with codecs.open("movie_recommender/movie_vectors.txt", "r", encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    ids = [int(line.split(":")[0]) for line in lines]
    embeddings = []
    for line in lines:
        line = line.strip().split(":")[1][1:-1]
        str_nums = line.split(",")
        emb = [float(x) for x in str_nums]
        embeddings.append(emb)
    return ids, embeddings

ids, embeddings = get_vectors()

# make it easier to look up an embedding from a movie id
id_to_embedding = dict(zip(ids, embeddings))

print(f"Dimensionality: {len(embeddings[0])}")
```

Now we can process the data in `movies.dat` to get the metadata of each movie (id, title, and genre), and get the vector embedding from above. We'll insert each movie into our database:

```python
cursor = conn.cursor()

def process_movie(lines):
    for line in lines:
        if len(line.strip()) == 0:
            continue
        tmp = line.strip().split("::")
        movie_id = int(tmp[0])
        title = tmp[1]
        genres = tmp[2]

        vector = id_to_embedding[movie_id]
        cursor.execute(f"INSERT INTO {TABLE_NAME} (id, title, genres, vector) VALUES (%s, %s, %s, %s);", (movie_id, title, genres, vector))


with codecs.open("movie_recommender/movies.dat", "r",encoding='utf-8',errors='ignore') as f:
        lines = f.readlines()
        process_movie(lines)


conn.commit()
cursor.close()
```

## Creating an Index

Now that we have inserted the embeddings into our database, we need to construct an index in postgres using lantern. This is important because the index will tell allow postgres to use lantern when performing vector search.

Note that we specify L2-squared distance as the distance metric. Also, as a good practice, we specify the dimension of vectors, 32 as mentioned above, in the index (although lantern can infer it from the vector's we've already inserted).

```python
cursor = conn.cursor()

cursor.execute(f"CREATE INDEX ON {TABLE_NAME} USING hnsw (vector dist_l2sq_ops) WITH (dim=32);")

conn.commit()
cursor.close()
```

## Getting Recommendations With Vector Search

Let's pick a movie below and assume that a user has really liked this movie. Let's find movies that are similar to this movie so we can recommend movies that they'll also like.

```python
query_movie_id = ids[69]
query_vector = str(id_to_embedding[query_movie_id])

cursor = conn.cursor()

cursor.execute(f"SELECT * FROM {TABLE_NAME} where id={query_movie_id};")

results = cursor.fetchall()
#print(results[0])
query_movie_title = results[0][1]
query_movie_genre = results[0][2]


print(f"The user really liked the movie: {query_movie_title}, genre: {query_movie_genre}")
print("Let's find similar movies they'll also like...")


conn.commit()
cursor.close()
```

```
The user really liked the movie: From Dusk Till Dawn (1996), genre: Action|Comedy|Crime|Horror|Thriller
Let's find similar movies they'll also like...
```

To find similar movies, we will perform a vector search using lantern. We pull the 10 most similar movies (movies whose embeddings are closet to the embedding of our query movie), and specify that we don't want the same movie as the query movie.

```python
cursor = conn.cursor()

cursor.execute("SET enable_seqscan = false;")
cursor.execute(f"SELECT id, title, genres FROM {TABLE_NAME} WHERE id != {query_movie_id} ORDER BY vector <-> ARRAY{query_vector} LIMIT 10;")

results = cursor.fetchall()

print(f"Recommendations if you liked '{query_movie_title}':\n")
for i,r in enumerate(results):
  print(f"#{i+1}. {r[1]}, Genre: {r[2]}")

conn.commit()
cursor.close()
```

```
Recommendations if you liked 'From Dusk Till Dawn (1996)':

#1. Strange Days (1995), Genre: Action|Crime|Sci-Fi
#2. Westworld (1973), Genre: Action|Sci-Fi|Thriller|Western
#3. Tank Girl (1995), Genre: Action|Comedy|Musical|Sci-Fi
#4. Thirteenth Floor, The (1999), Genre: Drama|Sci-Fi|Thriller
#5. Alien Nation (1988), Genre: Crime|Drama|Sci-Fi
#6. Puppet Master (1989), Genre: Horror|Sci-Fi|Thriller
#7. Village of the Damned (1960), Genre: Horror|Sci-Fi|Thriller
#8. Dog Day Afternoon (1975), Genre: Comedy|Crime|Drama
#9. Young Guns (1988), Genre: Action|Comedy|Western
#10. Sneakers (1992), Genre: Crime|Drama|Sci-Fi
```

## Conclusion

To cleanup, close the Postgres connection.

```python
# Close the postgres connection
conn.close()
```

As we can see, we get movies that are similar to the original movie that we can now recommend to the user. A lot of these share the same genre as the original movie.

And that's how you can implement a movie recommendation system using Lantern.
