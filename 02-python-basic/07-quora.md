# Semantic Search for Quora Questions

Here, we will use Lantern to implement a semantic similarity search for questions. We will be able to search for semantically similar questions to some query question, like "How can I be a better software engineer?".

We will use questions from the Quora dataset from Hugging Face's datasets.

If you are running this in a colab, note that enabling a gpu-enabled runtime will be faster when we compute the embeddings. A cpu runtime will take significantly longer.

A Jupyter notebook with this code can be found [here](https://github.com/lanterndata/examples/blob/main/jupyter-notebooks/Semantic_Search_for_Quora_Questions_Lantern_and_psycopg2.ipynb).

## Installing other Prerequisites

```bash
pip install -qU datasets==2.12.0 sentence-transformers==2.2.2
```

## Gathering and preprocessing Quora data

We will use the Quora dataset from Hugging Face datasets (the `datasets` package we installed above). It contains around 400K pairs of questions from the question-answering site, Quora. Let's use a subset of these pairs

```python
from datasets import load_dataset

dataset = load_dataset('quora', split='train[100000:150000]')

# Some example samples of this dataset
dataset[:4]
```

```
{'questions': [{'id': [165932, 165933],
   'text': ['What should I ask my friend to get from UK to India?',
    'What is the process of getting a surgical residency in UK after completing MBBS from India?']},
  {'id': [123111, 39307],
   'text': ['How can I learn hacking for free?',
    'How can I learn to hack seriously?']},
  {'id': [165934, 165935],
   'text': ['Which is the best website to learn programming language C++?',
    'Which is the best website to learn C++ Programming language for free?']},
  {'id': [165936, 165937],
   'text': ['What did Werner Heisenberg mean when he said, “The first gulp from the glass of natural sciences will turn you into an atheist, but at the bottom of the glass God is waiting for you”?',
    'What did God mean when He said "an eye for an eye "?']}],
 'is_duplicate': [False, True, False, False]}
```

Let's get all the questions into a single list.

```python
questions = []

for record in dataset['questions']:
    questions.extend(record['text'])

# Remove duplicates
questions = list(set(questions))
print('\n'.join(questions[:4]))
print(f"Number of questions: {len(questions)}")
```

```
How do I check if website uses schema.org?
How do I integrate Maven with selenium?
In Batman v Superman, what did Lex Luthor want the painting upside- down?
Number of questions: 88720
```

## Getting our embeddings

By embedding the questions above, the embeddings we obtain are the vectors that we will soon insert into lantern/postgres. Then, by performing a vector search in our database, we will get the "closest" embeddings/vectors to some other embedding/vector, which translates into semantic "similarity." This is the essence of semantic search!

To create our embeddings, we use the `MiniLM-L6` sentence transformer model, from the `sentence-transformers` package we installed. We first need to initialize it.

Note that when we print the details of the model in the last line, we can notice three things:

1. `max_seq_length` is 256, which means that the maximum number of tokens (which is a unit of length, kind of like "words") that can be encoded into a single vector embedding is 256. If we are dealing with more tokens than 256, we must truncate first.

2. `word_embedding_dimension` is 384, which means that each embedding we obtain is a vector with 384 dimensions. We will use this later with lantern

3. `Normalize()` This model has a final normalization step, which means that when measuring distance between embeddings, we can use either cosine similarity or dotproduct similarity metric (they are equivalent in this case, since the vectors are normalized). Hence, we will later use the cosine distance

```python
from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print(f"You are using {device}. This is much slower than using "
          "a CUDA-enabled GPU. If on Colab you can change this by "
          "clicking Runtime > Change runtime type > GPU.")

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
model
```

This is how we go from a question (query) to a vector (embedding).

```python
query = 'How do I become a better software engineer?'

embedded_query = model.encode(query)
embedded_query.shape
```

```
(384,)
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

Now let's create the table that we will use to store these embeddings. We'll call the table `questions`, and it will have a primary key `id`, the actual text content of the question `content`, and the embedding for the question `vector`. Note that we make vector of type real array (`real[]`). We can add a dimension, like `real[384]`, but note that this dimension specified here is just syntactic sugar in postgres, and is not enforced.

```python
# Create the table
cursor = conn.cursor()

create_table_query = "CREATE TABLE questions (id serial PRIMARY key, content text, vector real[]);"

cursor.execute(create_table_query)

conn.commit()
cursor.close()
```

## Inserting embeddings into our database

Now that we have a table created, let's create and insert the embeddings for the questions we prepared earlier.

The majority of the time spent here is computing the embeddings for our questions, using the model we set up before.

```python

from tqdm.auto import tqdm

cursor = conn.cursor()

# The questions we want to embed
# To make this faster, we will only insert the first 1000 questions
Qs = questions[:1000]

for i in tqdm(range(0, len(Qs))):
    content = Qs[i]

    # Create embedding for the question
    vector = [float(x) for x in model.encode(Qs[i])]

    # Insert the content of the question as well as the embedding into our db
    cursor.execute("INSERT INTO questions (content, vector) VALUES (%s, %s);", (content, vector))

conn.commit()
cursor.close()
```

## Creating an Index

Now that we have inserted the embeddings into our database, we need to construct an index in postgres using lantern. This is important because the index will tell allow postgres to use lantern when performing vector search.

Note that we specify cosine distance as the distance metric, as we mentioned earlier. Also, as a good practice, we specify the dimension of the index (although lantern can infer it from the vector's we've already inserted).

```python
cursor = conn.cursor()

cursor.execute("CREATE INDEX ON questions USING hnsw (vector dist_cos_ops) WITH (dim=384);")

conn.commit()
cursor.close()
```

## Performing Similarity Search

Now that we have embedded our questions, we can now perform vector search amongst our questions, and find out semantically similar questions! Recall the example query we had earlier:

```python
query = 'How do I become a better software engineer?'

embedded_query = model.encode(query)
embedded_query = [float(x) for x in embedded_query]
```

Let's do a vector search on our database to find the 5 most semantically similar questions to this query (which we accomplish by finding which questions' embeddings are closest to this query's embedding)

```python
cursor = conn.cursor()

# We only need to set this at the beginning of a session
cursor.execute("SET enable_seqscan = false;")
cursor.execute(f"SELECT content, cos_dist(vector, ARRAY{embedded_query}) AS dist FROM questions ORDER BY vector <-> ARRAY{embedded_query} LIMIT 5;")

record = cursor.fetchone()
while record:
    print(f"{record[0]}  (dist: {record[1]})")
    record = cursor.fetchone()

cursor.close()
```

```
How can I become a good software engineer by myself?  (dist: 0.13243657)
What are the best steps (1-10) to become a excellent programmer?  (dist: 0.33947504)
How do I become a qualified and professional ethical hacker?  (dist: 0.457249)
I am a 2nd year computer science engineering student. Other than studying, what should I be doing (like any extra studies, any internship, etc.)?  (dist: 0.4769982)
What are the requirements to be a programmer?  (dist: 0.48091978)
```

## Conclusion

As we can see, the questions with a lower distance rank "closer," in the semantic sense, to our query question!

And that's how you can implement similarity search for questions using Quora's database.

To cleanup, close the Postgres connection.

```python
# Close the postgres connection
conn.close()
```
