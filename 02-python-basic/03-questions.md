# Question Answering Engine

In this notebook, we will use Lantern to implement a question answering engine. From a given database of pre-existing questions and their answers, we will be able to fetch answers from completely different questions.

For the question-answer data that will make up our knowledge base, we will be using the [InsuranceQA Corpus](https://github.com/shuzi/insuranceQA).

We make the core functionality very simple using the python library Towhee, so that you can start hacking your own question answering engine.

A Jupyter notebook with this code can be found [here](https://github.com/lanterndata/examples/blob/main/jupyter-notebooks/Question_Answering_Engine_Lantern_and_psycopg2.ipynb).

## Installing other Prerequisites

```bash
python -m pip install -q towhee towhee.models
```

## Gathering Data

Let's download a subset of the [InsuranceQA corpus](https://github.com/shuzi/insuranceQA) we mentioned above. It contains 1000 pairs of questions and answers related to insurance.

The data contains three columns: an `id`, a `question`, and its corresponding `answer`.

```bash
curl -L https://github.com/towhee-io/examples/releases/download/data/question_answer.csv -O
```

```python
import pandas as pd

df = pd.read_csv('question_answer.csv')
df.head()
```

```table
| ID     | Question                                   | Answer                                     |
|--------|--------------------------------------------|--------------------------------------------|
| 0      | Is Disability Insurance Required By Law?   | Not generally. There are five states that requ...
| 1      | Can Creditors Take Life Insurance After... | If the person who passed away was the one with... |
| 2      | Does Travelers Insurance Have Renters Ins...	| One of the insurance carriers I represent is T...|
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

Now let's create the table that we will use to store the data we will reference against. We'll call the table `questions_answers`, and it will have 4 columns. The first 3 correspond to the columns in our dataset we downloaded above: an id, the text content of the `question`, and its corresponding `answer`. Lastly, we will store the embedding we compute for each question in the column `vector`. Note that we make `vector` of type real array (`real[]`). We can add a dimension, like `real[768]`, but note that this dimension specified here is just syntactic sugar in postgres, and is not enforced.

```python
# Create the table
cursor = conn.cursor()

TABLE_NAME = "questions_answers"
create_table_query = f"CREATE TABLE {TABLE_NAME} (id integer, question text, answer text, vector real[]);"

cursor.execute(create_table_query)

conn.commit()
cursor.close()
```

## Computing Embeddings and Inserting Data

Let's compute the embeddings of the questions in our dataset and insert them into our database. To do this, we will use Facebook's `dpr-ctx_encoder-single-nq-base` model that is included in Towhee. Note that this model creates vector of size 768, and so this is the dimensionality that we will specify later.

Towhee provides a method-chaining style API that we will use to create a pipeline to allow us to compute the embedding, and insert it into our database-- all in one pipeline.

Note that we include a normalization step in the pipeline, which is done so that we can use the `L2-squared` metric in our index later, when running vector search.

Also, note that the majority of the time here is spent on computing the embedding using the aforementioned model

```python
from towhee import pipe, ops
import numpy as np
from towhee.datacollection import DataCollection

# Define the processing pipeline
def insert_row(id, vec, question, answer):
    vector = [float(x) for x in vec]
    cursor.execute(f"INSERT INTO {TABLE_NAME} (id, question, answer, vector) VALUES (%s, %s, %s, %s);", (id, question, answer, vector))
    return True

insert_pipe = (
    pipe.input('id', 'question', 'answer')
        .map('question', 'vec', ops.text_embedding.dpr(model_name='facebook/dpr-ctx_encoder-single-nq-base'))
        # We normalize the embedding here
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
        .map(('id', 'vec', 'question', 'answer'), 'insert_status', insert_row)
        .output()
)

# Insert data
import csv
cursor = conn.cursor()

with open('question_answer.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        insert_pipe(*row)

conn.commit()
cursor.close()
```

## Creating an Index

Now that we have inserted the embeddings into our database, we need to construct an index in postgres using lantern. This is important because the index will tell allow postgres to use lantern when performing vector search.

Note that we specify L2-squared (squared Euclidean distance) as the distance metric, as we mentioned earlier. Also, as a good practice, we specify the dimension of the index (although lantern can infer it from the vector's we've already inserted).

```python
cursor = conn.cursor()

cursor.execute(f"CREATE INDEX ON {TABLE_NAME} USING hnsw (vector dist_l2sq_ops) WITH (dim=768);")

conn.commit()
cursor.close()
```

## Performing Similarity Search

Now that we have embedded our questions, let's implement the bulk of the question-answering engine: the vector search!

The idea here is that we will start with a query question. You can imagine that this is a question that a user has submitted to our engine. Our goal is to find an answer to this question.

We will do this by embedding the query question. Then, we will perform a vector search to find the most semantically similar question in our database. Then, we will retrieve the answer to this most similar question from our database, and we will serve that as the answer to the query question.

Let's see this in action by specifying the pipeline we will use, and an example of this process with `QUERY_QUESTION` below:

```python
conn.rollback()
cursor = conn.cursor()

# We only need to set this at the beginning of a session
cursor.execute("SET enable_seqscan = false;")
conn.commit()

def vector_search(vec):
  query_vector = str([float(x) for x in vec])
  cursor.execute(f"SELECT question AS similar_question, answer FROM {TABLE_NAME} ORDER BY vector <-> ARRAY{query_vector} LIMIT 1;")
  record = cursor.fetchall()[0]
  return record

ans_pipe = (
    pipe.input('question')
        .map('question', 'vec', ops.text_embedding.dpr(model_name="facebook/dpr-ctx_encoder-single-nq-base"))
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
        .map('vec', ('similar_question','answer'), vector_search)
        .output('question', 'similar_question', 'answer')
)

QUERY_QUESTION = "How much does disability insurance cost?"

ans = ans_pipe(QUERY_QUESTION)
ans = DataCollection(ans)
ans.show()

cursor.close()
```

```table
| Question | Similar Question | Answer |
|----------|------------------|--------|
| How much does disability insurance cost? | How Much Is Disability Insurance On Average? | On average, long term disability insurance costs 1% to 3% of the annual salary for men; slightly higher for women. There are man.. |
```

## Conclusion

To cleanup, close the Postgres connection.

```python
# Close the postgres connection
conn.close()
```

As we can see, we are able to obtain a very similar question to the original query question, and we retrieve the answer we stored alongside this similar question to answer the original question.

And that's how you can implement a simple Question Answering engine using Lantern! There are many approaches to how we go from the query question to a certain row in our database, but the one outlined above is a very straightforward, simple, and efficient approach. The premise behind all these approaches remains the same, however: use vector search to make the connection between the user question and our database which holds our unstructured knowledge-base data.
