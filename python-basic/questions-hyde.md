# Question Answering Engine with HyDE

In this notebook, we will use Lantern to implement a question answering engine. From a given database of pre-existing questions and their answers, we will be able to fetch answers from completely different questions.

We will be using a technique known as [HyDE (Hypothetical Document Embeddings)](https://arxiv.org/abs/2212.10496) to power this question-answering engine.

For the question-answer data that will make up our knowledge base, we will be using the [InsuranceQA Corpus](https://github.com/shuzi/insuranceQA).

We make the core functionality very simple using the python library `Towhee`, so that you can start hacking your own question answering engine.

A Jupyter notebook with this code can be found [here](https://github.com/lanterndata/examples/blob/main/jupyter-notebooks/Question_Answering_Engine_with_HyDE_Lantern_and_psycopg2.ipynb).

## Installing other Prerequisites

```bash
python -m pip install -q towhee towhee.models nltk
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

## Overview of HyDE

As mentioned earlier, we will be using the HyDE technique.

What exactly will we be embedding? We will be embedding the answers in our dataset, and storing them in our database. Then, our question-engine will operate as follows:

1. Start with a query question (for example, one that a user asks)
2. Use a Large Language Model (LLM) to hallucinate an answer to our question. It doesn't matter if the specific details in this answer are wrong. We are looking to get a "structurally similar" answer to this question, even if it's factually incorrect
3. We will embed this hallucinated answer
4. We use Lantern to perform a vector search to find the nearest neighbors to this hallucinated answer. Since we are storing the embeddings of the answers in our database, this translates to finding the "most similar" answer in our database to this hallucinated answer.
5. We present this nearest-neighbor answer from our DB as the answer to the original question. The idea is that we will retain the correct "structure" from the hallucinated answer but this answer will have the actually correct facts instead.

## Create Postgres Table

Now let's set up `psycopg2` with postgres, and enable the Lantern extension.

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

Let's compute the embeddings of all the answers in our dataset and insert them into our database. To do this, we will use Facebook's `dpr-ctx_encoder-single-nq-base` model that is included in `Towhee`. Note that this model creates vector of size 768, and so this is the dimensionality that we will specify later. Also note that we have to truncate our answers so that they do not exceed the maximum length allowed for the model.

Towhee provides a method-chaining style API that we will use to create a pipeline to allow us to compute the embedding, and insert it into our database-- all in one pipeline.

Note that we include a normalization step in the pipeline, which is done so that we can use the `L2-squared` metric in our index later, when running vector search.

Also, note that the majority of the time here is spent on computing the embedding using the aforementioned model (under 8 min on a Google Colab cpu instance at the time of this notebook being written).

```python
import nltk
from nltk.tokenize import sent_tokenize

# We use nltk to truncate our answers
nltk.download('punkt')
```

```python
from towhee import pipe, ops
import numpy as np
from towhee.datacollection import DataCollection


def truncate_answer(answer):
  # The model we use has a maximum token length

  # So, we must make sure our answers are of the appropriate length
  # before passing them to the model. The model actually counts length in terms
  # of units known as tokens, but we will implement an approximate technique
  # below where we just operate in terms of sentences.

  # We will take the first 7 sentences  of the answer as an example.
  # An in-production system would make sure that this still obeys the max length
  # in terms of tokens, which is how the model gauges length
  sentences = sent_tokenize(answer)
  return ' '.join(sentences[:7])


# Define the processing pipeline
def insert_row(id, vec, question, answer):
    vector = [float(x) for x in vec]
    cursor.execute(f"INSERT INTO {TABLE_NAME} (id, question, answer, vector) VALUES (%s, %s, %s, %s);", (id, question, answer, vector))
    return True

insert_pipe = (
    pipe.input('id', 'question', 'raw_answer')
        .map('raw_answer', 'answer', truncate_answer)
        .map('answer', 'vec', ops.text_embedding.dpr(model_name='facebook/dpr-ctx_encoder-single-nq-base'))
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

Now that we have embedded our answers, let's implement the bulk of the question-answering engine: the vector search!

As mentioned earlier in our overview of HyDE, we will hallucinate an answer to this question using an LLM (which we skip for this notebook, and use a hardcoded example), embed this hallucinated answer, and then run similarity search on our answer embeddings to find the closest answer in our database to this hallucinated answer. Then, we will serve this answer as the answer to the original question.

Let's see this in action by specifying the pipeline we will use, and an example of this process with `QUERY_QUESTION` below:

```python
QUERY_QUESTION = "How much does disability insurance cost?"

def hallucinate_answer(question):
  # Here is where you would use an LLM (like OpenAi's GPT, Anthropic, LLaMA, etc.) to hallucinate an answer to this question
  # Here, we will use an example hallucinated answer that corresponds to our example query question

  example_hallucinated_answer = """
  The cost of disability insurance varies widely depending on factors such as your age, health, occupation, coverage amount, and policy features.
  On average, it can range from 1% to 3% of your annual income. To get an accurate quote, you should contact insurance providers and request a personalized quote based on your specific circumstances.
  """

  return example_hallucinated_answer

HALLUCINATED_ANSWER = hallucinate_answer(QUERY_QUESTION)


cursor = conn.cursor()

# We only need to set this at the beginning of a session
cursor.execute("SET enable_seqscan = false;")
conn.commit()

def vector_search(vec):
  query_vector = str([float(x) for x in vec])
  cursor.execute(f"SELECT question AS associated_question, answer FROM {TABLE_NAME} ORDER BY vector <-> ARRAY{query_vector} LIMIT 1;")
  record = cursor.fetchall()[0]
  return record

ans_pipe = (
    pipe.input('answer')
        .map('answer', 'vec', ops.text_embedding.dpr(model_name="facebook/dpr-ctx_encoder-single-nq-base"))
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
        .map('vec', ('associated_question','answer'), vector_search)
        .output('answer', 'associated_question')
)

ans = ans_pipe(HALLUCINATED_ANSWER)
ans = DataCollection(ans)[0]

print(f"Original Question:\n{QUERY_QUESTION}\n\n")
print(f"Proposed Answer:\n {ans['answer']}")

cursor.close()
```

```
Original Question:
How much does disability insurance cost?

Proposed Answer:
Any disability insurance policy is priced based on several factors of the applicant. These include age, gender, occupation and duties of that occupation, tobacco use, and the amount of coverage needed or desired. The amount of coverage is often dictated by the person's earned income; the more someone earns, the more coverage that is available. There are several policy design features that can be included in the plan (or not) that will also affect the price. The person's medical history can also play an important role in pricing. As you can see, there are lots of moving parts to a disability policy that will affect the price. Doctors often buy coverage to protect their specific medical specialty.
```

```python
print(f"Original Question:\n{QUERY_QUESTION}\n\n")
print(f"Hallucinated Answer:\n{HALLUCINATED_ANSWER}\n\n")
print(f"Nearest-neighbor Answer:\n {ans['answer']}\n\n")
print(f"Question associated in DB with nearest-neightbor answer:\n {ans['associated_question']}\n\n")
```

```
Original Question:
How much does disability insurance cost?

Hallucinated Answer:
The cost of disability insurance varies widely depending on factors such as your age, health, occupation, coverage amount, and policy features.
On average, it can range from 1% to 3% of your annual income. To get an accurate quote, you should contact insurance providers and request a personalized quote based on your specific circumstances.

Nearest-neighbor Answer:
Any disability insurance policy is priced based on several factors of the applicant. These include age, gender, occupation and duties of that occupation, tobacco use, and the amount of coverage needed or desired. The amount of coverage is often dictated by the person's earned income; the more someone earns, the more coverage that is available. There are several policy design features that can be included in the plan (or not) that will also affect the price. The person's medical history can also play an important role in pricing. As you can see, there are lots of moving parts to a disability policy that will affect the price. Doctors often buy coverage to protect their specific medical specialty.

Question associated in DB with nearest-neightbor answer:
How  Much  Is  Disability  Insurance  For  Doctors?
```

As we can see, we are able to obtain an answer by finding the nearest neighbor to the hallucinated answer, which we obtained by using an LLM to come up with an answer to our original query question. Notice also that the question associated with the nearest-neighbor answer is very similar to our original query question, which we would expect.

And that's how you can implement a simple Question Answering engine using Lantern! There are many approaches to how we go from the query question to a certain row in our database, and this notebook used HyDE. The premise behind all these approaches remains the same, however: use vector search to make the connection between the user question and our database which holds our unstructured knowledge-base data.

## Conclusion

To cleanup, close the Postgres connection.

```python
# Close the postgres connection
conn.close()
```
