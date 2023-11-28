# Reverse Video Search

Here, we will use Lantern to implement a reverse video search engine. That is, given a video, we will return videos that are "similar" to that video from a collection of video that we index. In this case, we will be considering the "action" displayed in that video as constituting similarity across videos.

The videos we will be indexing will be a subset of the [Kinetics dataset](https://www.deepmind.com/open-source/kinetics).

To generate the video embeddings, we will use the `Omnivore` [model](https://towhee.io/action-classification/omnivore), using the `towhee` python library.

If you are running this in a colab, note that enabling a gpu-enabled runtime will be faster when we compute the embeddings. A cpu runtime will take significantly longer.

A Jupyter notebook with this code can be found [here](https://github.com/lanterndata/examples/blob/main/jupyter-notebooks/Reverse_Video_Search_Lantern_and_psycopg2.ipynb).

## Installing other Dependencies

```bash
python -m pip install -q towhee towhee.models pillow ipython
```

### Downloading Video dataset

```bash
curl -L https://github.com/towhee-io/examples/releases/download/data/reverse_video_search.zip -O
unzip -q -o reverse_video_search.zip
```

We also downloaded the file `reverse_video_search.csv` which contains the id, path, and label of our videos, as seen below:

```python
import pandas as pd

df = pd.read_csv('./reverse_video_search.csv')

# We'll use this to get a video from an id
id_video = df.set_index('id')['path'].to_dict()
label_ids = {}
for label in set(df['label']):
    label_ids[label] = list(df[df['label']==label].id)

# Visualize the first few rows...
df.head(3)
```

```table
| ID     | Path     | Label     |
|--------|----------|-----------|
|0|./train/country_line_dancing/bTbC3w_NIvM.mp4	|country_line_dancing|
|1|./train/country_line_dancing/n2dWtEmNn5c.mp4	|country_line_dancing|
|2|./train/country_line_dancing/zta-Iv-xK7I.mp4	|country_line_dancing|
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

Now let's create the table that we will use to store these embeddings. We'll call the table `videos_search`, and it will an integer `id` corresponding to the video's id in the dataset, and the embedding for the video `vector`. Note that we make `vector` of type real array (`real[]`).

```python
# Create the table
cursor = conn.cursor()

TABLE_NAME = "videos_search"

create_table_query = f"CREATE TABLE {TABLE_NAME} (id integer, vector real[]);"

cursor.execute(create_table_query)

conn.commit()
cursor.close()
```

## Inserting video embeddings into our database

Now that we have a table created, let's create and insert the embeddings for the videos we have.

Note that the majority of the time spent here is on computing the embeddings for the videos.

Note that the dimensionality of the embeddings we compute is 768, and so this is the dimensionality of the vectors that we will be inserting into our database.

Let's specify the insert pipeline we will use and run it:

```python
from towhee import pipe, ops
from towhee.datacollection import DataCollection

def read_csv(csv_file):
    import csv
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        data = csv.DictReader(f)
        for line in data:
            yield line['id'], line['path'], line['label']


cursor = conn.cursor()

def single_insert(id, features):
    vector = [float(x) for x in features]
    cursor.execute(f"INSERT INTO {TABLE_NAME} (id, vector) VALUES (%s, %s);", (id, vector))

insert_pipe = (
    pipe.input('csv_path')
        .flat_map('csv_path', ('id', 'path', 'label'), read_csv)
        .map('id', 'id', lambda x: int(x))
        .map('path', 'frames', ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 16}))
        .map('frames', ('labels', 'scores', 'features'), ops.action_classification.omnivore(model_name='omnivore_swinT'))
        .map(('id', 'features'), 'insert_res', single_insert)
        .output()
)

insert_pipe('reverse_video_search.csv')

conn.commit()
cursor.close()
```

## Creating an Index

Now that we have inserted the embeddings into our database, we need to construct an index in postgres using lantern. This is important because the index will tell allow postgres to use lantern when performing vector search.

Note that we use L2-squared (squared Euclidean distance) as the distance metric. Also, as a good practice, we specify the dimension of the vectors in the index (although lantern can infer it from the vectors we've already inserted).

```python
cursor = conn.cursor()

cursor.execute(f"CREATE INDEX ON {TABLE_NAME} USING hnsw (vector dist_l2sq_ops) WITH (dim=768);")

conn.commit()
cursor.close()
```

## Querying Videos

Now that we have our index, we can start querying for videos and utilizing vector search.

We define our query pipeline below, and try out a sample query.

```python
cursor = conn.cursor()

def single_query(features):
    query_vec = str([float(x) for x in features])
    cursor.execute(f"SELECT id FROM {TABLE_NAME} ORDER BY vector <-> ARRAY{query_vec} LIMIT 5;")
    results = cursor.fetchall()
    return results


query_pipe = (
    pipe.input('path')
        .map('path', 'frames', ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 16}))
        .map('frames', ('labels', 'scores', 'features'), ops.action_classification.omnivore(model_name='omnivore_swinT'))
        .map('features', 'result', single_query)
        .map('result', 'candidates', lambda x: [id_video[i[0]] for i in x])
        .output('path', 'candidates')
)


# Let's try a sample query
query_path = './test/eating_carrots/ty4UQlowp0c.mp4'

res = DataCollection(query_pipe(query_path))
res.show()
```

We can see what these videos look like:

```python
import os
from IPython import display
from IPython.display import Image as IPImage
from PIL import Image

tmp_dir = './tmp'
os.makedirs(tmp_dir, exist_ok=True)

def video_to_gif(video_path):
    gif_path = os.path.join(tmp_dir, video_path.split('/')[-1][:-4] + '.gif')
    p = (
        pipe.input('path')
            .map('path', 'frames', ops.video_decode.ffmpeg(sample_type='uniform_temporal_subsample', args={'num_samples': 16}))
            .output('frames')
    )
    frames = p(video_path).get()[0]
    imgs = [Image.fromarray(frame) for frame in frames]
    imgs[0].save(fp=gif_path, format='GIF', append_images=imgs[1:], save_all=True, loop=0)
    return gif_path


query_gif = video_to_gif(query_path)

results_paths = []
for path in res[0]['candidates'][:3]:
    gif_path = video_to_gif(path)
    results_paths.append(gif_path)

print("QUERY VID:")
IPImage(filename=query_gif)

print("SEARCH RESULT 1:")
IPImage(filename=results_paths[0])

print("SEARCH RESULT 2:")
IPImage(filename=results_paths[1])

print("SEARCH RESULT 3:")
IPImage(filename=results_paths[2])
```

## Conclusion

You should see that the top 3 videos are all also videos of people eating carrots, which is what our query video was!

And that's how you can implement reverse video search using Lantern and a video embedding model.

To cleanup, close the Postgres connection.

```python
# Close the postgres connection
conn.close()
```
