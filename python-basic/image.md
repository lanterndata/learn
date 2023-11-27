# Reverse Image Search

Here, we will use Lantern to implement a reverse image search. That is, given an image, we will return images that are "similar" to that image from a collection of images that we index.

The images we will be indexing will be a subset of the [ImageNet dataset](https://www.image-net.org/).

To generate the image embeddings, we will use the `MobileNetV2` model, using the `towhee` python library.

If you are running this in a colab, note that enabling a gpu-enabled runtime will be faster when we compute the embeddings. A cpu runtime will take significantly longer.

A Jupyter notebook with this code can be found [here](https://github.com/lanterndata/examples/blob/main/jupyter-notebooks/Reverse_Image_Search_Lantern_and_psycopg2.ipynb).

## Installing other Dependencies

```bash
python -m pip install -q towhee opencv-python pillow
```

### Downloading Image Data

The example image data (from ImageNet) we will use can be found on [Github](https://github.com/towhee-io/examples/releases/download/data/reverse_image_search.zip).

That data is organized as follows:

`train`: directory of candidate images, 10 images per class from ImageNet train data

`test`: directory of query images, 1 image per class from ImageNet test data

`reverse_image_search.csv`: a csv file containing id, path, and label for each candidate image

Let's download it:

```python
cd /content
pwd
curl -L https://github.com/towhee-io/examples/releases/download/data/reverse_image_search.zip -O
unzip -q -o reverse_image_search.zip
```

### Configuration

Let's import our dependencies and set up some configuration variables:

```python
import csv
from glob import glob
from pathlib import Path
from statistics import mean

from towhee import pipe, ops, DataCollection

# Towhee parameters
MODEL = 'mobilenetv2_100'
DEVICE = None # if None, use default device (cuda is enabled if available)

# Path to csv (column_1 indicates image path) OR a pattern of image paths
INSERT_SRC = './reverse_image_search.csv'
QUERY_SRC = './test/*/*.JPEG'
```

## Embedding Pipeline

Let's see how we will turn images into embeddings. These embeddings are precisely the vectors that we will insert into Lantern later, and will perform vector search over to find the most "similar" images to some query image.

This is how we will use the MobileNetV2 model we specified earlier to generate these embeddings:

```python

# Load image path
def load_image(x):
    if x.endswith('csv'):
        with open(x) as f:
            reader = csv.reader(f)
            next(reader)
            for item in reader:
                yield item[1]
    else:
        for item in glob(x):
            yield item

# Embedding pipeline
p_embed = (
    pipe.input('src')
        .flat_map('src', 'img_path', load_image)
        .map('img_path', 'img', ops.image_decode())
        .map('img', 'vec', ops.image_embedding.timm(model_name=MODEL, device=DEVICE))
)
```

Let's see an example embedding result. Note that the result vector has a shape of 1280, and so this is the dimensionality of the vectors we will insert into Lantern

```python
# Display example embedding result
p_display = p_embed.output('img_path', 'img', 'vec')
DataCollection(p_display('./test/goldfish/*.JPEG')).show()
```

You should see a picture of a blue fish.

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

Now let's create the table that we will use to store these embeddings. We'll call the table `images_search`, and it will have a primary key `id`, the path of the image on our filesystem `path`, and the embedding for the image `vector`. Note that we make `vector` of type real array (`real[]`). We can add a dimension, like `real[2048]`, but note that this dimension specified here is just syntactic sugar in postgres, and is not enforced.

```python
# Create the table
cursor = conn.cursor()

TABLE_NAME = "images_search"

create_table_query = f"CREATE TABLE {TABLE_NAME} (id serial PRIMARY key, path text, vector real[]);"

cursor.execute(create_table_query)

conn.commit()
cursor.close()
```

## Inserting image embeddings into our database

Now that we have a table created, let's create and insert the embeddings for the images we have specified in `reverse_image_search.csv`.

Note that the majority of the time spent here is on computing the embeddings for the images.

Let's specify the insert pipeline we will use and run it:

```python
cursor = conn.cursor()

# Inserts a single image into our database
def single_insert(img_path, vec):
    vector = [float(x) for x in vec]
    cursor.execute(f"INSERT INTO {TABLE_NAME}(path, vector) VALUES (%s, %s);", (img_path, vector))


# Insert pipeline
p_insert = (
    p_embed.map(('img_path', 'vec'), 'mr', single_insert)
    .output('mr')
)

p_insert(INSERT_SRC)

conn.commit()
cursor.close()
```

## Creating an Index

Now that we have inserted the embeddings into our database, we need to construct an index in postgres using lantern. This is important because the index will tell allow postgres to use lantern when performing vector search.

Note that we use L2-squared (squared Euclidean distance) as the distance metric. Also, as a good practice, we specify the dimension of the vectors in the index (although lantern can infer it from the vectors we've already inserted).

```python
cursor = conn.cursor()

cursor.execute(f"CREATE INDEX ON {TABLE_NAME} USING hnsw (vector dist_l2sq_ops) WITH (dim=1280);")

conn.commit()
cursor.close()
```

## Performing Similarity Search

Now that we have embedded our images, we can now perform vector search amongst our images and find similar images this way.

Let's define a search pipeline that we will use. We will return the 10 images that are closest to our query image, and display their paths.

```python
cursor = conn.cursor()

def single_search(vec):
  query_vec = str([float(x) for x in vec])
  cursor.execute(f"SELECT path, cos_dist(vector, ARRAY{query_vec}) AS dist FROM {TABLE_NAME} ORDER BY vector <-> ARRAY{query_vec} LIMIT 10;")
  results = cursor.fetchall()
  return results


# Search pipeline
p_search_pre = (
        p_embed.map('vec', ('search_res'), single_search)
               .map('search_res', 'pred', lambda x: [str(Path(y[0]).resolve()) for y in x])
)
p_search = p_search_pre.output('img_path', 'pred')

# Search for example query image(s)
dc = p_search('test/goldfish/*.JPEG')

# Display search results with image paths
DataCollection(dc).show()

#cursor.close()
```

To see these images:

```python
import cv2
from towhee.types.image import Image

def read_images(img_paths):
    imgs = []
    for p in img_paths:
        imgs.append(Image(cv2.imread(p), 'BGR'))
    return imgs

p_search_img = (
    p_search_pre.map('pred', 'pred_images', read_images)
                .output('img', 'pred_images')
)
DataCollection(p_search_img('test/goldfish/*.JPEG')).show()

cursor.close()
```

You should see a set of images of blue fish.

## Conclusion

To cleanup, close the Postgres connection.

```python
# Close the postgres connection
conn.close()
```
