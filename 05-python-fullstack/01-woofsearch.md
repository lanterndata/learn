# WoofSearch - Reverse Image Search for Dog Images

In this article, we'll go over how we built our [reverse image search demo](https://demos.lantern.dev/woofsearch).

How does a reverse image search engine work? It takes in an image and returns images that are similar to it. So, the first thing we need is a collection of images. For this demo, we used the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), which has over 20,000 images of over 120 breeds of dogs.

But what does this have to do with Lantern, a vector database? To power our image search, we will first be converting our images into vectors called embeddings, and then performing search over these embeddings. That is, given a query image, we turn it into the query vector/embedding, and then find the image vectors/embeddings from our collection of images that is _closest_ to this query vector. We will then return the images associated with these closest vectors.

To turn our images into embeddings, we will use OpenAI's CLIP model. You can read more about CLIP [here](https://openai.com/research/clip), and install it from OpenAI's [github](https://github.com/openai/CLIP). CLIP can actually embed both text and images, but for this project we will be using it for just images. CLIP produces embeddings that encapsulate many elements from an image, such as the subject, action, and background. This means that our image search engine will not only return images of the same breed of a particular dog, but will also return images of dogs doing the same action (i.e. jumping in the air), or standing in a very similar pose.

Now that we've walked through the design, let's go through the code:

## Computing Embeddings

Once we set up Lantern (via our cloud-hosted instances or by installing it locally), our first task is to generate the embeddings and store them in Lantern. This only needs to be done once: as soon as we have our reference images and their embeddings stored, we can perform queries on them.

First let's download and extract the images from the Stanford Dogs Dataset mentioned above:

```bash
wget -c http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar -O - | tar -xv
```

Now let's write a script to compute the embeddings for all of the images we've downloaded, using CLIP.

Let's import our dependencies, and load the CLIP model in:

```python
from pathlib import Path
import torch
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle

# Directory where your images are stored
root_dir = 'Images'

# CLIP requires a specific preprocessing pipeline
preprocess = Compose([
    Resize(256, interpolation=3),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
```

Note that we define a preprocessing pipeline above, since CLIP expects images to be inputted in a specific manner. So, we're resizing, cropping, and normalizing the image using parameters given by OpenAI for this model.

We'll need to read in all the images in our `Images` directory, so let's define a way to do that in `pytorch`:

```python
# A custom dataset to read images from a list of files
class ImageDataset(Dataset):
    def __init__(self, file_paths, transform=None):
   	 self.file_paths = [str(path) for path in file_paths]
   	 self.transform = transform

    def __len__(self):
   	 return len(self.file_paths)

    def __getitem__(self, idx):
   	 image_path = self.file_paths[idx]
   	 image = Image.open(image_path).convert('RGB')
   	 if self.transform:
   		 image = self.transform(image)
   	 return image, image_path

# Function to get all image paths
def get_image_paths(directory):
    return list(Path(directory).rglob('*.jpg')) + list(Path(directory).rglob('*.jpeg'))
```

We'll be using a GPU to compute our embeddings faster, and so we can batch images to make this more efficient, by loading in several images at a time:

```python
# Batch size for embedding
batch_size = 32 # You can adjust this according to your GPU memory

# Get all image file paths
image_paths = get_image_paths(root_dir)
image_dataset = ImageDataset(image_paths, transform=preprocess)
image_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)
```

Now let's write the function to embed our images:

```python
def process_and_embed_images(dataloader, model):
    model.eval() # Put the model in evaluation mode
    embeddings = []
    paths = []

    with torch.no_grad(): # No need to track gradients
    # Initialize the tqdm progress bar
   	 progress_bar = tqdm(dataloader, desc="Processing Images", mininterval=1)

   	 for images, image_paths in progress_bar:
   		 images = images.to(device)
   		 # Get the embeddings for this batch
   		 batch_embeddings = model.encode_image(images)

   		 embeddings.append(batch_embeddings.cpu())
   		 paths.extend(image_paths)

   		 # Update the progress bar
   		 progress_bar.update()
   		 progress_bar.refresh()

    # Concatenate all the embeddings from each batch into a single Tensor

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, paths
```

And we can run this function to get all our embeddings. We'll be running this on a computer with a GPU instance, so we can speed this process up considerably. So, we'll save our results using `pickle` and then process them later:

```python
# Process the images and get their embeddings
embeddings, paths = process_and_embed_images(image_dataloader, model)

embeddings_list = embeddings.tolist()
paths_list = [str(path) for path in paths]

# Pair each embedding with its corresponding path
embeddings_paths_pairs = list(zip(embeddings_list, paths_list))

# Serialize with pickle and save to a file
with open('embeddings_paths_pairs.pkl', 'wb') as f:
    pickle.dump(embeddings_paths_pairs, f)
```

## Storing Embeddings in Lantern

Now that we have our embeddings, let's insert them into Lantern. We'll do this using another script, which will open the `pkl` files we saved above and populate our database with the saved embedding and path values.

First, though, we need to do a few things to create and prepare our database. We'll create a database called `dog_images`:

```sql
CREATE DATABASE dog_images;
```

We'll also create a table called `images`, which will store a path to the image on our filesystem as well as its corresponding embedding in the `vector` column:

```sql
CREATE TABLE images (id SERIAL PRIMARY KEY, path text, vector real[]);
```

Lastly, we need to enable the Lantern extension with:

```sql
ENABLE EXTENSION lantern;
```

Now let's look at the script we used to populate our database with our embeddings:

```python
import pickle
import psycopg2

# Load the embeddings and paths from the .pkl file
with open('embeddings_paths_pairs.pkl', 'rb') as f:
    loaded_embeddings_paths_pairs = pickle.load(f)


# Connect to Postgres
conn = psycopg2.connect(
    dbname="dog_images",
    user="postgres",
    password="password",
    host="localhost",
    port="5432"
)

TABLE_NAME = "images"

# Get a new cursor
cursor = conn.cursor()

for embedding, path in loaded_embeddings_paths_pairs:
    cursor.execute(f"INSERT INTO {TABLE_NAME}(path, vector) VALUES (%s, %s);", (path, embedding))

conn.commit()
cursor.close()
```

Now that we've inserted all of our embeddings into our database, we need to create an index on our table so that Lantern can efficiently perform vector search on it. We will use the cosine distance as the distance metric.

```sql
CREATE INDEX on images USING lantern_hnsw (vector dist_cos_ops);
```

## Implementing the search

Now that we have completed the one-time step to process all our unstructured data (our images) into Lantern, we are ready to implement the actual backend logic of the search. We will use `gradio` for the frontend and backend of our app.

Let's first import our prerequisites and do some setup: initialize the CLIP model, and connect to PostgreSQL as we did in our script above.

```python
import gradio as gr
import time
from PIL import Image

import torch
import clip

import os
import psycopg2

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Prepare postgres
conn = psycopg2.connect(
    dbname="dog_images",
    user="postgres",
    password="password",
    host="localhost",
    port="5432"
)

TABLE_NAME = "images"
```

Now let's write the function that performs the search. It will take a vector representing the embedding of the query image, and it will return the 9 closest neighbors in our database, which will be the images in the search result:

```python
# Performs a vector search using lantern
def single_search(vec):
    cursor = conn.cursor()

    cursor.execute(f"SELECT path, cos_dist(vector, ARRAY{vec}) AS dist FROM {TABLE_NAME} ORDER BY vector <-> ARRAY{vec} LIMIT 9;")

    results = cursor.fetchall()
    cursor.close()

    return results
```

Now all that's left for the search logic is a function that takes the query image and computes its embedding using CLIP, so we can then pass it as an input to our function above. We also compute how long this entire process (computing the embedding and the actual search) takes, just so we can log it on our frontend:

```python
def process_image(uploaded_image):
    start_time = time.time()

    image = preprocess(uploaded_image).unsqueeze(0).to(device)

    # Get vector embedding of the query
    with torch.no_grad():
   	 image_features = model.encode_image(image)

    query_embedding = image_features.tolist()

    # Perform the vector search
    search_results = single_search(query_embedding)

    gallery_items = [(path, f"Distance: {dist:.4f}") for path, dist in search_results]

    end_time = time.time()
    operation_time = end_time - start_time

    return gallery_items, f"Total time for the search: {operation_time:.2f} seconds"
```

Finally, let's make a simple frontend using `gradio` that allows a user to upload an image and see the 9 result images in a gallery:

```python
with gr.Blocks() as app:
    gr.Markdown("# WoofSearch: Reverse Image Search For Dogs")

    with gr.Row():
   	 # Input image
   	 with gr.Column():

   		 image_input = gr.Image(type="pil", label="Upload Input Image", width=512, height=512)

   	 # Gallery for output images
   	 with gr.Column():
   		 output_image = gr.Gallery(columns=3, label="Results", show_label=True)

    gr.Markdown("## Examples")


    example_images = [
   	 "gradio_example_images/0.webp",
   	 "gradio_example_images/1.jpeg",
   	 "gradio_example_images/2.jpeg",
   	 "gradio_example_images/3.webp",
   	 "gradio_example_images/4.webp",
    ]

    gr.Examples(
   	 examples=example_images,
   	 inputs=image_input,
    )

    generate_button = gr.Button("Search")
    output_text = gr.Textbox(label="Operation Time")

    generate_button.click(
   	 fn=process_image,
   	 inputs=image_input,
   	 outputs=[output_image, output_text]
    )


# Run the Gradio app
app.launch()
```

And that's it! Let's search using one of our example images, of a dog sitting at the steering wheel of a car:

As we can see, the search results are all other images of dogs sitting behind a steering wheel, or inside a car, which indicates that our image search engine is indeed performing a semantic search within our collection of images of other dogs.

![Woofsearch Demo Screenshot](https://storage.googleapis.com/lantern-web/woofsearch.png)

Thanks for reading along how we built this! All the code is available on [github](https://github.com/lanterndata/examples/tree/main/full-stack-demos/woof-search).
