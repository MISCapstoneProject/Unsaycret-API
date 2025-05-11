import weaviate
import os

import weaviate.classes.config as wc


# Instantiate your client (not shown). e.g.:
# client = weaviate.connect_to_weaviate_cloud(..., headers=headers) or
client = weaviate.connect_to_local()

client.collections.create(
    name="MovieCustomVector",
    properties=[
        wc.Property(name="title", data_type=wc.DataType.TEXT),
        wc.Property(name="overview", data_type=wc.DataType.TEXT),
        wc.Property(name="vote_average", data_type=wc.DataType.NUMBER),
        wc.Property(name="genre_ids", data_type=wc.DataType.INT_ARRAY),
        wc.Property(name="release_date", data_type=wc.DataType.DATE),
        wc.Property(name="tmdb_id", data_type=wc.DataType.INT),
    ],
    # Define the vectorizer module (none, as we will add our own vectors)
    vectorizer_config=wc.Configure.Vectorizer.none(),
    # Define the generative module
    generative_config=wc.Configure.Generative.cohere()
)

client.close()