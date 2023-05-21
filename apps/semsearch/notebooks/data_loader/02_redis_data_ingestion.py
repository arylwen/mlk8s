import functools
import json
import os
import pathlib
import shutil
import tarfile
from pathlib import Path

import click
import pandas as pd
import redis
import requests
from redis.commands.json.path import Path as redPath
from tqdm import tqdm
from collections import OrderedDict


#execute MODULE LOAD /usr/lib/redis/modules/rejson.so in redis-commander

data_dir = Path('abo/')

url = 'https://amazon-berkeley-objects.s3.amazonaws.com/archives/'
filename = 'abo-images-small.tar'
tar_filename = os.path.join('abo', filename)
path = pathlib.Path('abo') / filename
listings = 'https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar'
path_list = pathlib.Path('abo') / 'abo-listings.tar'
listings_dir = data_dir/ 'listings'/'metadata'
csv = data_dir / 'images' /'metadata'/'images.csv.gz'

r = redis.Redis(host="redis", port=31781, password="")
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

class OrderedGroup(click.Group):
    def __init__(self, name=None, commands=None, **attrs):
        super(OrderedGroup, self).__init__(name, commands, **attrs)
        #: the registered subcommands by their exported names.
        self.commands = commands or OrderedDict()

    def list_commands(self, ctx):
        return self.commands


@click.group(cls=OrderedGroup, context_settings=CONTEXT_SETTINGS)
def cli():
    pass

@cli.command()
def add():
    """Adds the untared metadata to redis db. Ensure redis is running.
    The image metadata is with `IMG:image_id` key.
    The photo (flat file) name to image_id mapping is with `MAP:name` key
    """
    _add()

def _add():
    listing_jsons = [f for f in listings_dir.iterdir()]
    def extract_values(x):
        if isinstance(x, list) and 'value' in x[0]:
            return [a['value'] for a in x]
        if pd.isna(x):
            return ''
        else:
            return x


    descriptors = ['item_id', 'item_name', 'model_name', 'brand', 'bullet_point']
    for listing_file in listing_jsons:
        df_list = pd.read_json(listing_file, lines=True)
        pipe = r.pipeline()
        for _, row in df_list.iterrows():
            feature = json.dumps({d: extract_values(row[d]) for d in descriptors})
            if not pd.isna(row['main_image_id']):
                key = "IMG:" + row['main_image_id']
                pipe.json().set(key, redPath.root_path(), feature)

            try:
                for img in row['other_image_id']:
                    key = "IMG:" + img
                    pipe.json().set(key, redPath.root_path(), feature)

            except TypeError:
                # ignore nans
                pass
        print("Starting to update redis")
        pipe.execute()

    df = pd.read_csv(csv)
    df['name'] = df['path'].str.extract(r'\/(.*).jpg')
    pipe = r.pipeline()
    for _, (name, id) in df[['name', 'image_id']].iterrows():
        pipe.set('MAP:'+ str(name), id )
    pipe.execute()


if __name__ == "__main__":
    cli()