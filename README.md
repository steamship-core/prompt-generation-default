# Steamship Tagger Plugin

This project implements a basic Steamship Tagger that you can customize and deploy for your own use.

In Steamship, **Taggers** add annotations to text that can be queried and composed later. Note that a file must have
first been blockified in order to be tagged.

This sample project adds paragraph and sentence tags for a sample text document, but the
implementation you create might:

* Use entity recognition to tag named entities in text block
* Use sentiment analysis to tag positive and negative sections of a transcribed conversation

Once a Tagger has generated data in Steamship, that data is ready for use by the rest of the ecosystem. 
For example, you could perform a query over the sentences or embed each sentence.

## First Time Setup

We recommend using Python virtual environments for development.
To set one up, run the following command from this directory:

```bash
python3 -m venv .venv
```

Activate your virtual environment by running:

```bash
source .venv/bin/activate
```

Your first time, install the required dependencies with:

```bash
python -m pip install -r requirements.dev.txt
python -m pip install -r requirements.txt
```

## Developing

All the code for this plugin is located in the `src/api.py` file:

* The TaggerPlugin class
* The `run` method that is invoked via the `File.tag` call

## Testing

Tests are located in the `test/test_api.py` file. You can run them with:

```bash
pytest
```

We have provided sample data in the `test_data/` folder.

## Deploying

Deploy your tagger to Steamship by running:

```bash
ship deploy --register-plugin
```

That will deploy your plugin to Steamship and register it as a plugin for use.

## Using

Once deployed, your Tagger Plugin can be referenced by the handle in your `steamship.json` file.

```python
from steamship import Steamship, Block, File, MimeTypes, Tag

MY_PLUGIN_HANDLE = ".. fill this out .."

client = Steamship()
tagger = client.use_plugin(plugin_handle=MY_PLUGIN_HANDLE, plugin_instance="unique-instance-id")

with open("./test_data/king_speech.txt", "r") as text:
    # here, we add an initial block, as tagging requires files have been blockified.
    content = text.read()
    file = File.create(client, content=content, mime_type=MimeTypes.TXT, blocks=[Block.CreateRequest(text=content)])
    
file.tag(tagger.handle).wait()

# now that our file has been tagged, we can access the new tags by refreshing the file.
file = file.refresh()
for block in file.blocks:
    print(block.text)
    print(block.tags)

# we can also query for tags in the file (here, finding sentences)
print("\n".join([content[t.start_idx:t.end_idx] for t in Tag.query(client, f'file_id "{file.id}" and kind "sentence"').tags]))
```

## Sharing

Please share what you've built with hello@steamship.com! 

We would love take a look, hear your suggestions, help where we can, and share what you've made with the community.