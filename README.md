# Steamship Prompt Text Generation Plugin

This project implements Steamship's default text-generation plugin. The goal is to take an arbitrary prompt
(represented in a File) and continue the text as best it can.


## Usage

This plugin is invoked in the standard Tagger interface:

```python
from steamship import Steamship, File

client = Steamship()

generator = client.use_plugin(plugin_handle='prompt-generation-default')
...
file: File
file.tag(plugin_instance=generator.handle).wait()
file.refresh()
```
 
## Output

For each `Block` in the `File`, the tagger will produce an additional `Tag` with `kind` `TagKind.GENERATION`, 
`name` `GenerationTag.PROMPT_COMPLETION`, and `value` `{'string_value':'<the generated text>'}`.

## Parameters

 * __max_words__ - The maximum number of tokens to generate after the prompt

##TODOS
 * actually pass the max_words param
 * Capture output scores
 * Param for # of alternatives

