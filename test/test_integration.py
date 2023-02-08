"""Test assemblyai-s2t-blockifier via integration tests."""
import json
import random
import string
from pathlib import Path

import pytest
from steamship import Block, File, PluginInstance, Steamship, TaskState
from steamship.data import GenerationTag, TagKind, TagValueKey

GENERATOR_HANDLE = "openai"
ENVIRONMENT = "prod"


@pytest.fixture
def steamship() -> Steamship:
    """Instantiate a Steamship client."""
    return Steamship(profile=ENVIRONMENT)


def random_name() -> str:
    """Returns a random name suitable for a handle that has low likelihood of colliding with another.

    Output format matches test_[a-z0-9]+, which should be a valid handle.
    """
    letters = string.digits + string.ascii_letters
    return f"test_{''.join(random.choice(letters) for _ in range(10))}".lower()  # noqa: S311


@pytest.fixture
def plugin_instance(steamship: Steamship) -> PluginInstance:
    """Instantiate a plugin instance."""
    plugin_instance = steamship.use_plugin(
        plugin_handle=GENERATOR_HANDLE,
        instance_handle=random_name(),
        config=json.load(Path("config.json").open()),
        fetch_if_exists=False,
    )
    assert plugin_instance is not None
    assert plugin_instance.id is not None
    return plugin_instance


def test_generator(steamship: Steamship, plugin_instance: PluginInstance):
    """Test the AssemblyAI Blockifier via an integration test."""
    content = Path("data/roses.txt").open().read()
    file = File.create(steamship, blocks=[Block(text=content, tags=[])])

    blockify_task = file.tag(plugin_instance=plugin_instance.handle)
    blockify_task.wait(max_timeout_s=3600, retry_delay_s=1)

    assert blockify_task.state == TaskState.succeeded
    file = blockify_task.output.file

    assert file is not None
    assert file.blocks is not None
    assert len(file.blocks) == 1
    tags = file.blocks[0].tags
    assert len(tags) == json.load(Path("config.json").open()).get("best_of", 1)
    for tag in tags:
        assert tag.kind == TagKind.GENERATION
        assert tag.name == GenerationTag.PROMPT_COMPLETION
        assert TagValueKey.STRING_VALUE.value is not None
        assert isinstance(TagValueKey.STRING_VALUE.value, str)
