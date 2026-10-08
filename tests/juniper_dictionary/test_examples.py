import os
import json
import io
from unittest.mock import patch, MagicMock
import pytest

from juniper_examples.examples import (
    is_valid_real_word,
    get_cached_data,
    write_cached_data,
    generate_ollama_example,
    process_word_example,
    WorkerState,
)


def test_is_valid_real_word():
    # Valid short words from set
    assert is_valid_real_word("a") is True
    assert is_valid_real_word("in") is True

    # Invalid non-alphabetic
    assert is_valid_real_word("123") is False
    assert is_valid_real_word("word!") is False

    # Short words not in VALID_SHORT_WORDS
    assert is_valid_real_word("zx") is False


def test_cache_read_write(tmp_path):
    cache_dir = tmp_path / "llm_cache"
    cache_dir.mkdir()

    # Missing cache file
    assert get_cached_data("apple", cache_dir=str(cache_dir)) is None

    # Write cache
    write_cached_data("apple", "An apple a day keeps the doctor away.", cache_dir=str(cache_dir))

    # Read written cache
    cached = get_cached_data("apple", cache_dir=str(cache_dir))
    assert cached == "An apple a day keeps the doctor away."


@patch("urllib.request.urlopen")
def test_generate_ollama_example_success(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "choices": [
            {"message": {"content": '"The cat sat on the mat."'}}
        ]
    }).encode("utf-8")
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    result = generate_ollama_example("cat", definition="a small domesticated carnivorous mammal")
    assert result == "The cat sat on the mat."


@patch("urllib.request.urlopen")
def test_generate_ollama_example_failure(mock_urlopen):
    mock_urlopen.side_effect = Exception("Connection refused")

    result = generate_ollama_example("cat")
    assert result is None


@patch("juniper_examples.examples.get_cached_data")
def test_process_word_example_hits_cache(mock_get_cached):
    mock_get_cached.return_value = "Cached sentence for cat."

    result = process_word_example("cat")
    assert result == "Cached sentence for cat."


@patch("juniper_examples.examples.generate_ollama_example")
@patch("juniper_examples.examples.write_cached_data")
@patch("juniper_examples.examples.get_cached_data")
def test_process_word_example_misses_cache(mock_get_cached, mock_write_cached, mock_generate):
    mock_get_cached.return_value = None
    mock_generate.return_value = "Freshly generated sentence."

    result = process_word_example("cat")
    assert result == "Freshly generated sentence."
    mock_write_cached.assert_called_once_with("cat", "Freshly generated sentence.")


def test_worker_state():
    state = WorkerState()
    assert state.current_word is None
    assert state.total_processed == 0

    with state.lock:
        state.current_word = "test"
        state.total_processed += 1

    assert state.current_word == "test"
    assert state.total_processed == 1
