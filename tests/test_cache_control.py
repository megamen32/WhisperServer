import uuid

import main
from main import app


def test_openai_endpoint_exposes_request_scoped_cache_flag():
    schema = app.openapi()
    props = schema['paths']['/v1/audio/transcriptions']['post']['requestBody']['content']['multipart/form-data']['schema']
    if '$ref' in props:
        name = props['$ref'].split('/')[-1]
        props = schema['components']['schemas'][name]
    cache = props['properties']['cache']
    assert cache['type'] == 'boolean'
    assert cache['default'] is True


def test_invalid_legacy_cache_value_is_treated_as_a_miss():
    key = f"test-invalid-cache-{uuid.uuid4()}"
    main.cache.set(key, True)
    try:
        assert main._get_cached_result(key) is None
        assert key not in main.cache
    finally:
        main.cache.delete(key)
