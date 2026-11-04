"""Fast unit/integration checks: SQLite state, access, bounds and races.

Each test: category=fast unit or focused integration; expected <1 s, max 5 s.
Race tests detect cancellation/publication and concurrent capacity regressions.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from webui_jobs import CapacityExceeded, InvalidTransition, JobNotFound, JobStore


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / 'jobs.sqlite', max_jobs=3, max_payload_bytes=100, max_summaries=2)


def processing(store):
    job = store.create('alice', {'enabled': True})
    store.enqueue('alice', job['id'], 1)
    store.start('alice', job['id'], 1)
    return job['id']


def test_lifecycle_persistence_and_summaries(store):
    job_id = processing(store)
    store.complete('alice', job_id, 1, {'text': 'привет'})
    store.put_summary('alice', job_id, 1, 'short', {'text': 'тезис'})
    reopened = JobStore(store.path)
    job = reopened.get('alice', job_id)
    assert (job['status'], job['result'], job['summaries']) == ('done', {'text': 'привет'}, {'short': {'text': 'тезис'}})
    assert reopened.list('alice')[0]['id'] == job_id
    assert reopened.list('bob') == []
    reopened.delete('alice', job_id)
    with pytest.raises(JobNotFound):
        reopened.get('alice', job_id)


def test_owner_checked_for_every_operation(store):
    job_id = processing(store)
    operations = [lambda: store.get('bob', job_id), lambda: store.enqueue('bob', job_id, 1),
                  lambda: store.start('bob', job_id, 1), lambda: store.request_cancel('bob', job_id, 1),
                  lambda: store.finish_cancel('bob', job_id, 1), lambda: store.complete('bob', job_id, 1, {}),
                  lambda: store.fail('bob', job_id, 1, 'x'), lambda: store.retry('bob', job_id, {}),
                  lambda: store.put_summary('bob', job_id, 1, 'a', {}), lambda: store.delete('bob', job_id)]
    for operation in operations:
        with pytest.raises(JobNotFound):
            operation()
    assert store.get('alice', job_id)['status'] == 'processing'


def test_cancel_retry_and_stale_worker(store):
    job_id = processing(store)
    assert store.request_cancel('alice', job_id, 1)['status'] == 'cancel_requested'
    for operation in (lambda: store.complete('alice', job_id, 1, 'late'),
                      lambda: store.fail('alice', job_id, 1, 'late')):
        with pytest.raises(InvalidTransition):
            operation()
    store.finish_cancel('alice', job_id, 1)
    job = store.retry('alice', job_id, {'enabled': False})
    assert (job['attempt'], job['status'], job['diarization']) == (2, 'queued', {'enabled': False})
    store.start('alice', job_id, 2)
    with pytest.raises(InvalidTransition):
        store.complete('alice', job_id, 1, 'stale')
    store.complete('alice', job_id, 2, 'fresh')
    with pytest.raises(InvalidTransition):
        store.put_summary('alice', job_id, 1, 'stale', 'x')
    store.put_summary('alice', job_id, 2, 's', 'x')
    job = store.retry('alice', job_id, {})
    assert job['result'] is None and job['summaries'] == {} and job['attempt'] == 3


def test_prestart_cancel_and_invalid_transitions(store):
    job = store.create('a', {})
    with pytest.raises(InvalidTransition):
        store.start('a', job['id'], 1)
    with pytest.raises(InvalidTransition):
        store.delete('a', job['id'])
    assert store.request_cancel('a', job['id'], 1)['status'] == 'cancelled'
    with pytest.raises(InvalidTransition):
        store.enqueue('a', job['id'], 1)
    job = store.retry('a', job['id'], {})
    assert store.fail('a', job['id'], job['attempt'], 'failed')['error'] == 'failed'


def test_bounds_and_rollback(store):
    with pytest.raises(CapacityExceeded):
        store.create('a', 'x' * 100)
    job_id = processing(store)
    with pytest.raises(CapacityExceeded):
        store.complete('alice', job_id, 1, 'я' * 100)
    assert store.get('alice', job_id)['status'] == 'processing'
    store.complete('alice', job_id, 1, None)
    store.put_summary('alice', job_id, 1, 'a', 1)
    store.put_summary('alice', job_id, 1, 'b', 2)
    store.put_summary('alice', job_id, 1, 'a', 3)
    with pytest.raises(CapacityExceeded):
        store.put_summary('alice', job_id, 1, 'c', 4)
    store.create('a', {})
    store.create('b', {})
    with pytest.raises(CapacityExceeded):
        store.create('c', {})


def test_cancel_completion_race(store):
    job_id = processing(store)
    barrier = Barrier(2)
    def run(action):
        barrier.wait(timeout=3)
        try:
            return action()['status']
        except InvalidTransition:
            return 'rejected'
    with ThreadPoolExecutor(max_workers=2) as pool:
        cancel = pool.submit(run, lambda: store.request_cancel('alice', job_id, 1))
        complete = pool.submit(run, lambda: store.complete('alice', job_id, 1, 'result'))
        outcomes = {cancel.result(timeout=5), complete.result(timeout=5)}
    job = store.get('alice', job_id)
    assert outcomes in ({'done', 'rejected'}, {'cancel_requested', 'rejected'})
    assert (job['status'], job['result']) in (('done', 'result'), ('cancel_requested', None))


def test_capacity_is_atomic_across_instances(tmp_path):
    path = tmp_path / 'race.sqlite'
    stores = [JobStore(path, max_jobs=1), JobStore(path, max_jobs=1)]
    barrier = Barrier(2)
    def create(store):
        barrier.wait(timeout=3)
        try:
            return store.create('a', {})['id']
        except CapacityExceeded:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, stores))
    assert sum(result is not None for result in results) == 1
