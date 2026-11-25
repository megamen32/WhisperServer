"""Small, bounded SQLite registry for Web UI jobs.

Each operation opens its own connection, so instances can be used by different
threads or processes. Owner identifiers must come from the authenticated caller,
not from untrusted request data. Payloads are JSON-compatible values.
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone


class JobNotFound(LookupError):
    """The job does not exist or is not owned by the caller."""


class InvalidTransition(ValueError):
    """The requested state transition is no longer valid."""


class CapacityExceeded(ValueError):
    """A configured storage limit would be exceeded."""


def _now():
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """Persistent job store; limits apply to the whole database, not per process.

    Use a file-backed database: each call uses a separate SQLite connection.
    max_payload_bytes limits each JSON value and max_summaries limits each job.
    Terminal jobs are explicitly deleted by the owner to reclaim capacity.
    """

    def __init__(self, path, *, max_jobs=1000, max_payload_bytes=1_000_000,
                 max_summaries=20, timeout=5):
        if str(path) == ':memory:':
            raise ValueError('a file-backed database is required')
        if min(max_jobs, max_payload_bytes, max_summaries) < 1:
            raise ValueError('limits must be positive')
        self.path = str(path)
        self.max_jobs = max_jobs
        self.max_payload_bytes = max_payload_bytes
        self.max_summaries = max_summaries
        self.timeout = timeout
        with self._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL,
                attempt INTEGER NOT NULL, diarization TEXT NOT NULL,
                result TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )''')
            db.execute('''CREATE TABLE IF NOT EXISTS summaries (
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                name TEXT NOT NULL, value TEXT NOT NULL,
                PRIMARY KEY (job_id, name)
            )''')
            db.execute('CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner, created_at)')
            db.execute("""CREATE TABLE IF NOT EXISTS job_streams (
                job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                attempt INTEGER NOT NULL, epoch INTEGER NOT NULL, segment_count INTEGER NOT NULL,
                payload_bytes INTEGER NOT NULL, progress TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS job_segments (
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                attempt INTEGER NOT NULL, epoch INTEGER NOT NULL, seq INTEGER NOT NULL,
                value TEXT NOT NULL, PRIMARY KEY(job_id, attempt, epoch, seq)
            )""")

    @contextmanager
    def _transaction(self, *, readonly=False):
        db = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA foreign_keys = ON')
            db.execute('BEGIN' if readonly else 'BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _json(self, value):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        if len(data.encode('utf-8')) > self.max_payload_bytes:
            raise CapacityExceeded('JSON payload exceeds max_payload_bytes')
        return data

    @staticmethod
    def _identity(value, name):
        if not isinstance(value, str) or not value or len(value.encode('utf-8')) > 512:
            raise ValueError(f'{name} must be a nonempty string of at most 512 bytes')

    @staticmethod
    def _owned(db, owner, job_id):
        row = db.execute('SELECT * FROM jobs WHERE id = ? AND owner = ?', (job_id, owner)).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return row

    @staticmethod
    def _record(db, row, *, include_result=True, include_segments=True, include_summaries=True):
        data = dict(row)
        data['diarization'] = json.loads(data['diarization'])
        raw = data.pop('result', None)
        stream = db.execute('SELECT * FROM job_streams WHERE job_id=? AND attempt=?',
                            (row['id'], row['attempt'])).fetchone()
        epoch, count = (stream['epoch'], stream['segment_count']) if stream else (0, 0)
        data['stream_cursor'] = f"{row['attempt']}:{epoch}:{count}"
        data['segment_count'] = count
        if include_result:
            result = json.loads(raw) if raw is not None else None
            if not stream and isinstance(result, dict):
                count = len(result.get('segments', []))
                data['stream_cursor'] = f"{row['attempt']}:0:{count}"
                data['segment_count'] = count
            if isinstance(result, dict) and result.pop('_streamed', False):
                if include_segments and stream:
                    segments = [json.loads(item[0]) for item in db.execute(
                        'SELECT value FROM job_segments WHERE job_id=? AND attempt=? AND epoch=? ORDER BY seq',
                        (row['id'], row['attempt'], epoch))]
                    result['segments'] = segments
                    result['text'] = ' '.join(segment.get('text', '') for segment in segments)
            elif isinstance(result, dict) and not include_segments:
                result = {key: value for key, value in result.items()
                          if key not in ('segments', 'text', 'speaker_text', 'words')}
            data['result'] = result
        if include_summaries:
            data['summaries'] = {r['name']: json.loads(r['value']) for r in db.execute(
                'SELECT name, value FROM summaries WHERE job_id = ? ORDER BY name', (row['id'],))}
        return data

    def _stream(self, db, row, *, reset=False):
        stream = db.execute('SELECT * FROM job_streams WHERE job_id=?', (row['id'],)).fetchone()
        if stream is None or stream['attempt'] != row['attempt'] or reset:
            epoch = stream['epoch'] + 1 if stream and stream['attempt'] == row['attempt'] else 0
            # A reset explicitly invalidates the old generation. Audio is never touched.
            db.execute('DELETE FROM job_segments WHERE job_id=?', (row['id'],))
            db.execute("""INSERT INTO job_streams VALUES (?, ?, ?, 0, 0, '{}')
                          ON CONFLICT(job_id) DO UPDATE SET attempt=excluded.attempt,
                          epoch=excluded.epoch, segment_count=0, payload_bytes=0, progress='{}' """,
                       (row['id'], row['attempt'], epoch))
            stream = dict(db.execute('SELECT * FROM job_streams WHERE job_id=?', (row['id'],)).fetchone())
            # Import a legacy snapshot only when its first new segment arrives.
            # Opening the upgraded store and read-only queries never rewrite it.
            legacy = json.loads(row['result']) if row['result'] and not reset else None
            if isinstance(legacy, dict) and isinstance(legacy.get('segments'), list):
                for segment in legacy['segments']:
                    self._append_encoded(db, row, stream, self._json(segment))
                packed = self._json(legacy.get('progress', {}))
                db.execute('UPDATE job_streams SET segment_count=?,payload_bytes=?,progress=? WHERE job_id=?',
                           (stream['segment_count'], stream['payload_bytes'], packed, row['id']))
                stream['progress'] = packed
        return dict(stream)

    def _append_encoded(self, db, row, stream, encoded):
        payload_bytes = stream['payload_bytes'] + len(encoded.encode('utf-8'))
        if payload_bytes > self.max_payload_bytes:
            raise CapacityExceeded('Transcript exceeds max_payload_bytes')
        seq = stream['segment_count'] + 1
        db.execute('INSERT INTO job_segments VALUES (?, ?, ?, ?, ?)',
                   (row['id'], row['attempt'], stream['epoch'], seq, encoded))
        stream['segment_count'], stream['payload_bytes'] = seq, payload_bytes

    def append_progress(self, owner, job_id, attempt, *, segment=None, progress=None, reset=False):
        """Commit one segment and small progress state without reading its prefix."""
        encoded = self._json(segment) if segment is not None else None
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['attempt'] != attempt or row['status'] != 'processing':
                raise InvalidTransition('stale attempt or invalid state')
            stream = self._stream(db, row, reset=reset)
            state = json.loads(stream['progress'])
            state.update(progress or {})
            if encoded is not None:
                self._append_encoded(db, row, stream, encoded)
            packed = self._json(state)
            db.execute('UPDATE job_streams SET segment_count=?, payload_bytes=?, progress=? WHERE job_id=?',
                       (stream['segment_count'], stream['payload_bytes'], packed, job_id))
            db.execute('UPDATE jobs SET result=?, updated_at=? WHERE id=?',
                       (self._json({'_streamed': True, 'progress': state}), _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id), include_segments=False)

    def _sync_segments(self, db, row, segments):
        """Compatibility/final reconciliation; the worker path uses append_progress."""
        stream = self._stream(db, row)
        encoded = [self._json(segment) for segment in segments]
        old = [item[0] for item in db.execute(
            'SELECT value FROM job_segments WHERE job_id=? AND attempt=? AND epoch=? ORDER BY seq',
            (row['id'], row['attempt'], stream['epoch']))]
        if len(old) > len(encoded) or old != encoded[:len(old)]:
            stream = self._stream(db, row, reset=True)
            old = []
        for value in encoded[len(old):]:
            self._append_encoded(db, row, stream, value)
        db.execute('UPDATE job_streams SET segment_count=?, payload_bytes=? WHERE job_id=?',
                   (stream['segment_count'], stream['payload_bytes'], row['id']))
        return stream

    @staticmethod
    def parse_cursor(value):
        try:
            parts = tuple(int(item) for item in str(value).split(':'))
        except (TypeError, ValueError):
            raise ValueError('Invalid stream cursor')
        if len(parts) != 3 or min(parts) < 0:
            raise ValueError('Invalid stream cursor')
        return parts

    def stream_delta(self, owner, job_id, cursor=None, *, limit=128):
        """Read one bounded page after a durable attempt/epoch/segment cursor."""
        requested = self.parse_cursor(cursor) if cursor is not None else None
        with self._transaction(readonly=True) as db:
            # Existing stream metadata lets SQLite avoid loading a large final result.
            row = db.execute('''SELECT id,owner,status,attempt,diarization,
                CASE WHEN EXISTS (SELECT 1 FROM job_streams s WHERE s.job_id=jobs.id AND s.attempt=jobs.attempt)
                     THEN NULL ELSE result END AS result,error,created_at,updated_at
                FROM jobs WHERE id=? AND owner=?''', (job_id, owner)).fetchone()
            if row is None:
                raise JobNotFound(job_id)
            job = self._record(db, row, include_segments=False, include_summaries=False)
            stream = db.execute('SELECT progress FROM job_streams WHERE job_id=? AND attempt=?',
                                (job_id, row['attempt'])).fetchone()
            if stream:
                job['result'] = {'progress': json.loads(stream['progress'])}
            attempt, epoch, count = self.parse_cursor(job['stream_cursor'])
            reset = requested is None or requested[:2] != (attempt, epoch) or requested[2] > count
            after = 0 if reset else requested[2]
            rows = db.execute("""SELECT seq, value FROM job_segments
                                WHERE job_id=? AND attempt=? AND epoch=? AND seq>?
                                ORDER BY seq LIMIT ?""", (job_id, attempt, epoch, after, limit)).fetchall()
            if not stream and row['result']:
                legacy = json.loads(row['result'])
                legacy_segments = legacy.get('segments', []) if isinstance(legacy, dict) else []
                count = len(legacy_segments)
                if requested is not None and requested[:2] == (attempt, epoch) and requested[2] <= count:
                    after, reset = requested[2], False
                segments = legacy_segments[after:after + limit]
                end = after + len(segments)
                job['segment_count'] = count
                job['stream_cursor'] = f'{attempt}:{epoch}:{count}'
            else:
                segments = [json.loads(item['value']) for item in rows]
                end = rows[-1]['seq'] if rows else after
            return {'type': 'delta', 'job': job, 'segments': segments, 'reset': reset,
                    'cursor': f'{attempt}:{epoch}:{end}', 'has_more': end < count}

    def list_metadata(self, owner):
        self._identity(owner, 'owner')
        with self._transaction(readonly=True) as db:
            rows = db.execute("""SELECT id,owner,status,attempt,diarization,error,created_at,updated_at
                                 FROM jobs WHERE owner=? ORDER BY created_at,id""", (owner,)).fetchall()
            return [self._record(db, row, include_result=False, include_summaries=False) for row in rows]

    def create(self, owner, diarization):
        self._identity(owner, 'owner')
        job_id = uuid.uuid4().hex
        settings = self._json(diarization)
        now = _now()
        with self._transaction() as db:
            if db.execute('SELECT count(*) FROM jobs').fetchone()[0] >= self.max_jobs:
                raise CapacityExceeded('max_jobs reached')
            db.execute('INSERT INTO jobs VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?)',
                       (job_id, owner, 'uploading', 1, settings, now, now))
            return self._record(db, self._owned(db, owner, job_id))

    def get(self, owner, job_id):
        with self._transaction(readonly=True) as db:
            return self._record(db, self._owned(db, owner, job_id))

    def list(self, owner):
        self._identity(owner, 'owner')
        with self._transaction(readonly=True) as db:
            rows = db.execute('SELECT * FROM jobs WHERE owner = ? ORDER BY created_at, id', (owner,)).fetchall()
            return [self._record(db, row) for row in rows]

    def _transition(self, owner, job_id, attempt, allowed, status, *, result=None, error=None):
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['attempt'] != attempt or row['status'] not in allowed:
                raise InvalidTransition('stale attempt or invalid state')
            db.execute('UPDATE jobs SET status=?, result=?, error=?, updated_at=? WHERE id=?',
                       (status, result, error, _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def enqueue(self, owner, job_id, attempt):
        return self._transition(owner, job_id, attempt, ('uploading',), 'queued')

    def start(self, owner, job_id, attempt):
        return self._transition(owner, job_id, attempt, ('queued',), 'processing')

    def request_cancel(self, owner, job_id, attempt):
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['attempt'] != attempt or row['status'] not in ('uploading', 'queued', 'processing', 'cancel_requested'):
                raise InvalidTransition('stale attempt or invalid state')
            status = 'cancel_requested' if row['status'] in ('processing', 'cancel_requested') else 'cancelled'
            db.execute('UPDATE jobs SET status=?, updated_at=? WHERE id=?', (status, _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def finish_cancel(self, owner, job_id, attempt):
        return self._transition(owner, job_id, attempt, ('cancel_requested',), 'cancelled')

    def update_progress(self, owner, job_id, attempt, result):
        """Accept old snapshots while persisting only new segments and small state."""
        self._json(result)
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['attempt'] != attempt or row['status'] != 'processing':
                raise InvalidTransition('stale attempt or invalid state')
            if isinstance(result, dict) and isinstance(result.get('segments'), list):
                self._sync_segments(db, row, result['segments'])
                progress = result.get('progress', {})
                db.execute('UPDATE job_streams SET progress=? WHERE job_id=?', (self._json(progress), job_id))
                payload = {key: value for key, value in result.items() if key not in ('text', 'segments')}
                payload['_streamed'] = True
            else:
                payload = result
            db.execute('UPDATE jobs SET result=?, updated_at=? WHERE id=?',
                       (self._json(payload), _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id), include_segments=False)

    def complete(self, owner, job_id, attempt, result):
        payload = self._json(result)
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            settings = json.loads(row['diarization'])
            if (row['attempt'] != attempt or row['status'] != 'processing' or
                    settings.get('desired_diarize', settings.get('diarize')) != settings.get('diarize')):
                raise InvalidTransition('stale attempt or changed settings')
            if isinstance(result, dict) and isinstance(result.get('segments'), list):
                self._sync_segments(db, row, result['segments'])
            db.execute('UPDATE jobs SET status=?, result=?, error=NULL, updated_at=? WHERE id=?',
                       ('done', payload, _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def set_diarization(self, owner, job_id, attempt, settings):
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['attempt'] != attempt or row['status'] not in ('uploading', 'queued', 'processing', 'done', 'error'):
                raise InvalidTransition('stale attempt or invalid state')
            desired = settings['desired_diarize']
            status, next_attempt = row['status'], attempt
            if status in ('uploading', 'queued'):
                settings['diarize'] = desired
            elif status in ('done', 'error') and settings['diarize'] != desired:
                settings['diarize'] = desired
                status, next_attempt = 'queued', attempt + 1
                db.execute('DELETE FROM summaries WHERE job_id=?', (job_id,))
            db.execute('UPDATE jobs SET diarization=?, status=?, attempt=?, result=?, error=NULL, updated_at=? WHERE id=?',
                       (self._json(settings), status, next_attempt, row['result'] if status == 'done' else None, _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def retry_if_diarization_changed(self, owner, job_id, attempt):
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            settings = json.loads(row['diarization'])
            desired = settings.get('desired_diarize', settings.get('diarize'))
            if row['attempt'] != attempt or row['status'] not in ('processing', 'error', 'done') or desired == settings.get('diarize'):
                return None
            settings['diarize'] = desired
            db.execute('DELETE FROM summaries WHERE job_id=?', (job_id,))
            db.execute('UPDATE jobs SET status=?, attempt=?, diarization=?, result=NULL, error=NULL, updated_at=? WHERE id=?',
                       ('queued', attempt + 1, self._json(settings), _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def interrupt_for_retry(self, owner, job_id, attempt):
        """Retry lost compute once per attempt; explicit cancellation always wins."""
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['attempt'] != attempt or row['status'] not in ('queued', 'processing', 'cancel_requested'):
                raise InvalidTransition('stale attempt or invalid state')
            if row['status'] == 'cancel_requested':
                db.execute("UPDATE jobs SET status='cancelled',result=NULL,updated_at=? WHERE id=?",
                           (_now(), job_id))
            else:
                settings = json.loads(row['diarization'])
                settings['diarize'] = settings.get('desired_diarize', settings.get('diarize'))
                db.execute("UPDATE jobs SET status='queued',attempt=attempt+1,diarization=?,result=NULL,error=NULL,updated_at=? WHERE id=?",
                           (self._json(settings), _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def recover_interrupted(self):
        with self._transaction() as db:
            rows = db.execute("SELECT * FROM jobs WHERE status IN ('queued', 'processing', 'cancel_requested', 'error')").fetchall()
            result = []
            for row in rows:
                if row['status'] == 'cancel_requested':
                    db.execute("UPDATE jobs SET status='cancelled', updated_at=? WHERE id=?", (_now(), row['id']))
                    continue
                settings = json.loads(row['diarization'])
                if row['status'] == 'error' and settings.get('desired_diarize', settings.get('diarize')) == settings.get('diarize'):
                    continue
                settings['diarize'] = settings.get('desired_diarize', settings.get('diarize'))
                db.execute("UPDATE jobs SET status='queued', attempt=attempt+1, diarization=?, result=NULL, error=NULL, updated_at=? WHERE id=?",
                           (self._json(settings), _now(), row['id']))
                result.append(self._record(db, self._owned(db, row['owner'], row['id'])))
            return result

    def fail(self, owner, job_id, attempt, error):
        if not isinstance(error, str):
            raise ValueError('error must be a string')
        if len(error.encode('utf-8')) > self.max_payload_bytes:
            raise CapacityExceeded('error exceeds max_payload_bytes')
        return self._transition(owner, job_id, attempt, ('uploading', 'queued', 'processing'), 'error', error=error)

    def retry(self, owner, job_id, diarization):
        settings = self._json(diarization)
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['status'] not in ('done', 'error', 'cancelled'):
                raise InvalidTransition('only terminal jobs can be retried')
            db.execute('DELETE FROM summaries WHERE job_id=?', (job_id,))
            db.execute('''UPDATE jobs SET attempt=attempt+1, status='queued', diarization=?,
                          result=NULL, error=NULL, updated_at=? WHERE id=?''', (settings, _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def put_summary(self, owner, job_id, attempt, name, value):
        self._identity(name, 'name')
        payload = self._json(value)
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['attempt'] != attempt or row['status'] != 'done':
                raise InvalidTransition('summary requires the current completed attempt')
            exists = db.execute('SELECT 1 FROM summaries WHERE job_id=? AND name=?', (job_id, name)).fetchone()
            if not exists and db.execute('SELECT count(*) FROM summaries WHERE job_id=?', (job_id,)).fetchone()[0] >= self.max_summaries:
                raise CapacityExceeded('max_summaries reached')
            db.execute('''INSERT INTO summaries(job_id,name,value) VALUES (?,?,?)
                          ON CONFLICT(job_id,name) DO UPDATE SET value=excluded.value''', (job_id, name, payload))
            db.execute('UPDATE jobs SET updated_at=? WHERE id=?', (_now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def delete(self, owner, job_id):
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['status'] not in ('done', 'error', 'cancelled'):
                raise InvalidTransition('only terminal jobs can be deleted')
            db.execute('DELETE FROM jobs WHERE id=?', (job_id,))
