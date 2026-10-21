"""Fast unit/focused integration, expected 12s/max 45s, no weights or GPU.

Detects speaker changes lost inside ASR segments, mixed caches, dropped API
fields, unlabelled streaming segments, and GigaAM merging a whole dialogue.
"""
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
import gigaam_backend
from diarization_backend import attribute_segment, speaker_for_interval, speaker_text
from metrics import transcription_response_payload

TURNS = [
    {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
    {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
    {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_00"},
]


def test_gated_model_error_explains_required_access():
    from diarization_backend import _load_pipeline
    from huggingface_hub.errors import GatedRepoError
    class FakeResponse:
        # huggingface_hub 2.x made `response` a required keyword-only argument,
        # and both versions read `headers` and `request` from it.
        status_code = 403
        headers: dict = {}
        request = None
    class Gated:
        @staticmethod
        def from_pretrained(*a, **k):
            raise GatedRepoError('restricted model', response=FakeResponse())
    with pytest.raises(RuntimeError, match='Community-1.*HF_TOKEN'):
        _load_pipeline(Gated)


def test_word_changes_split_segment_and_preserve_returning_speaker():
    words = [dict(start=0.1, end=0.8, word=" Привет!"),
             dict(start=1.1, end=1.8, word=" Здравствуйте!"),
             dict(start=3.1, end=3.8, word=" Как дела?")]
    segments = attribute_segment(dict(start=0, end=4, text="Привет! Здравствуйте! Как дела?"), words, TURNS)
    assert [s["speaker"] for s in segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert [(s["start"], s["end"]) for s in segments] == [(0.1, 0.8), (1.1, 1.8), (3.1, 3.8)]
    assert speaker_text(segments) == "Спикер 1: Привет!\nСпикер 2: Здравствуйте!\nСпикер 1: Как дела?"
    assert [w["speaker"] for w in words] == [s["speaker"] for s in segments]


def test_overlap_and_unknown_do_not_invent_identity():
    assert speaker_for_interval(0.9, 1.8, TURNS) == "SPEAKER_01"
    assert speaker_for_interval(2.2, 2.8, TURNS) is None
    assert speaker_for_interval(1, 1, TURNS) == "SPEAKER_01"
    assert attribute_segment(dict(start=2.2, end=2.8, text="Неясно"), [], TURNS)[0]["speaker"] is None


def test_parakeet_tokens_keep_spaces_when_speaker_changes():
    words = [dict(start=.1, end=.3, word="Привет"), dict(start=.4, end=.8, word="мир"),
             dict(start=1.1, end=1.8, word="Пока")]
    groups = attribute_segment(dict(start=0, end=2, text="Привет мир Пока"), words, TURNS)
    assert [s["text"] for s in groups] == ["Привет мир", "Пока"]


def test_diarization_cache_is_separate_and_old_keys_stay_valid():
    args = ("native", "large-v3", "ru", "hash")
    plain = main.transcription_cache_key(*args, vad_filter=True, words=False)
    diarized = main.transcription_cache_key(*args, vad_filter=True, words=False, diarize=True)
    assert plain == "native:vad-v1:1:large-v3:ru:False:hash"
    assert plain != diarized


@pytest.mark.parametrize("stream", [False, True])
def test_worker_returns_labelled_segments_and_words(monkeypatch, stream):
    options = []
    class ASR:
        def transcribe(self, *args, **kwargs):
            options.append(kwargs)
            words = [SimpleNamespace(start=.1, end=.8, word=" Привет"),
                     SimpleNamespace(start=1.1, end=1.8, word=" Пока")]
            return [SimpleNamespace(start=0, end=2, text="Привет Пока", words=words)], SimpleNamespace(language="ru", language_probability=1)
    monkeypatch.setattr(main, "diarize_audio", lambda p, **k: TURNS)
    monkeypatch.setattr(main, "_create_model_entry", lambda p: ASR())
    monkeypatch.setattr(main, "ManagedModel", None)
    monkeypatch.setattr(main, "audio_duration_seconds", lambda p: 4)
    monkeypatch.setattr(main, "append_metric", lambda m: None)
    monkeypatch.setattr(main, "reset_cuda_peak_memory", lambda: None)
    monkeypatch.setattr(main, "peak_vram_mb", lambda: 0)
    requests, responses = Queue(), Queue()
    requests.put(dict(request_id="test", model="large-v3", audio_bytes=b"audio", diarize=True, words=True, stream=stream))
    checks = iter([False, True])
    main.model_worker(requests, responses, SimpleNamespace(is_set=lambda: next(checks)), {})
    messages = []
    while not responses.empty():
        messages.append(responses.get_nowait())
    result = messages[-1]["result"]
    assert options[0]["word_timestamps"] is True
    assert [s["speaker"] for s in result["segments"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert result["speaker_text"] == "Спикер 1: Привет\nСпикер 2: Пока"
    assert result["words"][1]["speaker"] == "SPEAKER_01"
    if stream:
        assert [m["segment"]["speaker"] for m in messages[:-1]] == ["SPEAKER_00", "SPEAKER_01"]
        assert [len(m["words"]) for m in messages[:-1]] == [1, 1]


def test_gigaam_transcribes_each_turn_with_native_offsets(monkeypatch):
    monkeypatch.setattr(gigaam_backend, "decode_audio", lambda *a, **k: np.zeros(4 * 16000))
    monkeypatch.setattr(gigaam_backend, "get_speech_timestamps", lambda *a: pytest.fail("turns already define speech"))
    calls = []
    class Decoder:
        def transcribe(self, path):
            calls.append(path)
            return "реплика"
    segments, _ = gigaam_backend.GigaAMModel(Decoder()).transcribe("audio.wav", speaker_turns=TURNS)
    assert [(s.start, s.end) for s in segments] == [(0, 1), (1, 2), (3, 4)]
    assert len(calls) == 3


@pytest.mark.parametrize("cached", [False, True])
def test_openai_fresh_and_cached_diarization(monkeypatch, cached):
    monkeypatch.setenv("API_KEY", "DIARIZATION_TEST_ONLY")
    monkeypatch.setattr(main, "pending_results", {})
    result = dict(text="Привет", speaker_text="Спикер 1: Привет", speaker_turns=TURNS,
                  diarization_model="pyannote/speaker-diarization-community-1",
                  segments=[dict(start=0, end=1, text="Привет", speaker="SPEAKER_00")])
    monkeypatch.setattr(main, "_get_cached_result", lambda k: result if cached else None)
    monkeypatch.setattr(main, "append_cache_metric", lambda *a: None)
    def put(request):
        assert request["diarize"] is True
        assert "community-1-v2" in request["cache_key"]
        main.pending_results.pop(request["request_id"]).set_result(result)
    monkeypatch.setattr(main.app.state, "request_queue", SimpleNamespace(put=put), raising=False)
    response = TestClient(main.app).post("/v1/audio/transcriptions",
        headers={"Authorization": "Bearer DIARIZATION_TEST_ONLY"},
        files={"file": ("voice.wav", b"fixture", "audio/wav")},
        data={"diarize": "true", "response_format": "verbose_json", "model": "large-v3"})
    assert response.status_code == 200
    assert response.json()["segments"][0]["speaker"] == "SPEAKER_00"
    assert response.json()["speaker_turns"] == TURNS
    assert transcription_response_payload(result)["speaker_text"] == "Спикер 1: Привет"


def test_native_cached_stream_keeps_word_speaker_labels(monkeypatch):
    result = dict(text="Привет", speaker_text="Спикер 1: Привет",
                  segments=[dict(start=0, end=1, text="Привет", speaker="SPEAKER_00")],
                  words=[dict(start=.1, end=.8, word="Привет", speaker="SPEAKER_00")])
    monkeypatch.setattr(main, "_get_cached_result", lambda k: result)
    monkeypatch.setattr(main, "append_cache_metric", lambda *a: None)
    monkeypatch.setattr(main, "ALLOWED_API_KEYS", {"DIARIZATION_TEST_ONLY"})
    response = TestClient(main.app).post("/transcribe",
        params=dict(model="large-v3", stream="true", words="true", diarize="true", api_key="DIARIZATION_TEST_ONLY"),
        files={"file": ("voice.wav", b"fixture", "audio/wav")})
    import json
    messages = [json.loads(line) for line in response.text.splitlines()]
    assert messages[0]["words"][0]["speaker"] == "SPEAKER_00"
    assert messages[-1]["result"]["speaker_text"] == "Спикер 1: Привет"


def test_openai_stream_surfaces_diarization_failure(monkeypatch):
    monkeypatch.setenv("API_KEY", "DIARIZATION_TEST_ONLY")
    monkeypatch.setattr(main, "pending_streams", {})
    monkeypatch.setattr(main, "_get_cached_result", lambda k: None)
    def put(request):
        queue = main.pending_streams[request["request_id"]]
        queue.put_nowait({"error": "Community-1 недоступна"})
        queue.put_nowait(None)
    monkeypatch.setattr(main.app.state, "request_queue", SimpleNamespace(put=put), raising=False)
    response = TestClient(main.app).post("/v1/audio/transcriptions",
        headers={"Authorization": "Bearer DIARIZATION_TEST_ONLY"},
        files={"file": ("voice.wav", b"fixture", "audio/wav")},
        data={"diarize": "true", "stream": "true", "model": "large-v3"})
    import json
    event = json.loads(response.text.removeprefix("data: ").strip())
    assert event == {"type": "error", "error": {"message": "Community-1 недоступна"}}
    assert not main.pending_streams


def test_long_gap_restores_original_timing_and_identity():
    from diarization_backend import _restore_turns
    turns = [dict(start=0, end=4, speaker="SPEAKER_00"),
             dict(start=4, end=6, speaker="SPEAKER_01")]
    restored = _restore_turns(turns, [(0, 2, 0), (2, 6, 1262)])
    assert [(t['start'], t['end'], t['speaker']) for t in restored] == [
        (0, 2, 'SPEAKER_00'), (1262, 1264, 'SPEAKER_00'), (1264, 1266, 'SPEAKER_01')]


@pytest.mark.parametrize('cached', [True, False])
@pytest.mark.parametrize('stream', [True, False])
def test_standard_openai_diarized_format(monkeypatch, cached, stream):
    import json
    from openai.types.audio.transcription_diarized import TranscriptionDiarized
    monkeypatch.setenv('API_KEY', 'DIARIZATION_TEST_ONLY')
    monkeypatch.setattr(main, 'pending_results', {})
    monkeypatch.setattr(main, 'pending_streams', {})
    result = dict(text='Привет Пока', duration=1266, segments=[
        dict(start=0, end=1, speaker='SPEAKER_00', text='Привет'),
        dict(start=1265, end=1266, speaker='SPEAKER_01', text='Пока')])
    monkeypatch.setattr(main, '_get_cached_result', lambda k: result if cached else None)
    monkeypatch.setattr(main, 'append_cache_metric', lambda *a: None)
    def put(request):
        assert request['diarize'] is True
        if stream:
            queue = main.pending_streams[request['request_id']]
            for seg in result['segments']:
                queue.put_nowait({'segment': seg})
            queue.put_nowait({'result': result})
            queue.put_nowait(None)
        else:
            main.pending_results.pop(request['request_id']).set_result(result)
    monkeypatch.setattr(main.app.state, 'request_queue', SimpleNamespace(put=put), raising=False)
    response = TestClient(main.app).post('/v1/audio/transcriptions',
        headers={'Authorization': 'Bearer DIARIZATION_TEST_ONLY'},
        files={'file': ('voice.flac', b'fixture', 'audio/flac')},
        data=dict(model='whisper-1', response_format='diarized_json', chunking_strategy='auto', stream=str(stream).lower()))
    assert response.status_code == 200
    if stream:
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        segments = [e for e in events if e['type'] == 'transcript.text.segment']
        assert [e['speaker'] for e in segments] == ['A', 'B']
        assert segments[1]['start'] == 1265
    else:
        parsed = TranscriptionDiarized.model_validate(response.json())
        assert parsed.duration == 1266
        assert parsed.task == 'transcribe'
        assert [s.speaker for s in parsed.segments] == ['A', 'B']


def test_vad_blocks_are_bounded_and_overlap_is_merged(monkeypatch, tmp_path):
    from diarization_backend import _compact_speech, SAMPLE_RATE
    import faster_whisper.vad
    lengths = []
    def vad(audio, options):
        lengths.append(len(audio))
        return [dict(start=0, end=len(audio))]
    monkeypatch.setattr(faster_whisper.vad, 'get_speech_timestamps', vad)
    samples = np.zeros(601 * SAMPLE_RATE, dtype=np.float32)
    compact, mapping = _compact_speech(samples, tmp_path / 'speech.pcm')
    assert max(lengths) <= 302 * SAMPLE_RATE
    assert len(compact) == len(samples)
    assert mapping == [(0, 601, 0)]


def test_community_gpu_load_uses_broker_and_no_unmanaged_fallback(monkeypatch):
    monkeypatch.setattr(main.torch.cuda, 'is_available', lambda: True)
    calls=[]
    monkeypatch.setattr(main, 'ManagedModel', lambda **kwargs: calls.append(kwargs) or kwargs)
    entry=main._create_diarization_entry()
    assert entry['model_id']=='whisper-community-1'
    assert entry['vram_mb']==1536
    assert entry['cpu_capable'] is False
    assert entry['client_id']==main.BROKER_CLIENT_ID
    monkeypatch.setattr(main, 'ManagedModel', None)
    with pytest.raises(RuntimeError,match='cudabroker_client'):
        main._create_diarization_entry()


@pytest.mark.parametrize('failure', [None, 'inference', 'idle', 'unload'])
def test_community_releases_gpu_even_on_errors_and_drops_live_reference(monkeypatch, failure):
    import weakref
    from contextlib import contextmanager
    events=[]
    class Model:
        device='cuda'
        def unload(self):
            events.append('unload')
            if failure=='unload':raise RuntimeError('unload failed')
    class Managed:
        def acquire(self):
            self.model=Model()
            self.ref=weakref.ref(self.model)
            events.append('load')
        @contextmanager
        def inference(self):
            events.append('active')
            try:yield self.model
            finally:
                events.append('idle')
                if failure=='idle':raise RuntimeError('idle failed')
        def release(self):
            self.model=None
            assert self.ref() is None,'live model reference prevents VRAM release'
            events.append('release')
    entry=Managed()
    monkeypatch.setattr(main,'ManagedModel',Managed)
    monkeypatch.setattr(main,'_create_diarization_entry',lambda:entry)
    def run():
        with main._diarization_model_session() as model:
            assert model.device=='cuda'
            model=None
            if failure=='inference':raise ValueError('inference failed')
    if failure:
        with pytest.raises((ValueError,RuntimeError)):run()
    else:run()
    assert events==['load','active','unload','idle','release']



def test_unload_drops_embedding_retained_by_frame_cache():
    import gc,weakref
    from functools import lru_cache
    from diarization_backend import Community1Model
    class Embedding:
        @lru_cache
        def num_frames(self,n):return n
    embedding=Embedding();embedding.num_frames(100)
    ref=weakref.ref(embedding)
    class Pipeline:
        _models={}
        _embedding=SimpleNamespace(model_=embedding)
        def to(self,device):pass
    pipeline=Pipeline();model=Community1Model(pipeline,'cuda')
    model.unload()
    assert embedding.num_frames.cache_info().currsize==0
    assert model.pipeline is None
    del embedding,pipeline,Pipeline
    gc.collect()
    assert ref() is None



def test_gigaam_tiny_turn_does_not_crash_fft_or_extend_timestamps(monkeypatch):
    import wave,torch
    monkeypatch.setattr(gigaam_backend,'decode_audio',lambda *a,**k:np.ones(16000,dtype=np.float32)*.1)
    class Decoder:
        def transcribe(self,path):
            with wave.open(path,'rb') as audio:
                waveform=torch.from_numpy(np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2').astype('float32'))
            # Real GigaAM failure: FFT window320 exceeded the270-sample input.
            spectrum=torch.stft(waveform,n_fft=320,hop_length=160,center=False,return_complex=True)
            assert spectrum.shape[1]>0
            return 'реплика'
    turns=[dict(start=0,end=270/16000,speaker='SPEAKER_00')]
    segments,_=gigaam_backend.GigaAMModel(Decoder()).transcribe('audio.wav',speaker_turns=turns)
    assert [(s.start,s.end) for s in segments]==[(0,270/16000)]
