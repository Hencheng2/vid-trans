import os
import uuid
import sys
import json
import shutil
import subprocess
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
import whisper
from deep_translator import GoogleTranslator
from datetime import timedelta

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# ── Folders ──
BASE_DIR    = Path(__file__).parent
UPLOAD_DIR  = BASE_DIR / 'uploads'
OUTPUT_DIR  = BASE_DIR / 'outputs'
JOBS_DIR    = BASE_DIR / 'jobs'          # persisted job JSON files
for d in (UPLOAD_DIR, OUTPUT_DIR, JOBS_DIR):
    d.mkdir(exist_ok=True)

_job_lock = threading.Lock()


# ══════════════════════════════════════════
#  Job persistence  (survives restarts)
# ══════════════════════════════════════════
def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def job_read(job_id: str) -> dict | None:
    p = _job_path(job_id)
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def job_write(job_id: str, data: dict):
    with _job_lock:
        _job_path(job_id).write_text(json.dumps(data))


def job_update(job_id: str, **kwargs):
    current = job_read(job_id) or {}
    current.update(kwargs)
    job_write(job_id, current)


def update_progress(job_id, stage, percent, message):
    job_update(job_id, stage=stage, percent=percent, message=message,
               done=False, error=None)


def set_job_done(job_id, output_file):
    job_update(job_id, done=True, percent=100,
               output_file=str(output_file),
               message='Processing complete!')


def set_job_error(job_id, msg):
    job_update(job_id, done=True, error=msg)
    print(f'[ERROR] {job_id}: {msg}', file=sys.stderr, flush=True)


# ══════════════════════════════════════════
#  FFmpeg detection
# ══════════════════════════════════════════
def find_ffmpeg():
    found = shutil.which('ffmpeg')
    if found:
        return found
    for p in [
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        str(Path.home() / 'ffmpeg' / 'bin' / 'ffmpeg.exe'),
    ]:
        if os.path.isfile(p):
            return p
    local = BASE_DIR / 'ffmpeg.exe'
    return str(local) if local.exists() else None


FFMPEG = find_ffmpeg()


def run_ffmpeg(cmd: list):
    cmd[0] = FFMPEG
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    print(f'[ffmpeg] {" ".join(str(c) for c in cmd)}', flush=True)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       creationflags=flags)
    if r.returncode != 0:
        raise RuntimeError(
            f'FFmpeg exit {r.returncode}:\n'
            + r.stderr.decode('utf-8', errors='replace')[-3000:]
        )


# ══════════════════════════════════════════
#  Languages
# ══════════════════════════════════════════
LANGUAGES = {
    "af":"Afrikaans","sq":"Albanian","am":"Amharic","ar":"Arabic",
    "hy":"Armenian","az":"Azerbaijani","eu":"Basque","be":"Belarusian",
    "bn":"Bengali","bs":"Bosnian","bg":"Bulgarian","ca":"Catalan",
    "ceb":"Cebuano","zh-CN":"Chinese (Simplified)","zh-TW":"Chinese (Traditional)",
    "co":"Corsican","hr":"Croatian","cs":"Czech","da":"Danish",
    "nl":"Dutch","en":"English","eo":"Esperanto","et":"Estonian",
    "fi":"Finnish","fr":"French","fy":"Frisian","gl":"Galician",
    "ka":"Georgian","de":"German","el":"Greek","gu":"Gujarati",
    "ht":"Haitian Creole","ha":"Hausa","haw":"Hawaiian","iw":"Hebrew",
    "hi":"Hindi","hmn":"Hmong","hu":"Hungarian","is":"Icelandic",
    "ig":"Igbo","id":"Indonesian","ga":"Irish","it":"Italian",
    "ja":"Japanese","jw":"Javanese","kn":"Kannada","kk":"Kazakh",
    "km":"Khmer","rw":"Kinyarwanda","ko":"Korean","ku":"Kurdish",
    "ky":"Kyrgyz","lo":"Lao","la":"Latin","lv":"Latvian",
    "lt":"Lithuanian","lb":"Luxembourgish","mk":"Macedonian","mg":"Malagasy",
    "ms":"Malay","ml":"Malayalam","mt":"Maltese","mi":"Maori",
    "mr":"Marathi","mn":"Mongolian","my":"Myanmar (Burmese)","ne":"Nepali",
    "no":"Norwegian","ny":"Nyanja (Chichewa)","or":"Odia (Oriya)","ps":"Pashto",
    "fa":"Persian","pl":"Polish","pt":"Portuguese","pa":"Punjabi",
    "ro":"Romanian","ru":"Russian","sm":"Samoan","gd":"Scots Gaelic",
    "sr":"Serbian","st":"Sesotho","sn":"Shona","sd":"Sindhi",
    "si":"Sinhala","sk":"Slovak","sl":"Slovenian","so":"Somali",
    "es":"Spanish","su":"Sundanese","sw":"Swahili","sv":"Swedish",
    "tl":"Tagalog (Filipino)","tg":"Tajik","ta":"Tamil","tt":"Tatar",
    "te":"Telugu","th":"Thai","tr":"Turkish","tk":"Turkmen",
    "uk":"Ukrainian","ur":"Urdu","ug":"Uyghur","uz":"Uzbek",
    "vi":"Vietnamese","cy":"Welsh","xh":"Xhosa","yi":"Yiddish",
    "yo":"Yoruba","zu":"Zulu"
}


# ══════════════════════════════════════════
#  SRT helpers
# ══════════════════════════════════════════
def fmt_time(s):
    td = timedelta(seconds=float(s))
    ts = int(td.total_seconds())
    ms = int((td.total_seconds() - ts) * 1000)
    return f'{ts//3600:02d}:{(ts%3600)//60:02d}:{ts%60:02d},{ms:03d}'


def build_srt(segments):
    out = []
    for i, seg in enumerate(segments, 1):
        out.append(f"{i}\n{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}\n{seg['text'].strip()}\n")
    return '\n'.join(out)


# ══════════════════════════════════════════
#  Core pipeline
# ══════════════════════════════════════════
def transcribe_video(video_path, job_id):
    update_progress(job_id, 'transcribe', 10, 'Loading Whisper model…')
    model = whisper.load_model('base')
    update_progress(job_id, 'transcribe', 22, 'Transcribing audio — please wait…')
    return model.transcribe(str(video_path), verbose=False)


def translate_segments(segments, target_lang, job_id):
    update_progress(job_id, 'translate', 40, f'Translating to {LANGUAGES.get(target_lang, target_lang)}…')
    translator = GoogleTranslator(source='auto', target=target_lang)
    out = []
    total = len(segments)
    for i, seg in enumerate(segments):
        try:
            text = translator.translate(seg['text'].strip()) or seg['text']
        except Exception:
            text = seg['text']
        out.append({'start': seg['start'], 'end': seg['end'], 'text': text})
        if i % 5 == 0:
            pct = 40 + int(i / max(total, 1) * 22)
            update_progress(job_id, 'translate', pct,
                            f'Translating segment {i+1}/{total}…')
    return out


def burn_subtitles(input_video: Path, srt_path: Path, output_path: Path, job_id: str):
    if not FFMPEG:
        raise RuntimeError(
            'FFmpeg not found. Install it and make sure it is on your PATH.'
        )
    update_progress(job_id, 'render', 70, 'Burning subtitles into video…')

    # Safe path for ffmpeg subtitles filter (forward slashes, escaped colons)
    srt_safe = str(srt_path.resolve()).replace('\\', '/').replace(':', r'\:')
    style = ('FontName=Arial,FontSize=20,'
             'PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,'
             'BackColour=&H80000000,Bold=1,Outline=2,Shadow=1,'
             'Alignment=2,MarginV=30')

    # Attempt 1 — hard-burn subtitles
    try:
        run_ffmpeg([
            'ffmpeg', '-y', '-i', str(input_video),
            '-vf', f"subtitles='{srt_safe}':force_style='{style}'",
            '-c:v', 'libx264', '-c:a', 'aac', '-preset', 'fast', '-crf', '23',
            str(output_path)
        ])
        print('[ffmpeg] Hard-burn succeeded.', flush=True)
        return
    except RuntimeError as e:
        print(f'[ffmpeg] Hard-burn failed, trying soft embed:\n{e}', flush=True)

    update_progress(job_id, 'render', 78, 'Embedding subtitle track…')

    # Attempt 2 — soft subtitle stream
    try:
        run_ffmpeg([
            'ffmpeg', '-y', '-i', str(input_video), '-i', str(srt_path),
            '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text',
            str(output_path)
        ])
        print('[ffmpeg] Soft embed succeeded.', flush=True)
        return
    except RuntimeError as e:
        print(f'[ffmpeg] Soft embed failed, plain copy:\n{e}', flush=True)

    update_progress(job_id, 'render', 84, 'Copying video (SRT available separately)…')

    # Attempt 3 — plain copy (subtitles downloadable as .srt)
    run_ffmpeg(['ffmpeg', '-y', '-i', str(input_video), '-c', 'copy', str(output_path)])
    print('[ffmpeg] Plain copy fallback succeeded.', flush=True)


def process_video_job(job_id: str, video_path: Path, target_lang: str):
    try:
        # 1 — Transcribe
        result = transcribe_video(video_path, job_id)
        segments = result.get('segments', [])
        if not segments:
            set_job_error(job_id, 'No speech detected. Ensure the video has clear, audible speech.')
            return

        # 2 — Translate
        translated = translate_segments(segments, target_lang, job_id)

        # 3 — SRT
        update_progress(job_id, 'subtitle', 65, 'Generating subtitle file…')
        srt_path = OUTPUT_DIR / f'{job_id}.srt'
        srt_path.write_text(build_srt(translated), encoding='utf-8')

        # 4 — Burn
        output_video = OUTPUT_DIR / f'{job_id}_translated.mp4'
        burn_subtitles(video_path, srt_path, output_video, job_id)

        # Cleanup upload
        try:
            video_path.unlink()
        except Exception:
            pass

        set_job_done(job_id, output_video)

    except Exception as e:
        import traceback
        set_job_error(job_id, str(e))
        traceback.print_exc()


# ══════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html', languages=LANGUAGES,
                           ffmpeg_ok=bool(FFMPEG), ffmpeg_path=FFMPEG or '')


@app.route('/status')
def status():
    return jsonify({'ffmpeg': FFMPEG, 'ffmpeg_ok': bool(FFMPEG)})


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    target_lang = request.form.get('language', 'es')
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    job_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or '.mp4'
    video_path = UPLOAD_DIR / f'{job_id}{ext}'
    file.save(str(video_path))

    # Write initial state to disk immediately
    job_write(job_id, {
        'stage': 'upload', 'percent': 5,
        'message': 'Video received, starting processing…',
        'done': False, 'error': None, 'output_file': None
    })

    threading.Thread(
        target=process_video_job,
        args=(job_id, video_path, target_lang),
        daemon=True
    ).start()

    return jsonify({'job_id': job_id})


@app.route('/progress/<job_id>')
def get_progress(job_id):
    data = job_read(job_id)
    if data is None:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(data)


@app.route('/download/<job_id>')
def download_video(job_id):
    data = job_read(job_id)
    if not data:
        return jsonify({'error': 'Job not found'}), 404
    if not data.get('done') or data.get('error'):
        return jsonify({'error': 'Video not ready'}), 400
    f = Path(data['output_file'])
    if not f.exists():
        return jsonify({'error': 'Output file missing'}), 404
    return send_file(str(f), as_attachment=True,
                     download_name=f'translated_{job_id[:8]}.mp4',
                     mimetype='video/mp4')


@app.route('/stream/<job_id>')
def stream_video(job_id):
    data = job_read(job_id)
    if not data:
        return jsonify({'error': 'Job not found'}), 404
    if not data.get('done') or data.get('error'):
        return jsonify({'error': 'Video not ready'}), 400
    f = Path(data['output_file'])
    if not f.exists():
        return jsonify({'error': 'Output file missing'}), 404
    return send_file(str(f), mimetype='video/mp4')


@app.route('/subtitle/<job_id>')
def get_subtitle(job_id):
    p = OUTPUT_DIR / f'{job_id}.srt'
    if not p.exists():
        return jsonify({'error': 'Subtitle not found'}), 404
    return send_file(str(p), mimetype='text/plain',
                     as_attachment=True,
                     download_name=f'subtitles_{job_id[:8]}.srt')


@app.route('/languages')
def get_languages():
    return jsonify(LANGUAGES)


if __name__ == '__main__':
    print(f'[Startup] FFmpeg: {FFMPEG or "NOT FOUND"}')
    app.run(debug=True, host='0.0.0.0', port=5000)
