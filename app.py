import os
import uuid
import sys
import shutil
import subprocess
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
import whisper
from deep_translator import GoogleTranslator
from datetime import timedelta

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('outputs')
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# ── Progress tracking ──
job_progress = {}

# ── Locate FFmpeg (handles Windows, Mac, Linux) ──
def find_ffmpeg():
    # 1. Already on PATH?
    found = shutil.which('ffmpeg')
    if found:
        return found

    # 2. Common Windows install locations
    windows_paths = [
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
        os.path.join(os.environ.get('USERPROFILE', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
    ]
    for p in windows_paths:
        if os.path.isfile(p):
            return p

    # 3. Same directory as app.py
    local = Path(__file__).parent / 'ffmpeg.exe'
    if local.exists():
        return str(local)

    return None

FFMPEG = find_ffmpeg()

def ffmpeg_available():
    return FFMPEG is not None

# ── Languages ──
LANGUAGES = {
    "af": "Afrikaans", "sq": "Albanian", "am": "Amharic", "ar": "Arabic",
    "hy": "Armenian", "az": "Azerbaijani", "eu": "Basque", "be": "Belarusian",
    "bn": "Bengali", "bs": "Bosnian", "bg": "Bulgarian", "ca": "Catalan",
    "ceb": "Cebuano", "zh-CN": "Chinese (Simplified)", "zh-TW": "Chinese (Traditional)",
    "co": "Corsican", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "en": "English", "eo": "Esperanto", "et": "Estonian",
    "fi": "Finnish", "fr": "French", "fy": "Frisian", "gl": "Galician",
    "ka": "Georgian", "de": "German", "el": "Greek", "gu": "Gujarati",
    "ht": "Haitian Creole", "ha": "Hausa", "haw": "Hawaiian", "iw": "Hebrew",
    "hi": "Hindi", "hmn": "Hmong", "hu": "Hungarian", "is": "Icelandic",
    "ig": "Igbo", "id": "Indonesian", "ga": "Irish", "it": "Italian",
    "ja": "Japanese", "jw": "Javanese", "kn": "Kannada", "kk": "Kazakh",
    "km": "Khmer", "rw": "Kinyarwanda", "ko": "Korean", "ku": "Kurdish",
    "ky": "Kyrgyz", "lo": "Lao", "la": "Latin", "lv": "Latvian",
    "lt": "Lithuanian", "lb": "Luxembourgish", "mk": "Macedonian", "mg": "Malagasy",
    "ms": "Malay", "ml": "Malayalam", "mt": "Maltese", "mi": "Maori",
    "mr": "Marathi", "mn": "Mongolian", "my": "Myanmar (Burmese)", "ne": "Nepali",
    "no": "Norwegian", "ny": "Nyanja (Chichewa)", "or": "Odia (Oriya)", "ps": "Pashto",
    "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "pa": "Punjabi",
    "ro": "Romanian", "ru": "Russian", "sm": "Samoan", "gd": "Scots Gaelic",
    "sr": "Serbian", "st": "Sesotho", "sn": "Shona", "sd": "Sindhi",
    "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "so": "Somali",
    "es": "Spanish", "su": "Sundanese", "sw": "Swahili", "sv": "Swedish",
    "tl": "Tagalog (Filipino)", "tg": "Tajik", "ta": "Tamil", "tt": "Tatar",
    "te": "Telugu", "th": "Thai", "tr": "Turkish", "tk": "Turkmen",
    "uk": "Ukrainian", "ur": "Urdu", "ug": "Uyghur", "uz": "Uzbek",
    "vi": "Vietnamese", "cy": "Welsh", "xh": "Xhosa", "yi": "Yiddish",
    "yo": "Yoruba", "zu": "Zulu"
}


# ── Helpers ──
def update_progress(job_id, stage, percent, message):
    if job_id not in job_progress:
        job_progress[job_id] = {}
    job_progress[job_id].update({
        'stage': stage,
        'percent': percent,
        'message': message,
        'done': False,
        'error': None,
        'output_file': job_progress[job_id].get('output_file'),
    })


def set_job_done(job_id, output_file):
    job_progress[job_id]['done'] = True
    job_progress[job_id]['percent'] = 100
    job_progress[job_id]['output_file'] = str(output_file)
    job_progress[job_id]['message'] = 'Processing complete!'


def set_job_error(job_id, error_msg):
    job_progress[job_id]['error'] = error_msg
    job_progress[job_id]['done'] = True
    print(f"[ERROR] Job {job_id}: {error_msg}", file=sys.stderr)


def format_srt_time(seconds):
    td = timedelta(seconds=float(seconds))
    total_s = int(td.total_seconds())
    ms = int((td.total_seconds() - total_s) * 1000)
    h  = total_s // 3600
    m  = (total_s % 3600) // 60
    s  = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segments):
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_time(seg['start'])
        end   = format_srt_time(seg['end'])
        text  = seg['text'].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


# ── Core pipeline ──
def transcribe_video(video_path, job_id):
    update_progress(job_id, 'transcribe', 10, 'Loading Whisper model...')
    model = whisper.load_model("base")
    update_progress(job_id, 'transcribe', 20, 'Transcribing audio — this may take a while...')
    result = model.transcribe(str(video_path), verbose=False)
    return result


def translate_segments(segments, target_lang, job_id):
    lang_name = LANGUAGES.get(target_lang, target_lang)
    update_progress(job_id, 'translate', 40, f'Translating to {lang_name}...')
    translator = GoogleTranslator(source='auto', target=target_lang)
    out = []
    total = len(segments)
    for i, seg in enumerate(segments):
        try:
            translated = translator.translate(seg['text'].strip()) or seg['text']
        except Exception:
            translated = seg['text']
        out.append({'start': seg['start'], 'end': seg['end'], 'text': translated})
        if i % 5 == 0:
            pct = 40 + int((i / max(total, 1)) * 22)
            update_progress(job_id, 'translate', pct, f'Translating segment {i+1}/{total}...')
    return out


def run_ffmpeg(cmd):
    """Run an ffmpeg command list. cmd[0] should be 'ffmpeg' placeholder."""
    cmd[0] = FFMPEG
    print(f"[FFmpeg] Running: {' '.join(str(c) for c in cmd)}", flush=True)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=flags
    )
    if result.returncode != 0:
        err = result.stderr.decode('utf-8', errors='replace')[-3000:]
        raise RuntimeError(f"FFmpeg exit code {result.returncode}:\n{err}")
    return result


def burn_subtitles(input_video, srt_path, output_path, job_id):
    update_progress(job_id, 'render', 70, 'Burning subtitles into video...')

    # Windows path fix: forward slashes + escape colons for FFmpeg subtitle filter
    srt_escaped = str(srt_path.resolve()).replace('\\', '/').replace(':', r'\:')

    subtitle_style = (
        "FontName=Arial,FontSize=20,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H80000000,"
        "Bold=1,Outline=2,Shadow=1,Alignment=2,MarginV=30"
    )

    burned = False
    try:
        run_ffmpeg([
            'ffmpeg', '-y',
            '-i', str(input_video),
            '-vf', f"subtitles='{srt_escaped}':force_style='{subtitle_style}'",
            '-c:v', 'libx264', '-c:a', 'aac',
            '-preset', 'fast', '-crf', '23',
            str(output_path)
        ])
        burned = True
        print("[FFmpeg] Subtitle burn succeeded.", flush=True)
    except RuntimeError as e:
        print(f"[FFmpeg] Burn failed, trying soft-sub embed:\n{e}", flush=True)

    if not burned:
        try:
            update_progress(job_id, 'render', 78, 'Embedding subtitles as soft track...')
            run_ffmpeg([
                'ffmpeg', '-y',
                '-i', str(input_video),
                '-i', str(srt_path),
                '-c:v', 'copy', '-c:a', 'copy',
                '-c:s', 'mov_text',
                str(output_path)
            ])
            burned = True
            print("[FFmpeg] Soft subtitle embed succeeded.", flush=True)
        except RuntimeError as e2:
            print(f"[FFmpeg] Soft embed failed, plain copy:\n{e2}", flush=True)

    if not burned:
        update_progress(job_id, 'render', 82, 'Copying video (subtitles as separate SRT)...')
        run_ffmpeg([
            'ffmpeg', '-y',
            '-i', str(input_video),
            '-c', 'copy',
            str(output_path)
        ])
        print("[FFmpeg] Plain copy fallback succeeded.", flush=True)

    update_progress(job_id, 'render', 90, 'Finalizing video...')


def process_video_job(job_id, video_path, target_lang):
    try:
        # 1. Transcribe
        transcription = transcribe_video(video_path, job_id)
        segments = transcription.get('segments', [])
        if not segments:
            set_job_error(job_id, 'No speech detected in the video. Ensure the video has clear audible speech.')
            return

        # 2. Translate
        translated = translate_segments(segments, target_lang, job_id)

        # 3. Build SRT
        update_progress(job_id, 'subtitle', 65, 'Generating subtitle file...')
        srt_content = build_srt(translated)
        srt_path = OUTPUT_FOLDER / f"{job_id}.srt"
        srt_path.write_text(srt_content, encoding='utf-8')

        # 4. Burn subtitles via FFmpeg
        if not ffmpeg_available():
            set_job_error(
                job_id,
                'FFmpeg not found on this system. '
                'Download from https://ffmpeg.org/download.html, '
                'extract the zip, and add the bin\\ folder to your Windows PATH, then restart the app.'
            )
            return

        output_video = OUTPUT_FOLDER / f"{job_id}_translated.mp4"
        burn_subtitles(video_path, srt_path, output_video, job_id)

        # Cleanup uploaded source
        try:
            video_path.unlink()
        except Exception:
            pass

        set_job_done(job_id, output_video)

    except Exception as e:
        import traceback
        set_job_error(job_id, str(e))
        traceback.print_exc()


# ── Routes ──
@app.route('/')
def index():
    return render_template('index.html', languages=LANGUAGES,
                           ffmpeg_ok=ffmpeg_available(), ffmpeg_path=FFMPEG or '')


@app.route('/status')
def status():
    return jsonify({'ffmpeg': FFMPEG, 'ffmpeg_ok': ffmpeg_available()})


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
    video_path = UPLOAD_FOLDER / f"{job_id}{ext}"
    file.save(str(video_path))

    job_progress[job_id] = {
        'stage': 'upload', 'percent': 5,
        'message': 'Video received, starting processing...',
        'done': False, 'error': None, 'output_file': None
    }

    threading.Thread(target=process_video_job,
                     args=(job_id, video_path, target_lang),
                     daemon=True).start()

    return jsonify({'job_id': job_id})


@app.route('/progress/<job_id>')
def get_progress(job_id):
    if job_id not in job_progress:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job_progress[job_id])


@app.route('/download/<job_id>')
def download_video(job_id):
    if job_id not in job_progress:
        return jsonify({'error': 'Job not found'}), 404
    info = job_progress[job_id]
    if not info.get('done') or info.get('error'):
        return jsonify({'error': 'Video not ready'}), 400
    f = Path(info['output_file'])
    if not f.exists():
        return jsonify({'error': 'Output file missing'}), 404
    return send_file(str(f), as_attachment=True,
                     download_name=f"translated_{job_id[:8]}.mp4",
                     mimetype='video/mp4')


@app.route('/stream/<job_id>')
def stream_video(job_id):
    if job_id not in job_progress:
        return jsonify({'error': 'Job not found'}), 404
    info = job_progress[job_id]
    if not info.get('done') or info.get('error'):
        return jsonify({'error': 'Video not ready'}), 400
    f = Path(info['output_file'])
    if not f.exists():
        return jsonify({'error': 'Output file missing'}), 404
    return send_file(str(f), mimetype='video/mp4')


@app.route('/subtitle/<job_id>')
def get_subtitle(job_id):
    srt_path = OUTPUT_FOLDER / f"{job_id}.srt"
    if not srt_path.exists():
        return jsonify({'error': 'Subtitle not found'}), 404
    return send_file(str(srt_path), mimetype='text/plain',
                     as_attachment=True, download_name=f"subtitles_{job_id[:8]}.srt")


@app.route('/languages')
def get_languages():
    return jsonify(LANGUAGES)


if __name__ == '__main__':
    print(f"[Startup] FFmpeg path: {FFMPEG or 'NOT FOUND'}")
    if not FFMPEG:
        print("[WARNING] FFmpeg not found! Subtitle burning will fail.")
        print("          Download: https://ffmpeg.org/download.html")
        print("          Extract and add the bin\\ folder to your Windows PATH.")
    app.run(debug=True, host='0.0.0.0', port=5000)