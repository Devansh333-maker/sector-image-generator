from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import sqlite3
import time
import uuid
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps
from pydantic import BaseModel, field_validator


# ============================================================
# App constants
# ============================================================
APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = APP_ROOT / "data"
OUTPUT_ROOT = APP_ROOT / "outputs"
UPLOAD_ROOT = DATA_ROOT / "uploads"
DB_PATH = DATA_ROOT / "jobs.db"
TARGET_SIZE = (584, 572)
DEFAULT_API_SIZE = "832x816"
SENTIMENTS = ("generic", "bullish", "neutral", "bearish")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

for folder in (DATA_ROOT, OUTPUT_ROOT, UPLOAD_ROOT):
    folder.mkdir(parents=True, exist_ok=True)

load_dotenv(APP_ROOT / ".env")


def get_secret_or_env(name: str, default: str = "") -> str:
    """Read Streamlit Cloud secrets first, then local environment variables."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, default)).strip()


TEXT_MODEL = get_secret_or_env("OPENAI_TEXT_MODEL", "gpt-5.6")
IMAGE_MODEL = get_secret_or_env("OPENAI_IMAGE_MODEL", "gpt-image-2")
API_SIZE = get_secret_or_env("OPENAI_IMAGE_SIZE", DEFAULT_API_SIZE)


# ============================================================
# Structured concept planning
# ============================================================
class Concept(BaseModel):
    concept_number: int
    concept_name: str
    base_scene: str
    main_subject: str
    environment: str
    camera_composition: str
    distinguishing_feature: str


class SectorPlan(BaseModel):
    sector_display_name: str
    sector_key: str
    concepts: list[Concept]

    @field_validator("concepts")
    @classmethod
    def exactly_five_concepts(cls, value: list[Concept]) -> list[Concept]:
        if len(value) != 5:
            raise ValueError("The planner must return exactly five concepts.")
        numbers = sorted(item.concept_number for item in value)
        if numbers != [1, 2, 3, 4, 5]:
            raise ValueError("Concept numbers must be exactly 1, 2, 3, 4 and 5.")
        return value


# ============================================================
# Generic helpers
# ============================================================
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_filename(value: str) -> str:
    return Path(value).name.replace("\x00", "")


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "sector"


def get_api_key() -> str:
    return get_secret_or_env("OPENAI_API_KEY", "")


def get_client() -> OpenAI:
    key = get_api_key()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Open the .env file, paste your API key, "
            "save it, and restart the app."
        )
    return OpenAI(api_key=key)


def decode_image(b64_data: str) -> Image.Image:
    raw = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def fit_and_save(image: Image.Image, path: Path) -> None:
    """Centre-crop to the target ratio, resize without stretching, save PNG."""
    final = ImageOps.fit(
        image.convert("RGB"),
        TARGET_SIZE,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    final.save(path, "PNG", optimize=True)
    with Image.open(path) as check:
        if check.size != TARGET_SIZE or check.format != "PNG":
            raise RuntimeError(f"Technical image check failed for {path.name}.")


def save_master(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, "PNG", optimize=True)


def normalise_existing_image(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        fit_and_save(image.convert("RGB"), destination)


def call_with_backoff(func, *, attempts: int = 5):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # API exceptions vary by SDK version
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(min(30, (2 ** attempt) + (attempt * 0.35)))
    raise RuntimeError(f"API request failed after {attempts} attempts: {last_error}") from last_error


def image_request_with_size_fallback(func_builder):
    """Use the configured close-ratio size; fall back to 1024x1024 if rejected."""
    sizes = [API_SIZE]
    if API_SIZE != "1024x1024":
        sizes.append("1024x1024")
    last_error: Exception | None = None
    for size in sizes:
        try:
            return call_with_backoff(lambda: func_builder(size))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Image request failed for all configured sizes: {last_error}")


def generate_from_text(client: OpenAI, prompt: str, quality: str) -> Image.Image:
    result = image_request_with_size_fallback(
        lambda size: client.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            size=size,
            quality=quality,
        )
    )
    return decode_image(result.data[0].b64_json)


def generate_edit(client: OpenAI, source_path: Path, prompt: str, quality: str) -> Image.Image:
    def build(size: str):
        with source_path.open("rb") as image_file:
            return client.images.edit(
                model=IMAGE_MODEL,
                image=image_file,
                prompt=prompt,
                size=size,
                quality=quality,
            )

    result = image_request_with_size_fallback(build)
    return decode_image(result.data[0].b64_json)


def create_zip(output_dir: Path, sector_key: str) -> Path:
    zip_path = output_dir / f"{sector_key}_image_library.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for png in sorted(output_dir.glob("main_*.png")):
            archive.write(png, arcname=png.name)
        manifest = output_dir / "manifest.json"
        plan = output_dir / "plan.json"
        if manifest.exists():
            archive.write(manifest, arcname="manifest.json")
        if plan.exists():
            archive.write(plan, arcname="plan.json")
    return zip_path


# ============================================================
# SQLite job queue
# ============================================================
def db_connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA busy_timeout=60000;")
    return connection


def init_db() -> None:
    with db_connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                sector TEXT NOT NULL,
                sector_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                output_dir TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def add_job(job_type: str, sector: str, sector_key: str, payload: dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    output_dir = OUTPUT_ROOT / job_type / sector_key / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    with db_connect() as connection:
        connection.execute(
            """
            INSERT INTO jobs
            (id, job_type, sector, sector_key, payload_json, status, progress,
             message, output_dir, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'pending', 0, 'Waiting in queue', ?, ?, ?)
            """,
            (
                job_id,
                job_type,
                sector,
                sector_key,
                json.dumps(payload),
                str(output_dir),
                now,
                now,
            ),
        )
        connection.commit()
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    allowed = {"status", "progress", "message", "payload_json"}
    clean_fields = {key: value for key, value in fields.items() if key in allowed}
    clean_fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{key} = ?" for key in clean_fields)
    values = list(clean_fields.values()) + [job_id]
    with db_connect() as connection:
        connection.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", values)
        connection.commit()


def get_job(job_id: str) -> sqlite3.Row:
    with db_connect() as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"Job {job_id} was not found.")
    return row


def list_jobs(status: str | None = None) -> list[sqlite3.Row]:
    with db_connect() as connection:
        if status:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC", (status,)
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            ).fetchall()
    return rows


def jobs_dataframe() -> pd.DataFrame:
    rows = list_jobs()
    if not rows:
        return pd.DataFrame(
            columns=["id", "type", "sector", "status", "progress", "message", "updated"]
        )
    return pd.DataFrame(
        [
            {
                "id": row["id"],
                "type": row["job_type"],
                "sector": row["sector"],
                "status": row["status"],
                "progress": f"{row['progress']}%",
                "message": row["message"],
                "updated": row["updated_at"],
            }
            for row in rows
        ]
    )


def set_failed_to_pending() -> int:
    with db_connect() as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET status = 'pending', message = 'Queued for retry', updated_at = ?
            WHERE status = 'failed'
            """,
            (utc_now(),),
        )
        connection.commit()
        return cursor.rowcount


def set_running_to_pending() -> int:
    """Recover jobs left marked running after sleep, shutdown, or interruption.

    Use only after the previous queue runner has been stopped or the app has been
    restarted. Saved output files are retained and skipped when the job resumes.
    """
    with db_connect() as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET status = 'pending',
                message = 'Recovered after interruption; resuming missing files',
                updated_at = ?
            WHERE status = 'running'
            """,
            (utc_now(),),
        )
        connection.commit()
        return cursor.rowcount


def delete_completed_jobs() -> int:
    with db_connect() as connection:
        cursor = connection.execute("DELETE FROM jobs WHERE status = 'completed'")
        connection.commit()
        return cursor.rowcount


# ============================================================
# Editable prompt studio
# ============================================================
STRICT_EXCLUSIONS_DEFAULT = """STRICTLY EXCLUDE all text, letters, numbers, percentages, currency symbols,
dates, company names, logos, trademarks, product labels, readable signage,
readable screens, UI, badges and watermarks. Use generic unbranded assets.
Avoid distorted machinery, impossible architecture, duplicate objects, crowds
and unrelated items."""

DEFAULT_PROMPTS: dict[str, str] = {
    "planner_system": """You are a financial-sector visual art director. Return a production-ready structured concept plan.""",
    "planner_user": """Create exactly five meaningfully different visual concepts for a reusable,
photorealistic financial-editorial main-image library for: {sector}

The five concepts must represent different business activities or sides of the
sector—not merely different weather, lighting or camera angles. Consider core
infrastructure, manufacturing/operations, technology/R&D, products/services,
customer use, logistics/distribution, testing and future innovation where relevant.

Every concept will later have four related versions: generic, bullish, neutral
and bearish. The scene must be generic and unbranded, with no names, logos,
readable signage, numbers, dates, percentages or article-specific facts.

Images will be used as premium right-side blog-banner visuals and cropped with
rounded corners. Keep important subjects away from extreme edges. Use realistic
machinery, architecture, products and spatial composition.

Additional user style notes:
{style_notes}

Return a short lowercase underscore sector_key and concept numbers 1–5.""",
    "generic_image": """Create a premium, photorealistic financial-editorial main image for the {sector}
sector. This is the GENERIC master for a reusable image library.

CONCEPT
Name: {concept_name}
Base scene: {base_scene}
Main subject: {main_subject}
Environment: {environment}
Camera/composition: {camera_composition}
Distinctive feature: {distinguishing_feature}

STYLE
Natural, balanced, non-directional lighting. Strong sector recognition, one clear
focal point, realistic materials, clean depth and premium business-publication
quality. No bullish, bearish or sideways market indicator. No chart or arrow.
Keep the scene suitable for rounded-corner cropping.

Additional notes: {style_notes}

{strict_exclusions}
Fully rendered photographic scene only.""",
    "edit_image": """Edit the supplied {source_sentiment} reference image into the {target_sentiment}
version of the same reusable {sector} sector concept.

PRESERVE the core industry scene, subject identity, camera angle, spatial layout,
environment and recognisable concept. Do not replace it with a new scene and do
not add unrelated objects.

TARGET CHANGE
{sentiment_instruction}

Additional notes: {style_notes}

{strict_exclusions}
Any financial overlay must contain no values, remain secondary and not cover the
main subject. Maintain realistic machinery, architecture and perspective.""",
    "bullish_instruction": """Transform the reference into a clearly positive growth version. Preserve the
same scene and composition. Use bright daylight, sunrise or golden-hour optimism,
healthy colours and active operations. Add only one subtle green, teal or cyan
rising financial line or arrow, naturally integrated and visually secondary.""",
    "neutral_instruction": """Transform the reference into a stable, balanced version. Preserve the same scene
and composition. Use natural, blue, cyan or silver-balanced lighting and normal
operations. Add at most one subtle sideways/stable line or restrained data grid.
It must not look bullish or bearish.""",
    "bearish_instruction": """Transform the reference into a realistic pressure/headwinds version. Preserve
the same scene and composition. Use overcast, muted, darker or evening lighting,
with reduced activity where appropriate. Add only one subtle red declining line
or downward arrow. Keep it professional—not destroyed, abandoned or apocalyptic.""",
    "generic_instruction": """Transform the reference into a clean generic sector image. Preserve the same
scene and composition. Use balanced natural lighting and remove all directional
market cues, arrows, charts, sentiment colours and exaggerated mood.""",
    "strict_exclusions": STRICT_EXCLUSIONS_DEFAULT,
}


def current_prompt_config() -> dict[str, str]:
    if "prompt_config" not in st.session_state:
        st.session_state["prompt_config"] = dict(DEFAULT_PROMPTS)
    return dict(st.session_state["prompt_config"])


def render_template(template: str, **values: Any) -> str:
    try:
        return template.format(**values).strip()
    except KeyError as exc:
        raise RuntimeError(
            f"Prompt template contains an unsupported placeholder: {exc}. "
            "Use only the placeholders shown in Prompt Studio."
        ) from exc

# ============================================================
# Prompt planning and image prompts
# ============================================================
def build_plan_prompt(sector: str, style_notes: str, prompt_config: dict[str, str]) -> str:
    return render_template(
        prompt_config["planner_user"],
        sector=sector,
        style_notes=style_notes or "No additional notes.",
    )


def plan_sector(
    client: OpenAI,
    sector: str,
    style_notes: str,
    prompt_config: dict[str, str],
) -> SectorPlan:
    response = call_with_backoff(
        lambda: client.responses.parse(
            model=TEXT_MODEL,
            input=[
                {"role": "system", "content": prompt_config["planner_system"]},
                {
                    "role": "user",
                    "content": build_plan_prompt(sector, style_notes, prompt_config),
                },
            ],
            text_format=SectorPlan,
        )
    )
    plan = response.output_parsed
    if plan is None:
        raise RuntimeError("The concept planner returned no structured output.")
    plan.sector_key = slugify(plan.sector_key or sector)
    return plan


def generic_prompt(
    sector: str,
    concept: dict[str, Any],
    style_notes: str,
    prompt_config: dict[str, str],
) -> str:
    return render_template(
        prompt_config["generic_image"],
        sector=sector,
        concept_name=concept["concept_name"],
        base_scene=concept["base_scene"],
        main_subject=concept["main_subject"],
        environment=concept["environment"],
        camera_composition=concept["camera_composition"],
        distinguishing_feature=concept["distinguishing_feature"],
        style_notes=style_notes or "None.",
        strict_exclusions=prompt_config["strict_exclusions"],
    )


def edit_prompt(
    sector: str,
    source_sentiment: str,
    target_sentiment: str,
    style_notes: str,
    prompt_config: dict[str, str],
) -> str:
    instruction_key = f"{target_sentiment}_instruction"
    return render_template(
        prompt_config["edit_image"],
        sector=sector,
        source_sentiment=source_sentiment,
        target_sentiment=target_sentiment,
        sentiment_instruction=prompt_config[instruction_key],
        style_notes=style_notes or "None.",
        strict_exclusions=prompt_config["strict_exclusions"],
    )


# ============================================================
# New-sector pack generation
# ============================================================
def load_or_create_plan(
    client: OpenAI,
    output_dir: Path,
    sector: str,
    style_notes: str,
    prompt_config: dict[str, str],
) -> dict[str, Any]:
    plan_path = output_dir / "plan.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text(encoding="utf-8"))
    plan = plan_sector(client, sector, style_notes, prompt_config).model_dump()
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def process_new_pack(job: sqlite3.Row, client: OpenAI) -> None:
    payload = json.loads(job["payload_json"])
    sector = job["sector"]
    sector_key = job["sector_key"]
    output_dir = Path(job["output_dir"])
    quality = payload.get("quality", "medium")
    style_notes = payload.get("style_notes", "")
    prompt_config = payload.get("prompt_config", dict(DEFAULT_PROMPTS))
    concept_limit = int(payload.get("concept_limit", 5))
    concepts_to_generate = max(1, min(5, concept_limit))
    masters_dir = output_dir / "_masters"
    masters_dir.mkdir(parents=True, exist_ok=True)

    plan = load_or_create_plan(client, output_dir, sector, style_notes, prompt_config)
    concepts = sorted(plan["concepts"], key=lambda item: item["concept_number"])[
        :concepts_to_generate
    ]
    total_images = len(concepts) * 4
    completed = sum(1 for path in output_dir.glob("main_*.png"))

    manifest: dict[str, Any] = {
        "job_type": "new_pack",
        "sector": sector,
        "sector_key": sector_key,
        "created_at": job["created_at"],
        "updated_at": utc_now(),
        "text_model": TEXT_MODEL,
        "image_model": IMAGE_MODEL,
        "api_size": API_SIZE,
        "final_size": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
        "quality": quality,
        "concepts": [],
    }

    for concept in concepts:
        number = int(concept["concept_number"])
        master_path = masters_dir / f"generic_master_{number:02d}.png"
        generic_path = output_dir / f"main_{sector_key}_generic_{number:02d}.png"

        if not master_path.exists():
            update_job(
                job["id"],
                message=f"Concept {number}: generating Generic master",
                progress=int((completed / total_images) * 100),
            )
            generic_image = generate_from_text(
                client, generic_prompt(sector, concept, style_notes, prompt_config), quality
            )
            save_master(generic_image, master_path)
        if not generic_path.exists():
            with Image.open(master_path) as generic_image:
                fit_and_save(generic_image, generic_path)
            completed += 1
            update_job(job["id"], progress=int((completed / total_images) * 100))

        concept_files = {"generic": generic_path.name}

        for sentiment in ("bullish", "neutral", "bearish"):
            target_path = output_dir / f"main_{sector_key}_{sentiment}_{number:02d}.png"
            if not target_path.exists():
                update_job(
                    job["id"],
                    message=f"Concept {number}: generating {sentiment.title()}",
                    progress=int((completed / total_images) * 100),
                )
                edited = generate_edit(
                    client,
                    master_path,
                    edit_prompt(sector, "generic", sentiment, style_notes, prompt_config),
                    quality,
                )
                fit_and_save(edited, target_path)
                completed += 1
                update_job(job["id"], progress=int((completed / total_images) * 100))
            concept_files[sentiment] = target_path.name

        manifest["concepts"].append(
            {
                "concept_number": number,
                "concept_name": concept["concept_name"],
                "files": concept_files,
            }
        )

    manifest["updated_at"] = utc_now()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    create_zip(output_dir, sector_key)


# ============================================================
# Existing-pack completion
# ============================================================
def parse_existing_filename(filename: str, default_sector: str) -> dict[str, Any]:
    stem = Path(filename).stem.lower()
    sentiment = "generic"
    for key in ("bullish", "neutral", "bearish", "generic"):
        if re.search(rf"(^|[_\-\s]){key}([_\-\s]|$)", stem):
            sentiment = key
            break
    if sentiment == "generic":
        if re.search(r"(^|[_\-\s])(up|rise|rising)([_\-\s]|$)", stem):
            sentiment = "bullish"
        elif re.search(r"(^|[_\-\s])(down|fall|falling)([_\-\s]|$)", stem):
            sentiment = "bearish"
        elif re.search(r"(^|[_\-\s])(sideways|stable)([_\-\s]|$)", stem):
            sentiment = "neutral"

    number_matches = re.findall(r"(?<!\d)(\d{1,2})(?!\d)", stem)
    concept_number = int(number_matches[-1]) if number_matches else 1
    concept_number = max(1, min(99, concept_number))

    sector_key = slugify(default_sector)
    standard_match = re.match(
        r"main_(.+?)_(bullish|neutral|bearish|generic)_(\d{1,2})$", stem
    )
    if standard_match:
        sector_key = slugify(standard_match.group(1))
        sentiment = standard_match.group(2)
        concept_number = int(standard_match.group(3))

    return {
        "include": True,
        "filename": filename,
        "sector_key": sector_key,
        "concept_number": concept_number,
        "current_variation": sentiment,
        "concept_notes": "",
    }


def stage_uploaded_files(uploaded_files, default_sector: str) -> tuple[Path, list[dict[str, Any]]]:
    stage_dir = UPLOAD_ROOT / uuid.uuid4().hex[:12]
    stage_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []

    for uploaded in uploaded_files:
        filename = safe_filename(uploaded.name)
        suffix = Path(filename).suffix.lower()
        saved_path = stage_dir / filename
        saved_path.write_bytes(uploaded.getvalue())

        if suffix == ".zip":
            extract_dir = stage_dir / f"extracted_{Path(filename).stem}"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(saved_path) as archive:
                for member in archive.infolist():
                    member_name = safe_filename(member.filename)
                    if not member_name:
                        continue
                    if Path(member_name).suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    destination = extract_dir / member_name
                    destination.write_bytes(archive.read(member))
                    image_paths.append(destination)
        elif suffix in IMAGE_EXTENSIONS:
            image_paths.append(saved_path)

    rows = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                image.verify()
            row = parse_existing_filename(path.name, default_sector)
            row["source_path"] = str(path)
            rows.append(row)
        except Exception:
            continue

    return stage_dir, rows


def source_priority(sentiment: str) -> int:
    return {"generic": 0, "neutral": 1, "bullish": 2, "bearish": 3}.get(sentiment, 9)


def process_existing_pack(job: sqlite3.Row, client: OpenAI) -> None:
    payload = json.loads(job["payload_json"])
    sector = job["sector"]
    sector_key = job["sector_key"]
    output_dir = Path(job["output_dir"])
    quality = payload.get("quality", "medium")
    target_variations = payload.get("target_variations", list(SENTIMENTS))
    style_notes = payload.get("style_notes", "")
    prompt_config = payload.get("prompt_config", dict(DEFAULT_PROMPTS))
    sources = payload.get("sources", [])

    grouped: dict[int, list[dict[str, Any]]] = {}
    for source in sources:
        grouped.setdefault(int(source["concept_number"]), []).append(source)

    total_targets = len(grouped) * len(target_variations)
    completed = sum(1 for path in output_dir.glob("main_*.png"))
    manifest: dict[str, Any] = {
        "job_type": "complete_existing",
        "sector": sector,
        "sector_key": sector_key,
        "created_at": job["created_at"],
        "updated_at": utc_now(),
        "image_model": IMAGE_MODEL,
        "api_size": API_SIZE,
        "final_size": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
        "quality": quality,
        "concepts": [],
    }

    for concept_number in sorted(grouped):
        entries = sorted(grouped[concept_number], key=lambda item: source_priority(item["current_variation"]))
        concept_notes = next((item.get("concept_notes", "") for item in entries if item.get("concept_notes")), "")
        existing_by_sentiment: dict[str, Path] = {}

        for entry in entries:
            source_path = Path(entry["source_path"])
            sentiment = entry["current_variation"]
            destination = output_dir / f"main_{sector_key}_{sentiment}_{concept_number:02d}.png"
            if not destination.exists():
                normalise_existing_image(source_path, destination)
                completed += 1
            existing_by_sentiment[sentiment] = destination

        # Create Generic first when it is requested and missing. The Generic then becomes
        # the clean master for the remaining sentiment variants.
        generic_path = output_dir / f"main_{sector_key}_generic_{concept_number:02d}.png"
        if "generic" in target_variations and not generic_path.exists():
            best_entry = entries[0]
            update_job(
                job["id"],
                message=f"Concept {concept_number}: creating Generic from existing image",
                progress=min(99, int((completed / max(1, total_targets)) * 100)),
            )
            generated = generate_edit(
                client,
                Path(best_entry["source_path"]),
                edit_prompt(
                    sector,
                    best_entry["current_variation"],
                    "generic",
                    concept_notes or style_notes,
                    prompt_config,
                ),
                quality,
            )
            fit_and_save(generated, generic_path)
            existing_by_sentiment["generic"] = generic_path
            completed += 1

        if generic_path.exists():
            master_path = generic_path
            master_sentiment = "generic"
        else:
            best_entry = entries[0]
            master_path = Path(best_entry["source_path"])
            master_sentiment = best_entry["current_variation"]

        files: dict[str, str] = {
            sentiment: path.name for sentiment, path in existing_by_sentiment.items()
        }

        for sentiment in target_variations:
            target_path = output_dir / f"main_{sector_key}_{sentiment}_{concept_number:02d}.png"
            if not target_path.exists():
                update_job(
                    job["id"],
                    message=f"Concept {concept_number}: generating {sentiment.title()}",
                    progress=min(99, int((completed / max(1, total_targets)) * 100)),
                )
                edited = generate_edit(
                    client,
                    master_path,
                    edit_prompt(
                        sector,
                        master_sentiment,
                        sentiment,
                        concept_notes or style_notes,
                        prompt_config,
                    ),
                    quality,
                )
                fit_and_save(edited, target_path)
                completed += 1
            files[sentiment] = target_path.name

        manifest["concepts"].append(
            {
                "concept_number": concept_number,
                "concept_notes": concept_notes,
                "files": files,
            }
        )

    manifest["updated_at"] = utc_now()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    create_zip(output_dir, sector_key)


# ============================================================
# Job runner
# ============================================================
def process_job(job_id: str) -> tuple[str, bool, str]:
    job = get_job(job_id)
    update_job(job_id, status="running", message="Starting", progress=max(0, job["progress"]))
    try:
        client = get_client()
        if job["job_type"] == "new_pack":
            process_new_pack(job, client)
        elif job["job_type"] == "complete_existing":
            process_existing_pack(job, client)
        else:
            raise RuntimeError(f"Unsupported job type: {job['job_type']}")
        update_job(job_id, status="completed", progress=100, message="Completed")
        return job_id, True, "Completed"
    except Exception as exc:
        update_job(job_id, status="failed", message=str(exc)[:500])
        return job_id, False, str(exc)


def run_pending_jobs(max_workers: int, job_limit: int | None = None) -> None:
    pending = list(reversed(list_jobs("pending")))
    if job_limit:
        pending = pending[:job_limit]
    if not pending:
        st.info("There are no pending jobs.")
        return

    placeholder = st.empty()
    result_box = st.container()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_job, row["id"]): row["id"] for row in pending}
        while futures:
            done, _ = wait(futures, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                job_id = futures.pop(future)
                try:
                    _, success, message = future.result()
                    with result_box:
                        if success:
                            st.success(f"Job {job_id}: completed")
                        else:
                            st.error(f"Job {job_id}: {message}")
                except Exception as exc:
                    with result_box:
                        st.error(f"Job {job_id}: {exc}")
            placeholder.dataframe(jobs_dataframe(), width="stretch", hide_index=True)


# ============================================================
# Streamlit interface
# ============================================================
init_db()
st.set_page_config(
    page_title="Acme Sector Image Library Generator V2.2",
    page_icon="🖼️",
    layout="wide",
)

st.title("Acme Sector Image Library Generator — V2.2")
st.caption(
    "Bulk-generate new sector packs, complete existing packs, and customise all production prompts."
)

if not get_api_key():
    st.error(
        "OPENAI_API_KEY is missing. For local use, add it to .env. For Streamlit "
        "Cloud, add it under App settings → Secrets."
    )
    st.stop()

with st.sidebar:
    st.header("Default generation settings")
    default_quality = st.selectbox("Image quality", ["low", "medium", "high"], index=1)
    max_workers = st.slider(
        "Parallel sector jobs",
        min_value=1,
        max_value=3,
        value=1,
        help="Start with 1. Use 2 after testing your API limits. Each job generates its images sequentially.",
    )
    st.write(f"Final size: **{TARGET_SIZE[0]} × {TARGET_SIZE[1]} px**")
    st.write(f"Planner model: `{TEXT_MODEL}`")
    st.write(f"Image model: `{IMAGE_MODEL}`")
    st.warning(
        "A complete new sector pack makes 20 image requests. Start with one sector or one concept before queuing 100 sectors."
    )

new_tab, existing_tab, prompt_tab, queue_tab = st.tabs(
    [
        "1. Generate New Sector Packs",
        "2. Complete Existing Sector Packs",
        "3. Prompt Studio",
        "4. Queue & Downloads",
    ]
)

with new_tab:
    st.subheader("Bulk new-sector generation")
    st.write(
        "Paste one sector per line. The app will create a separate persistent queue job for each sector."
    )
    sectors_text = st.text_area(
        "Sectors or industries",
        height=220,
        placeholder="Agriculture Industry\nAutomobile Industry\nHealthcare Industry",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        concept_limit = st.selectbox(
            "Concepts per sector",
            [1, 2, 3, 4, 5],
            index=4,
            help="Use 1 for testing. Full library = 5 concepts × 4 sentiments = 20 images.",
        )
    with col_b:
        new_quality = st.selectbox(
            "Quality for these jobs",
            ["low", "medium", "high"],
            index=["low", "medium", "high"].index(default_quality),
            key="new_quality",
        )
    style_notes = st.text_area(
        "Optional common style notes",
        placeholder="Example: premium realistic Indian business-editorial aesthetic; no people unless essential.",
        height=90,
    )

    if st.button("Add sectors to queue", type="primary"):
        sectors = []
        seen = set()
        for line in sectors_text.splitlines():
            value = line.strip(" \t,;")
            if value and value.lower() not in seen:
                sectors.append(value)
                seen.add(value.lower())
        if not sectors:
            st.warning("Enter at least one sector.")
        else:
            created = []
            for sector in sectors:
                sector_key = slugify(sector)
                job_id = add_job(
                    "new_pack",
                    sector,
                    sector_key,
                    {
                        "quality": new_quality,
                        "concept_limit": concept_limit,
                        "style_notes": style_notes,
                        "prompt_config": current_prompt_config(),
                    },
                )
                created.append(job_id)
            st.success(f"Added {len(created)} sector job(s) to the queue.")
            st.dataframe(jobs_dataframe().head(len(created)), width="stretch", hide_index=True)

with existing_tab:
    st.subheader("Complete missing variations from existing images")
    st.write(
        "Upload individual PNG/JPG/WEBP images or a ZIP. The app preserves your scene, standardises filenames, "
        "and generates only the missing Generic, Bullish, Neutral and Bearish versions."
    )
    default_existing_sector = st.text_input(
        "Sector/industry for uploaded images",
        placeholder="Example: Agriculture Industry",
    )
    uploads = st.file_uploader(
        "Upload existing images or ZIP files",
        type=["png", "jpg", "jpeg", "webp", "zip"],
        accept_multiple_files=True,
    )
    if st.button("Analyse uploaded files", disabled=not uploads or not default_existing_sector.strip()):
        stage_dir, rows = stage_uploaded_files(uploads, default_existing_sector.strip())
        if not rows:
            st.error("No valid images were found in the upload.")
        else:
            st.session_state["existing_stage_dir"] = str(stage_dir)
            st.session_state["existing_rows"] = rows
            st.success(f"Prepared {len(rows)} image(s). Review the detected values below.")

    if "existing_rows" in st.session_state:
        source_df = pd.DataFrame(st.session_state["existing_rows"])
        display_columns = [
            "include",
            "filename",
            "sector_key",
            "concept_number",
            "current_variation",
            "concept_notes",
            "source_path",
        ]
        edited_df = st.data_editor(
            source_df[display_columns],
            width="stretch",
            hide_index=True,
            disabled=["filename", "source_path"],
            column_config={
                "include": st.column_config.CheckboxColumn("Use", default=True),
                "sector_key": st.column_config.TextColumn(
                    "Sector key", help="Lowercase underscore key used in output filenames."
                ),
                "concept_number": st.column_config.NumberColumn(
                    "Concept no.", min_value=1, max_value=99, step=1
                ),
                "current_variation": st.column_config.SelectboxColumn(
                    "Current variation", options=list(SENTIMENTS), required=True
                ),
                "concept_notes": st.column_config.TextColumn(
                    "Optional concept notes",
                    help="Example: tractor in irrigated farmland with silos. Helps guide edits.",
                ),
                "source_path": None,
            },
            key="existing_editor",
        )
        target_variations = st.multiselect(
            "Variations required in the completed pack",
            options=list(SENTIMENTS),
            default=list(SENTIMENTS),
        )
        existing_quality = st.selectbox(
            "Quality for missing variations",
            ["low", "medium", "high"],
            index=["low", "medium", "high"].index(default_quality),
            key="existing_quality",
        )
        existing_notes = st.text_area(
            "Optional common edit notes",
            placeholder="Preserve the exact machinery and camera angle; keep overlays subtle.",
            height=80,
        )

        if st.button("Add completion jobs to queue", type="primary"):
            clean_rows = edited_df[edited_df["include"] == True].to_dict("records")  # noqa: E712
            if not clean_rows:
                st.warning("Select at least one image.")
            elif not target_variations:
                st.warning("Select at least one target variation.")
            else:
                grouped_rows: dict[str, list[dict[str, Any]]] = {}
                for row in clean_rows:
                    sector_key = slugify(str(row["sector_key"]))
                    row["sector_key"] = sector_key
                    row["concept_number"] = int(row["concept_number"])
                    grouped_rows.setdefault(sector_key, []).append(row)

                created = []
                for sector_key, rows_for_sector in grouped_rows.items():
                    display_sector = default_existing_sector.strip()
                    job_id = add_job(
                        "complete_existing",
                        display_sector,
                        sector_key,
                        {
                            "quality": existing_quality,
                            "target_variations": target_variations,
                            "style_notes": existing_notes,
                            "prompt_config": current_prompt_config(),
                            "sources": rows_for_sector,
                        },
                    )
                    created.append(job_id)
                st.success(f"Added {len(created)} completion job(s) to the queue.")

with prompt_tab:
    st.subheader("Prompt Studio")
    st.write(
        "Edit the prompts used for all NEW jobs. Each queued job stores a snapshot, "
        "so changing prompts later will not alter jobs already in the queue."
    )
    st.info(
        "Prompt settings are kept for the current browser session. Download the JSON "
        "preset to your device so it can be uploaded again after a cloud restart."
    )

    uploaded_preset = st.file_uploader(
        "Load a prompt preset JSON",
        type=["json"],
        key="prompt_preset_upload",
    )
    if uploaded_preset is not None and st.button("Apply uploaded preset"):
        try:
            loaded = json.loads(uploaded_preset.getvalue().decode("utf-8"))
            missing = [key for key in DEFAULT_PROMPTS if key not in loaded]
            if missing:
                st.error(f"Preset is missing: {', '.join(missing)}")
            else:
                st.session_state["prompt_config"] = {
                    key: str(loaded[key]) for key in DEFAULT_PROMPTS
                }
                st.success("Prompt preset applied.")
                st.rerun()
        except Exception as exc:
            st.error(f"Could not read preset: {exc}")

    prompts = current_prompt_config()
    st.markdown("### Concept planner")
    prompts["planner_system"] = st.text_area(
        "Planner system prompt",
        value=prompts["planner_system"],
        height=100,
    )
    prompts["planner_user"] = st.text_area(
        "Planner user template",
        value=prompts["planner_user"],
        height=320,
        help="Allowed placeholders: {sector}, {style_notes}",
    )

    st.markdown("### Generic master generation")
    prompts["generic_image"] = st.text_area(
        "Generic image template",
        value=prompts["generic_image"],
        height=420,
        help=(
            "Allowed placeholders: {sector}, {concept_name}, {base_scene}, "
            "{main_subject}, {environment}, {camera_composition}, "
            "{distinguishing_feature}, {style_notes}, {strict_exclusions}"
        ),
    )

    st.markdown("### Variation editing")
    prompts["edit_image"] = st.text_area(
        "Shared edit template",
        value=prompts["edit_image"],
        height=360,
        help=(
            "Allowed placeholders: {sector}, {source_sentiment}, {target_sentiment}, "
            "{sentiment_instruction}, {style_notes}, {strict_exclusions}"
        ),
    )
    c1, c2 = st.columns(2)
    with c1:
        prompts["bullish_instruction"] = st.text_area(
            "Bullish instruction", value=prompts["bullish_instruction"], height=190
        )
        prompts["bearish_instruction"] = st.text_area(
            "Bearish instruction", value=prompts["bearish_instruction"], height=190
        )
    with c2:
        prompts["neutral_instruction"] = st.text_area(
            "Neutral instruction", value=prompts["neutral_instruction"], height=190
        )
        prompts["generic_instruction"] = st.text_area(
            "Generic conversion instruction", value=prompts["generic_instruction"], height=190
        )

    prompts["strict_exclusions"] = st.text_area(
        "Strict exclusions used by image prompts",
        value=prompts["strict_exclusions"],
        height=180,
    )

    save_col, reset_col, download_col = st.columns(3)
    with save_col:
        if st.button("Save prompts for new jobs", type="primary"):
            st.session_state["prompt_config"] = prompts
            st.success("Saved. New jobs will use this prompt configuration.")
    with reset_col:
        if st.button("Reset to Acme defaults"):
            st.session_state["prompt_config"] = dict(DEFAULT_PROMPTS)
            st.success("Default prompts restored.")
            st.rerun()
    with download_col:
        preset_bytes = json.dumps(prompts, indent=2).encode("utf-8")
        st.download_button(
            "Download prompt preset",
            data=preset_bytes,
            file_name="acme_image_prompt_preset.json",
            mime="application/json",
        )

    with st.expander("Show a rendered prompt preview"):
        preview_sector = st.text_input("Preview sector", value="Automobile")
        preview_concept = {
            "concept_name": "Modern vehicle manufacturing",
            "base_scene": "Robotic automobile assembly line",
            "main_subject": "partially assembled passenger vehicle",
            "environment": "clean modern manufacturing plant",
            "camera_composition": "wide editorial three-quarter angle",
            "distinguishing_feature": "precision robotic arms and production depth",
        }
        try:
            st.code(
                generic_prompt(
                    preview_sector,
                    preview_concept,
                    "Premium realistic Indian business-editorial aesthetic.",
                    prompts,
                )
            )
        except Exception as exc:
            st.error(str(exc))

with queue_tab:
    st.subheader("Persistent job queue")
    st.write(
        "Jobs remain saved if the app closes. Failed or interrupted jobs can be retried; existing output files are skipped."
    )
    queue_placeholder = st.empty()
    queue_placeholder.dataframe(jobs_dataframe(), width="stretch", hide_index=True)

    st.caption(
        "Recovery is file-based: plan.json, master images and completed PNG files "
        "are reused. Only missing files are generated again."
    )
    run_col, retry_col, recover_col, clear_col = st.columns(4)
    with run_col:
        jobs_per_run = st.number_input(
            "Maximum jobs to run now",
            min_value=1,
            max_value=500,
            value=10,
            step=1,
        )
        if st.button("Run pending jobs", type="primary"):
            run_pending_jobs(max_workers=max_workers, job_limit=int(jobs_per_run))
            st.rerun()
    with retry_col:
        st.write("")
        st.write("")
        if st.button("Retry all failed jobs"):
            count = set_failed_to_pending()
            st.success(f"Moved {count} failed job(s) back to pending.")
            st.rerun()
    with recover_col:
        st.write("")
        st.write("")
        if st.button(
            "Recover interrupted running jobs",
            help="First stop/restart the old queue runner. This moves jobs stuck in Running back to Pending.",
        ):
            count = set_running_to_pending()
            st.success(
                f"Recovered {count} running job(s). Click Run pending jobs to continue."
            )
            st.rerun()
    with clear_col:
        st.write("")
        st.write("")
        if st.button("Remove completed jobs from queue"):
            count = delete_completed_jobs()
            st.success(f"Removed {count} completed queue record(s). Output files were not deleted.")
            st.rerun()

    st.subheader("Completed downloads")
    completed_jobs = list_jobs("completed")
    if not completed_jobs:
        st.info("No completed jobs yet.")
    else:
        for row in completed_jobs:
            output_dir = Path(row["output_dir"])
            zip_candidates = list(output_dir.glob("*_image_library.zip"))
            with st.expander(f"{row['sector']} — {row['job_type']} — {row['id']}"):
                st.code(str(output_dir.resolve()))
                previews = sorted(output_dir.glob("main_*.png"))[:8]
                if previews:
                    cols = st.columns(4)
                    for index, path in enumerate(previews):
                        with cols[index % 4]:
                            st.image(str(path), caption=path.name, width="stretch")
                if zip_candidates:
                    zip_path = zip_candidates[0]
                    st.download_button(
                        "Download ZIP",
                        data=zip_path.read_bytes(),
                        file_name=zip_path.name,
                        mime="application/zip",
                        key=f"download_{row['id']}",
                    )
