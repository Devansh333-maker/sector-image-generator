# Acme Sector Image Library Generator V2.2

V2.2 includes two production modes and an editable Prompt Studio.

## Production modes

1. **Generate New Sector Packs**
   - Paste one or many sectors.
   - Generate 1–5 concepts per sector.
   - Each concept creates Generic, Bullish, Neutral and Bearish assets.

2. **Complete Existing Sector Packs**
   - Upload PNG/JPG/WEBP files or a ZIP.
   - Confirm the sector, concept number and existing sentiment.
   - Generate only the missing variations.

3. **Prompt Studio**
   - Edit the concept-planning prompt.
   - Edit the generic master-image prompt.
   - Edit the shared image-edit prompt.
   - Edit Bullish, Neutral, Bearish and Generic transformation rules.
   - Edit strict exclusions.
   - Preview a rendered prompt.
   - Download/upload prompt presets as JSON.

4. **Queue & Downloads**
   - Persistent local queue when running on your own computer.
   - Retry failed jobs and recover interrupted jobs.
   - Download completed sector packs as ZIP files.

## Local Windows setup

1. Run `setup_windows.bat` once.
2. Add your OpenAI API key to `.env`.
3. Run `start_app.bat`.
4. Open `http://localhost:8501` or the port configured in `start_app.bat`.

## Cloud deployment

Read `DEPLOY_STREAMLIT_CLOUD.md`.

## Download behaviour

When deployed, click **Download ZIP** to save images to the device currently using the browser. The server cannot silently save into a local folder on the user's computer because browsers block that for security.
