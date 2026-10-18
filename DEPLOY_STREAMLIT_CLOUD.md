# Deploy Acme Sector Image Generator V2.2 on Streamlit Community Cloud

## What this cloud version does

- Generates new sector packs.
- Completes missing variations from existing images.
- Includes Prompt Studio for changing planner, generic-image and sentiment-edit prompts.
- Lets users download completed ZIP files to the device currently using the browser.
- Lets users download/upload prompt presets as JSON.

## Important storage behaviour

The deployed app generates files on the Streamlit server. The user must click **Download ZIP** to save them to the laptop or device being used. A browser app cannot silently write to an arbitrary folder on the user's device.

Free Streamlit hosting may restart or clear its temporary server storage. Download completed ZIP files promptly. Prompt presets should also be downloaded as JSON if you want to preserve custom prompt settings.

## Files to upload to GitHub

Upload the full project except local/private files. The included `.gitignore` excludes:

- `.env`
- `.venv/`
- `outputs/`
- `data/`
- Python cache files

Never upload your real `.env` file or API key.

## GitHub steps

1. Create a private GitHub repository.
2. Upload all files in this folder.
3. Confirm `.env` is not present in GitHub.
4. Commit the files to the `main` branch.

## Streamlit Cloud steps

1. Sign in to Streamlit Community Cloud using GitHub.
2. Click **Create app**.
3. Select your repository and the `main` branch.
4. Set the entrypoint to `app.py`.
5. Open **Advanced settings**.
6. Select Python 3.12.
7. Paste the following under Secrets:

```toml
OPENAI_API_KEY = "your-real-api-key"
OPENAI_TEXT_MODEL = "gpt-5.6"
OPENAI_IMAGE_MODEL = "gpt-image-2"
OPENAI_IMAGE_SIZE = "832x816"
```

8. Deploy the app.
9. Keep the app private and share access only with approved team members.

## First cloud test

1. Open Prompt Studio and leave the defaults unchanged.
2. Add one sector with one concept and Low quality.
3. Run the job.
4. Download the ZIP to your device.
5. Refresh the app and confirm it still loads.
6. Then test a complete five-concept sector.

## Prompt Studio usage

- Changes apply only after clicking **Save prompts for new jobs**.
- Jobs already queued keep the prompt snapshot they were created with.
- Download the prompt preset JSON to your local device.
- Upload the JSON later to restore the same prompt configuration.
