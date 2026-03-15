
import io
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/drive"]  # needed for download+upload+delete


ManifestItem = Tuple[str, str, str]  # (file_id, file_name, local_path)


# ----------------------------
# Drive helpers
# ----------------------------
def build_drive_service(service_account_json_path: str):
    creds = service_account.Credentials.from_service_account_file(
        service_account_json_path, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_or_create_folder_id(drive, folder_name: str, parent_id: str = None) -> str:
    q = (
        "mimeType='application/vnd.google-apps.folder' "
        f"and name='{folder_name}' and trashed=false"
    )
    if parent_id:
        q += f" and '{parent_id}' in parents"

    res = drive.files().list(
        q=q,
        fields="files(id, name)",
        pageSize=50,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    folders = res.get("files", [])

    if len(folders) == 1:
        return folders[0]["id"]

    if len(folders) > 1:
        ids = ", ".join([f["id"] for f in folders])
        raise RuntimeError(f"Multiple folders named '{folder_name}' found. IDs: {ids}")

    metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        metadata["parents"] = [parent_id]

    created = drive.files().create(
        body=metadata, fields="id", supportsAllDrives=True
    ).execute()
    print(f"Created folder '{folder_name}'")
    return created["id"]


def list_files_in_folder(drive, folder_id: str) -> List[Dict]:
    files: List[Dict] = []
    page_token = None
    while True:
        res = drive.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files.extend(res.get("files", []))
        page_token = res.get("nextPageToken")
        if not page_token:
            break
    return files


def download_file(drive, file_id: str, out_path: str):
    request = drive.files().get_media(fileId=file_id, supportsAllDrives=True)
    with io.FileIO(out_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def upload_file(drive, local_path: str, dest_folder_id: str, filename: str) -> str:
    media = MediaFileUpload(local_path, resumable=True)
    body = {"name": filename, "parents": [dest_folder_id]}
    try:
        created = drive.files().create(
            body=body,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()
        return created["id"]
    except HttpError as e:
        msg = str(e)
        if "storageQuotaExceeded" in msg:
            raise RuntimeError(
                "Drive upload failed: service account has no storage quota. "
                "Use a Shared Drive and add this service account as a member, "
                "or use OAuth delegation."
            ) from e
        raise


def delete_file(drive, file_id: str):
    drive.files().delete(fileId=file_id, supportsAllDrives=True).execute()


# ----------------------------
# Config helpers
# ----------------------------
def read_config_download_path(config_path: str) -> str:
    """
    config.json example:
    {
      "download_path": "/tmp/pgtele_downloads"
    }
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    download_path = cfg.get("download_path") or os.getenv("DRIVE_DOWNLOAD_PATH")
    if not download_path or not isinstance(download_path, str):
        # Fallback to a safe default under the config directory
        download_path = os.path.join(os.path.dirname(config_path), "drive_downloads")

    if not os.path.isabs(download_path):
        download_path = os.path.join(os.path.dirname(config_path), download_path)

    os.makedirs(download_path, exist_ok=True)
    return download_path


# ----------------------------
# Context (all values shared across functions)
# ----------------------------
@dataclass(frozen=True)
class DriveContext:
    drive: any
    source_folder_name: str
    dest_folder_name: str
    parent_folder_id: Optional[str] = None
    config_json_path: str = "config.json"


# ----------------------------
# Your 3 pipeline functions
# ----------------------------
def download_pgtele(ctx: DriveContext) -> List[ManifestItem]:
    """
    Downloads all files from source folder to local download_path from config.json.
    Returns manifest list for later delete.
    """
    download_path = read_config_download_path(ctx.config_json_path)

    src_folder_id = get_or_create_folder_id(
        ctx.drive, ctx.source_folder_name, parent_id=ctx.parent_folder_id
    )

    files = list_files_in_folder(ctx.drive, src_folder_id)
    if not files:
        print("No files found in source folder.")
        return []

    manifest: List[ManifestItem] = []
    for f in files:
        if f["mimeType"].startswith("application/vnd.google-apps."):
            print(f"Skipping Google workspace file: {f['name']} ({f['mimeType']})")
            continue

        local_path = os.path.join(download_path, f["name"])
        print(f"Downloading: {f['name']} -> {local_path}")
        download_file(ctx.drive, f["id"], local_path)
        manifest.append((f["id"], f["name"], local_path))

    return manifest


def upload_video(ctx: DriveContext, output_video_path: str) -> str:
    """
    Uploads ONE local file to destination folder.
    Returns 'ok' on success.
    """
    if not output_video_path or not os.path.exists(output_video_path):
        raise FileNotFoundError(f"output_video_path not found: {output_video_path}")

    dst_folder_id = get_or_create_folder_id(
        ctx.drive, ctx.dest_folder_name, parent_id=ctx.parent_folder_id
    )

    filename = os.path.basename(output_video_path)
    print(f"Uploading: {output_video_path} -> '{ctx.dest_folder_name}' as '{filename}'")
    _new_id = upload_file(ctx.drive, output_video_path, dst_folder_id, filename)
    return "ok"


def delete_from_pgtele(ctx: DriveContext, manifest: List[ManifestItem]) -> str:
    """
    Deletes the original files from source folder using manifest file_ids.
    Returns 'ok' when done.
    """
    if not manifest:
        print("Manifest empty; nothing to delete.")
        return "ok"

    print(f"Deleting {len(manifest)} files from '{ctx.source_folder_name}'...")
    for file_id, name, _local_path in manifest:
        print(f"Deleting: {name} ({file_id})")
        delete_file(ctx.drive, file_id)

    return "ok"


# ----------------------------
# main (3 variables + call flow)
# ----------------------------
def main():
    # =========================
    # ONLY 3 REQUIRED VARIABLES
    # =========================
    SERVICE_ACCOUNT_JSON = "scibugai-0c6edfa92c76.json"
    SOURCE_FOLDER_NAME = "pgtele"
    DEST_FOLDER_NAME = "pguploadvideo"

    # Optional but recommended:
    
    PARENT_FOLDER_ID = os.getenv("pgfolder")
    CONFIG_JSON_PATH = "config.json"
    # =========================

    if not os.path.exists(SERVICE_ACCOUNT_JSON):
        raise FileNotFoundError(f"Service account json not found: {SERVICE_ACCOUNT_JSON}")

    drive = build_drive_service(SERVICE_ACCOUNT_JSON)

    ctx = DriveContext(
        drive=drive,
        source_folder_name=SOURCE_FOLDER_NAME,
        dest_folder_name=DEST_FOLDER_NAME,
        parent_folder_id=PARENT_FOLDER_ID,
        config_json_path=CONFIG_JSON_PATH,
    )

    # 1) download everything from pgtele
    manifest = download_pgtele(ctx)

    # 2) upload ONE output video (example: first downloaded file)
    # Replace this with your real output video path
    if manifest:
        first_local_path = manifest[0][2]
        status = upload_video(ctx, first_local_path)
        print("upload status:", status)

        # 3) delete originals from pgtele AFTER successful upload
        if status == "ok":
            del_status = delete_from_pgtele(ctx, manifest)
            print("delete status:", del_status)
    else:
        print("No files downloaded; skipping upload/delete.")


if __name__ == "__main__":
    main()
