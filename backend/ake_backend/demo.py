"""Opt-in demo seeding against a running server, using a user-supplied PDF."""

import argparse
import asyncio
import getpass
import json
import os
from pathlib import Path
from uuid import uuid4

import httpx


async def seed_demo(
    client: httpx.AsyncClient,
    email: str,
    password: str,
    filename: str,
    data: bytes,
    poll_seconds: float = 2,
    max_polls: int = 300,
) -> dict:
    if not data.startswith(b"%PDF-"):
        raise ValueError("The demo input must be a PDF")
    login = await client.post("auth/login", json={"email": email, "password": password})
    login.raise_for_status()
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    base = await client.post(
        "knowledge-bases", json={"name": "Demo: Attention Is All You Need"}, headers=headers
    )
    base.raise_for_status()
    base_id = base.json()["id"]
    upload = await client.post(
        f"knowledge-bases/{base_id}/documents",
        files={"file": (filename, data, "application/pdf")},
        headers={**headers, "Idempotency-Key": str(uuid4())},
    )
    upload.raise_for_status()
    document_id = upload.json()["id"]
    for _ in range(max_polls):
        response = await client.get(f"documents/{document_id}/status", headers=headers)
        response.raise_for_status()
        status = response.json()["status"]
        if status == "ready":
            return {"knowledge_base_id": base_id, "document_id": document_id, "status": status}
        if status in {"failed", "cancelled"}:
            raise RuntimeError(
                f"Indexing ended with status {status}; inspect the document status in the application"
            )
        await asyncio.sleep(poll_seconds)
    raise TimeoutError("Indexing did not finish before the demo seeding deadline")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a user-supplied open paper to a running AI Knowledge Engine"
    )
    parser.add_argument("pdf", type=Path, help="Path to a legally obtained PDF")
    parser.add_argument("--api", default=os.getenv("AKE_API", "http://127.0.0.1:8000/api/v1/"))
    parser.add_argument("--email", default=os.getenv("AKE_EMAIL"))
    args = parser.parse_args()
    email = args.email or input("Account email: ")
    password = getpass.getpass("Account password: ")
    data = args.pdf.read_bytes()
    async with httpx.AsyncClient(base_url=args.api.rstrip("/") + "/", timeout=60) as client:
        result = await seed_demo(client, email, password, args.pdf.name, data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
