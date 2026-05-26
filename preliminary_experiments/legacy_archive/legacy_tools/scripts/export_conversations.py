"""
Export Claude Code conversation transcripts from JSONL to readable Markdown.

Usage:
    python scripts/export_conversations.py                    # Export all sessions
    python scripts/export_conversations.py --session <uuid>   # Export one session
    python scripts/export_conversations.py --latest           # Export most recent session

Output goes to conversation_history/ directory.
"""

import json
import os
import sys
import glob
from datetime import datetime
from pathlib import Path

JSONL_DIR = os.path.expanduser(
    "~/.claude/projects/-Users-chrishe-Downloads-PokerRL-Vanilla"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "conversation_history",
)


def extract_text_from_content(content):
    """Extract readable text from message content (string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool = block.get("name", "unknown")
                    inp = block.get("input", {})
                    if tool == "Read":
                        texts.append(f"*[Reading {inp.get('file_path', '?')}]*")
                    elif tool == "Write":
                        texts.append(f"*[Writing {inp.get('file_path', '?')}]*")
                    elif tool == "Edit":
                        texts.append(f"*[Editing {inp.get('file_path', '?')}]*")
                    elif tool == "Bash":
                        cmd = inp.get("command", "?")
                        if len(cmd) > 100:
                            cmd = cmd[:100] + "..."
                        texts.append(f"*[Running: `{cmd}`]*")
                    elif tool == "Grep":
                        texts.append(
                            f"*[Searching for '{inp.get('pattern', '?')}']*"
                        )
                    elif tool == "Glob":
                        texts.append(
                            f"*[Finding files: {inp.get('pattern', '?')}]*"
                        )
                    elif tool == "Task":
                        desc = inp.get("description", "?")
                        texts.append(f"*[Launching agent: {desc}]*")
                    elif tool == "TodoWrite":
                        pass  # Skip todo updates in export
                    else:
                        texts.append(f"*[Using tool: {tool}]*")
                elif block.get("type") == "tool_result":
                    pass  # Skip tool results (too verbose)
        return "\n".join(texts)
    return str(content)


def convert_session(jsonl_path, output_path):
    """Convert a single JSONL session to markdown."""
    messages = []
    session_meta = {}

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type", "")

            if msg_type == "user":
                content = obj.get("message", {}).get("content", "")
                text = extract_text_from_content(content)
                if text.strip():
                    messages.append(("user", text.strip()))
                if not session_meta:
                    session_meta = {
                        "session_id": obj.get("sessionId", "unknown"),
                        "slug": obj.get("slug", ""),
                        "branch": obj.get("gitBranch", ""),
                        "version": obj.get("version", ""),
                    }

            elif msg_type == "assistant":
                content = obj.get("message", {}).get("content", "")
                text = extract_text_from_content(content)
                if text.strip():
                    messages.append(("assistant", text.strip()))

    if not messages:
        return False

    # Get file modification time for the date
    mtime = os.path.getmtime(jsonl_path)
    date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    date_short = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    # Build markdown
    slug = session_meta.get("slug", "session")
    md_lines = [
        f"# Conversation: {slug}",
        f"",
        f"> **Date**: {date_str}",
        f"> **Session ID**: {session_meta.get('session_id', 'unknown')}",
        f"> **Branch**: {session_meta.get('branch', 'unknown')}",
        f"> **Claude Code Version**: {session_meta.get('version', 'unknown')}",
        f"",
        f"---",
        f"",
    ]

    for role, text in messages:
        if role == "user":
            md_lines.append(f"## User")
            md_lines.append(f"")
            md_lines.append(text)
            md_lines.append(f"")
        else:
            md_lines.append(f"## Assistant")
            md_lines.append(f"")
            md_lines.append(text)
            md_lines.append(f"")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(md_lines))

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export Claude Code conversations")
    parser.add_argument("--session", help="Export a specific session by UUID")
    parser.add_argument(
        "--latest", action="store_true", help="Export only the most recent session"
    )
    args = parser.parse_args()

    jsonl_files = sorted(
        glob.glob(os.path.join(JSONL_DIR, "*.jsonl")),
        key=os.path.getmtime,
    )

    if not jsonl_files:
        print("No conversation files found.")
        return

    if args.session:
        jsonl_files = [f for f in jsonl_files if args.session in f]
        if not jsonl_files:
            print(f"Session {args.session} not found.")
            return
    elif args.latest:
        jsonl_files = [jsonl_files[-1]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exported = 0

    for jsonl_path in jsonl_files:
        session_id = Path(jsonl_path).stem
        mtime = os.path.getmtime(jsonl_path)
        date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

        # Read slug from first user message
        slug = "session"
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if obj.get("type") == "user" and obj.get("slug"):
                        slug = obj["slug"]
                        break
                except (json.JSONDecodeError, KeyError):
                    continue

        # Use short session ID suffix to avoid collisions
        short_id = session_id[:8]
        output_name = f"{date_str}_{slug}_{short_id}.md"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        if convert_session(jsonl_path, output_path):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"  Exported: {output_name} ({size_kb:.0f} KB)")
            exported += 1

    print(f"\n{exported} sessions exported to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
