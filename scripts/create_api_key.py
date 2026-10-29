"""Manage API keys. The key is printed once; only its hash is stored.

    python scripts/create_api_key.py mentor                 # full key: search + add images
    python scripts/create_api_key.py lovable --scope web    # search-only, for a public frontend
    python scripts/create_api_key.py --list
    python scripts/create_api_key.py --revoke mentor

Each new key also prints a "label:scope:hash" entry: put entries (joined with ';')
into the VS_API_KEYS secret on hosts where data/api_keys.json is not deployed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.auth import SCOPES, KeyStore, hash_key  # noqa: E402
from app.config import Settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label", nargs="?", help="who the key is for, e.g. mentor")
    parser.add_argument("--scope", choices=SCOPES, default="full")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--revoke", metavar="LABEL")
    args = parser.parse_args()

    keys = KeyStore(Settings.from_env().keys_path)
    if args.list:
        for entry in keys.list():
            state = "revoked" if entry["revoked"] else "active"
            print(f"{entry['label']:<20} {entry.get('scope', 'full'):<5} {state:<8} created {entry['created_at']}")
    elif args.revoke:
        print("Revoked." if keys.revoke(args.revoke) else f"No active key labelled '{args.revoke}'.")
    elif args.label:
        try:
            key = keys.create(args.label, args.scope)
        except ValueError as exc:
            sys.exit(str(exc))
        print(f"API key for '{args.label}' ({args.scope}), shown only once, store it now:\n\n{key}\n")
        print(f"VS_API_KEYS entry:\n{args.label}:{args.scope}:{hash_key(key)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
