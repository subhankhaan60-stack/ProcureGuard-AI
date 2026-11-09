"""Create the first ProcureGuard administrator interactively.

Run from the repository root:
    python scripts/create_admin.py

This bootstrap command is intended for a trusted local/server console only.
It never prints or stores the plaintext password.
"""

import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings
from core.database import Database, User


def main() -> int:
    db = Database(db_path=settings.db_path)
    with db.session() as session:
        existing_admin = session.query(User).filter(User.role == "ADMIN").first()
    if existing_admin:
        print("An administrator account already exists. Use the authenticated admin API to add users.")
        return 1

    email = input("Administrator email: ").strip()
    full_name = input("Full name: ").strip()
    password = getpass.getpass("Password (minimum 12 characters): ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        print("Passwords do not match.")
        return 1

    try:
        user = db.create_user(
            email=email,
            full_name=full_name,
            password=password,
            role="ADMIN",
        )
    except ValueError as exc:
        print(f"Unable to create administrator: {exc}")
        return 1

    print(f"Administrator account created for {user['email']} (user ID {user['id']}).")
    print("Keep USER_SESSION_SECRET and API_AUTH_TOKEN outside source control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
